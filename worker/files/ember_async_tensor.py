import os
from datetime import datetime
import argparse
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.distributed as dist
from apex.parallel import DistributedDataParallel as DDP
from apex import amp
from sklearn.metrics import precision_score, f1_score
from tqdm import tqdm
import threading
import io
import grpc
import pickle
from rpc import dist_data_pb2, dist_data_pb2_grpc
from utils import ember, modelFile
import queue
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import time
from sklearn.metrics import accuracy_score
import json
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader, DistributedSampler

def calculate_top_k_accuracy(outputs, labels, k=1):
    """Compute Top-k accuracy for the given outputs and labels."""
    _, top_k_predictions = outputs.topk(k, dim=1, largest=True, sorted=True)
    correct = top_k_predictions.eq(labels.view(-1, 1).expand_as(top_k_predictions))
    return correct.any(dim=1).float().sum().item()

#gpu logic
gpu_data = []
import threading
import csv
import time 
import pynvml

batch_queue = queue.Queue()
all_images = []
all_labels = []
new_data_available = False

def gpu_monitoring_thread(interval, gpu_data):
    global stop_monitoring
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()

    while not stop_monitoring:
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_data.append((time.time(), i, utilization.gpu))
        time.sleep(interval)
    
    pynvml.nvmlShutdown()

def fetch_batches_in_thread(api_host, api_port, num_replicas, rank):
    global new_data_available
    max_msg = 1000 * 1024 * 1024
    channel = grpc.insecure_channel(
        f'{api_host}:{api_port}',
        options=[
            ('grpc.max_send_message_length', max_msg),
            ('grpc.max_receive_message_length', max_msg)
        ]
    )
    stub = dist_data_pb2_grpc.TrainLoaderServiceStub(channel)

    batch_idx = 0
    while True:
        request = dist_data_pb2.BatchRequest(
            num_replicas=num_replicas,
            rank=rank,
            batch_idx=batch_idx,
            batch_size=5000
        )
        response = stub.GetBatch(request)

        # fim das batches
        if not response.data:
            batch_queue.put(None)
            break

        raw = pickle.loads(response.data)
        converted_batch = []
        for tensor_bytes, label_data in raw:
            buf = io.BytesIO(tensor_bytes)
            # desserializa exato torch.Tensor
            img_tensor = torch.load(buf)
            label_tensor = torch.tensor(label_data)
            converted_batch.append((img_tensor, label_tensor))

        batch_queue.put(converted_batch)
        new_data_available = True
        batch_idx += 1

def create_dataloader_from_queue(batch_queue, num_replicas, rank, batch_size=100):
    # 1) Inicializa o buffer persistente na primeira chamada
    if not hasattr(create_dataloader_from_queue, "images"):
        create_dataloader_from_queue.images = []
        create_dataloader_from_queue.labels = []

    # 2) Move tudo da queue pro buffer, mas não zera o buffer
    while True:
        try:
            batch = batch_queue.get_nowait()
        except queue.Empty:
            break
        if batch is None:
            break
        for img_tensor, label_tensor in batch:
            create_dataloader_from_queue.images.append(img_tensor)
            create_dataloader_from_queue.labels.append(label_tensor)

    # 3) Empilha o buffer inteiro em tensores
    if create_dataloader_from_queue.images:
        images_tensor = torch.stack(create_dataloader_from_queue.images)
        labels_tensor = torch.stack(create_dataloader_from_queue.labels)
    else:
        # buffer ainda vazio → dataset vazio
        images_tensor = torch.empty((0, 3, 32, 32))
        labels_tensor = torch.empty((0,), dtype=torch.long)

    dataset = TensorDataset(images_tensor, labels_tensor)

    # 4) Cria sampler distribuído sobre todo o dataset acumulado
    sampler = DistributedSampler(
        dataset,
        num_replicas=num_replicas,
        rank=rank,
        shuffle=True,
        drop_last=False
    )

    # 5) DataLoader
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=True
    )

    return dataloader, sampler

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the JSON configuration file')
    args = ember.set_ambient(parser.parse_args().config)
    os.environ['MASTER_ADDR'] = 'grworker1'
    os.environ['MASTER_PORT'] = '8888'
    start = datetime.now()
    # GPU monitoring
    global stop_monitoring
    stop_monitoring = False
    monitor_thread = threading.Thread(target=gpu_monitoring_thread, args=(0.1, gpu_data))
    monitor_thread.start()
    try:
        mp.spawn(train, nprocs=args.gpus, args=(args,))
    finally:
        stop_monitoring = True
        monitor_thread.join()
    with open("gpu_utilization_async.csv", "w") as file:
        writer = csv.writer(file)
        writer.writerow(["Time", "GPU_ID", "Utilization"])
        writer.writerows(gpu_data)
    print("Training complete in: " + str(datetime.now() - start))


def train(gpu, args):
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)
    torch.manual_seed(0)
    model = modelFile.getModel()
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    batch_size = 400
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    # Data loading code
    print("--------- TRAINING STATUS --------")
    print("World Size: ", args.world_size)
    print("Number of GPU's: ", args.gpus)
    print("Machine Rank: ", args.nr)
    print("Number of Epochs in the Training: ", args.epochs)
    print("Number of Nodes: ", args.nodes)
    print("----------------------------------")
    api_host = "grserver"  
    api_port = 8040
    fetch_thread = threading.Thread(
        target=fetch_batches_in_thread, 
        args=(api_host, api_port, args.world_size, rank)
    )
    fetch_thread.start()
    epoch_counter = 0
    global new_data_available
    loop_check = 0

    epoch_losses = []
    epoch_accuracies = []

    # Define the filename to save training metrics
    metrics_filename = "async_metrics.json"
    writer = SummaryWriter(log_dir="runs/async")

    while epoch_counter < args.epochs:

        #if new_data_available or epoch_counter == 0:
        #print("Creating new dataloader")

        dataloader, sampler = create_dataloader_from_queue(batch_queue, args.world_size, rank)
        new_data_available = False
        #print("Dataloader created in: ", datetime.now() - start)

        if dataloader:
            # ---------------------- ALLREDUCE STEP -----------------------------
            epoch_start_time = time.time()
            local_batch_count = len(dataloader)
            local_batch_count_tensor = torch.tensor(
                local_batch_count, dtype=torch.int, device=torch.device(f'cuda:{gpu}')
            )
            min_batch_count_tensor = local_batch_count_tensor.clone()
            dist.all_reduce(min_batch_count_tensor, op=dist.ReduceOp.MIN)
            min_batch_count = min_batch_count_tensor.item()
            if local_batch_count != min_batch_count:
                truncated_batches = []
                for i, batch in enumerate(dataloader):
                    if i >= min_batch_count:
                        break
                    truncated_batches.append(batch)
            else:
                truncated_batches = dataloader
            # -------------------------------------------------------------------
            #print("Allreduce in: ", datetime.now() - start)
            #print(f"Epoch {epoch_counter + 1}/{args.epochs}, Min Batch Count Across Nodes: {min_batch_count}")
            total_step = len(truncated_batches)

            # epoch metrics
            epoch_loss = 0
            correct_predictions = 0
            total_predictions = 0
            train_loss_sum = 0.0
            train_total = 0
            train_correct_top1 = 0
            train_correct_top5 = 0

            sampler.set_epoch(epoch_counter)
            for images, labels in tqdm(truncated_batches, desc=f"Train Epoch {epoch_counter}"):
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)

                outputs = model(images)
                loss = criterion(outputs, labels)
                epoch_loss += loss.item() 

                # Update accuracy
                _, predicted = outputs.max(1)
                correct_predictions += predicted.eq(labels).sum().item()
                total_predictions += labels.size(0)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Calculate average loss and accuracy for the epoch
            epoch_loss /= total_step
            epoch_accuracy = 100.0 * correct_predictions / total_predictions
            epoch_losses.append(epoch_loss)
            epoch_accuracies.append(epoch_accuracy)

            batch_size = images.size(0)
            train_loss_sum += loss.item() * batch_size
            train_total += batch_size
            _, pred_top1 = outputs.topk(1, dim=1)
            train_correct_top1 += (pred_top1.squeeze() == labels).sum().item()
            _, pred_top5 = outputs.topk(5, dim=1)
            train_correct_top5 += (pred_top5 == labels.unsqueeze(1)).sum().item()

            train_loss_avg = train_loss_sum / train_total
            train_top1_acc = train_correct_top1 / train_total
            train_top5_acc = train_correct_top5 / train_total

            writer.add_scalar('Loss/Train', train_loss_avg, epoch_counter)
            writer.add_scalar('Accuracy/Top1', train_top1_acc, epoch_counter)
            writer.add_scalar('Accuracy/Top5', train_top5_acc, epoch_counter)

            #print(f"End of epoch {epoch_counter + 1}/{args.epochs}")
            epoch_counter += 1
            epoch_duration = time.time() - epoch_start_time
            writer.add_scalar('Time/Epoch', epoch_duration, epoch_counter)
            # Save metrics to file after each epoch
            #with open(metrics_filename, "w") as file:
            #    json.dump({"losses": epoch_losses, "accuracies": epoch_accuracies}, file)

        else:
            print("No new data available at this moment. ", loop_check)
            loop_check += 1
            time.sleep(1)
    if gpu == 0:
        print("Training complete")
    if rank == 0:
        print("Teste gpu 0")

        # Test loading code
        test_loader = ember.fetch_test_loader(api_host="grserver", api_port="8040")
        model.eval()
        
        all_labels = []
        all_predictions = []
        total_loss = 0
        total_samples = 0

        total_top1_correct = 0
        total_top5_correct = 0

        
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                            # Top-1 and Top-5 accuracy
                total_top1_correct += calculate_top_k_accuracy(outputs, labels, k=1)
                total_top5_correct += calculate_top_k_accuracy(outputs, labels, k=5)
           

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
        
        accuracy = 100 * correct / total
        average_loss = total_loss / total
        precision = precision_score(all_labels, all_predictions, average='weighted')
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        top1_accuracy = 100 * total_top1_correct / total
        top5_accuracy = 100 * total_top5_correct / total
        top5_error = 100 - top5_accuracy
        top1_error = 100 - top1_accuracy

        filename = "metrics_async.txt"
        with open(filename, "w") as file:
            print('Test Accuracy of the model on the test images: {} %'.format(accuracy))
            file.write('Test Accuracy of the model on the test images: {} %\n'.format(accuracy))
            
            print('Test Loss: {:.4f}'.format(average_loss))
            file.write('Test Loss: {:.4f}\n'.format(average_loss))
            
            print('Test Precision: {:.4f}'.format(precision))
            file.write('Test Precision: {:.4f}\n'.format(precision))
            
            print('Test F1 Score: {:.4f}'.format(f1))
            file.write('Test F1 Score: {:.4f}\n'.format(f1))
            
            print('Test Top-1 Accuracy: {:.4f} %'.format(top1_accuracy))
            file.write('Test Top-1 Accuracy: {:.4f} %\n'.format(top1_accuracy))
            
            print('Test Top-5 Accuracy: {:.4f} %'.format(top5_accuracy))
            file.write('Test Top-5 Accuracy: {:.4f} %\n'.format(top5_accuracy))
            
            print('Test Top-5 Error: {:.4f} %'.format(top5_error))
            file.write('Test Top-5 Error: {:.4f} %\n'.format(top5_error))
            
            print('Test Top-1 Error: {:.4f} %'.format(top1_error))
            file.write('Test Top-1 Error: {:.4f} %\n'.format(top1_error))
    writer.close()
    

if __name__ == '__main__':
    main()
