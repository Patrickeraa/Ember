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
import ember
import io
import dist_data_pb2
import dist_data_pb2_grpc
import modelFile
import queue
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import time

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
    max_message_length = 1000 * 1024 * 1024
    channel = grpc.insecure_channel(
        f'{api_host}:{api_port}',
        options=[
            ('grpc.max_send_message_length', max_message_length),
            ('grpc.max_receive_message_length', max_message_length)
        ]
    )
    
    stub = dist_data_pb2_grpc.TrainLoaderServiceStub(channel)
    
    batch_idx = 0

    while True:
        request = dist_data_pb2.BatchRequest(
            num_replicas=num_replicas,
            rank=rank,
            batch_idx=batch_idx,
            batch_size=10000
        )

        response = stub.GetBatch(request)

        if not response.data:
            batch_queue.put(None)
            break

        batch = pickle.loads(response.data)

        converted_batch = []
        for img_data, label_data in batch:
            # Deserialize PNG back to Tensor directly
            img_bytes = io.BytesIO(img_data)
            img = Image.open(img_bytes)

            # Convert image directly into tensor without applying redundant transformations
            img_tensor = transforms.functional.to_tensor(img)  # Only basic conversion
            label_tensor = torch.tensor(label_data)

            converted_batch.append((img_tensor, label_tensor))

        batch_queue.put(converted_batch)
        new_data_available = True
        batch_idx += 1

def create_dataloader_from_queue(batch_queue, stop_signal=None):
    while not batch_queue.empty():
        batch = batch_queue.get()
        if batch is None:
            break
        for img_tensor, label in batch:
            all_images.append(img_tensor)
            all_labels.append(label)

    if all_images and all_labels:
        images_tensor = torch.stack(all_images)
        labels_tensor = torch.tensor(all_labels)
        dataset = TensorDataset(images_tensor, labels_tensor)
        dataloader = DataLoader(dataset, batch_size=200, shuffle=True, pin_memory=True, num_workers=0)
        return dataloader
    else:
        return None

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
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
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
    while epoch_counter < args.epochs:

        if new_data_available or epoch_counter == 0:
            print("Creating new dataloader")
            start = datetime.now()
            dataloader = create_dataloader_from_queue(batch_queue)
            new_data_available = False
            print("Dataloader created in: ", datetime.now() - start)

        if dataloader:
            # ---------------------- ALLREDUCE STEP -----------------------------
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
            print(f"Epoch {epoch_counter + 1}/{args.epochs}, Min Batch Count Across Nodes: {min_batch_count}")
            total_step = len(truncated_batches)

            for i, (images, labels) in enumerate(truncated_batches):
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)

                outputs = model(images)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i + 1) % 100 == 0 or i == total_step - 1:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                        epoch_counter + 1, args.epochs, i + 1, total_step, loss.item()))

            print(f"End of epoch {epoch_counter + 1}/{args.epochs}")
            epoch_counter += 1
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
                
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
        
        accuracy = 100 * correct / total
        average_loss = total_loss / total
        precision = precision_score(all_labels, all_predictions, average='weighted')
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        print('Test Accuracy of the model on the test images: {} %'.format(accuracy))
        print('Test Loss: {:.4f}'.format(average_loss))
        print('Test Precision: {:.4f}'.format(precision))
        print('Test F1 Score: {:.4f}'.format(f1))

if __name__ == '__main__':
    main()
