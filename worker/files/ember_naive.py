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
import time
import io
import grpc
import pickle
import io
from rpc import dist_data_pb2, dist_data_pb2_grpc
from utils import ember, modelFile
from torch.utils.tensorboard import SummaryWriter
import pynvml
from pyinstrument import Profiler
from torch.profiler import profile, record_function, ProfilerActivity, schedule
from torch.profiler import tensorboard_trace_handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the JSON configuration file')
    args = ember.set_ambient(parser.parse_args().config)
    os.environ['MASTER_ADDR'] = 'grworker1'
    os.environ['MASTER_PORT'] = '8888'
    start = datetime.now()
    mp.spawn(train, nprocs=args.gpus, args=(args,))
    print("Training complete in: " + str(datetime.now() - start))


def train(gpu, args):
    train_start = datetime.now()
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)
    torch.manual_seed(0)
    model = modelFile.getModel()
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    batch_size = 100
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu],output_device=gpu, static_graph=True)

    # Data loading code
    print("--------- TRAINING STATUS --------")
    print("World Size: ", args.world_size)
    print("Number of GPU's: ", args.gpus)
    print("Machine Rank: ", args.nr)
    print("Number of Epochs in the Training: ", args.epochs)
    print("Number of Nodes: ", args.nodes)
    print("----------------------------------")

    LOG_DIR = "/workspace/runs/tensorboard_run"
    custom_log_dir = os.path.join(LOG_DIR, f"worker_{args.nr}")
    os.makedirs(custom_log_dir, exist_ok=True)
    writer = SummaryWriter(custom_log_dir)
    writer.add_text('Training Configuration', f"World Size: {args.world_size}, GPUs: {args.gpus}, Epochs: {args.epochs}, Nodes: {args.nodes}")
    start = datetime.now()
    train_loader, sampler = ember.fetch_train_loader(api_host="grserver", api_port="8040", num_replicas=args.world_size, rank=rank, batch_size=batch_size)
    print("\n----------------------------------")
    print("Data loaded in: " + str(datetime.now() - start))
    print("----------------------------------\n")


    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", leave=False)
        sampler.set_epoch(epoch)
        batch_step = 0
        train_loss_sum = 0.0
        train_total = 0
        train_correct_top1 = 0
        train_correct_top5 = 0

        times_data     = []
        times_forward  = []
        times_loss     = []
        times_backward = []
        times_step     = []
        for i, (images, labels) in enumerate(progress_bar):
            t0 = time.time()
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            t1 = time.time()
            times_data.append(t1 - t0)

            # Forward pass
            t0 = time.time()
            outputs = model(images)
            t1 = time.time()
            times_forward.append(t1 - t0)

            t0 = time.time()
            loss = criterion(outputs, labels)
            t1 = time.time()
            times_loss.append(t1 - t0)


            # Backward and optimize
            t0 = time.time()
            optimizer.zero_grad()
            loss.backward()
            t1 = time.time()
            times_backward.append(t1 - t0)


            t0 = time.time()
            optimizer.step()
            t1 = time.time()
            times_step.append(t1 - t0)

            
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

            writer.add_scalar('Loss/Train', train_loss_avg, epoch)
            writer.add_scalar('Accuracy/Top1', train_top1_acc, epoch)
            writer.add_scalar('Accuracy/Top5', train_top5_acc, epoch)
            epoch_duration = time.time() - epoch_start_time
            writer.add_scalar('Time/Epoch', epoch_duration, epoch)
            progress_bar.set_description(f"Epoch {epoch + 1}/{args.epochs}, Loss: {loss.item():.4f}")

    import numpy as np
    avg_data     = np.mean(times_data)
    std_data     = np.std(times_data)
    avg_forward  = np.mean(times_forward)
    std_forward  = np.std(times_forward)
    avg_loss     = np.mean(times_loss)
    std_loss     = np.std(times_loss)
    avg_backward = np.mean(times_backward)
    std_backward = np.std(times_backward)
    avg_step     = np.mean(times_step)
    std_step     = np.std(times_step)

    # Caminho do arquivo de saída
    timings_path = os.path.join("/workspace", "timings_ember.txt")
    with open(timings_path, "w") as f:
        f.write("=== Médias de Tempo por Etapa (s) ===\n")
        f.write(f"Data transfer     : {avg_data:.6f} ± {std_data:.6f}\n")
        f.write(f"Forward           : {avg_forward:.6f} ± {std_forward:.6f}\n")
        f.write(f"Loss computation  : {avg_loss:.6f} ± {std_loss:.6f}\n")
        f.write(f"Backward          : {avg_backward:.6f} ± {std_backward:.6f}\n")
        f.write(f"Optimizer step    : {avg_step:.6f} ± {std_step:.6f}\n")
    writer.close()

    print("Training complete in: " + str(datetime.now() - train_start))
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
