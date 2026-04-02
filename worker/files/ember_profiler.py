import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
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
from utils import ember, modelFile, monitoring, ember_dataset, model_wrapper
import queue
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, IterableDataset
import time
from sklearn.metrics import accuracy_score
import json
from torch.utils.data import TensorDataset, DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import torch.distributed.checkpoint as dcp

# GPU TORCH PROFILER
torch.cuda.memory._record_memory_history(max_entries=100000, stacks='all')

def calculate_top_k_accuracy(outputs, labels, k=1):
    _, top_k_predictions = outputs.topk(k, dim=1, largest=True, sorted=True)
    correct = top_k_predictions.eq(labels.view(-1, 1).expand_as(top_k_predictions))
    return correct.any(dim=1).float().sum().item()

#gpu logic
gpu_data = []
import threading
import csv
import time 
import pynvml

all_images = []
all_labels = []
new_data_available = False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the JSON configuration file')
    args = ember.set_ambient(parser.parse_args().config)
    os.environ['MASTER_ADDR'] = 'grworker1'
    os.environ['MASTER_PORT'] = '8888'
    start = datetime.now()

    mp.spawn(train, nprocs=args.gpus, args=(args,))

    print("Training complete in: " + str(datetime.now() - start))


LOG_DIR = "/workspace/logs"
CHECKPOINT_DIR = "/workspace/checkpoints"


def train(gpu, args):
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(backend='nccl', init_method='env://',
                            world_size=args.world_size, rank=rank)
    torch.cuda.set_device(gpu)



    model = modelFile.getModel().cuda(gpu)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    #checkpoint loading
    if args.checkpoint_load and os.path.isdir(CHECKPOINT_DIR):
        state_dict = { "app": model_wrapper.AppState(model, optimizer)}
        dcp.load(
            state_dict=state_dict,
            checkpoint_id=CHECKPOINT_DIR,
        )


    # Streaming dataset and loader
    api_host = "grserver"  
    api_port = 8040
    request_size = args.request_size
    train_batch = args.batch_size
    print(f"REQUEST SIZE: {request_size}")

    custom_log_dir = os.path.join(LOG_DIR, f"worker_{args.nr}")
    os.makedirs(custom_log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=custom_log_dir)

    dataset = ember_dataset.RPCIterableDataset(api_host=api_host, api_port=api_port, world_size=args.world_size, rank=rank, request_size=request_size, writer=writer)
    loader = DataLoader(dataset,
                        batch_size=train_batch,
                        num_workers=0,
                        drop_last=False,
                        pin_memory=True)



    for epoch in range(args.epochs):
        start_t = time.time()
        epoch_loss = 0.0
        correct = 0
        total = 0
        batch_count = 0
        model.train()

        train_loss_sum = 0.0
        train_total = 0
        train_correct_top1 = 0
        train_correct_top5 = 0
        epoch_start_time = time.time()
        #monitoring.log_gpu_metrics(writer, epoch, device=torch.device('cuda:0'), reset_peak=True)
        dataset.set_epoch(epoch)
        for images, labels in tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            images = images.cuda(gpu, non_blocking=True)
            labels = labels.cuda(gpu, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
            batch_count += 1

        train_loss_avg = train_loss_sum / train_total
        train_top1_acc  = train_correct_top1 / train_total
        train_top5_acc  = train_correct_top5 / train_total
        
        writer.add_scalar('Loss/Train',     train_loss_avg, epoch)
        writer.add_scalar('Accuracy/Top1',  train_top1_acc,  epoch)
        writer.add_scalar('Accuracy/Top5',  train_top5_acc,  epoch)

        epoch_duration = time.time() - epoch_start_time
        writer.add_scalar('Time/Epoch', epoch_duration, epoch)
        #if args.checkpoint_save:
        #    state_dict = { "app": model_wrapper.AppState(model, optimizer) }
        #    dcp.save(state_dict, checkpoint_id=CHECKPOINT_DIR)
    writer.close()
    torch.cuda.memory._dump_snapshot(f"/workspace/profile_{args.nr}.pkl")
    torch.cuda.memory._record_memory_history(enabled=None)
    if gpu == 0:
        print("Training complete")
    if rank == 0:
        print("Teste gpu 0")
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
    dist.destroy_process_group()
    

if __name__ == '__main__':
    main()
