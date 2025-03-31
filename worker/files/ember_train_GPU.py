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

#gpu logic
gpu_data = []
import threading
import csv
import time 
import pynvml
import json

import io
import grpc
import pickle
import io
import dist_data_pb2
import dist_data_pb2_grpc
import ember
import modelFile
from sklearn.metrics import accuracy_score

def calculate_top_k_accuracy(outputs, labels, k=1):
    """Compute Top-k accuracy for the given outputs and labels."""
    _, top_k_predictions = outputs.topk(k, dim=1, largest=True, sorted=True)
    correct = top_k_predictions.eq(labels.view(-1, 1).expand_as(top_k_predictions))
    return correct.any(dim=1).float().sum().item()

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
    with open("gpu_utilization.csv", "w") as file:
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
    batch_size = 128
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    # Data loading code
    print("--------- TRAINING STATUS --------")
    print("World Size: ", args.world_size)
    print("Number of GPU's: ", args.gpus)
    print("Machine Rank: ", args.nr)
    print("Number of Epochs in the Training: ", args.epochs)
    print("Number of Nodes: ", args.nodes)
    print("----------------------------------")


    start = datetime.now()
    train_loader = ember.fetch_train_loader(api_host="grserver", api_port="8040", num_replicas=args.world_size, rank=rank, batch_size=batch_size)
    print("\n----------------------------------")
    print("Data loaded in: " + str(datetime.now() - start))
    print("----------------------------------\n")

    epoch_losses = []
    epoch_accuracies = []
    metrics_filename = "sync_metrics.json"
    total_step = len(train_loader)
    for epoch in range(args.epochs):
        dist.barrier() 
            # epoch metrics
        epoch_loss = 0
        correct_predictions = 0
        total_predictions = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", leave=False)
        for i, (images, labels) in enumerate(progress_bar):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item() 

                        # Update accuracy
            _, predicted = outputs.max(1)
            correct_predictions += predicted.eq(labels).sum().item()
            total_predictions += labels.size(0)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar.set_description(f"Epoch {epoch + 1}/{args.epochs}, Loss: {loss.item():.4f}")
        scheduler.step()
                # Calculate average loss and accuracy for the epoch
        epoch_loss /= total_step
        epoch_accuracy = 100.0 * correct_predictions / total_predictions
        epoch_losses.append(epoch_loss)
        epoch_accuracies.append(epoch_accuracy)
        # Save metrics to file after each epoch
        with open(metrics_filename, "w") as file:
            json.dump({"losses": epoch_losses, "accuracies": epoch_accuracies}, file)
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
        
        filename = "metrics_sync.txt"
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



if __name__ == '__main__':
    main()
