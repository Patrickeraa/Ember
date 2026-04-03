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
from utils import ember, modelFile, monitoring
import queue
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, IterableDataset
import time
from sklearn.metrics import accuracy_score
import json
from torch.utils.data import TensorDataset, DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

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

def bytes_to_gib(x_bytes):
    return float(x_bytes) / (1024 ** 3)  # GiB

def log_gpu_metrics(writer, epoch, device=None, reset_peak=False):
    if device is None:
        device = torch.cuda.current_device()

    torch.cuda.synchronize(device)

    if reset_peak:
        torch.cuda.reset_peak_memory_stats(device)

    mem_alloc = torch.cuda.memory_allocated(device)
    mem_res = torch.cuda.memory_reserved(device)
    peak_alloc = torch.cuda.max_memory_allocated(device) 
    peak_res = torch.cuda.max_memory_reserved(device) 
    mem_free, mem_total = torch.cuda.mem_get_info(device)

    # gb
    alloc_gib = bytes_to_gib(mem_alloc)
    res_gib = bytes_to_gib(mem_res)
    peak_alloc_gib = bytes_to_gib(peak_alloc)
    peak_res_gib = bytes_to_gib(peak_res)
    free_gib = bytes_to_gib(mem_free)
    total_gib = bytes_to_gib(mem_total)

    pct_used = (alloc_gib / total_gib * 100.0) if total_gib > 0 else 0.0

    writer.add_scalar("GPU/MemoryAllocated_GiB", alloc_gib, epoch)
    writer.add_scalar("GPU/MemoryReserved_GiB", res_gib, epoch)
    writer.add_scalar("GPU/PeakMemoryAllocated_GiB", peak_alloc_gib, epoch)
    writer.add_scalar("GPU/PeakMemoryReserved_GiB", peak_res_gib, epoch)
    writer.add_scalar("GPU/MemoryFree_GiB", free_gib, epoch)
    writer.add_scalar("GPU/MemoryTotal_GiB", total_gib, epoch)
    writer.add_scalar("GPU/AllocatedPct", pct_used, epoch)

    try:
        summary = torch.cuda.memory_summary(device=device, abbreviated=True)
        writer.add_text(f"GPU{device}/MemorySummary", summary, epoch)
    except Exception as e:
        writer.add_text(f"GPU{device}/MemorySummary_Error", str(e), epoch)

    try:
        max_alloc = torch.cuda.max_memory_allocated(device)
        max_res = torch.cuda.max_memory_reserved(device)
        writer.add_scalar(f"GPU{device}/MaxMemoryAllocated_GiB", bytes_to_gib(max_alloc), epoch)
        writer.add_scalar(f"GPU{device}/MaxMemoryReserved_GiB", bytes_to_gib(max_res), epoch)
    except Exception as e:
        writer.add_text(f"GPU{device}/MaxMemory_Error", str(e), epoch)


class RPCIterableDataset(IterableDataset):
    def __init__(self, api_host, api_port, world_size, rank, request_size=10000):
        self.api_host     = api_host
        self.api_port     = api_port
        self.world_size   = world_size
        self.rank         = rank
        self.request_size = request_size
        self.current_epoch = 0

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch

    def fetch_loop(self, output_queue):
        max_msg = 1000 * 1024 * 1024
        channel = grpc.insecure_channel(
            f"{self.api_host}:{self.api_port}",
            options=[
                ('grpc.max_send_message_length', max_msg),
                ('grpc.max_receive_message_length', max_msg),
            ]
        )
        stub = dist_data_pb2_grpc.TrainLoaderServiceStub(channel)
        batch_idx = 0

        while True:
            req = dist_data_pb2.BatchRequest(
                num_replicas=self.world_size,
                rank=self.rank,
                batch_idx=batch_idx,
                batch_size=self.request_size,
                epoch=self.current_epoch,
            )
            resp = stub.GetBatch(req)
            if not resp.data:
                break

            raw = pickle.loads(resp.data)
            for tensor_bytes, label_data in raw:
                buf = io.BytesIO(tensor_bytes)
                img = torch.load(buf)
                lbl = torch.tensor(label_data, dtype=torch.long)
                output_queue.put((img, lbl))

            batch_idx += 1

        output_queue.put(None)

    def __iter__(self):
        q = queue.Queue(maxsize=2 * self.request_size)
        thread = threading.Thread(target=self.fetch_loop, args=(q,), daemon=True)
        thread.start()

        while True:
            item = q.get()
            if item is None:
                break
            yield item

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


def train(gpu, args):
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(backend='nccl', init_method='env://',
                            world_size=args.world_size, rank=rank)
    torch.cuda.set_device(gpu)



    model = modelFile.getModel().cuda(gpu)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Streaming dataset and loader
    api_host = "grserver"  
    api_port = 8040
    request_size = args.request_size
    train_batch = args.batch_size
    print(f"REQUEST SIZE: {request_size}")
    dataset = RPCIterableDataset(api_host=api_host, api_port=api_port, world_size=args.world_size, rank=rank, request_size=request_size)
    loader = DataLoader(dataset,
                        batch_size=train_batch,
                        num_workers=0,
                        drop_last=False,
                        pin_memory=True)

    custom_log_dir = os.path.join(LOG_DIR, f"worker_{args.nr}")
    os.makedirs(custom_log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=custom_log_dir)

    for epoch in range(args.epochs):
        start_t = time.time()
        epoch_loss = 0.0
        correct = 0
        total = 0
        batch_count = 0
        model.train()
        log_gpu_metrics(writer, epoch, device=torch.device('cuda:0'), reset_peak=True)

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

    

if __name__ == '__main__':
    main()
