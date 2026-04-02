import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
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
from utils import ember, modelFile, monitoring, model_wrapper
import queue
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, IterableDataset
import time
from sklearn.metrics import accuracy_score
import json
from torch.utils.data import TensorDataset, DataLoader, DistributedSampler
import torch.distributed.checkpoint as dcp
import csv
from torch.distributed.fsdp import fully_shard
import pandas as pd

# GPU TORCH PROFILER
torch.cuda.memory._record_memory_history(max_entries=100000, stacks='all')

#gpu logic
gpu_data = []
import threading
import csv
import time 
import pynvml

all_images = []
all_labels = []
new_data_available = False

request_sizes = [100, 200, 500, 1000, 2000, 5000, 10000]
batch_sizes = [16, 32, 64, 128, 256, 512]

results = []
training_times = []
avg_epoch_duration = []

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

CHECKPOINT_DIR = "/workspace/checkpoints"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the JSON configuration file')
    args = ember.set_ambient(parser.parse_args().config)
    os.environ['MASTER_ADDR'] = 'grworker1'
    os.environ['MASTER_PORT'] = '8888'

    for i in range(len(request_sizes)):
        for j in range(len(batch_sizes)):
            args.request_size = request_sizes[i]
            args.batch_size = batch_sizes[j]
            start = datetime.now()
            mp.spawn(train, nprocs=args.gpus, args=(args, request_sizes[i], batch_sizes[j]))
            elapsed = (datetime.now() - start).total_seconds()
            print("Training complete in: " + str(datetime.now() - start))
            training_times.append(elapsed)

            results.append({
                'request_size': request_sizes[i],
                'batch_size': batch_sizes[j],
                'training_time': elapsed
            })

    df = pd.DataFrame(results)
    df.to_csv('training_times.csv', index=False)

def train(gpu, args, request_size, batch_size):
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(backend='nccl', init_method='env://',
                            world_size=args.world_size, rank=rank)
    torch.cuda.set_device(gpu)

    model = modelFile.getModel().to(gpu)
    #model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    model = fully_shard(model)
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
    train_batch = batch_size
    print(f"REQUEST SIZE: {request_size}")
    dataset = RPCIterableDataset(api_host=api_host, 
                                 api_port=api_port, 
                                 world_size=args.world_size, 
                                 rank=rank, request_size=request_size)
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

        if args.checkpoint_save:
            state_dict = { "app": model_wrapper.AppState(model, optimizer) }
            dcp.save(state_dict, checkpoint_id=CHECKPOINT_DIR)
    torch.cuda.memory._dump_snapshot(f"/workspace/profile_{args.nr}.pkl")
    torch.cuda.memory._record_memory_history(enabled=None)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
