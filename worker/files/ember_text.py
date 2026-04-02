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
from torch.optim import AdamW
from torch.distributed.fsdp import fully_shard
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup


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

class RPCIterableDataset(IterableDataset):
    def __init__(self, api_host, api_port, world_size, rank, request_size=10000, data_type="text"):
        self.api_host     = api_host
        self.api_port     = api_port
        self.world_size   = world_size
        self.rank         = rank
        self.request_size = request_size
        self.current_epoch = 0
        self.data_type = data_type  # "image" or "text"

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
                data = torch.load(buf)
                lbl = torch.tensor(label_data, dtype=torch.long)
                
                if self.data_type == "text":
                    # For text data, 'data' is a dictionary with 'input_ids', 'attention_mask', etc.
                    output_queue.put((data, lbl))
                else:
                    # For image data, 'data' is a tensor
                    output_queue.put((data, lbl))

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
    start = datetime.now()

    mp.spawn(train, nprocs=args.gpus, args=(args,))

    print("Training complete in: " + str(datetime.now() - start))


def train(gpu, args):
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(backend='nccl', init_method='env://',
                            world_size=args.world_size, rank=rank)
    torch.cuda.set_device(gpu)

    # Configuration
    model_name = "prajjwal1/bert-tiny"
    num_classes = 2
    batch_size = 16
    num_epochs = 3
    learning_rate = 2e-5
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=num_classes
    )

    # Create RPC dataset
    api_host = "grserver"  
    api_port = 8040
    request_size = args.request_size
    train_batch = args.batch_size
    rpc_dataset = RPCIterableDataset(
        api_host=api_host,
        api_port=api_port,
        world_size=args.world_size,  # Adjust based on your setup
        rank=rank,
        request_size=request_size,
        data_type="text"
    )

    # Create dataloader
    dataloader = DataLoader(rpc_dataset, batch_size=batch_size)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        batch_count = 0
        
        for batch_data, batch_labels in dataloader:
            # Move data to device
            input_ids = batch_data['input_ids']
            attention_mask = batch_data['attention_mask']
            labels = batch_labels

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            batch_count += 1

            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch_count % 100 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_count}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / batch_count
        print(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.4f}")

    # Save model
    model.save_pretrained("./fine_tuned_rpc_model")
    tokenizer.save_pretrained("./fine_tuned_rpc_model")
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
