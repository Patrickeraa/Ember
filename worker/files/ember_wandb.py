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
from utils import ember, modelFile, monitoring
import queue
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, IterableDataset
import time
from sklearn.metrics import accuracy_score
import json
from torch.utils.data import TensorDataset, DataLoader, DistributedSampler

# --- imports para W&B e monitoramento extra ---
import wandb
import psutil
import pynvml

# Inicializa NVML para GPU metrics
pynvml.nvmlInit()
gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

def calculate_top_k_accuracy(outputs, labels, k=1):
    _, top_k_predictions = outputs.topk(k, dim=1, largest=True, sorted=True)
    correct = top_k_predictions.eq(labels.view(-1, 1).expand_as(top_k_predictions))
    return correct.any(dim=1).float().sum().item()

# ... (RPCIterableDataset e funções auxiliares inalteradas) ...
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
            batch_size=10000
        )
        response = stub.GetBatch(request)

        if not response.data:
            batch_queue.put(None)
            break

        raw = pickle.loads(response.data)
        converted_batch = []
        for tensor_bytes, label_data in raw:
            buf = io.BytesIO(tensor_bytes)

            img_tensor = torch.load(buf)
            label_tensor = torch.tensor(label_data)
            converted_batch.append((img_tensor, label_tensor))

        batch_queue.put(converted_batch)
        new_data_available = True
        batch_idx += 1

def create_dataloader_from_queue(batch_queue, num_replicas, rank, batch_size=100):
    if not hasattr(create_dataloader_from_queue, "images"):
        create_dataloader_from_queue.images = []
        create_dataloader_from_queue.labels = []


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

    if create_dataloader_from_queue.images:
        images_tensor = torch.stack(create_dataloader_from_queue.images)
        labels_tensor = torch.stack(create_dataloader_from_queue.labels)
    else:
        images_tensor = torch.empty((0, 3, 32, 32))
        labels_tensor = torch.empty((0,), dtype=torch.long)

    # tira os itens da fila
    create_dataloader_from_queue.images.clear()
    create_dataloader_from_queue.labels.clear()

    dataset = TensorDataset(images_tensor, labels_tensor)
    sampler = DistributedSampler(
        dataset,
        num_replicas=num_replicas,
        rank=rank,
        shuffle=True,
        drop_last=False
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0
    )

    return dataloader, sampler

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the JSON configuration file')
    args = ember.set_ambient(parser.parse_args().config)

    os.environ['MASTER_ADDR'] = 'grworker1'
    os.environ['MASTER_PORT'] = '8888'
    start = datetime.now()

    mp.spawn(train, nprocs=args.gpus, args=(args,))

    duration = datetime.now() - start
    print("Training complete in: " + str(duration))

def train(gpu, args):
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(backend='nccl', init_method='env://',
                            world_size=args.world_size, rank=rank)
    torch.cuda.set_device(gpu)

    run = wandb.init(
        project="Ember",
        entity="patrickera_",
        name=f"train-rank{rank}",         # opcionalmente diferenciar por rank
        config={k: v for k, v in vars(args).items()},
        reinit=True,
        tags=["distributed", "streaming"]
    )

    model = modelFile.getModel().cuda(gpu)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Streaming dataset e loader
    api_host = "grserver"
    api_port = 8040
    request_size = args.request_size
    train_batch = args.batch_size

    dataset = RPCIterableDataset(api_host, api_port, args.world_size, rank, request_size)
    loader = DataLoader(dataset,
                        batch_size=train_batch,
                        num_workers=0,
                        drop_last=False,
                        pin_memory=True)
    samples_processed = 0
    for epoch in range(args.epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        correct = 0
        total = 0
        batch_count = 0

        for batch_idx, (images, labels) in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")):
            # envia métricas de dataset
            samples_processed += labels.size(0)
            run.log({
                "epoch": epoch + 1,
                "batch_idx": batch_idx,
                "samples_processed": samples_processed
            })

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

            # log de GPU e RAM a cada batch
            ram_gb = psutil.virtual_memory().used / (1024**3)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
            gpu_used_gb = mem_info.used / (1024**3)
            gpu_total_gb = mem_info.total / (1024**3)

            run.log({
                "batch_loss": loss.item(),
                "batch_accuracy": 100.0 * preds.eq(labels).sum().item() / labels.size(0),
                "ram_used_gb": ram_gb,
                "gpu_used_gb": gpu_used_gb,
                "gpu_total_gb": gpu_total_gb
            })

        # métricas de época
        avg_loss = epoch_loss / batch_count
        acc = 100.0 * correct / total
        epoch_time = time.time() - epoch_start

        print(f"Epoch {epoch+1}: loss={avg_loss:.4f}, acc={acc:.2f}%, time={epoch_time:.1f}s")

        run.log({
            "epoch_avg_loss": avg_loss,
            "epoch_accuracy": acc,
            "epoch_time_s": epoch_time
        })

    if rank == 0:
        test_loader = ember.fetch_test_loader(api_host="grserver", api_port="8040")
        model.eval()

        all_labels, all_predictions = [], []
        total_loss = total = 0
        total_top1 = total_top5 = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
                outputs = model(images)
                loss = criterion(outputs, labels)

                total_loss += loss.item() * labels.size(0)
                total += labels.size(0)
                _, preds = outputs.max(1)
                total_top1 += calculate_top_k_accuracy(outputs, labels, k=1)
                total_top5 += calculate_top_k_accuracy(outputs, labels, k=5)

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(preds.cpu().numpy())

        accuracy = 100 * total_top1 / total
        avg_loss = total_loss / total
        precision = precision_score(all_labels, all_predictions, average='weighted')
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        top5_acc = 100 * total_top5 / total

        # log final no W&B
        run.log({
            "test_accuracy": accuracy,
            "test_loss": avg_loss,
            "test_precision": precision,
            "test_f1": f1,
            "test_top1_accuracy": accuracy,
            "test_top5_accuracy": top5_acc,
            "test_top1_error": 100 - accuracy,
            "test_top5_error": 100 - top5_acc
        })

        # grava também em arquivo local
        metrics = {
            "accuracy": accuracy,
            "loss": avg_loss,
            "precision": precision,
            "f1": f1,
            "top1_acc": accuracy,
            "top5_acc": top5_acc
        }
        with open("metrics_async.txt", "w") as f:
            json.dump(metrics, f, indent=2)

    dist.destroy_process_group()

if __name__ == '__main__':
    main()
