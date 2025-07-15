import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
import io
import time
import json
import queue
import grpc
import pickle
import threading
import argparse
from datetime import datetime

import psutil
import pynvml

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, IterableDataset, TensorDataset, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from apex import amp
from apex.parallel import DistributedDataParallel as DDP
from sklearn.metrics import precision_score, f1_score

from rpc import dist_data_pb2, dist_data_pb2_grpc
from utils import ember, modelFile

# ─── Instrumentação ───────────────────────────────────────────────────────────

def init_profiling(gpu, args):
    """Inicializa PSUtil, NVML e TensorBoard para este processo."""
    proc = psutil.Process(os.getpid())
    pynvml.nvmlInit()
    nv_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu)

    exp_name = "ddp_experiment"
    logdir = os.path.join("runs", exp_name, f"gpu_{args.nr * args.gpus + gpu}")
    writer = SummaryWriter(log_dir=logdir)

    return proc, nv_handle, writer

def log_metrics(step, batch_idx, proc, nv_handle, writer, start_time, net_before, args):
    """Coleta e envia métricas para o SummaryWriter."""
    # CPU
    rss = proc.memory_info().rss / 1024**2
    cpu_pct = proc.cpu_percent(interval=None)

    # GPU PyTorch
    torch.cuda.synchronize()
    peak_alloc = torch.cuda.max_memory_allocated(args.gpu) / 1024**2
    torch.cuda.reset_peak_memory_stats(args.gpu)
    # Rede
    net_after = psutil.net_io_counters()
    sent_mb = (net_after.bytes_sent - net_before.bytes_sent) / 1024**2
    recv_mb = (net_after.bytes_recv - net_before.bytes_recv) / 1024**2
    # Throughput
    elapsed = time.time() - start_time
    throughput = args.batch_size / elapsed if elapsed > 0 else 0.0

    tag = f"batch/{batch_idx}"
    writer.add_scalar(f"{tag}/cpu_rss_mb", rss, step)
    writer.add_scalar(f"{tag}/cpu_percent", cpu_pct, step)
    #writer.add_scalar(f"{tag}/gpu_mem_used_mb", mem.used/1024**2, step)
    #writer.add_scalar(f"{tag}/gpu_mem_total_mb", mem.total/1024**2, step)
    #writer.add_scalar(f"{tag}/gpu_util_percent", util.gpu, step)
    #writer.add_scalar(f"{tag}/gpu_mem_util_percent", util.memory, step)
    writer.add_scalar(f"{tag}/torch_peak_alloc_mb", peak_alloc, step)
    writer.add_scalar(f"{tag}/net_sent_mb", sent_mb, step)
    writer.add_scalar(f"{tag}/net_recv_mb", recv_mb, step)
    writer.add_scalar(f"{tag}/throughput_samples_s", throughput, step)

# ─── Dataset de RPC (sem alterações) ──────────────────────────────────────────

class RPCIterableDataset(IterableDataset):
    def __init__(self, api_host, api_port, world_size, rank, writer: SummaryWriter, request_size=10000):
        self.api_host       = api_host
        self.api_port       = api_port
        self.world_size     = world_size
        self.rank           = rank
        self.request_size   = request_size
        self.current_epoch  = 0

        # TensorBoard writer
        self.writer = writer

        # Contadores
        self.batches_fetched  = 0
        self.batches_consumed = 0

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch
        # reset counters a cada época
        self.batches_fetched  = 0
        self.batches_consumed = 0

    def fetch_loop(self, output_queue):
        """Thread que busca batches via gRPC e os coloca na fila."""
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

            # descriptografa e empurra cada par (img, lbl)
            raw = pickle.loads(resp.data)
            for tensor_bytes, label_data in raw:
                buf = io.BytesIO(tensor_bytes)
                img = torch.load(buf)
                lbl = torch.tensor(label_data, dtype=torch.long)
                output_queue.put((img, lbl))

            # log de batches fetched
            self.batches_fetched += 1
            global_step = self.current_epoch * 1_000_000 + self.batches_fetched
            self.writer.add_scalar("queue/batches_fetched", self.batches_fetched, global_step)

            batch_idx += 1

        # sinal de término
        output_queue.put(None)

    def __iter__(self):
        q = queue.Queue(maxsize=2 * self.request_size)
        thread = threading.Thread(target=self.fetch_loop, args=(q,), daemon=True)
        thread.start()

        while True:
            # log backlog size antes de tentar consumir
            backlog = q.qsize()
            global_step = self.current_epoch * 1_000_000 + self.batches_consumed
            self.writer.add_scalar("queue/backlog_size", backlog, global_step)

            item = q.get()
            if item is None:
                break

            # log batch consumido
            self.batches_consumed += 1
            global_step = self.current_epoch * 1_000_000 + self.batches_consumed
            self.writer.add_scalar("queue/batches_consumed", self.batches_consumed, global_step)

            yield item

# ─── Função de treino ─────────────────────────────────────────────────────────

def calculate_top_k_accuracy(outputs, labels, k=1):
    _, top_k_predictions = outputs.topk(k, dim=1, largest=True, sorted=True)
    correct = top_k_predictions.eq(labels.view(-1, 1).expand_as(top_k_predictions))
    return correct.any(dim=1).float().sum().item()

def train(gpu, args):
    # inicialização DDP
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(backend='nccl', init_method='env://',
                            world_size=args.world_size, rank=rank)
    torch.cuda.set_device(gpu)
    args.gpu = gpu
    args.rank = rank

    # init profiling e tensorboard
    proc, nv_handle, writer = init_profiling(gpu, args)
    api_host = "grserver"  
    api_port = 8040
    # modelo, otimizador, etc
    model = modelFile.getModel().cuda(gpu)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # dataset + loader
    dataset = RPCIterableDataset(api_host=api_host, api_port=api_port, world_size=args.world_size, rank=rank, writer=writer, request_size=args.request_size)
    loader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        num_workers=0,
                        drop_last=False,
                        pin_memory=True)

    # treino
    global_step = 0
    for epoch in range(args.epochs):
        dataset.set_epoch(epoch)
        for batch_idx, (images, labels) in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")):
            start_time = time.time()
            net_before = psutil.net_io_counters()

            images = images.cuda(gpu, non_blocking=True)
            labels = labels.cuda(gpu, non_blocking=True)

            # reset pico PyTorch
            torch.cuda.reset_peak_memory_stats(gpu)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # métricas do próprio treino
            writer.add_scalar("train/loss", loss.item(), global_step)
            preds = outputs.argmax(dim=1)
            acc = (preds == labels).float().mean().item()
            writer.add_scalar("train/acc", acc, global_step)

            global_step += 1
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
    writer.close()
    dist.destroy_process_group()

# ─── Função principal ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the JSON configuration file')
    args = ember.set_ambient(parser.parse_args().config)
    os.environ['MASTER_ADDR'] = 'grworker1'
    os.environ['MASTER_PORT'] = '8888'
    start = datetime.now()

    mp.spawn(train, nprocs=args.gpus, args=(args,))

    print("Training complete in: " + str(datetime.now() - start))

if __name__ == '__main__':
    main()
