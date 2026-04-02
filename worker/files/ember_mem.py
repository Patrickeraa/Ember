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
from utils import ember, modelFile, monitoring
import queue
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, IterableDataset
import time
from sklearn.metrics import accuracy_score
import json
from torch.utils.data import TensorDataset, DataLoader, DistributedSampler

#gpu logic
gpu_data = []
import threading
import csv
import time 
import pynvml

all_images = []
all_labels = []
new_data_available = False

def get_process_memory():
    """
    Retorna (rss_bytes, total_virtual_bytes_or_None, percent_used_or_None).
    Usa psutil se disponível; senão tenta /proc/self/status; senão resource.ru_maxrss.
    """
    try:
        p = psutil.Process(os.getpid())
        mem = p.memory_info().rss
        vm = psutil.virtual_memory()
        return mem, vm.total, vm.percent
    except Exception:
        # fallback Linux /proc
        try:
            with open('/proc/self/status', 'r') as f:
                for line in f:
                    if line.startswith('VmRSS:'):
                        parts = line.split()
                        rss_kb = int(parts[1])
                        return rss_kb * 1024, None, None
        except Exception:
            import resource
            ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # ru_maxrss pode estar em KB ou bytes dependendo do SO; multiplicar por 1024 costuma funcionar
            return int(ru) * 1024, None, None

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
    dataset = RPCIterableDataset(api_host=api_host, 
                                 api_port=api_port, 
                                 world_size=args.world_size, 
                                 rank=rank, request_size=request_size)
    loader = DataLoader(dataset,
                        batch_size=train_batch,
                        num_workers=0,
                        drop_last=False,
                        pin_memory=True)
    
    mem_interval = getattr(args, 'mem_monitor_interval', 1.0)  # segundos entre amostras contínuas
    mem_csv_path = getattr(args, 'mem_csv_path', f"memory_profile_rank{rank}.csv")
    mem_samples = []   # lista de dicionários {timestamp, type, epoch, batch_idx, rss_bytes, ...}
    stop_monitor = threading.Event()

    # monitor thread: amostras periódicas independentes do batch
    def monitor_loop():
        while not stop_monitor.is_set():
            ts = time.time()
            rss, vtotal, vpercent = get_process_memory()
            mem_samples.append({
                'timestamp': ts,
                'type': 'sample',
                'epoch': None,
                'batch_idx': None,
                'rss_bytes': rss,
                'vm_total': vtotal,
                'vm_percent': vpercent,
                'rank': rank,
                'pid': os.getpid()
            })
            # aguarda com timeout para permitir término rápido
            stop_monitor.wait(mem_interval)

    monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
    monitor_thread.start()

    for epoch in range(args.epochs):
        start_t = time.time()
        epoch_loss = 0.0
        correct = 0
        total = 0
        batch_count = 0

    
        for batch_idx, (images, labels) in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")):
            ts = time.time()
            rss, vtotal, vpercent = get_process_memory()
            mem_samples.append({
                'timestamp': ts,
                'type': 'after_batch_received',
                'epoch': epoch,
                'batch_idx': batch_idx,
                'rss_bytes': rss,
                'vm_total': vtotal,
                'vm_percent': vpercent,
                'rank': rank,
                'pid': os.getpid()
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

    stop_monitor.set()
    monitor_thread.join(timeout=5.0)

    # escrever csv (ordenando por timestamp)
    mem_samples.sort(key=lambda x: x['timestamp'])
    csv_fields = ['timestamp', 'iso_time', 'rank', 'pid', 'type', 'epoch', 'batch_idx',
                  'rss_bytes', 'rss_mb', 'vm_total', 'vm_total_gb', 'vm_percent']
    with open(mem_csv_path, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for s in mem_samples:
            iso = datetime.utcfromtimestamp(s['timestamp']).isoformat() + 'Z'
            rss_mb = round(s['rss_bytes'] / (1024**2), 3) if s['rss_bytes'] is not None else None
            vm_total_gb = round(s['vm_total'] / (1024**3), 3) if s.get('vm_total') else None
            writer.writerow({
                'timestamp': s['timestamp'],
                'iso_time': iso,
                'rank': s['rank'],
                'pid': s['pid'],
                'type': s['type'],
                'epoch': s['epoch'],
                'batch_idx': s['batch_idx'],
                'rss_bytes': s['rss_bytes'],
                'rss_mb': rss_mb,
                'vm_total': s.get('vm_total'),
                'vm_total_gb': vm_total_gb,
                'vm_percent': s.get('vm_percent')
            })
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
                total_top1_correct += (outputs.argmax(dim=1) == labels).sum().item()
                total_top5_correct += (outputs.topk(5, dim=1).indices == labels.view(-1, 1)).sum().item()

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
