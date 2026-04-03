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
from torch.profiler import profile, record_function, ProfilerActivity, schedule
from torch.profiler import tensorboard_trace_handler
import psutil
import threading
import csv

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


    mem_interval = getattr(args, 'mem_monitor_interval', 1.0)
    mem_csv_path = getattr(args, 'mem_csv_path', f"nostream_profile_rank{rank}.csv")
    mem_samples = []  
    stop_monitor = threading.Event()

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
            stop_monitor.wait(mem_interval)

    monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
    monitor_thread.start()


    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", leave=False)
        sampler.set_epoch(epoch)
        batch_step = 0
        train_loss_sum = 0.0
        train_total = 0
        train_correct_top1 = 0
        train_correct_top5 = 0
        for i, (images, labels) in enumerate(progress_bar):
            ts = time.time()
            rss, vtotal, vpercent = get_process_memory()
            mem_samples.append({
                'timestamp': ts,
                'type': 'after_batch_received',
                'epoch': epoch,
                'batch_idx': i,
                'rss_bytes': rss,
                'vm_total': vtotal,
                'vm_percent': vpercent,
                'rank': rank,
                'pid': os.getpid()
            })
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
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
    writer.close()
    stop_monitor.set()
    monitor_thread.join(timeout=5.0)

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
