from datetime import datetime
import grpc
from concurrent import futures
import torch
import pickle
import io
from torchvision import datasets, transforms
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from torchvision.datasets import ImageFolder

from torch.utils.tensorboard import SummaryWriter
import psutil
import time 
import os

import threading
import csv

import dist_data_pb2
import dist_data_pb2_grpc
import ember_server
import argparse
import json

from transformers import AutoTokenizer
from utils.text_loader import TextDataset

class TrainLoaderService(dist_data_pb2_grpc.TrainLoaderServiceServicer):
    def __init__(self):
        if transform_json['data_type'] == "image":
            self.dataset = self.get_custom_dataset_json(train=True)
            self.test_set = self.get_custom_dataset_json(train=False)
            #self.val_set = self.get_custom_dataset_json(train=False)
        if transform_json['data_type'] == "text":
            self.dataset = TextDataset(
                path="/workspace/datasets/wiki_text",
                tokenizer=AutoTokenizer.from_pretrained("bert-base-uncased"),
                max_length=128,
                text_field='text',
                label_field='label',
                file_type='txt'
            )

    def GetTrainLoader(self, request, context):
        try:
            data = []
            for tensor_img, label in self.dataset:
                buffer = io.BytesIO()
                torch.save(tensor_img, buffer)
                data.append((buffer.getvalue(), label))

            response_data = pickle.dumps(data)
            return dist_data_pb2.TrainLoaderResponse(data=response_data)

        except Exception as e:
            context.set_details(f"Error: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            raise

    def GetTestLoader(self, request, context):
        try:
            data = []
            for tensor_img, label in self.test_set:
                buf = io.BytesIO()

                torch.save(tensor_img, buf)
                data.append((buf.getvalue(), label))

            response_data = pickle.dumps(data)
            return dist_data_pb2.TrainLoaderResponse(data=response_data)

        except Exception as e:
            context.set_details(f"Error: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            raise

    def GetSoloLoader(self, request, context):
        try:
            dataset = self.get_custom_dataset(train=True)
            data = []
            to_pil = transforms.ToPILImage() 
            print("message received")
            for img, label in dataset:
                if isinstance(img, torch.Tensor):
                    img = to_pil(img)

                buffer = io.BytesIO()
                img.save(buffer, format="PNG")
                data.append((buffer.getvalue(), label))
            print(type(data), data[:5])  
            response_data = pickle.dumps(data)
            return dist_data_pb2.TrainLoaderResponse(data=response_data)
        except Exception as e:
            context.set_details(f"Error: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            raise

    def GetSoloTest(self, request, context):
        try:
            dataset = self.get_custom_dataset(train=False)
            buffer = io.BytesIO()
            pickle.dump(dataset, buffer)
            buffer.seek(0)

            return dist_data_pb2.TrainLoaderResponse(data=buffer.read())
        except Exception as e:
            context.set_details(f"Error: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            raise

    def get_custom_dataset_json(self, train=True):
        transform = ember_server.parse_transform(transform_json['transforms'])
        dataset_path = transform_json['dataset_path']['train'] if train else transform_json['dataset_path']['test']
        print(f"Loading dataset from {dataset_path}")
        dataset = ImageFolder(root=dataset_path, transform=transform)
        class_names = dataset.classes
        num_classes = len(class_names)
        
        print(f"Number of classes: {num_classes}")
        print(f"Class names: {class_names}")
        return dataset
    
    def GetBatch(self, request, context):
        rank = request.rank
        num_replicas = request.num_replicas
        batch_idx = request.batch_idx
        batch_size = request.batch_size
        epoch = request.epoch

        sampler = DistributedSampler(
            dataset=self.dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=True,
            drop_last=False
        )
        sampler.set_epoch(epoch)

        all_indices = list(sampler)

        start = batch_idx * batch_size
        end = min(start + batch_size, len(all_indices))
        if start >= len(all_indices):
            return dist_data_pb2.TrainLoaderResponse(data=b"")

        batch_indices = all_indices[start:end]

        data = []
        for idx in batch_indices:
            item, label = self.dataset[idx]
            
            # Handle both text and image data the same way since torch.save can handle both
            buf = io.BytesIO()
            torch.save(item, buf)
            data.append((buf.getvalue(), label))

        response_data = pickle.dumps(data)
        return dist_data_pb2.TrainLoaderResponse(data=response_data)
    


        
def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10),
                         options=[('grpc.max_send_message_length', 1000 * 1024 * 1024),
                                  ('grpc.max_receive_message_length', 1000 * 1024 * 1024)])
    dist_data_pb2_grpc.add_TrainLoaderServiceServicer_to_server(TrainLoaderService(), server)
    server.add_insecure_port('[::]:8040')
    server.start()
    print("Server started, listening on port 8040.")
    server.wait_for_termination()

def monitor_cpu(csv_filename: str = "cpu_usage.csv", interval: float = 1.0, process: psutil.Process = None):
    if process is None:
        process = psutil.Process(os.getpid())
    
    num_cpus = psutil.cpu_count()
    
    try:
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            headers = ['timestamp', 'total_cpu_percent', 'normalized_cpu_percent', 'elapsed_time_seconds']
            headers.extend([f'cpu_core_{i}' for i in range(num_cpus)])
            writer.writerow(headers)
        print(f"Detailed CPU monitoring started. Number of cores: {num_cpus}")
    except Exception as e:
        print(f"monitor_cpu: failed to create CSV file: {e}")
        return

    process.cpu_percent(interval=None)
    start_time = time.time()
    
    try:
        while True:
            time.sleep(interval)
            
            total_cpu_pct = process.cpu_percent(interval=None)
            normalized_cpu_pct = total_cpu_pct / num_cpus
            per_cpu_pct = psutil.cpu_percent(percpu=True)
            
            current_time = time.time()
            elapsed_time = current_time - start_time
            
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            try:
                with open(csv_filename, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    row = [timestamp, total_cpu_pct, normalized_cpu_pct, elapsed_time]
                    row.extend(per_cpu_pct)
                    writer.writerow(row)
            except Exception as e:
                print(f"monitor_cpu: failed to write to CSV: {e}")
                break
                
    except KeyboardInterrupt:
        print("CPU monitoring stopped by user")
    except Exception as e:
        print("monitor_cpu: error during sampling:", e)

def start_cpu_monitoring_thread(csv_filename: str = "cpu_usage.csv", interval: float = 1.0, daemon: bool = True):
    process = psutil.Process(os.getpid())  # Create the process here
    thread = threading.Thread(
        target=monitor_cpu,
        args=(csv_filename, interval, process),  # Pass the process object
        daemon=daemon
    )
    thread.start()
    return thread

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load transformation pipeline from JSON")
    parser.add_argument('--config', type=str, required=True, help="Path to the JSON file with transform definitions")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        transform_json = json.load(f)

    # Start CPU monitoring BEFORE starting the server
    monitor_thread = start_cpu_monitoring_thread(
        csv_filename="my_cpu_usage.csv",
        interval=5.0,
        daemon=True
    )
    serve()
