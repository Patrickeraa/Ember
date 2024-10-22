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

import json
import io
import grpc
import pickle
import io
import dist_data_pb2
import dist_data_pb2_grpc


class Args:
    def __init__(self, config):
        self.nodes = config.get('nodes', 1)
        self.gpus = config.get('gpus', 1)
        self.nr = config.get('nr', 0)
        self.epochs = config.get('epochs', 10)
        self.world_size = self.gpus * self.nodes

def set_ambient(json_file):
    with open(json_file, 'r') as f:
        config = json.load(f)

    return Args(config)

def fetch_solo_loader(api_host, api_port):
    max_message_length = 1000 * 1024 * 1024
    channel = grpc.insecure_channel(
        f'{api_host}:{api_port}', 
        options=[
            ('grpc.max_send_message_length', max_message_length),
            ('grpc.max_receive_message_length', max_message_length)
        ]
    )
    
    stub = dist_data_pb2_grpc.TrainLoaderServiceStub(channel)
    request = dist_data_pb2.Empty()
    response = stub.GetSoloLoader(request)

    buffer = io.BytesIO(response.data)
    dataset = pickle.load(buffer)

    solo_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=100,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return solo_loader

def fetch_solo_test(api_host, api_port):
    max_message_length = 1000 * 1024 * 1024
    channel = grpc.insecure_channel(
        f'{api_host}:{api_port}', 
        options=[
            ('grpc.max_send_message_length', max_message_length),
            ('grpc.max_receive_message_length', max_message_length)
        ]
    )
    
    stub = dist_data_pb2_grpc.TrainLoaderServiceStub(channel)
    request = dist_data_pb2.Empty()
    response = stub.GetSoloTest(request)

    buffer = io.BytesIO(response.data)
    dataset = pickle.load(buffer)

    solo_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=100,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return solo_loader


def fetch_train_loader(api_host, api_port, num_replicas, rank, batch_size):
    max_message_length = 1000 * 1024 * 1024
    channel = grpc.insecure_channel(
        f'{api_host}:{api_port}', 
        options=[
            ('grpc.max_send_message_length', max_message_length),
            ('grpc.max_receive_message_length', max_message_length)
        ]
    )
    stub = dist_data_pb2_grpc.TrainLoaderServiceStub(channel)
    request = dist_data_pb2.TrainLoaderRequest(num_replicas=num_replicas, rank=rank)
    response = stub.GetTrainLoader(request)

    buffer = io.BytesIO(response.data)
    partitioned_dataset = pickle.load(buffer)

    classes = partitioned_dataset.dataset.classes
    print(f"Classes: {classes}, Number of classes: {len(classes)}")

    print(f"Received dataset partition for rank {rank}")
    print(f"Number of samples in received partition: {len(partitioned_dataset)}")

    train_loader = torch.utils.data.DataLoader(
        dataset=partitioned_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader


def fetch_test_loader(api_host, api_port):
    max_message_length = 1000 * 1024 * 1024
    channel = grpc.insecure_channel(
        f'{api_host}:{api_port}', 
        options=[
            ('grpc.max_send_message_length', max_message_length),
            ('grpc.max_receive_message_length', max_message_length)
        ]
    )
    
    stub = dist_data_pb2_grpc.TrainLoaderServiceStub(channel)
    request = dist_data_pb2.Empty()
    response = stub.GetTestLoader(request)

    buffer = io.BytesIO(response.data)
    dataset = pickle.load(buffer)

    test_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=100,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return test_loader