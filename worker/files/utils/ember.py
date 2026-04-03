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
from PIL import Image
import json
import io
import grpc
import pickle
import io
from rpc import dist_data_pb2, dist_data_pb2_grpc


class Args:
    def __init__(self, config):
        self.nodes = config.get('nodes', 1)
        self.gpus = config.get('gpus', 1)
        self.nr = config.get('nr', 0)
        self.epochs = config.get('epochs', 10)
        self.world_size = self.gpus * self.nodes
        self.request_size = config.get('request_size', 10000)
        self.batch_size = config.get('batch_size', 100)

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
    
    data = pickle.loads(response.data)
    to_tensor = transforms.ToTensor()
    
    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            img_bytes, label = self.data[idx]
            img = Image.open(io.BytesIO(img_bytes))
            img = to_tensor(img)
            return img, label

    solo_loader = torch.utils.data.DataLoader(
        dataset=CustomDataset(data),
        batch_size=100,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        persistent_workers=True
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


class PartitionedDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        # data: lista de (torch.Tensor, label)
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tensor_img, label = self.data[idx]
        return tensor_img, label

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def fetch_train_loader(api_host, api_port, num_replicas, rank, batch_size):
    # setup do canal gRPC
    max_msg = 1000 * 1024 * 1024
    channel = grpc.insecure_channel(
        f'{api_host}:{api_port}',
        options=[
            ('grpc.max_send_message_length', max_msg),
            ('grpc.max_receive_message_length', max_msg)
        ]
    )
    stub = dist_data_pb2_grpc.TrainLoaderServiceStub(channel)
    request = dist_data_pb2.TrainLoaderRequest(num_replicas=num_replicas, rank=rank)
    response = stub.GetTrainLoader(request)

    # unpack pickle → lista de (bytes_do_tensor, label)
    raw = pickle.loads(response.data)
    # converte cada entrada de volta para torch.Tensor
    data = []
    for tensor_bytes, label in raw:
        buf = io.BytesIO(tensor_bytes)
        tensor = torch.load(buf)   # recupera o tensor original
        data.append((tensor, label))

    # monta seu Dataset particionado
    dataset = PartitionedDataset(data)

    # DistributedSampler para shuffle por epoch
    train_sampler = DistributedSampler(
        dataset,
        num_replicas=num_replicas,
        rank=rank,
        shuffle=True,
        drop_last=False
    )

    # DataLoader sem shuffle (já no sampler)
    train_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=0,
        pin_memory=True
    )

    return train_loader, train_sampler



def fetch_test_loader(api_host, api_port):
    # monta canal gRPC com limites altos de mensagem
    max_msg = 1000 * 1024 * 1024
    channel = grpc.insecure_channel(
        f'{api_host}:{api_port}',
        options=[
            ('grpc.max_send_message_length', max_msg),
            ('grpc.max_receive_message_length', max_msg)
        ]
    )
    stub = dist_data_pb2_grpc.TrainLoaderServiceStub(channel)
    request = dist_data_pb2.Empty()
    response = stub.GetTestLoader(request)

    # desempacota pickle → lista de (bytes_do_tensor, label)
    raw = pickle.loads(response.data)
    data = []
    for tensor_bytes, label in raw:
        buf = io.BytesIO(tensor_bytes)
        tensor = torch.load(buf)   # recupera o tensor exato
        data.append((tensor, label))

    # monta o PartitionedDataset (já devolvendo tensor puro)
    dataset = PartitionedDataset(data)

    # DataLoader de teste — só um nó, sem sampler
    test_loader = DataLoader(
        dataset=dataset,
        batch_size=100,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    return test_loader
