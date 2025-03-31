import torchvision.transforms as transforms
import json
import pickle
import io
import grpc
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset

def parse_transform(transform_config):
    print('Applying transformations...')
    print("transforms:", transform_config)
    transform_list = []
    for t in transform_config:
        if t['name'] == 'Grayscale':
            print('Grayscale')
            transform_list.append(transforms.Grayscale(num_output_channels=t['args'][0]))
        elif t['name'] == 'ToTensor':
            print('ToTensor')
            transform_list.append(transforms.ToTensor())
        elif t['name'] == 'Normalize':
            print('Normalize')
            transform_list.append(transforms.Normalize(mean=t['args'][0], std=t['args'][1]))
        elif t['name'] == 'Resize':
            print('Resize')
            size = t['args']
            if len(size) == 1:
                size = size[0]
            elif len(size) == 2:
                size = tuple(size)
            transform_list.append(transforms.Resize(size))
        elif t['name'] == 'RandomCrop':
            print('RandomCrop')
            transform_list.append(transforms.RandomCrop(size=t['args'][0]))
        elif t['name'] == 'RandomHorizontalFlip':
            print('RandomHorizontalFlip')
            p = t['args'][0] if 'args' in t and len(t['args']) > 0 else 0.5
            transform_list.append(transforms.RandomHorizontalFlip(p=p))

    print(transform_list)
    return transforms.Compose(transform_list)

def partition_dataset(dataset, rank, num_replicas):
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        partition_size = dataset_size // num_replicas
        start_idx = rank * partition_size
        end_idx = start_idx + partition_size if rank != num_replicas - 1 else dataset_size

        subset_indices = indices[start_idx:end_idx]
        print(f"Rank {rank}: Partition size = {len(subset_indices)}, Indices = {start_idx} to {end_idx - 1}")
        partitioned_dataset = Subset(dataset, subset_indices)
        return partitioned_dataset
        