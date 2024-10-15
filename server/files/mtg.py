import torchvision.transforms as transforms
import json
import pickle
import io
import grpc
from torchvision.datasets import ImageFolder

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
            transform_list.append(transforms.Resize(size=t['args'][0]))

    print(transform_list)
    return transforms.Compose(transform_list)