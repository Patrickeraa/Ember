import torchvision.transforms as transforms
import json
import pickle
import io
import grpc
from torchvision.datasets import ImageFolder

def parse_transform(transform_json_str):
    transform_config = json.loads(transform_json_str)
    
    transform_list = []
    for t in transform_config['transforms']:
        if t['name'] == 'Grayscale':
            transform_list.append(transforms.Grayscale(num_output_channels=t['args'][0]))
        elif t['name'] == 'ToTensor':
            transform_list.append(transforms.ToTensor())
        elif t['name'] == 'Normalize':
            transform_list.append(transforms.Normalize(mean=t['args'][0], std=t['args'][1]))
        elif t['name'] == 'Resize':
            transform_list.append(transforms.Resize(size=t['args'][0]))

    return transforms.Compose(transform_list)