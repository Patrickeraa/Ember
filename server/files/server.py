import grpc
from concurrent import futures
import torch
import pickle
import io
from torchvision import datasets, transforms
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from torchvision.datasets import ImageFolder


import dist_data_pb2
import dist_data_pb2_grpc
import mtg
import argparse
import json

class TrainLoaderService(dist_data_pb2_grpc.TrainLoaderServiceServicer):
    def GetTrainLoader(self, request, context):
        try:
            dataset = self.get_custom_dataset(train=True)

            partitioned_dataset = mtg.partition_dataset(dataset, request.rank, request.num_replicas)

            buffer = io.BytesIO()

            pickle.dump(partitioned_dataset, buffer)
            buffer.seek(0)

            return dist_data_pb2.TrainLoaderResponse(data=buffer.read())
        except Exception as e:
            context.set_details(f"Error: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            raise

    def GetTestLoader(self, request, context):
        try:
            dataset = self.get_custom_dataset(train=False)

            buffer = io.BytesIO()

            pickle.dump(dataset, buffer)
            buffer.seek(0)

            return dist_data_pb2.TrainLoaderResponse(
                data=buffer.read()
            )
        except Exception as e:
            context.set_details(f"Error: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            raise

    def GetSoloLoader(self, request, context):
        try:
            dataset = self.get_custom_dataset(train=True)
            buffer = io.BytesIO()
            pickle.dump(dataset, buffer)
            buffer.seek(0)
            return dist_data_pb2.TrainLoaderResponse(data=buffer.read())
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
        transform = mtg.parse_transform(transform_json['transforms'])
        dataset_path = transform_json['dataset_path']['train'] if train else transform_json['dataset_path']['test']

        dataset = ImageFolder(root=dataset_path, transform=transform)
        
        return dataset
    
    def get_custom_dataset(self, train=True):
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
            transforms.Resize((56, 56))
        ])
        dataset_path = "/workspace/dataset/Vegetable_Images/train" if train else "/workspace/dataset/Vegetable_Images/test"

        dataset = ImageFolder(root=dataset_path, transform=transform)
        print(f"Total number of samples in the {'train' if train else 'test'} dataset: {len(dataset)}")
        return dataset
    
def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10),
                         options=[('grpc.max_send_message_length', 1000 * 1024 * 1024),
                                  ('grpc.max_receive_message_length', 1000 * 1024 * 1024)])
    dist_data_pb2_grpc.add_TrainLoaderServiceServicer_to_server(TrainLoaderService(), server)
    server.add_insecure_port('[::]:8040')
    server.start()
    print("Server started, listening on port 8040.")
    server.wait_for_termination()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load transformation pipeline from JSON")
    parser.add_argument('--config', type=str, required=True, help="Path to the JSON file with transform definitions")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        transform_json = json.load(f)
    serve()