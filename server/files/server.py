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


import dist_data_pb2
import dist_data_pb2_grpc
import ember_server
import argparse
import json

class TrainLoaderService(dist_data_pb2_grpc.TrainLoaderServiceServicer):
    def __init__(self):
        self.dataset = self.get_custom_dataset_json(train=True)
        self.test_set = self.get_custom_dataset_json(train=False)

    def GetTrainLoader(self, request, context):
        try:
            partitioned_dataset = ember_server.partition_dataset(self.dataset, request.rank, request.num_replicas)
            
            data = []
            to_pil = transforms.ToPILImage()

            for img, label in partitioned_dataset:
                if isinstance(img, torch.Tensor):
                    img = to_pil(img)
                
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG")
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
            to_pil = transforms.ToPILImage()

            for img, label in self.test_set:
                if isinstance(img, torch.Tensor):
                    img = to_pil(img)
                
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG")
                data.append((buffer.getvalue(), label))

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
                img.save(buffer, format="JPEG")
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
        
        return dataset
    
    def get_custom_dataset(self, train=True):
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
            transforms.Resize((256, 256))
        ])
        dataset_path = "/workspace/dataset/vegetable_images/train" if train else "/workspace/dataset/vegetable_images/test"

        dataset = ImageFolder(root=dataset_path, transform=transform)
        print(f"Total number of samples in the {'train' if train else 'test'} dataset: {len(dataset)}")
        return dataset
    
    def GetBatch(self, request, context):
        try:
            rank = request.rank
            num_replicas = request.num_replicas
            batch_idx = request.batch_idx
            batch_size = request.batch_size

            start = datetime.now()
            partitioned_dataset = ember_server.partition_dataset(self.dataset, rank, num_replicas)

            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(partitioned_dataset))

            if start_idx >= len(partitioned_dataset):
                print(f"All batches sent to client {rank}")
                return dist_data_pb2.TrainLoaderResponse(data=b"")
            print(f"Serving batch {batch_idx} for rank {rank}: {start_idx} to {end_idx}")
            batch = [partitioned_dataset[i] for i in range(start_idx, end_idx)]
            data = []
            to_pil = transforms.ToPILImage()
            for img, label in batch:
                if isinstance(img, torch.Tensor):
                    img = to_pil(img)
                
                buffer = io.BytesIO()
                img.save(buffer, format="PNG")
                data.append((buffer.getvalue(), label))

            response_data = pickle.dumps(data)
            print("Batch sent in: " + str(datetime.now() - start))
            return dist_data_pb2.TrainLoaderResponse(data=response_data)

        except Exception as e:
            context.set_details(f"Error: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            print(f"Error serving batch {batch_idx} for rank {rank}: {str(e)}")
            raise
        
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