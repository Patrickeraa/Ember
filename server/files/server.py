import grpc
from concurrent import futures
import torch
import pickle
import io
from torchvision import datasets, transforms
from torch.utils.data.distributed import DistributedSampler
import numpy as np

import dist_data_pb2
import dist_data_pb2_grpc

class TrainLoaderService(dist_data_pb2_grpc.TrainLoaderServiceServicer):
    def GetTrainLoader(self, request, context):
        try:
            dataset = self.get_svhn_train()

            train_sampler = DistributedSampler(
                dataset,
                num_replicas=request.num_replicas,
                rank=request.rank
            )
            
            buffer = io.BytesIO()
            pickle.dump((dataset, train_sampler, request.batch_size), buffer)
            buffer.seek(0)

            return dist_data_pb2.TrainLoaderResponse(data=buffer.read())
        except Exception as e:
            context.set_details(f"Error: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            raise

    def GetTestLoader(self, request, context):
        try:
            dataset = self.get_svhn_test()
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

        
    def GetCifarTrainLoader(self, request, context):
        try:
            dataset = self.get_cifar100_data(train=True)

            train_sampler = DistributedSampler(
                dataset,
                num_replicas=request.num_replicas,
                rank=request.rank
            )

            buffer = io.BytesIO()
            torch.save((dataset, train_sampler, request.batch_size), buffer)
            buffer.seek(0)
            
            return dist_data_pb2.TrainLoaderResponse(data=buffer.read())

        except Exception as e:
            context.set_details(f"Error: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            raise

    def GetCifarTestLoader(self, request, context):
        try:
            dataset = self.get_cifar100_data(train=False)

            buffer = io.BytesIO()
            pickle.dump(dataset, buffer)
            buffer.seek(0)

            return dist_data_pb2.TrainLoaderResponse(data=buffer.read())
        except Exception as e:
            context.set_details(f"Error: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            raise

    def get_mnist_data(self, train=True):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        return datasets.MNIST(root='./data', train=train, download=True, transform=transform)
    
    def get_svhn_train(self):
        transform = transforms.Compose([
                        transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_dataset = datasets.SVHN(root='/app/dataset', split='train', download=False, transform=transform)

        return train_dataset
    
    def get_svhn_test(self):
        transform = transforms.Compose([
                        transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        test_dataset = datasets.SVHN(root='/app/dataset', split='test', download=False, transform=transform)

        return test_dataset

    def get_cifar100_data(self, train=True):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # CIFAR-100 normalization
        ])
        return datasets.CIFAR100(root='./data', train=train, download=True, transform=transform)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10),
                         options=[('grpc.max_send_message_length', 500 * 1024 * 1024),
                                  ('grpc.max_receive_message_length', 500 * 1024 * 1024)])
    dist_data_pb2_grpc.add_TrainLoaderServiceServicer_to_server(TrainLoaderService(), server)
    server.add_insecure_port('[::]:8040')
    server.start()
    print("Server started, listening on port 8040.")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()