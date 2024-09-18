import grpc
from concurrent import futures
import torch
import io
import dist_data_pb2
import dist_data_pb2_grpc
from torchvision import datasets, transforms
from torch.utils.data.distributed import DistributedSampler
import numpy as np

class TrainLoaderService(dist_data_pb2_grpc.TrainLoaderServiceServicer):
    def GetCifarTrainLoader(self, request, context):
        try:
            dataset = self.get_cifar100_data(train=True)

            train_sampler = DistributedSampler(
                dataset,
                num_replicas=request.num_replicas,
                rank=request.rank
            )

            image_data_list = []
            for image, label in dataset:
                image_bytes = image.numpy().tobytes()
                image_data_list.append(
                    dist_data_pb2.ImageData(image=image_bytes, label=label)
                )
            
            return dist_data_pb2.TrainLoaderResponse(
                data=image_data_list,
                batch_size=request.batch_size
            )
        except Exception as e:
            context.set_details(f"Error: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            raise

    def GetCifarTestLoader(self, request, context):
        try:
            dataset = self.get_cifar100_data(train=False)

            image_data_list = []
            for image, label in dataset:
                image_bytes = image.numpy().tobytes()
                image_data_list.append(
                    dist_data_pb2.ImageData(image=image_bytes, label=label)
                )
            
            return dist_data_pb2.TrainLoaderResponse(
                data=image_data_list,
                batch_size=100  # Adjust batch size as needed
            )
        except Exception as e:
            context.set_details(f"Error: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            raise

    def get_cifar100_data(self, train=True):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # CIFAR-100 normalization
        ])
        return datasets.CIFAR100(root='./data', train=train, download=True, transform=transform)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10),
                         options=[('grpc.max_send_message_length', 200 * 1024 * 1024),
                                  ('grpc.max_receive_message_length', 200 * 1024 * 1024)])
    dist_data_pb2_grpc.add_TrainLoaderServiceServicer_to_server(TrainLoaderService(), server)
    server.add_insecure_port('[::]:8040')
    server.start()
    print("Server started, listening on port 8040.")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()