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
import numpy as np

import io
import grpc
import pickle
import io
import dist_data_pb2
import dist_data_pb2_grpc
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


def fetch_train_loader(api_host, api_port, num_replicas, rank, batch_size):
    max_message_length = 200 * 1024 * 1024  # 200MB
    channel = grpc.insecure_channel(
        f'{api_host}:{api_port}', 
        options=[
            ('grpc.max_send_message_length', max_message_length),
            ('grpc.max_receive_message_length', max_message_length)
        ]
    )
    
    stub = dist_data_pb2_grpc.TrainLoaderServiceStub(channel)
    request = dist_data_pb2.TrainLoaderRequest(
        num_replicas=num_replicas,
        rank=rank,
        batch_size=batch_size
    )
    response = stub.GetTrainLoader(request)

    images = []
    labels = []
    for image_data in response.data:     
        image = torch.Tensor(np.frombuffer(image_data.image, dtype=np.float32).reshape(1, 28, 28))
        images.append(image)
        labels.append(image_data.label)

    dataset = TensorDataset(torch.stack(images), torch.tensor(labels))
    train_sampler = DistributedSampler(dataset, num_replicas=num_replicas, rank=rank)

    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, sampler=train_sampler)
    
    return train_loader


def fetch_test_loader(api_host, api_port):
    max_message_length = 200 * 1024 * 1024 
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

    images = []
    labels = []
    for image_data in response.data:
        image = torch.Tensor(np.frombuffer(image_data.image, dtype=np.float32).reshape(1, 28, 28))
        images.append(image)
        labels.append(image_data.label)
    
    dataset = TensorDataset(torch.stack(images), torch.tensor(labels))
    
    test_loader = DataLoader(dataset=dataset, batch_size=100, shuffle=False, num_workers=0, pin_memory=True)
    
    return test_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run')
    args = parser.parse_args()
    args.world_size = args.gpus * args.nodes
    os.environ['MASTER_ADDR'] = 'grworker1'
    os.environ['MASTER_PORT'] = '8888'
    start = datetime.now()
    mp.spawn(train, nprocs=args.gpus, args=(args,))
    print("Training complete in: " + str(datetime.now() - start))


class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

def train(gpu, args):
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)
    torch.manual_seed(0)
    model = ConvNet()
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    batch_size = 100
    
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    optimizer = torch.optim.SGD(model.parameters(), 1e-4)

    # Wrap the model
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    # Data loading code
    train_loader = fetch_train_loader(api_host="grserver", api_port="8040", num_replicas=args.world_size, rank=rank, batch_size=batch_size)


    total_step = len(train_loader)
    for epoch in range(args.epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0 and gpu == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, args.epochs, i + 1, total_step,
                                                                         loss.item()))
    if gpu == 0:
        print("Training complete")
    if rank == 0:
        print("Teste gpu 0")
        test_loader = fetch_test_loader(api_host="grserver", api_port="8040")
        model.eval()
        
        all_labels = []
        all_predictions = []
        total_loss = 0
        total_samples = 0
        
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
        
        accuracy = 100 * correct / total
        average_loss = total_loss / total
        precision = precision_score(all_labels, all_predictions, average='weighted')
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        print('Test Accuracy of the model on the test images: {} %'.format(accuracy))
        print('Test Loss: {:.4f}'.format(average_loss))
        print('Test Precision: {:.4f}'.format(precision))
        print('Test F1 Score: {:.4f}'.format(f1))

if __name__ == '__main__':
    main()
