import threading
import queue
import torch
from torch.utils.data import DataLoader, TensorDataset
import dist_data_pb2
import dist_data_pb2_grpc
import grpc
import pickle
import time
from PIL import Image
import numpy as np
import io


batch_queue = queue.Queue()
all_images = []
all_labels = []

def fetch_batches_in_thread(api_host, api_port, num_replicas, rank):
    max_message_length = 1000 * 1024 * 1024
    channel = grpc.insecure_channel(
        f'{api_host}:{api_port}',
        options=[
            ('grpc.max_send_message_length', max_message_length),
            ('grpc.max_receive_message_length', max_message_length)
        ]
    )
    
    stub = dist_data_pb2_grpc.TrainLoaderServiceStub(channel)
    
    batch_idx = 0
    while True:
        request = dist_data_pb2.BatchRequest(
            num_replicas=num_replicas,
            rank=rank,
            batch_idx=batch_idx,
            batch_size=10000
        )

        response = stub.GetBatch(request)

        if not response.data:
            batch_queue.put(None)
            break

        batch = pickle.loads(response.data)
        converted_batch = []
        for img_data, label_data in batch:
            # Deserialize image
            img_bytes = io.BytesIO(img_data)
            img = Image.open(img_bytes)
            img = img.convert('RGB')
            img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1)
            label_tensor = torch.tensor(label_data)
            converted_batch.append((img_tensor, label_tensor))

        batch_queue.put(converted_batch)
        batch_idx += 1

def create_dataloader_from_queue(batch_queue, stop_signal=None):
    while not batch_queue.empty():
        batch = batch_queue.get()
        if batch is None:
            break
        for img_tensor, label in batch:
            all_images.append(img_tensor)
            all_labels.append(label)
    if all_images and all_labels:
        images_tensor = torch.stack(all_images)
        labels_tensor = torch.tensor(all_labels)
        dataset = TensorDataset(images_tensor, labels_tensor)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        return dataloader
    else:
        return None


def main():
    api_host = "grserver"  
    api_port = 8040
    num_replicas = 1   
    rank = 0         

    fetch_thread = threading.Thread(
        target=fetch_batches_in_thread, 
        args=(api_host, api_port, num_replicas, rank)
    )
    fetch_thread.start()

    for _ in range(10):
        dataloader = create_dataloader_from_queue(batch_queue)
        if dataloader:
            total_images = sum(len(batch[0]) for batch in dataloader)
            total_labels = sum(len(batch[1]) for batch in dataloader)

            print(f"Total number of images: {total_images}")
            print(f"Total number of labels: {total_labels}")
        else:
            print("No new data available at this moment.")

        time.sleep(2)
    fetch_thread.join()

if __name__ == "__main__":
    main()
