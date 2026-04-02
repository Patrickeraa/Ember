import threading, queue, pickle, io
import grpc
import torch
from torch.utils.data import IterableDataset, DataLoader
from rpc import dist_data_pb2, dist_data_pb2_grpc
import time
from torchvision import transforms


class RPCIterableDataset(IterableDataset):
    def __init__(self, api_host, api_port, world_size, rank, request_size=10000, writer=None):
        self.api_host     = api_host
        self.api_port     = api_port
        self.world_size   = world_size
        self.rank         = rank
        self.request_size = request_size
        self.current_epoch = 0
        self.writer = writer

        self.total_bytes = 0
        self.start_time = None

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch

    def fetch_loop(self, output_queue):
        max_msg = 1000 * 1024 * 1024
        channel = grpc.insecure_channel(
            f"{self.api_host}:{self.api_port}",
            options=[
                ('grpc.max_send_message_length', max_msg),
                ('grpc.max_receive_message_length', max_msg),
            ]
        )
        stub = dist_data_pb2_grpc.TrainLoaderServiceStub(channel)
        batch_idx = 0

        self.start_time = time.time()

        while True:
            req = dist_data_pb2.BatchRequest(
                num_replicas=self.world_size,
                rank=self.rank,
                batch_idx=batch_idx,
                batch_size=self.request_size,
                epoch=self.current_epoch,
            )
            resp = stub.GetBatch(req)
            if not resp.data:
                break

            raw = pickle.loads(resp.data)
            for tensor_bytes, label_data in raw:
                self.total_bytes += len(tensor_bytes)
                buf = io.BytesIO(tensor_bytes)
                img = torch.load(buf)
                lbl = torch.tensor(label_data, dtype=torch.long)
                output_queue.put((img, lbl))

            batch_idx += 1
            elapsed = time.time() - self.start_time
            if elapsed > 0 and self.writer is not None:
                mb_processed = self.total_bytes / (1024 * 1024)
                throughput = mb_processed / elapsed
                self.writer.add_scalar("dataset/throughput_MBps", throughput, self.current_epoch)
            output_queue.put(None)

    def __iter__(self):
        q = queue.Queue(maxsize=2 * self.request_size)
        thread = threading.Thread(target=self.fetch_loop, args=(q,), daemon=True)
        thread.start()

        while True:
            item = q.get()
            if item is None:
                break
            yield item