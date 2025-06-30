import threading, queue, pickle, io
import grpc
import torch
from torch.utils.data import IterableDataset, DataLoader
from rpc import dist_data_pb2, dist_data_pb2_grpc

class RPCIterableDataset(IterableDataset):
    def __init__(self, api_host, api_port, world_size, rank, batch_size=10000):
        self.api_host = api_host
        self.api_port = api_port
        self.world_size = world_size
        self.rank = rank
        self.batch_size = batch_size

    def fetch_loop(self, q: queue.Queue):
        """Background thread: fetch pickled batches and push to queue until done."""
        max_msg = 1000 * 1024 * 1024
        channel = grpc.insecure_channel(
            f'{self.api_host}:{self.api_port}',
            options=[
                ('grpc.max_send_message_length', max_msg),
                ('grpc.max_receive_message_length', max_msg),
            ]
        )
        stub = dist_data_pb2_grpc.TrainLoaderServiceStub(channel)
        batch_idx = 0

        while True:
            req = dist_data_pb2.BatchRequest(
                num_replicas=self.world_size,
                rank=self.rank,
                batch_idx=batch_idx,
                batch_size=self.batch_size
            )
            resp = stub.GetBatch(req)
            if not resp.data:
                break
            raw = pickle.loads(resp.data)
            # Push each sample, or you could push the entire batch if you prefer
            for tensor_bytes, label_data in raw:
                buf = io.BytesIO(tensor_bytes)
                img = torch.load(buf)
                lbl = torch.tensor(label_data, dtype=torch.long)
                q.put((img, lbl))
            batch_idx += 1

        # signal end‑of‑stream
        q.put(None)

    def __iter__(self):
        q = queue.Queue(maxsize=2 * self.batch_size)  # cap queue size for back‐pressure
        # Start gRPC fetch thread
        t = threading.Thread(target=self.fetch_loop, args=(q,), daemon=True)
        t.start()

        while True:
            item = q.get()
            if item is None:
                break
            yield item
