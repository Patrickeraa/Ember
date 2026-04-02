import io
import pickle
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import dist_data_pb2
import grpc


class TextDataset(Dataset):
    def __init__(self, path, tokenizer, max_length=128, text_field='text', label_field='label', file_type='csv'):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_field = text_field
        self.label_field = label_field

        if file_type == 'csv':
            df = pd.read_csv(path)
            self.texts = df[text_field].astype(str).tolist()
            self.labels = df[label_field].tolist()
        elif file_type == 'jsonl':
            df = pd.read_json(path, lines=True)
            self.texts = df[text_field].astype(str).tolist()
            self.labels = df[label_field].tolist()
        elif file_type == 'txt':
            files = sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith('.txt')])
            self.texts = []
            self.labels = []
            for f in files:
                with open(f, 'r', encoding='utf-8') as fh:
                    self.texts.append(fh.read())
                self.labels.append(0)
        else:
            raise ValueError("file_type deve ser 'csv', 'jsonl' ou 'txt_dir'")

        assert len(self.texts) == len(self.labels), "texts and labels must be same length"

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = int(self.labels[idx])

        encoded = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        encoded = {k: v.squeeze(0) for k, v in encoded.items()}
        return encoded, label
