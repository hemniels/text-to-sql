# dataset.py
import json
import numpy as np
from datasets import load_dataset
from torch.utils.data import Dataset

class SQLDataset(Dataset):
    def __init__(self, data_path, glove_path):
        self.data = load_dataset(data_path)
        self.word_to_idx, self.embeddings = self.load_glove(glove_path)
        self.tokenizer = self.tokenize

    def load_glove(self, glove_path):
        word_to_idx = {}
        embeddings = []
        with open(glove_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                word_to_idx[word] = idx
                embeddings.append(vector)
        embeddings = np.stack(embeddings)
        return word_to_idx, embeddings

    def tokenize(self, text):
        return [self.word_to_idx.get(word, 0) for word in text.split()]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_seq = self.tokenizer(item['input_prompt'])
        target_seq = self.tokenizer(item['sql_query'])
        return input_seq, target_seq