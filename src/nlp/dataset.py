# dataset.py
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader

class SQLDataset(Dataset):
    def __init__(self, data_path, glove_path):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
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

def get_dataloader(data_path, glove_path, batch_size=32):
    dataset = SQLDataset(data_path, glove_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

if __name__ == "__main__":
    data_loader = get_dataloader('data/pg-wikiSQL-sql-instructions-80k.json', 'data/glove.6B/glove.6B.50d.txt')
    for batch in data_loader:
        print(batch)
        break
