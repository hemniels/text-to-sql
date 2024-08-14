import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import random


class Glove2SQLDataset(Dataset):
    def __init__(self, data, glove_model, max_len):
        self.data = data
        self.glove_model = glove_model
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Sicherstellen, dass idx als int interpretiert wird
        idx = int(idx)
        item = self.data[idx]
        text = item['question']
        sql_query = item['sql_query']
        
        input_ids = self._pad_sequence(text_to_tokens(text, self.glove_model))
        labels = self._pad_sequence(text_to_tokens(sql_query, self.glove_model))
        return {'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'labels': torch.tensor(labels, dtype=torch.long)}

    def _pad_sequence(self, sequence):
        if len(sequence) < self.max_len:
            sequence = sequence + [0] * (self.max_len - len(sequence))
        return sequence[:self.max_len]
    
class Transformer2SQLDataset(Dataset):
    def __init__(self, questions, sql_queries, tokenizer=None):
        self.questions = questions
        self.sql_queries = sql_queries
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        sql_query = self.sql_queries[idx]

        if self.tokenizer:
            encoding = self.tokenizer(
                question,
                truncation=True,
                padding='max_length',
                return_tensors='pt',
                max_length=128
            )
            labels = self.tokenizer(
                sql_query,
                truncation=True,
                padding='max_length',
                return_tensors='pt',
                max_length=128
            )

            return {
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze(),
                'labels': labels['input_ids'].squeeze()
            }
        return {}


def load_pg_wikiSQL_dataset():
    dataset = load_dataset('kaxap/pg-wikiSQL-sql-instructions-80k')
    return dataset


def text_to_tokens(text, glove_model):
    tokens = text.lower().split()
    return [glove_model.key_to_index.get(token, 0) for token in tokens]  # 0 als unbekanntes Token

def create_transformer_dataloaders(dataset, sample_size=50, batch_size=8, tokenizer=None):
    random.seed(42)
    sampled_indices = random.sample(range(len(dataset['train'])), sample_size)
    sampled_questions = [dataset['train'][i]['question'] for i in sampled_indices]
    sampled_sql_queries = [dataset['train'][i]['sql_query'] for i in sampled_indices]

    seq2sql_dataset = Transformer2SQLDataset(sampled_questions, sampled_sql_queries, tokenizer)
    dataloader = DataLoader(seq2sql_dataset, batch_size=batch_size, shuffle=True)

    return dataloader
