import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import random

# Laden Sie das Dataset
def load_pg_wikiSQL_dataset():
    dataset = load_dataset('kaxap/pg-wikiSQL-sql-instructions-80k')
    return dataset

# Dataset-Klasse für PyTorch
class Seq2SQLDataset(Dataset):
    def __init__(self, questions, sql_queries, tokenizer=None, glove_model=None):
        self.questions = questions
        self.sql_queries = sql_queries
        self.tokenizer = tokenizer
        self.glove_model = glove_model
        self.glove_vocab_size = len(glove_model.key_to_index) if glove_model else 0

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        sql_query = self.sql_queries[idx]

        if self.tokenizer:
            encoding = self.tokenizer(question, truncation=True, padding='max_length', return_tensors='pt')
            input_ids = encoding['input_ids'].squeeze()
            attention_mask = encoding['attention_mask'].squeeze()
            labels = self.tokenizer(sql_query, truncation=True, padding='max_length', return_tensors='pt')['input_ids'].squeeze()
            return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}
        
        if self.glove_model:
            question_tokens = text_to_tokens(question, self.glove_model)
            sql_tokens = text_to_tokens(sql_query, self.glove_model)
            return {'input_ids': torch.tensor(question_tokens, dtype=torch.long),
                    'labels': torch.tensor(sql_tokens, dtype=torch.long)}

        return {}

def text_to_tokens(text, glove_model):
    tokens = text.lower().split()
    return [glove_model.key_to_index.get(token, 0) for token in tokens]  # 0 als unbekanntes Token

def create_dataloaders(dataset, sample_size=50, batch_size=8, tokenizer=None, glove_model=None):
    # Sample random subsets
    random.seed(42)
    sampled_indices = random.sample(range(len(dataset['train'])), sample_size)
    
    # Verwenden der tatsächlichen Dataset-Schlüssel
    sampled_questions = [dataset['train'][i]['question'] for i in sampled_indices]
    sampled_sql_queries = [dataset['train'][i]['sql_query'] for i in sampled_indices]

    # Erstellen des Datasets und Dataloaders
    seq2sql_dataset = Seq2SQLDataset(sampled_questions, sampled_sql_queries, tokenizer, glove_model)
    dataloader = DataLoader(seq2sql_dataset, batch_size=batch_size, shuffle=True)

    return dataloader
