import numpy as np
import torch
import re
from transformers import AutoTokenizer
from datasets import load_dataset
from models.agg_predictor import AggPredictor
from models.select_predictor import SelectPredictor

class GloveEmbeddings:
    def __init__(self, glove_file, embedding_dim):
        self.embedding_dim = embedding_dim
        self.word_to_index = {}
        self.index_to_word = []
        self.embeddings = []
        self.load_glove(glove_file)

    def load_glove(self, glove_file):
        with open(glove_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.split()
                word = parts[0]
                vector = np.array(parts[1:], dtype=np.float32)
                if len(vector) == self.embedding_dim:
                    self.word_to_index[word] = len(self.index_to_word)
                    self.index_to_word.append(word)
                    self.embeddings.append(vector)
        self.embeddings = np.array(self.embeddings)

    def get_embedding(self, word):
        idx = self.word_to_index.get(word)
        if idx is not None:
            return torch.tensor(self.embeddings[idx], dtype=torch.float32)
        else:
            return torch.zeros(self.embedding_dim, dtype=torch.float32)

    def get_vocab_size(self):
        return len(self.word_to_index)

    def get_embedding_dim(self):
        return self.embedding_dim

    def get_embeddings_matrix(self):
        return torch.tensor(self.embeddings, dtype=torch.float32)

def get_glove_embeddings(tokens, glove):
    embeddings = [glove.get_embedding(token) for token in tokens]
    return torch.stack(embeddings)

def extract_column_names(create_table_statement):
    # Extrahiere Spaltennamen aus CREATE TABLE Statement
    pattern = re.compile(r'\b(\w+)\s+\w+')
    matches = pattern.findall(create_table_statement)
    return matches

def preprocess_data(example, tokenizer, glove, agg_predictor, select_predictor):
    # Tokenisierung des Beispiels
    tokens = tokenizer(example['question'], padding='max_length', truncation=True, max_length=128, return_tensors="pt")
    input_ids = tokens['input_ids'].squeeze(0)
    attention_mask = tokens['attention_mask'].squeeze(0)

    # GloVe-Embeddings erhalten
    tokens_list = example['question'].split()  # Beispielhafte Tokenisierung
    input_encodings = get_glove_embeddings(tokens_list, glove)
    
    # Sicherstellen, dass die Dimensionen übereinstimmen
    input_encodings_tensor = torch.tensor(input_encodings, dtype=torch.float32).unsqueeze(0)  # [1, seq_len, embedding_dim]

    # Aggregationsvorhersagen
    agg_scores = agg_predictor(input_encodings_tensor)
    target_agg = torch.tensor([agg_scores.argmax().item()], dtype=torch.long)

    # Berechne Spalten-Embeddings
    create_table_statement = example['create_table_statement']
    column_names = extract_column_names(create_table_statement)
    col_embeddings = torch.stack([
        torch.tensor(get_glove_embeddings([col], glove), dtype=torch.float32).mean(dim=0)
        for col in column_names
    ])

    # Selektionsvorhersagen
    col_embeddings_tensor = col_embeddings.unsqueeze(0)  # [1, num_columns, embedding_dim]
    sel_scores = select_predictor(col_embeddings_tensor)
    target_sel = torch.tensor([sel_scores.argmax().item()], dtype=torch.long)

    generated_query = example.get('generated_query', '')  # Generierte Abfrage simulieren
    ground_truth = example['sql_query']  # Wahrheitsgemäße SQL-Abfrage

    # Rückgabe der verarbeiteten Daten
    return {
        'input_encodings': input_encodings_tensor.squeeze(0),
        'attention_mask': attention_mask,
        'col_embeddings': col_embeddings,
        'target_agg': target_agg,
        'target_sel': target_sel,
        'generated_query': generated_query,
        'ground_truth': ground_truth
    }



def load_and_preprocess_data(glove_file):
    dataset_name = "kaxap/pg-wikiSQL-sql-instructions-80k"
    dataset = load_dataset(dataset_name)

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    glove = GloveEmbeddings(glove_file, embedding_dim=64)

    # Initialisiere die Modelle
    agg_predictor = AggPredictor(embedding_dim=64, hidden_dim=128, num_agg_ops=4, glove_embeddings=glove)
    select_predictor = SelectPredictor(embedding_dim=64, hidden_dim=128, num_columns=10, glove_embeddings=glove)

    def preprocess(example):
        return preprocess_data(example, tokenizer, glove, agg_predictor, select_predictor)

    train_data = dataset['train'].map(preprocess, remove_columns=['question', 'sql_query'])

    return train_data, glove
