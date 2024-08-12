# scripts/train.py

import torch
from models.agg_predictor import AggPredictor
from models.select_predictor import SelectPredictor
from models.where_predictor import WherePredictor
from utils import GloveEmbeddings

def train():
    # Hyperparameter
    embedding_dim = 50
    hidden_dim = 64
    num_columns = 100  # Beispielwert
    glove_file = 'path/to/glove.txt'
    
    # Initialisierung der GloVe-Embeddings
    glove_embeddings = GloveEmbeddings(glove_file, embedding_dim)
    
    # Erstellen der Modelle
    agg_pred = AggPredictor(embedding_dim, hidden_dim, glove_embeddings)
    sel_pred = SelectPredictor(embedding_dim, hidden_dim, num_columns, glove_embeddings)
    where_pred = WherePredictor(hidden_dim, glove_embeddings)
    
    # Dummy-Daten
    input_encodings = torch.randn(32, 10, hidden_dim)  # [batch_size, seq_len, hidden_dim]
    col_embeddings = torch.randn(32, num_columns, embedding_dim)  # [batch_size, num_columns, embedding_dim]
    target_agg = torch.randint(0, 5, (32,))  # Beispielziel für Aggregationen
    target_sel = torch.randint(0, num_columns, (32,))  # Beispielziel für die Auswahl
    generated_query = "SELECT * FROM table WHERE condition"
    ground_truth = "SELECT * FROM table WHERE condition"
    
    # Trainiere die Modelle
    loss_agg = agg_pred.train_step(col_embeddings, target_agg)
    loss_sel = sel_pred.train_step(col_embeddings, target_sel)
    loss_whe = where_pred.train_step(input_encodings, generated_query, ground_truth)
    
    print(f"Aggregation Loss: {loss_agg.item()}")
    print(f"Selection Loss: {loss_sel.item()}")
    print(f"Where Clause Loss: {loss_whe.item()}")

if __name__ == '__main__':
    train()
