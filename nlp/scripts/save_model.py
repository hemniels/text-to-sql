# save_model.py
import torch
from model import Seq2SQL
from dataset import SQLDataset

def save_model(model, path):
    torch.save(model.state_dict(), path)

if __name__ == "__main__":
    data_path = "kaxap/pg-wikiSQL-sql-instructions-80k"
    glove_path = 'data/glove.6B/glove.6B.50d.txt'
    input_size = 400000  # Adjust based on GloVe vocabulary size
    hidden_size = 200
    output_size = 400000  # Adjust based on output vocabulary size
    dataset = SQLDataset(data_path, glove_path)
    model = Seq2SQL(input_size, hidden_size, output_size, dataset.embeddings)
    save_model(model, 'seq2sql_model.pth')
