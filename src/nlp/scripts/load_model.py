# load_model.py
import torch
from model import Seq2SQL
from dataset import SQLDataset

def load_model(path, input_size, hidden_size, output_size, embeddings):
    model = Seq2SQL(input_size, hidden_size, output_size, embeddings)
    model.load_state_dict(torch.load(path))
    return model

if __name__ == "__main__":
    data_path = 'data/pg-wikiSQL-sql-instructions-80k.json'
    glove_path = 'data/glove.6B/glove.6B.50d.txt'
    input_size = 400000  # Adjust based on GloVe vocabulary size
    hidden_size = 200
    output_size = 400000  # Adjust based on output vocabulary size
    dataset = SQLDataset(data_path, glove_path)
    model = load_model('seq2sql_model.pth', input_size, hidden_size, output_size, dataset.embeddings)
    print(model)
