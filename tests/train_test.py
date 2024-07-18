import torch.nn as nn
import torch.optim as optim

from dataset import SQLDataset
from train import train
from model import Seq2SQL
from tests.dataset_test import get_dataloader

#Test of Training module

if __name__ == "__main__":
    data_path = 'data/pg-wikiSQL-sql-instructions-80k.json'
    glove_path = 'data/glove.6B/glove.6B.50d.txt'
    input_size = 400000  # Adjust based on GloVe vocabulary size
    hidden_size = 200
    output_size = 400000  # Adjust based on output vocabulary size
    dataloader = get_dataloader(data_path, glove_path)

    dataset = SQLDataset(data_path, glove_path)
    model = Seq2SQL(input_size, hidden_size, output_size, dataset.embeddings)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train(model, dataloader, criterion, optimizer, num_epochs=10)