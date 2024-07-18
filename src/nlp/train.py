# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloader
from model import Seq2SQL

def train(model, dataloader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for batch in dataloader:
            input_seq, target_seq = batch
            optimizer.zero_grad()
            output = model(input_seq, target_seq)
            loss = criterion(output.view(-1, output.shape[-1]), target_seq.view(-1))
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

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
