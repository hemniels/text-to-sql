# train.py
import torch
import torch.nn as nn
import torch.optim as optim
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