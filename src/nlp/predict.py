# predict.py
import torch
from model import Seq2SQL
from dataset import SQLDataset

def predict(model, input_prompt, tokenizer):
    model.eval()
    input_seq = tokenizer(input_prompt)
    input_seq = torch.tensor(input_seq).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(input_seq)
    return output