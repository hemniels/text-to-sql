import torch
import torch.nn as nn
import sys
import os

# Füge das Hauptverzeichnis zum Python-Pfad hinzu
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class DecoderNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(DecoderNetwork, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, bidirectional=False, batch_first=True)
        self.W_ptr = nn.Linear(hidden_dim, 1)
        self.U_ptr = nn.Linear(hidden_dim, hidden_dim)
        self.V_ptr = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, input_sequence):
        output, (hn, cn) = self.lstm(input_sequence)
        return output, hn

    def scalar_attention(self, state, output_t):
        g = state
        s = output_t
        alpha = self.W_ptr(torch.tanh(self.U_ptr(g) + self.V_ptr(s)))
        return torch.argmax(alpha, dim=-1)
