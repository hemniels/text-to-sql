# models/agg_predictor.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class AggPredictor(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, glove_embeddings):
        super(AggPredictor, self).__init__()
        self.glove_embeddings = glove_embeddings
        self.hidden_dim = hidden_dim
        self.W_inp = nn.Linear(hidden_dim, 1)
        self.W_agg = nn.Linear(hidden_dim, hidden_dim)
        self.V_agg = nn.Linear(hidden_dim, hidden_dim)
        self.b_agg = nn.Parameter(torch.zeros(1))
        self.c_agg = nn.Parameter(torch.zeros(1))

    def compute_alpha(self, input_encodings):
        alpha = self.W_inp(input_encodings)
        alpha_normalize = F.softmax(alpha, dim=-1)
        return alpha_normalize

    def input_representation(self, input_encodings, alpha):
        a = torch.sum(alpha * input_encodings, dim=1)
        return a

    def compute_agg_scores(self, input_encodings):
        alpha = self.compute_alpha(input_encodings)
        a = self.input_representation(input_encodings, alpha)
        score = self.W_agg(torch.tanh(self.V_agg(a) + self.b_agg))
        score = F.softmax(score, dim=-1)
        return score

    def train_step(self, input_encodings, target_agg):
        pred_scores = self.compute_agg_scores(input_encodings)
        loss = F.cross_entropy(pred_scores, target_agg)
        return loss

    def forward(self, input_encodings):
        return self.compute_agg_scores(input_encodings)
