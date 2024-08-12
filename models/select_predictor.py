# models/select_predictor.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class SelectPredictor(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_columns, glove_embeddings):
        super(SelectPredictor, self).__init__()
        self.glove_embeddings = glove_embeddings
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=False, batch_first=True)
        self.W_sel = nn.Linear(hidden_dim, hidden_dim)
        self.V_sel = nn.Linear(hidden_dim, hidden_dim)
        self.num_columns = num_columns

    def column_representations(self, col_embeddings):
        output, (hn, cn) = self.lstm(col_embeddings)
        e_c_j = hn[-1]
        return e_c_j

    def question_representation(self, e_c_j):
        norm_e_c_j = F.softmax(e_c_j, dim=-1)
        agg_sel = torch.sum(norm_e_c_j * e_c_j, dim=0)
        return agg_sel

    def compute_sel_score(self, col_embeddings):
        e_c_j = self.column_representations(col_embeddings)
        score_agg_sel = self.question_representation(e_c_j)
        score = self.W_sel(torch.tanh(self.V_sel(score_agg_sel)))
        return F.softmax(score, dim=-1)

    def train_step(self, col_embeddings, target_sel):
        pred_scores = self.compute_sel_score(col_embeddings)
        loss = F.cross_entropy(pred_scores, target_sel)
        return loss

    def forward(self, col_embeddings):
        return self.compute_sel_score(col_embeddings)
