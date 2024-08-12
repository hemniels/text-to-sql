import torch
import torch.nn as nn
import sys
import os

# Füge das Hauptverzeichnis zum Python-Pfad hinzu
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class SqlNet(nn.Module):
    def __init__(self, agg_pred, sel_pred, where_pred):
        super(SqlNet, self).__init__()
        self.agg_pred = agg_pred
        self.sel_pred = sel_pred
        self.where_pred = where_pred

    def forward(self, col_embeddings, input_encodings):
        agg_scores = self.agg_pred(input_encodings)
        sel_scores = self.sel_pred(col_embeddings)
        where_scores = self.where_pred(input_encodings)
        return agg_scores, sel_scores, where_scores

    def train_step(self, col_embeddings, input_encodings, target_agg, target_sel, generated_query, ground_truth):
        loss_agg = self.agg_pred.train_step(input_encodings, target_agg)
        loss_sel = self.sel_pred.train_step(col_embeddings, target_sel)
        loss_whe = self.where_pred.train_step(input_encodings, generated_query, ground_truth)
        total_loss = loss_agg + loss_sel + loss_whe
        return total_loss

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def store(self, path):
        torch.save(self.state_dict(), path)
