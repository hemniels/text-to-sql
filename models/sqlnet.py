import torch
import torch.nn as nn
import torch.nn.functional as F
from models import AggPredictor,SelectPredictor,WherePredictor

class SQLNet(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_agg_ops, num_columns, glove_embeddings):
        super(SQLNet, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_agg_ops = num_agg_ops
        self.num_columns = num_columns
        self.glove_embeddings = glove_embeddings
        
        # Modelle für die Vorhersagen
        self.agg_predictor = AggPredictor(embedding_dim, hidden_dim, num_agg_ops, glove_embeddings)
        self.select_predictor = SelectPredictor(embedding_dim, hidden_dim, num_columns, glove_embeddings)
        self.where_predictor = WherePredictor(hidden_dim, glove_embeddings)
        
    def forward(self, input_encodings, col_embeddings):
        # Vorhersage der Aggregationsoperationen
        agg_scores = self.agg_predictor(input_encodings)
        # Vorhersage der SELECT-Klausel
        select_scores = self.select_predictor(col_embeddings)
        # Vorhersage der WHERE-Klausel
        where_scores = self.where_predictor(input_encodings)
        
        return agg_scores, select_scores, where_scores

    def compute_loss(self, input_encodings, col_embeddings, target_agg, target_sel, target_where):
        agg_scores, select_scores, where_scores = self(input_encodings, col_embeddings)
        
        # Berechnung der Verluste
        agg_loss = self.agg_predictor.train_step(input_encodings, target_agg)
        sel_loss = F.cross_entropy(select_scores, target_sel)
        where_loss = F.cross_entropy(where_scores, target_where)
        
        total_loss = agg_loss + sel_loss + where_loss
        return total_loss

