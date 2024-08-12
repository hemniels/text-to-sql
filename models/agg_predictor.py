import torch
import torch.nn as nn
import torch.nn.functional as F

class AggPredictor(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_agg_ops, glove_embeddings):
        super(AggPredictor, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_agg_ops = num_agg_ops
        self.glove_embeddings = glove_embeddings

        # Initialisiere die linearen Schichten
        self.W_inp = nn.Linear(embedding_dim, hidden_dim)  # [embedding_dim -> hidden_dim]
        self.W_agg = nn.Linear(hidden_dim, hidden_dim)  # [hidden_dim -> hidden_dim]
        self.V_agg = nn.Linear(hidden_dim, num_agg_ops)  # [hidden_dim -> num_agg_ops]
        self.b_agg = nn.Parameter(torch.zeros(hidden_dim))  # Bias für die Aggregation (hidden_dim)

    def compute_alpha(self, input_encodings):
        alpha = self.W_inp(input_encodings)  # [batch_size, seq_len, hidden_dim]
        alpha = torch.bmm(alpha, alpha.transpose(1, 2))  # [batch_size, seq_len, seq_len]
        alpha = F.softmax(alpha, dim=-1)  # [batch_size, seq_len, seq_len]
        return alpha

    def input_representation(self, input_encodings):
        alpha = self.compute_alpha(input_encodings)  # [batch_size, seq_len, seq_len]
        a = torch.bmm(alpha, input_encodings)  # [batch_size, seq_len, hidden_dim]
        a = torch.sum(a, dim=1)  # [batch_size, hidden_dim]
        return a

    def compute_agg_scores(self, input_encodings):
        a = self.input_representation(input_encodings)  # [batch_size, hidden_dim]
        a = F.tanh(self.W_agg(a) + self.b_agg)  # [batch_size, hidden_dim]
        score = self.V_agg(a)  # [batch_size, num_agg_ops]
        score = F.softmax(score, dim=-1)  # [batch_size, num_agg_ops]
        return score

    def train_step(self, input_encodings, target_agg):
        scores = self.compute_agg_scores(input_encodings)
        loss = F.cross_entropy(scores, target_agg)
        return loss

    def forward(self, input_encodings):
        return self.compute_agg_scores(input_encodings)
