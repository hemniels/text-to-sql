import torch
import torch.nn as nn
import torch.nn.functional as F

class WherePredictor(nn.Module):
    def __init__(self, hidden_dim, glove_embeddings):
        super(WherePredictor, self).__init__()
        self.glove_embeddings = glove_embeddings
        self.hidden_dim = hidden_dim
        self.pointer_network = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.W_ptr = nn.Linear(hidden_dim, hidden_dim)
        self.U_ptr = nn.Linear(hidden_dim, hidden_dim)
        self.V_ptr = nn.Linear(hidden_dim, hidden_dim)

    def scalar_attention(self, scores, hidden_states):
        attn_weights = F.softmax(scores, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), hidden_states).squeeze(1)
        return context

    def compute_scores(self, hidden_states, query):
        return torch.matmul(query, hidden_states.permute(0, 2, 1))

    def train_step(self, input_encodings, generated_query, ground_truth):
        hidden_states, _ = self.pointer_network(input_encodings)
        scores = self.compute_scores(hidden_states, generated_query)
        context = self.scalar_attention(scores, hidden_states)
        loss = F.cross_entropy(context, ground_truth)
        return loss

    def forward(self, input_encodings, generated_query):
        hidden_states, _ = self.pointer_network(input_encodings)
        scores = self.compute_scores(hidden_states, generated_query)
        context = self.scalar_attention(scores, hidden_states)
        return context
