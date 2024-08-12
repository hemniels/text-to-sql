import torch
import torch.nn as nn
import numpy as np

class SimpleNet(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(SimpleNet, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        outputs = self.fc(lstm_out)
        return outputs

def load_glove_embeddings(glove_file, embedding_dim=100):
    word_to_index = {}
    embeddings = []
    with open(glove_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            if len(vector) == embedding_dim:
                word_to_index[word] = len(word_to_index)
                embeddings.append(vector)
    embeddings = np.array(embeddings)
    return word_to_index, embeddings

def create_embedding_layer(word_to_index, embeddings):
    vocab_size = len(word_to_index)
    embedding_dim = embeddings.shape[1]
    embedding_layer = nn.Embedding(vocab_size, embedding_dim)
    embedding_layer.weight.data.copy_(torch.from_numpy(embeddings))
    embedding_layer.weight.requires_grad = False
    return embedding_layer
