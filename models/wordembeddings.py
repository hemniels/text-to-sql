# models/wordembeddings.py

import torch
import torch.nn as nn
import torch.optim as optim

class WordEmbeddingsNet(nn.Module):
    def __init__(self, vocab_size, embedding_dim, window_size=5):
        super(WordEmbeddingsNet, self).__init__()
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.optimizer = optim.SGD(self.parameters(), lr=0.01)
        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, center_words, context_words):
        center_embeddings = self.embeddings(center_words)
        context_embeddings = self.embeddings(context_words)
        scores = torch.matmul(center_embeddings, context_embeddings.t())
        return scores

    def train_model(self, corpus, word_to_index):
        """
        Trainiert das Modell auf dem gegebenen Korpus.
        """
        word_pairs = self.create_word_pairs(corpus, word_to_index)
        for center, context in word_pairs:
            center_word = torch.tensor([center], dtype=torch.long)
            context_word = torch.tensor([context], dtype=torch.long)

            self.optimizer.zero_grad()
            output = self(center_word, context_word)
            # Die Loss-Funktion erwartet, dass `context_word` der Target-Index ist
            loss = self.loss_function(output.squeeze(0), context_word)
            loss.backward()
            self.optimizer.step()

    def create_word_pairs(self, corpus, word_to_index):
        """
        Erstellt Paare von Wörtern für das Training, konvertiert Wörter in Indizes.
        """
        pairs = []
        for sentence in corpus:
            sentence_indices = [word_to_index[word] for word in sentence if word in word_to_index]
            for i, center_word_idx in enumerate(sentence_indices):
                start = max(0, i - self.window_size)
                end = min(len(sentence_indices), i + self.window_size + 1)
                for j in range(start, end):
                    if i != j:
                        context_word_idx = sentence_indices[j]
                        pairs.append((center_word_idx, context_word_idx))
        return pairs

    def get_embedding(self, word_index):
        """
        Gibt das Embedding für ein Wort zurück.
        """
        with torch.no_grad():
            return self.embeddings(torch.tensor([word_index], dtype=torch.long)).numpy()

    def save_model(self, path):
        """
        Speichert das Modell.
        """
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        """
        Lädt ein gespeichertes Modell.
        """
        self.load_state_dict(torch.load(path))
        self.eval()
