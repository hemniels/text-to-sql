import sys
import os
import numpy as np
import torch

# Füge das Hauptverzeichnis zum Python-Pfad hinzu
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class Utils:
    @staticmethod
    def input_sequence(col_names, sql_vocab, prompt):
        # Konkatenieren der Spaltennamen, des SQL-Vokabulars und des Prompts zu einer Eingabesequenz
        return col_names + sql_vocab + prompt

class GloveEmbeddings:
    def __init__(self, glove_file, embedding_dim):
        self.embedding_dim = embedding_dim
        self.word_to_index = {}
        self.index_to_word = []
        self.embeddings = []

        # Laden der GloVe-Datei
        self.load_glove(glove_file)

    def load_glove(self, glove_file):
        with open(glove_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.split()
                word = parts[0]
                vector = np.array(parts[1:], dtype=np.float32)
                if len(vector) == self.embedding_dim:
                    self.word_to_index[word] = len(self.index_to_word)
                    self.index_to_word.append(word)
                    self.embeddings.append(vector)
        
        self.embeddings = np.array(self.embeddings)
    
    def get_embedding(self, word):
        idx = self.word_to_index.get(word)
        if idx is not None:
            return torch.tensor(self.embeddings[idx], dtype=torch.float32)
        else:
            # Rückgabewert für unbekannte Wörter, z.B. ein Vektor von Nullen
            return torch.zeros(self.embedding_dim, dtype=torch.float32)

    def get_vocab_size(self):
        return len(self.word_to_index)
    
    def get_embedding_dim(self):
        return self.embedding_dim

    def get_embeddings_matrix(self):
        return torch.tensor(self.embeddings, dtype=torch.float32)

# Beispiel für eine Funktion zur Tokenisierung und Umwandlung in GloVe-Embeddings
def get_glove_embeddings(text, glove):
    tokens = text.split()  # Tokenisierung des Textes
    embeddings = [glove.get_embedding(token) for token in tokens]
    return torch.stack(embeddings)
