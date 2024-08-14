import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from gensim.models import KeyedVectors
from text2sql.utils import Glove2SQLDataset, load_dataset
from text2sql.models import SimpleSeq2SQLModel

# Lade das Dataset von Hugging Face
dataset = load_dataset("kaxap/pg-wikiSQL-sql-instructions-80k", split='train')  # Lade den 'train'-Split

# Nehme 50 zufällige Samples
indices = np.random.choice(len(dataset), 50, replace=False)
indices = list(map(int, indices))  # Stelle sicher, dass alle Indizes als int vorliegen
train_subset = Subset(dataset, indices)

# Lade die GloVe-Embeddings
glove_path = 'data/glove.6B.100d.txt'
glove_model = KeyedVectors.load_word2vec_format(glove_path, binary=False, no_header=True)

# Hyperparameter
embedding_dim = 100
hidden_dim = 128
num_epochs = 25
batch_size = 16
max_len = 50  # Beispiel-Länge, sollte passend zur Länge der Daten gewählt werden

# Dataset und Dataloader erstellen

seq2sql_dataset = Glove2SQLDataset(train_subset, glove_model, max_len)
dataloader = DataLoader(seq2sql_dataset, batch_size=batch_size, shuffle=True)

# Modell und Optimizer erstellen
vocab_size = len(glove_model.key_to_index)
model = SimpleSeq2SQLModel(embedding_dim, hidden_dim, vocab_size)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

SimpleSeq2SQLModel.train_model(dataloader, criterion, optimizer, num_epochs)

# Testen der Vorhersage
test_text = "Find all orders greater than 100 dollars"
predicted_tokens = SimpleSeq2SQLModel.predict_sql_query(test_text)
decoded_query = SimpleSeq2SQLModel.decode_sql_tokens(predicted_tokens)
print("Predicted SQL Query:", decoded_query)