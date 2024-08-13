import torch
import torch.nn as nn
import torch.optim as optim
from gensim.models import KeyedVectors
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
from datasets import load_dataset
import numpy as np
from utils import text_to_tokens  # Importiere text_to_tokens aus utils.py

# LSTM-Modell definieren
class Seq2SQLModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        super(Seq2SQLModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x

class Seq2SQLDataset(Dataset):
    def __init__(self, data, glove_model, max_len):
        self.data = data
        self.glove_model = glove_model
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Sicherstellen, dass idx als int interpretiert wird
        idx = int(idx)
        item = self.data[idx]
        text = item['question']
        sql_query = item['sql_query']
        
        input_ids = self._pad_sequence(text_to_tokens(text, self.glove_model))
        labels = self._pad_sequence(text_to_tokens(sql_query, self.glove_model))
        return {'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'labels': torch.tensor(labels, dtype=torch.long)}

    def _pad_sequence(self, sequence):
        if len(sequence) < self.max_len:
            sequence = sequence + [0] * (self.max_len - len(sequence))
        return sequence[:self.max_len]

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
seq2sql_dataset = Seq2SQLDataset(train_subset, glove_model, max_len)
dataloader = DataLoader(seq2sql_dataset, batch_size=batch_size, shuffle=True)

# Modell und Optimizer erstellen
vocab_size = len(glove_model.key_to_index)
model = Seq2SQLModel(embedding_dim, hidden_dim, vocab_size)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit='batch'):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids)
        loss = criterion(outputs.view(-1, vocab_size), labels.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {epoch_loss / len(dataloader)}")

print("Training abgeschlossen.")

# Beispiel für die SQL-Abfragevorhersage
def predict_sql_query(text):
    model.eval()
    input_ids = torch.tensor(text_to_tokens(text, glove_model), dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_ids)
        _, predicted = torch.max(outputs, dim=-1)
        return predicted.squeeze().tolist()

def decode_sql_tokens(tokens):
    reverse_vocab = {v: k for k, v in glove_model.key_to_index.items()}
    sql_query = ' '.join([reverse_vocab.get(token, '<UNK>') for token in tokens])
    return sql_query

# Testen der Vorhersage
test_text = "Find all orders greater than 100 dollars"
predicted_tokens = predict_sql_query(test_text)
decoded_query = decode_sql_tokens(predicted_tokens)
print("Predicted SQL Query:", decoded_query)
