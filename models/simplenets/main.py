import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from simplenet_base import SimpleNet, load_glove_embeddings, create_embedding_layer

class ExampleDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

def train_model(model, dataloader, epochs=50):
    model.train()
    optimizer = optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Flatten outputs and labels for CrossEntropyLoss
            outputs_reshaped = outputs.view(-1, outputs.size(-1))
            labels_reshaped = labels.view(-1)

            # Ensure the labels_reshaped is valid
            if labels_reshaped.size(0) != outputs_reshaped.size(0):
                print(f"Error: mismatched shapes. Labels shape: {labels_reshaped.size()}, Outputs shape: {outputs_reshaped.size()}")
                continue
            
            loss = criterion(outputs_reshaped, labels_reshaped)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f'Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}')

def generate_sql(model, input_seq, index_to_word, word_to_index, device):
    model.eval()
    input_seq = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_seq)
        output = output.squeeze(0)
        _, predicted_ids = torch.max(output, dim=1)
        predicted_words = [index_to_word[idx.item()] for idx in predicted_ids]
        return ' '.join(predicted_words)

def main():
    # Load GloVe embeddings
    glove_file = 'data/glove.6B.100d.txt'
    word_to_index, embeddings = load_glove_embeddings(glove_file, embedding_dim=100)

    # Create embedding layer
    embedding_layer = create_embedding_layer(word_to_index, embeddings)

    # Initialize model
    vocab_size = len(word_to_index)
    embedding_dim = embeddings.shape[1]
    hidden_dim = 64
    output_dim = vocab_size
    model = SimpleNet(vocab_size, embedding_dim, hidden_dim, output_dim)
    model.embedding = embedding_layer

    # Create dummy dataset
    data = np.array([[word_to_index.get('word1', 0), word_to_index.get('word2', 0), word_to_index.get('word3', 0)]])
    labels = np.array([[word_to_index.get('word1', 0), word_to_index.get('word2', 0), word_to_index.get('word3', 0)]])
    dataset = ExampleDataset(data, labels)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Train the model
    train_model(model, dataloader, epochs=25)

    # Test SQL generation
    test_input_seq = [word_to_index.get('word1', 0), word_to_index.get('word2', 0), word_to_index.get('word3', 0)]
    index_to_word = {v: k for k, v in word_to_index.items()}
    sql_query = generate_sql(model, test_input_seq, index_to_word, word_to_index, device='cpu')
    print(f'Generated SQL Query: {sql_query}')

if __name__ == "__main__":
    main()
