import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Dummy Tokenizer and Dataset classes
class SimpleTokenizer:
    def __init__(self, word_to_index=None):
        if word_to_index is None:
            word_to_index = {'<pad>': 0, '<unk>': 1}
        self.word_to_index = word_to_index
        self.index_to_word = {v: k for k, v in word_to_index.items()}
    
    def __call__(self, text):
        return [self.word_to_index.get(word, self.word_to_index['<unk>']) for word in text.split()]

    def build_vocab(self, texts):
        word_count = {}
        for text in texts:
            for word in text.split():
                if word not in word_count:
                    word_count[word] = len(word_count) + 2  # Reserve 0 for <pad> and 1 for <unk>
        self.word_to_index = {word: idx for word, idx in word_count.items()}
        self.word_to_index.update({'<pad>': 0, '<unk>': 1})
        self.index_to_word = {v: k for k, v in self.word_to_index.items()}

class SimpleDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence, sql = self.data[idx]
        input_ids = torch.tensor(self.tokenizer(sentence))
        labels = torch.tensor(self.tokenizer(sql))
        return {'input_ids': input_ids, 'labels': labels}

def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['labels'] for item in batch]
    
    max_len_input = max([x.size(0) for x in input_ids])
    max_len_labels = max([x.size(0) for x in labels])
    
    input_ids_padded = torch.stack([torch.cat([x, torch.zeros(max_len_input - x.size(0))]) for x in input_ids])
    labels_padded = torch.stack([torch.cat([x, torch.zeros(max_len_labels - x.size(0))]) for x in labels])
    
    return {'input_ids': input_ids_padded.long(), 'labels': labels_padded.long()}

class SimpleNet(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(SimpleNet, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        out = self.fc(lstm_out)
        return out

def test_model():
    # Test Data
    questions = ["What is the time?", "Who is the president?"]
    sql_queries = ["SELECT time", "SELECT president"]
    
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(questions + sql_queries)
    
    dataset = SimpleDataset(list(zip(questions, sql_queries)), tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    
    vocab_size = len(tokenizer.word_to_index)
    embedding_dim = 5
    hidden_dim = 10
    output_dim = vocab_size
    
    model = SimpleNet(vocab_size, embedding_dim, hidden_dim, output_dim)
    
    # Simple Training Loop
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.word_to_index['<pad>'])
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(1):
        model.train()
        for batch in dataloader:
            input_ids = batch['input_ids']
            labels = batch['labels']
            
            print(f'Input IDs Shape: {input_ids.shape}')
            print(f'Labels Shape: {labels.shape}')
            
            optimizer.zero_grad()
            outputs = model(input_ids)
            
            # Print shapes of the outputs and labels
            print(f'Outputs Shape: {outputs.shape}')
            
            # Flatten outputs and labels for CrossEntropyLoss
            outputs_reshaped = outputs.view(-1, outputs.size(-1))
            labels_reshaped = labels.view(-1)
            
            print(f'Reshaped Outputs Shape: {outputs_reshaped.shape}')
            print(f'Reshaped Labels Shape: {labels_reshaped.shape}')
            
            # Create mask for non-padding elements
            non_pad_mask = labels_reshaped != tokenizer.word_to_index['<pad>']
            
            # Print mask shape and type
            print(f'Non-pad Mask Shape: {non_pad_mask.shape}')
            
            # Check if mask size matches reshaped outputs size
            if non_pad_mask.size(0) != outputs_reshaped.size(0):
                print(f'Warning: Mask size ({non_pad_mask.size(0)}) does not match reshaped outputs size ({outputs_reshaped.size(0)})')
                print(f'non_pad_mask: {non_pad_mask}')
                print(f'outputs_reshaped: {outputs_reshaped}')
            
            # Apply mask
            try:
                filtered_outputs = outputs_reshaped[non_pad_mask]
                filtered_labels = labels_reshaped[non_pad_mask]
            except IndexError as e:
                print(f'IndexError: {e}')
                print(f'outputs_reshaped: {outputs_reshaped}')
                print(f'non_pad_mask: {non_pad_mask}')
                print(f'filtered_outputs size: {filtered_outputs.size() if "filtered_outputs" in locals() else "N/A"}')
                print(f'filtered_labels size: {filtered_labels.size() if "filtered_labels" in locals() else "N/A"}')
                raise

            # Print filtered shapes
            print(f'Filtered Outputs Shape: {filtered_outputs.shape}')
            print(f'Filtered Labels Shape: {filtered_labels.shape}')
            
            # Ensure filtered outputs and labels have the same shape
            if filtered_outputs.size(0) != filtered_labels.size(0):
                raise ValueError(f'Filtered outputs size ({filtered_outputs.size(0)}) does not match filtered labels size ({filtered_labels.size(0)})')
            
            loss = criterion(filtered_outputs, filtered_labels)
            loss.backward()
            optimizer.step()
            
            print(f'Epoch {epoch + 1}, Loss: {loss.item()}')
    
    # Example Inference
    def decode_predictions(predictions, index_to_word, max_len=32):
        _, predicted_ids = torch.max(predictions, dim=-1)
        predicted_ids = predicted_ids.view(-1).cpu().numpy()
        decoded_tokens = [index_to_word.get(idx, '<unk>') for idx in predicted_ids]
        decoded_text = ' '.join(decoded_tokens).replace('<pad>', '').strip()
        return decoded_text
    
    def infer(model, sentence, tokenizer, index_to_word, max_len=32):
        model.eval()
        with torch.no_grad():
            input_ids = torch.tensor(tokenizer(sentence)).unsqueeze(0)
            output = model(input_ids)
            probabilities = torch.softmax(output, dim=-1)
            decoded_sql = decode_predictions(probabilities.squeeze(0), index_to_word, max_len)
        return decoded_sql
    
    sentence = "What is the time?"
    index_to_word = tokenizer.index_to_word
    decoded_sql = infer(model, sentence, tokenizer, index_to_word)
    print(f'Decoded SQL Query: {decoded_sql}')

if __name__ == '__main__':
    test_model()
