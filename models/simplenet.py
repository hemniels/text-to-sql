from transformers import BertForTokenClassification, BertTokenizer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Dummy Dataset
class SQLDataset(Dataset):
    def __init__(self, questions, sql_queries, tokenizer, max_len):
        self.questions = questions
        self.sql_queries = sql_queries
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        sql_query = self.sql_queries[idx]
        
        inputs = self.tokenizer(question, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')
        target = self.tokenizer(sql_query, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': target['input_ids'].squeeze()
        }

# Transformer Model
class SQLTransformer(nn.Module):
    def __init__(self, model_name, num_labels):
        super(SQLTransformer, self).__init__()
        self.transformer = BertForTokenClassification.from_pretrained(model_name, num_labels=num_labels)
    
    def forward(self, input_ids, attention_mask):
        return self.transformer(input_ids=input_ids, attention_mask=attention_mask).logits

# Training Function
def train_model(model, dataloader, epochs=5):
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in dataloader:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f'Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}')

# Inference Function
def generate_prediction(question, tokenizer, model, max_len):
    inputs = tokenizer(question, max_length=max_len, padding='max_length', truncation=True, return_tensors='pt')
    input_ids, attention_mask = inputs['input_ids'], inputs['attention_mask']
    
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
    
    # Decode the predicted token IDs into SQL query
    predicted_ids = torch.argmax(outputs, dim=-1)
    predicted_query = tokenizer.decode(predicted_ids.squeeze().tolist(), skip_special_tokens=True)
    
    return predicted_query

# Example Data
questions = ["What is the time of the match?", "Which player scored the highest?"]
sql_queries = ["SELECT time FROM matches", "SELECT player FROM scores WHERE points = (SELECT MAX(points) FROM scores)"]

# Tokenizer and Dataset
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
num_labels = tokenizer.vocab_size  # Set this to the actual number of labels you have
dataset = SQLDataset(questions, sql_queries, tokenizer, max_len=32)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Model
model = SQLTransformer('bert-base-uncased', num_labels=num_labels)
train_model(model, dataloader)

# Example Inference
question = "What is the time of the match?"
predicted_sql = generate_prediction(question, tokenizer, model, max_len=32)

print(f'Input question: {question}')
print(f'Predicted SQL query: {predicted_sql}')
