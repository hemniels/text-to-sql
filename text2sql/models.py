import torch
import torch.nn as nn
import torch.optim as optim
from transformers import T5ForConditionalGeneration, T5Tokenizer, TrainingArguments, Trainer
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm import tqdm
from utils import text_to_tokens


# LSTM-Modell definieren
class SimpleSeq2SQLModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, glove_model):
        super(SimpleSeq2SQLModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.glove_model = glove_model

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x
    
    def train_model(self, dataloader, criterion, optimizer, num_epochs):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        for epoch in range(num_epochs):
            self.train()
            epoch_loss = 0
            for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit='batch'):
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)

                outputs = self(input_ids)
                loss = criterion(outputs.view(-1, self.fc.out_features), labels.view(-1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {epoch_loss / len(dataloader)}")

        print("Training abgeschlossen.")

    def predict_sql_query(self,text):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eval()
        input_ids = torch.tensor(text_to_tokens(text, self.glove_model), dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = self(input_ids)
            _, predicted = torch.max(outputs, dim=-1)
            return predicted.squeeze().tolist()

    def decode_sql_tokens(self,tokens):
        reverse_vocab = {v: k for k, v in self.glove_model.key_to_index.items()}
        sql_query = ' '.join([reverse_vocab.get(token, '<UNK>') for token in tokens])
        return sql_query

    
# Transformer Model
class SimpleSqlTransformerNet:
    def __init__(self, model_name='t5-small'):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train(self, dataloader, output_dir='./models/t5', epochs=1):
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=8,
            save_steps=10,
            save_total_limit=1,
            logging_dir='./logs',
            logging_steps=10,
            prediction_loss_only=True,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataloader.dataset
        )
        
        trainer.train()

    def generate_sql(self, question, max_length=50):
        self.model.eval()
        inputs = self.tokenizer(question, return_tensors='pt').to(self.device)
        outputs = self.model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=max_length
        )
        sql_query = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return sql_query

# BERT Classifier
class SimpleSqlClassifier:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Beispiel-Label, passen Sie diese an Ihre Klassifikationsaufgabe an
        self.LABEL_LIST = {
            'aggregate': ['SUM', 'COUNT', 'AVG'],
            'select': ['order_id', 'customer_id', 'amount'],
            'where': ['greater than', 'less than', 'equals']
        }

    def classify(self, text, label_type):
        labels = self.LABEL_LIST[label_type]
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True).to(self.device)
        outputs = self.model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        return labels[predictions.item()]