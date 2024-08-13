# nets.py

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer, TrainingArguments, Trainer
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset
import random

# SQL Transformer Model
class SqlTransformerNet:
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
class SqlClassifier:
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


# Dataset for PyTorch
class Seq2SQLDataset(Dataset):
    def __init__(self, questions, sql_queries, tokenizer=None):
        self.questions = questions
        self.sql_queries = sql_queries
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        sql_query = self.sql_queries[idx]

        if self.tokenizer:
            encoding = self.tokenizer(
                question,
                truncation=True,
                padding='max_length',
                return_tensors='pt',
                max_length=128
            )
            labels = self.tokenizer(
                sql_query,
                truncation=True,
                padding='max_length',
                return_tensors='pt',
                max_length=128
            )

            return {
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze(),
                'labels': labels['input_ids'].squeeze()
            }
        return {}


def load_pg_wikiSQL_dataset():
    dataset = load_dataset('kaxap/pg-wikiSQL-sql-instructions-80k')
    return dataset


def create_dataloaders(dataset, sample_size=50, batch_size=8, tokenizer=None):
    random.seed(42)
    sampled_indices = random.sample(range(len(dataset['train'])), sample_size)
    sampled_questions = [dataset['train'][i]['question'] for i in sampled_indices]
    sampled_sql_queries = [dataset['train'][i]['sql_query'] for i in sampled_indices]

    seq2sql_dataset = Seq2SQLDataset(sampled_questions, sampled_sql_queries, tokenizer)
    dataloader = DataLoader(seq2sql_dataset, batch_size=batch_size, shuffle=True)

    return dataloader


# main.py

def main():
    # Laden des Datasets
    dataset = load_pg_wikiSQL_dataset()
    
    # Initialisierung des Tokenizers
    tokenizer = T5Tokenizer.from_pretrained('t5-small')

    # Erstellen der Dataloaders
    dataloader = create_dataloaders(dataset, tokenizer=tokenizer)

    # Initialisierung des SQL Transformer Netzwerks
    sql_transformer_net = SqlTransformerNet(model_name='t5-small')
    
    # Training des Modells
    sql_transformer_net.train(dataloader, output_dir='./models/t5', epochs=1)
    
    # Initialisierung des SQL Classifiers
    sql_classifier = SqlClassifier()
    
    # Beispiel-Frage
    question = "Show all orders greater than 100 dollars"
    
    # Klassifikation mit BERT
    aggregate_pred = sql_classifier.classify(question, 'aggregate')
    select_pred = sql_classifier.classify(question, 'select')
    where_pred = sql_classifier.classify(question, 'where')
    
    print("Aggregate Prediction:", aggregate_pred)
    print("Select Prediction:", select_pred)
    print("Where Prediction:", where_pred)
    
    # Generierung der SQL-Abfrage mit T5
    generated_sql = sql_transformer_net.generate_sql(question)
    print("Generated SQL Query:", generated_sql)

if __name__ == "__main__":
    main()
