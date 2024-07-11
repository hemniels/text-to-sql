import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

class TextToSQLModel:
    def __init__(self, model_name='t5-small'):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)

    def train(self, dataset, batch_size=8, epochs=3, lr=5e-5):
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.model.train()

        for epoch in range(epochs):
            for batch in data_loader:
                optimizer.zero_grad()
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['labels']
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                print(f"Epoch {epoch}, Loss: {loss.item()}")

    def predict(self, text):
        self.model.eval()
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        outputs = self.model.generate(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
