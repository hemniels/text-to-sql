import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

def load_model(model_path, tokenizer_path):
    tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    return model, tokenizer

def main():
    model_path = 'model.py'
    tokenizer_path = 'dataset.py'

    model, tokenizer = load_model(model_path, tokenizer_path)

    text = "Your natural language question here"
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
