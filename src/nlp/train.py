import torch
from transformers import T5Tokenizer
from dataset import TextToSQLDataset
from model import TextToSQLModel

def main():
    # Load and prepare the dataset
    data_path = 'path/to/your/dataset.json'
    model_name = 't5-small'
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    dataset = TextToSQLDataset(data_path, tokenizer)

    # Initialize and train the model
    model = TextToSQLModel(model_name)
    model.train(dataset)

    # Save the model
    model.model.save_pretrained('path/to/save/model')
    tokenizer.save_pretrained('path/to/save/tokenizer')

if __name__ == "__main__":
    main()
