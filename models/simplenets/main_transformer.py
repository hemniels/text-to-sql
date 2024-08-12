import torch
from simplenet_transformer import SQLDataset, SQLTransformer, train_model, generate_prediction
from transformers import BertTokenizer
from torch.utils.data import DataLoader

def main():
    # Example Data
    questions = ["What is the time of the match?", "Which player scored the highest?"]
    sql_queries = ["SELECT time FROM matches", "SELECT player FROM scores WHERE points = (SELECT MAX(points) FROM scores)"]

    # Tokenizer and Dataset
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    num_labels = tokenizer.vocab_size  # Set this to the actual number of labels you have
    dataset = SQLDataset(questions, sql_queries, tokenizer, max_len=32)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SQLTransformer('bert-base-uncased', num_labels=num_labels)

    # Train the model
    train_model(model, dataloader, device, tokenizer, epochs=50)

    # Example Inference
    question = "What is the time of the match?"
    predicted_sql = generate_prediction(question, tokenizer, model, max_len=32, device=device)

    print(f'Input question: {question}')
    print(f'Predicted SQL query: {predicted_sql}')

if __name__ == "__main__":
    main()
