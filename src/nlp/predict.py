# predict.py
import torch
from model import Seq2SQL
from dataset import SQLDataset

def predict(model, input_prompt, tokenizer):
    model.eval()
    input_seq = tokenizer(input_prompt)
    input_seq = torch.tensor(input_seq).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(input_seq)
    return output

if __name__ == "__main__":
    data_path = 'data/pg-wikiSQL-sql-instructions-80k.json'
    glove_path = 'data/glove.6B/glove.6B.50d.txt'
    input_size = 400000  # Adjust based on GloVe vocabulary size
    hidden_size = 200
    output_size = 400000  # Adjust based on output vocabulary size
    dataset = SQLDataset(data_path, glove_path)
    model = Seq2SQL(input_size, hidden_size, output_size, dataset.embeddings)
    model.load_state_dict(torch.load('seq2sql_model.pth'))
    tokenizer = dataset.tokenizer  # Use the same tokenizer as in the dataset
    input_prompt = "Show the names of all employees"
    sql_query = predict(model, input_prompt, tokenizer)
    print(sql_query)
