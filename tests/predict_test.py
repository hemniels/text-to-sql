
import torch
from nlp.model import Seq2SQL
from nlp.tokenizer import Tokenizer
# Test of predicting results, given prompt, from Seq2SQL model

if __name__ == "__main__":
    data_path = "kaxap/pg-wikiSQL-sql-instructions-80k"
    glove_path = 'data/glove.6B/glove.6B.50d.txt'
    input_size = len(data_path)  # Adjust based on GloVe vocabulary size
    hidden_size = 200
    output_size = 400000  # Adjust based on output vocabulary size
    dataset = Tokenizer(data_path, glove_path)
    model = Seq2SQL(input_size, hidden_size, output_size, dataset.embeddings)
    model.load_state_dict(torch.load('seq2sql_model.pth'))
    tokenizer = dataset.tokenizer  # Use the same tokenizer as in the dataset
    input_prompt = "Show the names of all employees"
    sql_query = Seq2SQL.predict(model, input_prompt, tokenizer)
    print(sql_query)
