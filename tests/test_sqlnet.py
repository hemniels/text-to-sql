import pytest
import torch
from transformers import AutoTokenizer
from utils.utils import preprocess_data, GloveEmbeddings
from models.sqlnet import SQLNet
from models.agg_predictor import AggPredictor
from models.select_predictor import SelectPredictor
from models.where_predictor import WherePredictor

@pytest.fixture
def setup_sqlnet_and_data():
    # Beispielhafte Dimensionen und Anzahl der Aggregationsoperationen
    embedding_dim = 64
    hidden_dim = 128
    num_agg_ops = 10
    num_columns = 5  # Beispielhafte Anzahl von Spalten

    # Initialisiere die Tokenizer und das GloVe-Embedding
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    glove = GloveEmbeddings(glove_file='data/glove.6B.50d.txt', embedding_dim=embedding_dim)

    # Erstelle Dummy-Modelle für SQLNet, Aggregation und Auswahl
    sqlnet = SQLNet(embedding_dim=embedding_dim, hidden_dim=hidden_dim, num_agg_ops=num_agg_ops, num_columns=num_columns, glove_embeddings=glove)
    
    # Dummy-Datenbeispiel
    example = {
        'question': 'What time was the match played with a score of 3-2?',
        'create_table_statement': 'CREATE TABLE matches (time TEXT, score TEXT);',
        'sql_query': 'SELECT time FROM matches WHERE score="3-2"'
    }
    
    # Dummy-Targets für Testzwecke
    target_agg = torch.tensor([1], dtype=torch.long)
    target_sel = torch.tensor([0], dtype=torch.long)
    target_where = torch.tensor([1], dtype=torch.long)
    
    # Dummy-Spalten-Embeddings (hier zufällige Werte)
    col_embeddings = torch.randn(num_columns, embedding_dim)
    
    # Dummy-Eingabecodierungen (hier zufällige Werte)
    input_encodings = torch.randn(1, 10, embedding_dim)  # [batch_size, seq_len, embedding_dim]

    return tokenizer, glove, sqlnet, example, col_embeddings, input_encodings, target_agg, target_sel, target_where

def test_sqlnet(setup_sqlnet_and_data):
    tokenizer, glove, sqlnet, example, col_embeddings, input_encodings, target_agg, target_sel, target_where = setup_sqlnet_and_data

    # Verarbeite die Daten
    processed_example = preprocess_data(
        example, tokenizer, glove, sqlnet.agg_predictor, sqlnet.select_predictor
    )

    # Berechne den Verlust für die Dummy-Daten
    loss = sqlnet.compute_loss(input_encodings, col_embeddings, target_agg, target_sel, target_where)

    # Überprüfe die Dimensionen der Verarbeitungsdaten
    assert processed_example['input_encodings'].shape == torch.Size([1, 10, 64])  # [batch_size, seq_len, embedding_dim]
    assert processed_example['col_embeddings'].shape == torch.Size([5, 64])  # [num_columns, embedding_dim]

    # Überprüfe den Verlust
    assert loss.item() >= 0

if __name__ == '__main__':
    pytest.main()
