import torch
import torch.nn as nn
import pytest

# Importieren Sie die Modelle hier
from models import AggregatePredictor, SelectPredictor, WherePredictor

# Testdaten
input_dim = 100  # Größe des Vokabulars
output_dim_aggregate = 3  # Anzahl der möglichen Aggregatfunktionen
output_dim_select = 10  # Anzahl der möglichen Spalten
output_dim_where = 20  # Anzahl der möglichen Bedingungen
embedding_dim = 64
hidden_dim = 128

@pytest.fixture
def setup_models():
    aggregate_model = AggregatePredictor(input_dim, output_dim_aggregate, embedding_dim, hidden_dim)
    select_model = SelectPredictor(input_dim, output_dim_select, embedding_dim, hidden_dim)
    where_model = WherePredictor(input_dim, output_dim_where, embedding_dim, hidden_dim)
    return aggregate_model, select_model, where_model

def test_aggregate_predictor_forward(setup_models):
    aggregate_model, _, _ = setup_models
    # Erstellen Sie einen Dummy-Eingangstensor
    input_tensor = torch.randint(0, input_dim, (10, 5))  # Batchgröße 10, Sequenzlänge 5
    output = aggregate_model(input_tensor)
    assert output.shape == (10, output_dim_aggregate)
    assert torch.all(output >= 0) and torch.all(output <= 1)  # Softmax-Ausgabe sollte zwischen 0 und 1 liegen

def test_select_predictor_forward(setup_models):
    _, select_model, _ = setup_models
    input_tensor = torch.randint(0, input_dim, (10, 5))
    output = select_model(input_tensor)
    assert output.shape == (10, output_dim_select)
    assert torch.all(output >= 0) and torch.all(output <= 1)

def test_where_predictor_forward(setup_models):
    _, _, where_model = setup_models
    input_tensor = torch.randint(0, input_dim, (10, 5))
    output = where_model(input_tensor)
    assert output.shape == (10, output_dim_where)
    assert torch.all(output >= 0) and torch.all(output <= 1)
