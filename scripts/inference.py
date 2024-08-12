import torch
from models.sqlnet import SqlNet
from models.agg_predictor import AggPredictor
from models.select_predictor import SelectPredictor
from models.where_predictor import WherePredictor

def infer():
    # Modellinitialisierung
    hidden_dim = 128
    embedding_dim = 300
    num_columns = 10
    sql_vocab_size = 4

    agg_pred = AggPredictor(hidden_dim, sql_vocab_size)
    sel_pred = SelectPredictor(embedding_dim, hidden_dim, num_columns)
    where_pred = WherePredictor(hidden_dim)
    sqlnet = SqlNet(agg_pred, sel_pred, where_pred)

    # Modell laden
    sqlnet.load('sqlnet_model.pth')

    # Dummy-Daten für die Inferenz
    col_embeddings = torch.randn(1, num_columns, embedding_dim)
    input_encodings = torch.randn(1, 20, hidden_dim)

    # Vorhersage
    agg_scores, sel_scores, where_scores = sqlnet(col_embeddings, input_encodings)
    print(f"Agg scores: {agg_scores}")
    print(f"Sel scores: {sel_scores}")
    print(f"Where scores: {where_scores}")

if __name__ == "__main__":
    infer()
