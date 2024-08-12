import torch
from torch.utils.data import DataLoader
from utils import load_and_preprocess_data
from models import AggPredictor, SelectPredictor, WherePredictor

def train():
    # Konfigurationsparameter
    glove_file = 'data/glove.6B.50d.txt'  # Pfad zur GloVe-Datei
    embedding_dim = 64
    hidden_dim = 64
    num_agg_ops = 4  # Anzahl der Aggregationsoperationen (z.B. MIN, MAX, COUNT, NULL)
    num_columns = 10  # Beispiel, anpassen auf Anzahl der Spalten
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001

    # Prüfe, ob eine GPU verfügbar ist
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Lade und preprocess Daten
    train_data, glove = load_and_preprocess_data(glove_file)

    # Erstelle DataLoader für das Training
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # Initialisiere Modelle
    agg_pred = AggPredictor(embedding_dim, hidden_dim, num_agg_ops, glove).to(device)
    select_pred = SelectPredictor(embedding_dim, hidden_dim, num_columns, glove).to(device)
    where_pred = WherePredictor(hidden_dim, glove).to(device)

    # Initialisiere Optimierer
    optimizer_agg = torch.optim.Adam(agg_pred.parameters(), lr=learning_rate)
    optimizer_select = torch.optim.Adam(select_pred.parameters(), lr=learning_rate)
    optimizer_where = torch.optim.Adam(where_pred.parameters(), lr=learning_rate)

    # Training Loop
    for epoch in range(num_epochs):
        agg_pred.train()
        select_pred.train()
        where_pred.train()

        epoch_loss_agg = 0
        epoch_loss_select = 0
        epoch_loss_where = 0

        for batch in train_loader:
            # Bereite die Eingaben vor und verschiebe sie auf die GPU
            input_encodings = batch['input_encodings'].to(device).float()
            col_embeddings = batch['col_embeddings'].to(device).float()
            target_agg = batch['target_agg'].to(device)
            target_sel = batch['target_sel'].to(device)
            generated_query = batch['generated_query']  # Beispiel, falls nötig
            ground_truth = batch['ground_truth']  # Beispiel, falls nötig

            # Aggregator-Modell Training
            optimizer_agg.zero_grad()
            loss_agg = agg_pred.train_step(input_encodings, target_agg)
            loss_agg.backward()
            optimizer_agg.step()
            epoch_loss_agg += loss_agg.item()

            # Selektor-Modell Training
            optimizer_select.zero_grad()
            loss_select = select_pred.train_step(col_embeddings, target_sel)
            loss_select.backward()
            optimizer_select.step()
            epoch_loss_select += loss_select.item()

            # Where-Modell Training
            optimizer_where.zero_grad()
            loss_where = where_pred.train_step(input_encodings, generated_query, ground_truth)
            loss_where.backward()
            optimizer_where.step()
            epoch_loss_where += loss_where.item()

        # Ausgabe der Verlustwerte für diese Epoche
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Aggregates Loss: {epoch_loss_agg / len(train_loader)}')
        print(f'Selector Loss: {epoch_loss_select / len(train_loader)}')
        print(f'Where Loss: {epoch_loss_where / len(train_loader)}')

if __name__ == "__main__":
    train()
