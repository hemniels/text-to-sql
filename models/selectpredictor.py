# Pseudocode für das SelectPredictor-Modell
class SelectPredictor:
    def __init__(input_dim, output_dim, embedding_dim=64, hidden_dim=128):
        """
        Konstruktor für das Modell:
        - input_dim: Größe des Vokabulars (Anzahl der einzigartigen Tokens).
        - output_dim: Anzahl der möglichen Spalten (z.B. 10 verschiedene Spalten in der Datenbank).
        - embedding_dim: Dimensionalität der Wort-Embeddings (z.B. 64).
        - hidden_dim: Anzahl der Neuronen im versteckten Zustand des LSTM (z.B. 128).
        """
        self.embedding = Embedding(input_dim, embedding_dim)
        self.lstm = LSTM(embedding_dim, hidden_dim)
        self.fc = Linear(hidden_dim, output_dim)
        self.softmax = Softmax(dim=1)
    
    def forward(x):
        """
        Vorwärtsdurchlauf durch das Modell:
        - x: Eingabesequenz (z.B. Token-ID-Sequenzen des Benutzer-Prompts).
        - Ausgabe: Wahrscheinlichkeiten für jede Spalte.
        """
        embedded = self.embedding(x)
        hidden, _ = self.lstm(embedded)
        logits = self.fc(hidden)
        output = self.softmax(logits)
        return output
