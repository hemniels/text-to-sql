# Pseudocode für das AggregatePredictor-Modell
class AggregatePredictor:
    def __init__(input_dim, output_dim, embedding_dim=64, hidden_dim=128):
        """
        Konstruktor für das Modell:
        - input_dim: Größe des Vokabulars (Anzahl der einzigartigen Tokens).
        - output_dim: Anzahl der möglichen Aggregatfunktionen (z.B. 3 für COUNT, MAX, AVG).
        - embedding_dim: Dimensionalität der Wort-Embeddings (z.B. 64).
        - hidden_dim: Anzahl der Neuronen im versteckten Zustand des LSTM (z.B. 128).
        """
        # Embedding-Schicht: transformiert Token-IDs in dichte Vektoren (Embeddings).
        self.embedding = Embedding(input_dim, embedding_dim)
        
        # LSTM-Schicht: verarbeitet die eingebetteten Vektoren sequentiell.
        self.lstm = LSTM(embedding_dim, hidden_dim)
        
        # Fully Connected Layer: transformiert den LSTM-Ausgang in die Ausgabe-Logits.
        self.fc = Linear(hidden_dim, output_dim)
        
        # Softmax-Funktion: wandelt Logits in Wahrscheinlichkeiten um.
        self.softmax = Softmax(dim=1)
    
    def forward(x):
        """
        Vorwärtsdurchlauf durch das Modell:
        - x: Eingabesequenz (z.B. Token-ID-Sequenzen des Benutzer-Prompts).
        - Ausgabe: Wahrscheinlichkeiten für jede Aggregatfunktion.
        """
        embedded = self.embedding(x)  # Eingabe in Vektoren umwandeln.
        hidden, _ = self.lstm(embedded)  # LSTM zur Sequenzverarbeitung.
        logits = self.fc(hidden)  # LSTM-Ausgang in Ausgabe-Logits umwandeln.
        output = self.softmax(logits)  # Wahrscheinlichkeiten berechnen.
        return output
