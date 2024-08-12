# scripts/prepare_data.py

from utils import load_and_preprocess_data

def main():
    glove_file = 'data/glove.6B.50d.txt'  # Pfad zur GloVe-Datei
    train_data, glove = load_and_preprocess_data(glove_file)

    # Speichern oder Weiterverarbeiten der vorbereiteten Daten
    # Zum Beispiel: Speichern als .pt-Dateien oder .csv-Dateien für spätere Verwendung
    torch.save(train_data, 'processed_train_data.pt')

if __name__ == "__main__":
    main()
