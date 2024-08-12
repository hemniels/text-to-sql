# scripts/download_glove.py

import os
import requests
import zipfile

def download_glove(glove_url, dest_folder='data'):
    """Download und extrahiere die GloVe-Embeddings."""
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # Bestimme den Namen der Datei aus der URL
    filename = glove_url.split('/')[-1]
    file_path = os.path.join(dest_folder, filename)
    
    # Überprüfe, ob die Datei bereits existiert
    if not os.path.exists(file_path):
        print(f"Downloading {filename} from {glove_url}...")
        response = requests.get(glove_url, stream=True)
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded {filename}")
        else:
            raise Exception(f"Failed to download file from {glove_url}")

    # Extrahiere die Datei, falls es sich um eine ZIP-Datei handelt
    if file_path.endswith('.zip'):
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(dest_folder)
        print(f"Extracted {filename} to {dest_folder}")

    return file_path

if __name__ == "__main__":
    # Beispiel für GloVe-URL und Zielordner
    GLOVE_URL = 'http://nlp.stanford.edu/data/glove.6B.zip'
    download_glove(GLOVE_URL)
