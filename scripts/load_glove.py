import os
import requests
import zipfile
from tqdm import tqdm

# Pfad zum Ordner /data + Erstelle den Ordner /data, falls er nicht existiert
data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Herunterladen + Entpacken der GloVe-Embeddings
glove_url = "http://nlp.stanford.edu/data/glove.6B.zip"
glove_zip_path = os.path.join(data_dir, "glove.6B.zip")

print(f"Herunterladen von {glove_url}...")
response = requests.get(glove_url, stream=True)
total_size = int(response.headers.get('content-length', 0))
block_size = 1024
tqdm_bar = tqdm(total=total_size, unit='iB', unit_scale=True, desc="Downloading glove.6B.zip")

with open(glove_zip_path, 'wb') as f:
    for data in response.iter_content(block_size):
        tqdm_bar.update(len(data))
        f.write(data)

tqdm_bar.close()

if total_size != 0 and tqdm_bar.n != total_size:
    print("Warnung: Die heruntergeladene Datei hat eine unerwartete Größe.")

print("Download abgeschlossen.")

print("Entpacken der GloVe-Embeddings...")
with zipfile.ZipFile(glove_zip_path, 'r') as zip_ref:
    zip_ref.extractall(data_dir)

print(f"Entpacken abgeschlossen. Dateien sind im Ordner {data_dir} gespeichert.")

os.remove(glove_zip_path)
print("ZIP-Datei wurde gelöscht.")
