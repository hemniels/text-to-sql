import requests
import os
from zipfile import ZipFile

def download_dataset(url, save_path):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Download the dataset
    response = requests.get(url)
    with open(save_path, 'wb') as file:
        file.write(response.content)

        
def download_glove(save_path, embedding_size):
    glove_urls = {
        50: "http://nlp.stanford.edu/data/glove.6B.zip",
        100: "http://nlp.stanford.edu/data/glove.6B.zip",
        200: "http://nlp.stanford.edu/data/glove.6B.zip",
        300: "http://nlp.stanford.edu/data/glove.6B.zip"
    }
    
    glove_url = glove_urls[embedding_size]
    zip_file_path = os.path.join(save_path, "glove.6B.zip")
    glove_file_path = os.path.join(save_path, f"glove.6B.{embedding_size}d.txt")
    
    # Download the GloVe zip file
    if not os.path.exists(zip_file_path):
        response = requests.get(glove_url, stream=True)
        with open(zip_file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
    
    # Extract the zip file
    if not os.path.exists(glove_file_path):
        print(f"Extracting {zip_file_path}...")
        with ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(glove_file_path)

if __name__ == "__main__":
    dataset_url = "https://huggingface.co/datasets/kaxap/pg-wikiSQL-sql-instructions-80k/resolve/main/pg-wikiSQL-sql-instructions-80k.json"
    dataset_path = "../data/pg-wikiSQL-sql-instructions-80k.json"
    glove_path = "../data/"
    download_dataset(dataset_url, dataset_path)
    glove_path = download_glove(glove_path, 50)