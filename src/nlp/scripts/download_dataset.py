import requests
import os

def download_dataset(url, save_path):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Download the dataset
    response = requests.get(url)
    with open(save_path, 'wb') as file:
        file.write(response.content)

if __name__ == "__main__":
    dataset_url = "https://huggingface.co/datasets/kaxap/pg-wikiSQL-sql-instructions-80k/resolve/main/pg-wikiSQL-sql-instructions-80k.json"
    save_path = "data/pg-wikiSQL-sql-instructions-80k.json"
    download_dataset(dataset_url, save_path)
