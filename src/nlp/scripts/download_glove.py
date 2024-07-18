import requests
import zipfile
import os

def download_glove(url, save_path, extract_to):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    response = requests.get(url)
    with open(save_path, 'wb') as file:
        file.write
