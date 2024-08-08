import torch
import torch.nn as nn
import pandas as pd
import gensim
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

# Laden und Vorverarbeiten des WikiSQL-Datasets
def load_and_preprocess_wikisql(data_path):
    """
    Laden und Vorverarbeiten des WikiSQL-Datasets.
    """
    data = pd.read_json(data_path, lines=True)
    stop_words = set(stopwords.words('english'))
    preprocessed_data = []
    
    for question in data['question']:
        tokens = word_tokenize(question.lower())
        tokens = [w for w in tokens if w not in stop_words and w not in string.punctuation]
        preprocessed_data.append(tokens)
    
    return preprocessed_data

# Trainieren des Word2Vec-Modells
def train_word2vec(corpus, embedding_dim=100, window=5, min_count=1, epochs=10):
    """
    Trainieren eines Word2Vec-Modells auf dem gegebenen Korpus.
    """
    model = Word2Vec(sentences=corpus, vector_size=embedding_dim, window=window, min_count=min_count, sg=0)
    model.train(corpus, total_examples=model.corpus_count, epochs=epochs)
    return model

# Einlesen und Erstellen von Word Embeddings für WikiSQL
def create_w2v_embeddings(data_path, embedding_dim=100):
    """
    Erstellt Word Embeddings für das WikiSQL-Dataset.
    """
    corpus = load_and_preprocess_wikisql(data_path)
    w2v_model = train_word2vec(corpus, embedding_dim)
    return w2v_model

# Beispielnutzung
data_path = 'path_to_wikisql_train.jsonl'
w2v_model = create_w2v_embeddings(data_path)

# Funktion, um ein einzelnes Wort in ein Embedding zu konvertieren
def get_word_embedding(word, w2v_model):
    if word in w2v_model.wv:
        return torch.tensor(w2v_model.wv[word])
    else:
        raise ValueError(f"Word '{word}' not in vocabulary.")
