# tests/w2vec_test.py

import pytest
import torch
from models.wordembeddings import WordEmbeddingsNet
from utils.text_processing import preprocess_corpus, build_vocab
from utils.downloader import load_wikisql_dataset

def test_word_embeddings():
    """
    Testet die Erstellung und Qualität der Word Embeddings.
    """
    # Laden des WikiSQL-Datasets
    dataset = load_wikisql_dataset()
    corpus = [entry['question'].split() for entry in dataset]  # Tokenisierung auf Basis von Leerzeichen

    # Vorverarbeitung des Datasets und Aufbau des Vokabulars
    processed_corpus = preprocess_corpus(corpus)
    word_to_index, index_to_word = build_vocab(processed_corpus)
    
    # Erstellen des Modells
    vocab_size = len(word_to_index)
    embedding_dim = 100
    model = WordEmbeddingsNet(vocab_size, embedding_dim)

    # Training des Modells
    model.train_model(processed_corpus, word_to_index)
    
    # Testen der Embeddings
    common_word = 'select'
    if common_word in word_to_index:
        word_index = word_to_index[common_word]
        embedding = model.get_embedding(word_index)
        assert embedding.shape == (embedding_dim,)
    else:
        pytest.fail(f"Das Wort '{common_word}' ist nicht im Vokabular enthalten.")

    # Testen, ob das Modell eine Ausnahme auslöst, wenn das Wort nicht im Vokabular ist
    unknown_word = 'unknownword'
    with pytest.raises(KeyError):
        if unknown_word in word_to_index:
            word_index = word_to_index[unknown_word]
            model.get_embedding(word_index)

if __name__ == "__main__":
    pytest.main([__file__])
