# utils/text_processing.py

import numpy as np

def preprocess_corpus(corpus):
    """
    Vorverarbeitet den Korpus: Tokenisiert die Sätze und entfernt Satzzeichen.
    """
    preprocessed_corpus = []
    for sentence in corpus:
        tokens = [word.lower() for word in sentence if word.isalnum()]
        preprocessed_corpus.append(tokens)
    return preprocessed_corpus

def build_vocab(corpus):
    """
    Baut den Wortschatz und gibt die Zuordnungen von Wörtern zu Indizes zurück.
    """
    words = list(set(word for sentence in corpus for word in sentence))
    word_to_index = {word: idx for idx, word in enumerate(words)}
    index_to_word = {idx: word for word, idx in word_to_index.items()}
    return word_to_index, index_to_word
