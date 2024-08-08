import pytest
from models import wordembeddings

def test_word_embeddings():
    """
    Testet die Erstellung und Qualität der Word Embeddings.
    """
    data_path = 'path_to_wikisql_train.jsonl'  # Beispielpfad zur Datei
    embedding_dim = 100
    w2v_model = create_w2v_embeddings(data_path, embedding_dim)
    
    # Prüfen, ob ein Embedding für ein häufiges Wort erzeugt werden kann
    word = 'select'
    try:
        embedding = get_word_embedding(word, w2v_model)
        assert embedding.shape == (embedding_dim,)
    except ValueError as e:
        pytest.fail(str(e))

    # Testen, ob unbekannte Wörter korrekt gehandhabt werden
    unknown_word = 'unknownword'
    with pytest.raises(ValueError):
        get_word_embedding(unknown_word, w2v_model)

    # Testen der Dimension des erzeugten Embeddings
    common_word = 'where'
    embedding = get_word_embedding(common_word, w2v_model)
    assert embedding.shape == (embedding_dim,)

# Ausführen des Tests
if __name__ == "__main__":
    pytest.main([__file__])
