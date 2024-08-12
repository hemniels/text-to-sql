import unittest
import torch
from utils import GloveEmbeddings

class TestGloveEmbeddings(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Set up a dummy GloVe file for testing
        cls.glove_file = 'test_glove.txt'
        cls.embedding_dim = 50
        
        with open(cls.glove_file, 'w') as f:
            # Schreiben von Dummy-GloVe-Daten in die Datei
            f.write('word1 ' + ' '.join(['0.1']*cls.embedding_dim) + '\n')
            f.write('word2 ' + ' '.join(['0.2']*cls.embedding_dim) + '\n')
        
        cls.glove = GloveEmbeddings(cls.glove_file, cls.embedding_dim)

    @classmethod
    def tearDownClass(cls):
        import os
        os.remove(cls.glove_file)
    
    def test_embedding_dim(self):
        # Testet die Dimension der geladenen Embeddings
        self.assertEqual(self.glove.get_embedding_dim(), self.embedding_dim)

    def test_vocabulary_size(self):
        # Testet die Größe des Vokabulars
        self.assertEqual(self.glove.get_vocab_size(), 2)

    def test_get_embedding(self):
        # Testet das Abrufen von Embeddings für bekannte und unbekannte Wörter
        embedding_word1 = self.glove.get_embedding('word1')
        embedding_word2 = self.glove.get_embedding('word2')
        embedding_unknown = self.glove.get_embedding('unknown')

        # Überprüfen der Dimension der zurückgegebenen Embeddings
        self.assertEqual(embedding_word1.shape, torch.Size([self.embedding_dim]))
        self.assertEqual(embedding_word2.shape, torch.Size([self.embedding_dim]))
        self.assertEqual(embedding_unknown.shape, torch.Size([self.embedding_dim]))

        # Überprüfen der Werte
        self.assertTrue(torch.all(embedding_word1 == torch.tensor([0.1] * self.embedding_dim)))
        self.assertTrue(torch.all(embedding_word2 == torch.tensor([0.2] * self.embedding_dim)))
        self.assertTrue(torch.all(embedding_unknown == torch.zeros(self.embedding_dim)))

    def test_get_embeddings_matrix(self):
        # Testet die Dimension der Embeddings-Matrix
        embeddings_matrix = self.glove.get_embeddings_matrix()
        self.assertEqual(embeddings_matrix.shape, (2, self.embedding_dim))

if __name__ == '__main__':
    unittest.main()
