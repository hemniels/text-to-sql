import torch
import unittest
from models.where_predictor import WherePredictor

class TestWherePredictor(unittest.TestCase):
    def setUp(self):
        hidden_dim = 20  # Beispielhafte Dimension für hidden_dim
        self.where_predictor = WherePredictor(hidden_dim)

    def test_scalar_attention(self):
        # Erstelle Dummy-Daten für den Test
        batch_size = 4
        seq_len = 5
        
        # Dummy Input Encodings [batch_size, seq_len, hidden_dim]
        input_encodings = torch.randn(batch_size, seq_len, 20)
        
        # Dummy Hidden States [num_layers, batch_size, hidden_dim]
        hidden_states = torch.randn(1, batch_size, 20)
        
        # Berechne Scores
        scores = torch.randn(batch_size, seq_len, 1)  # Dummy Scores [batch_size, seq_len, 1]

        # Teste die scalar_attention Methode direkt
        attention_weights = self.where_predictor.scalar_attention(scores, hidden_states)
        
        # Erwartete Dimensionen
        self.assertEqual(attention_weights.size(), (batch_size, seq_len))
        
        # Optional: Teste Wertebereich oder spezifische Eigenschaften der Attention-Gewichte
        self.assertTrue(torch.all(attention_weights >= 0))  # Positive Werte
        self.assertTrue(torch.all(attention_weights <= 1))  # Maximalwerte <= 1

if __name__ == '__main__':
    unittest.main()
