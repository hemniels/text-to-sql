# model.py
import torch
import torch.nn as nn
import numpy as np

class Seq2SQL(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, embeddings):
        super(Seq2SQL, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embeddings))
        self.encoder_lstm = nn.LSTM(embeddings.shape[1], hidden_size, batch_first=True)
        self.decoder_lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, input_seq, target_seq=None, teacher_forcing_ratio=0.5):
        embedded = self.dropout(self.embedding(input_seq))
        encoder_output, (hidden, cell) = self.encoder_lstm(embedded)
        decoder_input = embedded[:, 0, :].unsqueeze(1)
        outputs = torch.zeros(target_seq.size(0), target_seq.size(1), self.hidden_size).to(input_seq.device)
        
        for t in range(1, target_seq.size(1)):
            decoder_output, (hidden, cell) = self.decoder_lstm(decoder_input, (hidden, cell))
            output = self.fc(decoder_output)
            outputs[:, t, :] = output.squeeze(1)
            teacher_force = np.random.random() < teacher_forcing_ratio
            decoder_input = target_seq[:, t, :].unsqueeze(1) if teacher_force else output
            
        return outputs
