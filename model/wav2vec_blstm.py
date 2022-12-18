import torch.nn as nn


class Wav2Vec_blstm(nn.Module):
    def __init__(self, input_dim=1024):
        super(Wav2Vec_blstm, self).__init__()        
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=32, num_layers=4, bidirectional=True, batch_first=True) 
        self.fc = nn.Linear(64, 1)
        
    def forward(self, x):
        # x dimension: torch.Size([batch_size, seq_len, input_size])
        x, (hn, cn)  = self.lstm(x)
        # x dimension: torch.Size([batch_size, seq_len, hidden_size])
        x = x.mean(-2)
        # x dimension: torch.Size([batch_size, hidden_size])
        out = self.fc(x)
        return out

