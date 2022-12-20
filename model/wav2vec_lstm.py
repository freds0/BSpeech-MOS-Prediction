import torch.nn as nn

class Wav2VecLSTM(nn.Module):
    def __init__(self, input_dim=1024, hidden_size=1024, num_layers=1, bidirectional=True):
        super(Wav2VecLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_size, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)
        if bidirectional:
            self.fc = nn.Linear(2*hidden_size, 1)
        else:
            self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):        
        x, (hn, cn)  = self.lstm(x) # x dimension: torch.Size([batch_size, seq_len, input_size])
        x = x[:, -1, :] # x dimension: torch.Size([batch_size, input_size])
        out = self.fc(x) # x dimension: torch.Size([batch_size])
        return out

