import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention_layer import DotAttention, SoftAttention

class Wav2VecLSTM_Attention(nn.Module):
    '''
    Source: https://github.com/shangeth/wavencoder/blob/master/wavencoder/models/lstm_classifier.py
    '''
    def __init__(self, input_dim=1024, hidden_size=1024, return_attn_weights=False, attn_type='dot'):
        super(Wav2VecLSTMAttention, self).__init__()
        self.return_attn_weights = return_attn_weights
        self.attn_type = attn_type
        
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_size, num_layers=1, bidirectional=False, batch_first=True) 
        self.fc = nn.Linear(hidden_size, 1)

        if self.attn_type == 'dot':
            self.attention = DotAttention()
        elif self.attn_type == 'soft':
            self.attention = SoftAttention(hidden_size, hidden_size)
        

    def forward(self, x):        
        lstm_out, (hidden, cell_state)  = self.lstm(x)
        if self.attn_type == 'dot':
            attn_output = self.attention(lstm_out, hidden)
            attn_weights = self.attention._get_weights(lstm_out, hidden)
        elif self.attn_type == 'soft':
            attn_output = self.attention(lstm_out)
            attn_weights = self.attention._get_weights(lstm_out)

        out = self.fc(attn_output)
        if self.return_attn_weights:
            return out, attn_weights
        else:
            return out
