import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LSTM_model(nn.Module):
    def __init__(self, feature_shape=128, num_layers_lstm=3, hidden_size=512, batch_size=16, bidirectional=True):
        super(LSTM_model, self).__init__()

        self.lstm = nn.LSTM(input_size=feature_shape, hidden_size=hidden_size, num_layers=num_layers_lstm, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.activation = nn.ReLU()

        self.linear = nn.Linear(hidden_size*2*150, 100)

    def forward(self, data):
        data = data.squeeze()
        data = data.reshape(data.shape[0], data.shape[2], data.shape[1])
        batch_shape = data.shape[0]

        output, (h, _) = self.lstm(data)
        return output, h


class Attention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim):
        super(Attention, self).__init__()
        self.scale = 1. / math.sqrt(query_dim)

    def forward(self, query, keys, values):
        query = query.unsqueeze(1) 
        keys = keys.transpose(1,2) 

        energy = torch.bmm(query, keys) 
        energy = F.softmax(energy.mul_(self.scale), dim=2)

        linear_combination = torch.bmm(energy, values).squeeze(1)
        return linear_combination

class Classifier(nn.Module):
    def __init__(self, encoder, attention, hidden_dim, num_classes):
        super(Classifier, self).__init__()
        self.encoder = encoder
        self.attention = attention
        self.decoder = nn.Linear(hidden_dim, num_classes)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input):
        outputs, hidden = self.encoder(input)
        hidden = torch.cat([hidden[-1], hidden[-2]], dim=1)
        linear_combination = self.attention(hidden, outputs, outputs) 
        logits = self.decoder(linear_combination)
        softmaxed = self.sigmoid(logits)
        return logits
