import torch.nn as nn

class Wav2VecFullyConnected(nn.Module):
    def __init__(self, input_dim=1024):
        super().__init__()
        self.dense1 = nn.Sequential(
            nn.Linear(input_dim, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 2048)
        )
        self.dense2 = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        # x = x['last_hidden_state'] # [Batch, time, feats]
        x = self.dense1(x)  # [batch, time, 1]
        x = self.dense2(x)  # [batch, time, 1]
        x = x.mean(dim=[1, 2], keepdims=True)  # [batch, 1, 1]
        return x

