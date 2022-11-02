import torch.nn as nn

class Wav2Vec2MOS(nn.Module):
    def __init__(self, input_dim=1024):
        super().__init__()
        self.dense = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # x = x['last_hidden_state'] # [Batch, time, feats]
        x = self.dense(x)  # [batch, time, 1]
        x = x.mean(dim=[1, 2], keepdims=True)  # [batch, 1, 1]
        return x