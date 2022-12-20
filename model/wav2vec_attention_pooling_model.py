import torch.nn as nn
import torch


class SelfAttentionPooling(nn.Module):
    """
    Implementation of SelfAttentionPooling 
    Original Paper: Self-Attention Encoding and Pooling for Speaker Recognition
    https://arxiv.org/pdf/2008.01077v1.pdf
    """
    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.W = nn.Linear(input_dim, 1)
        
    def forward(self, batch_rep):
        """
        input:
            batch_rep : size (N, T, H), N: batch size, T: sequence length, H: Hidden dimension
        
        attention_weight:
            att_w : size (N, T, 1)
        
        return:
            utter_rep: size (N, H)
        """
        softmax = nn.functional.softmax
        att_w = softmax(self.W(batch_rep).squeeze(-1), dim=1).unsqueeze(-1)
        utter_rep = torch.sum(batch_rep * att_w, dim=1)

        return utter_rep


class Wav2VecAttentionPooling(nn.Module):
    def __init__(self, input_dim=1024):
        super().__init__()
        self.att_pool = SelfAttentionPooling(input_dim)
        self.dense = nn.Sequential(
            nn.Linear(input_dim, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 1)
        )
    def forward(self, x):
        # x = x['last_hidden_state'] # [Batch, time, feats]
        x = self.att_pool(x)
        # [Batch, feats]
        x = self.dense(x)  # [batch, time, 1]
        return x
