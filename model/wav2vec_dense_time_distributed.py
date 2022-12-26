import torch.nn as nn
'''
https://discuss.pytorch.org/t/any-pytorch-function-can-work-as-keras-timedistributed/1346/4
'''
class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, input_seq):
        assert len(input_seq.size()) > 2
        reshaped_input = input_seq.contiguous().view(-1, input_seq.size(-1))
        output = self.module(reshaped_input)
        if self.batch_first:
            output = output.contiguous().view(input_seq.size(0), -1, output.size(-1))
        else:
            output = output.contiguous().view(-1, input_seq.size(1), output.size(-1))
        return output


class Wav2VecFullyConnectedTimeDistributed(nn.Module):
    def __init__(self, input_dim=1024):
        super(Wav2VecFullyConnectedTimeDistributed, self).__init__()
        self.time_distributed = TimeDistributed(
            nn.Linear(in_features=input_dim, out_features=4096), batch_first=True)
        self.dense1 = nn.Sequential(
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
        # x [batch, time, feat]
        x = self.time_distributed(x) 
        # x [batch, time, 4096]
        x = self.dense1(x)  # [batch, time, 1]
        # x [batch, time, 2048]
        x = self.dense2(x)  # [batch, time, 1]
        # x [batch, time, 1]
        x = x.mean(dim=[1,2], keepdims=False)  # [batch, 1, 1]
        return x

