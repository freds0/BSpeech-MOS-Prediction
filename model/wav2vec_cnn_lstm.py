import torch.nn as nn

class Conv1dBlock(nn.Module):
    def __init__(self, input_dim=1024):
        super(Conv1dBlock, self).__init__()

        self.conv1 = nn.Sequential(            
            nn.Conv1d(input_dim, 512, 
                            kernel_size=5, stride=1, padding=0),
            nn.Dropout(p=0.25),            
            nn.LeakyReLU(negative_slope=0.1),  
        ) 
        self.conv2 = nn.Sequential(
            nn.Conv1d(512,256, 
                            kernel_size=7, stride=1, padding=0),    
            nn.Dropout(p=0.25), 
            nn.LeakyReLU(negative_slope=0.1),        
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(256, 128, 
                            kernel_size=11, stride=1, padding=0),        
            nn.Dropout(p=0.25), 
            nn.LeakyReLU(negative_slope=0.1),    
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        out = self.conv3(x)
        return out


class Wav2VecCNN_LSTM(nn.Module):
    def __init__(self, input_dim=1024):
        super(Wav2VecCNN_LSTM, self).__init__()
        self.conv = Conv1dBlock(input_dim)
        self.lstm = nn.LSTM(input_size=128, hidden_size=32, num_layers=2, bidirectional=True, batch_first=True) 
        # First fully connected layer
        self.fc = nn.Linear(2*32, 1)
        
    def forward(self, x):
        # x dimension: torch.Size([batch_size, seq_len, input_size])
        x = x.transpose(1,2)
        # x dimension: torch.Size([batch_size, input_size, seq_len])
        x = self.conv(x)
        # x dimension: torch.Size([batch_size, conv_out, seq_len])
        x = x.transpose(1,2)
        # x dimension: torch.Size([batch_size, seq_len, conv_out])
        out, (hidden_state, cell_state) = self.lstm(x)
        x = out.mean(-2)
        # x dimension: torch.Size([batch_size, conv_out])
        out = self.fc(x)
        return out

