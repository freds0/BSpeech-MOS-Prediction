import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, input_dim=1024):
        super(ConvBlock, self).__init__()

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


class Wav2VecCNN1D(nn.Module):
    def __init__(self, input_dim=1024):
        super(Wav2VecCNN1D, self).__init__()        
        self.conv = ConvBlock(input_dim)
        # First fully connected layer
        self.fc = nn.Linear(128, 1)
        
    def forward(self, x):
        x = x.transpose(1,2)
        x = self.conv(x)
        x = x.mean(-1)
        out = self.fc(x)
        return out

