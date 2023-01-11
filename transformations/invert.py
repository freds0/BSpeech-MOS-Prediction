import random
import torch
import torchaudio
#import librosa

class Invert(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, wav):
        if random.random() < self.p:
            inverted_wav = torch.flip(wav, dims=(0,1))
            return inverted_wav
        else:
            return wav


if __name__ == '__main__':
    waveform = torch.range(1, 1000).reshape(1, -1)
    invert_function = Invert(p=1.0)
    transformed_waveform = invert_function(waveform,)
    print(transformed_waveform)
