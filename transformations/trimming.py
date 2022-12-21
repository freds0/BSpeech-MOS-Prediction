import random
import torch
import librosa

class Trimming(object):

    def __init__(self, top_db_range=[60, 80], p=0.5):
        self.top_db_range = top_db_range
        self.top_db = random.uniform(self.top_db_range[0], self.top_db_range[1]) 
        self.p = p

    def __call__(self, wav):
        if random.random() < self.p:
            timmed_wav, index = librosa.effects.trim(wav, top_db=self.top_db)
            return timmed_wav
        else:
            return wav


if __name__ == '__main__':
    waveform = torch.randn(1, 16000)
    trimming_function = Trimming(top_db_range=(30, 60), p=1.0)
    transformed_waveform = trimming_function(waveform)
    print(transformed_waveform.shape)
