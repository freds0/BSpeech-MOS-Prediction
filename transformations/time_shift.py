import random
import torch

class TimeShift(object):

    def __init__(self, shift_factor=(30, 50), p=0.5):
        self.shift_factor = shift_factor
        self.p = p

    def __call__(self, wav):
        #[1, 16000]
        if random.random() < self.p:
            shift = int(wav.shape[1] * random.randint(self.shift_factor[0], self.shift_factor[1]) / 100)
            shifted_wav = torch.roll(wav, shift, 1)
            return shifted_wav
        else:
            return wav


if __name__ == '__main__':
    waveform = torch.randn(1, 16000)
    timeshift_function = TimeShift(shift_factor=(30, 50), p=1.0)
    transformed_waveform = timeshift_function(waveform)
    print(transformed_waveform.shape)
