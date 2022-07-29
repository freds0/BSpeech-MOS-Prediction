import librosa
import torchaudio
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import os
import numpy as np

from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VCC2018SpecDatasetLoad(Dataset):
    def __init__(self,
                 input_csv,
                 input_dir,
                 transformation,
                 target_sample_rate = 16000,
                 same_length_samples = False,
                 num_samples = 160000 # 10s
                 ):

        self.annotations = pd.read_csv(input_csv)
        self.input_dir = input_dir
        self.target_sample_rate = target_sample_rate
        self.same_length_samples = same_length_samples
        self.num_samples = num_samples
        self.transformation = transformation

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = self._resample_if_necessary(signal, sr)
        if self.same_length_samples:
            signal = self._cut_if_necessary(signal)
            signal = self._right_pad_if_necessary(signal)
        signal = self.transformation(signal).unsqueeze(1)

        return {"data": signal, "target": label}

    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def get_transform_spec(self, idx, data_dir='./'):
        # get audio path
        specs = []
        audio_name = self.audio_names[idx]
        audio_path = os.path.join(data_dir, audio_name)
          
        # load audio and get its melspectrogram
        audio_wave, sr = librosa.load(audio_path, sr=None)
        mel_spec = librosa.feature.melspectrogram(y=audio_wave, sr=sr)
        
        mel_spec = librosa.power_to_db(mel_spec)
        mel_spec = torch.Tensor(self.normalize(mel_spec)).unsqueeze(0)
        return mel_spec        

    def _get_audio_sample_path(self, index):
        path = os.path.join(self.input_dir, self.annotations.iloc[
            index, 0])
        return path

    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 1]

class VCC2018WavDatasetLoad(Dataset):
    def __init__(self, audio_names: list, labels: list):
        self.audio_names = audio_names
        self.labels = labels

        #self.label_to_id = dict((mos,id) for id, mos in enumerate(labels))
        
    def __len__(self):
        return len(self.audio_names)

    def __getitem__(self, idx):
        filename = self.audio_names[idx]
        waveform, sample_rate = torchaudio.load(filename)
        #target = self.label_to_id[self.labels[idx]]
        target = self.labels[idx]
        
        return {"data": waveform, "target": target}


def custom_spec_collate_fn(data):
    """
       data: is a list of tuples with (example, label, length)
             where 'example' is a tensor of arbitrary shape
             and label/length are scalars
    """
    features = [torch.tensor(d['data']) for d in data] #(3)
    labels = torch.tensor([d['target']  for d in data]) 
    #new_features = pad_sequence([f.T for f in features], batch_first=True).squeeze()
    new_features = pad_sequence([f for f in features], batch_first=True).squeeze()

    return  {
        'data': new_features.to(device),
        'target': labels.to(device)
    }


