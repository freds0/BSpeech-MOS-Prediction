import torch
import torchaudio
from datasets.vcc2018_dataset import VCC2018SpecDatasetLoad, custom_spec_collate_fn
from torch.utils.data import DataLoader
from models.vanilla_lstm import lstm_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transformation_fn = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=1024,
    hop_length=512,
    n_mels=80
)

dataset_train = VCC2018SpecDatasetLoad(input_csv='data/train.csv', input_dir='/media/fred/DADOS/DATASET_MOS/', transformation = transformation_fn, same_length_samples = True)
train_loader = DataLoader(dataset_train, batch_size=10, shuffle=True, collate_fn=custom_spec_collate_fn)

it = iter(train_loader)
batch = next(it)
data = batch['data']
label = batch['target']
print(data.shape)
print(label)

model = lstm_model(313,128,15).to(device)

out = model(data)
print(out)