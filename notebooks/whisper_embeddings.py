import torch, torchaudio
from transformers import WhisperModel, WhisperFeatureExtractor, AutoFeatureExtractor

from tqdm import tqdm
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = WhisperModel.from_pretrained("openai/whisper-base")
model = model.encoder
model = model.to(device)
model.eval()

#processor = AutoFeatureExtractor.from_pretrained("openai/whisper-base")
#feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-base")

feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-base")

filepath = "/home/fred/Projetos/DATASETS/MOS/VCC2018_MOS_preprocessed/wav/B00_VCC2TF1_VCC2SF1_30003_HUB.wav"
data, sr = torchaudio.load(filepath)

inputs = feature_extractor(
    data.squeeze(), sampling_rate=16000, return_tensors="pt"
)

input_features = inputs.input_features
input_features = input_features.to(device)
outputs = model(input_features)
print(outputs.last_hidden_state.shape)




metadata_filepath = '/home/fred/Projetos/DATASETS/MOS/VCC2018_MOS_preprocessed/mos_list.txt'
wavs_filepath = '/home/fred/Projetos/DATASETS/MOS/VCC2018_MOS_preprocessed/wav'

with open(metadata_filepath, encoding="utf-8") as f:
  content_file = f.readlines()[1:]

output_dir = "./VCC2018_whisper_embeddings"
os.makedirs(output_dir, exist_ok=True)

for line in tqdm(content_file):
    #filepath, mos, condition, database = line.split(',')
    filepath, mos = line.split(',')
    filename = os.path.basename(filepath)
    filepath = os.path.join(wavs_filepath, filepath)
    if not os.path.exists(filepath):
      continue
    audio_data, sr = torchaudio.load(filepath)
    audio_data = audio_data.to(device)
    # Extract Embedding      
    inputs_features = feature_extractor(
        data.squeeze(), sampling_rate=16000, return_tensors="pt"
    ).to(device).input_features
    input_features = input_features.to(device)
    file_embedding = model(inputs_features).last_hidden_state
    # Saving embedding
    output_filename = filename.split(".")[0] + ".pt"
    output_filepath = os.path.join(output_dir, output_filename)
    #file_embedding = torch.from_numpy(file_embedding.numpy())
    torch.save(file_embedding, output_filepath)    
