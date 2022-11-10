import argparse
import sys
import torch
from pathlib import Path
from os.path import join, exists, basename
from os import makedirs
from tqdm import tqdm
from glob import glob
import numpy as np

# git clone https://github.com/CorentinJ/Real-Time-Voice-Cloning.git
sys.path.append('Real-Time-Voice-Cloning')
from encoder.model import SpeakerEncoder
from encoder.inference import compute_partial_slices
from encoder.audio import normalize_volume, preprocess_wav, wav_to_mel_spectrogram

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## Mel-filterbank
mel_window_length = 25  # In milliseconds
mel_window_step = 10    # In milliseconds
mel_n_channels = 40

## Audio
sampling_rate = 16000
# Number of spectrogram frames in a partial utterance
partials_n_frames = 160     # 1600 ms
# Number of spectrogram frames at inference
inference_n_frames = 80     #  800 ms
## Audio volume normalization
audio_norm_target_dBFS = -30

def load_model(model_path):
    # wget https://github.com/Edresson/Real-Time-Voice-Cloning/releases/download/checkpoints/pretrained.zip
    weights_fpath = Path(model_path)
    model = SpeakerEncoder(device, torch.device("cpu"))
    checkpoint = torch.load(weights_fpath, device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model

def embed_frames_batch(model, frames_batch):
    """
    Computes embeddings for a batch of mel spectrogram. Original function at Real-Time-Voice-Cloning/encoder/inference.py

    :param frames_batch: a batch mel of spectrogram as a numpy array of float32 of shape
    (batch_size, n_frames, n_channels)
    :return: the embeddings as a numpy array of float32 of shape (batch_size, model_embedding_size)
    """
    if model is None:
        raise Exception("Model was not loaded. Call load_model() before inference.")

    frames = torch.from_numpy(frames_batch).to(device)
    embed = model.forward(frames).detach().cpu().numpy()
    return embed

def extract_embed_utterance(model, wav, using_partials=True, return_partials=False, **kwargs):
    """
    Computes an embedding for a single utterance. . Original function at Real-Time-Voice-Cloning/encoder/inference.py named embed_utterance

    # TODO: handle multiple wavs to benefit from batching on GPU
    :param wav: a preprocessed (see audio.py) utterance waveform as a numpy array of float32
    :param using_partials: if True, then the utterance is split in partial utterances of
    <partial_utterance_n_frames> frames and the utterance embedding is computed from their
    normalized average. If False, the utterance is instead computed from feeding the entire
    spectogram to the network.
    :param return_partials: if True, the partial embeddings will also be returned along with the
    wav slices that correspond to the partial embeddings.
    :param kwargs: additional arguments to compute_partial_splits()
    :return: the embedding as a numpy array of float32 of shape (model_embedding_size,). If
    <return_partials> is True, the partial utterances as a numpy array of float32 of shape
    (n_partials, model_embedding_size) and the wav partials as a list of slices will also be
    returned. If <using_partials> is simultaneously set to False, both these values will be None
    instead.
    """
    # Process the entire utterance if not using partials
    if not using_partials:
        frames = wav_to_mel_spectrogram(wav)
        embed = embed_frames_batch(model, frames[None, ...])[0]
        if return_partials:
            return embed, None, None
        return embed

    # Compute where to split the utterance into partials and pad if necessary
    wave_slices, mel_slices = compute_partial_slices(len(wav), **kwargs)
    max_wave_length = wave_slices[-1].stop
    if max_wave_length >= len(wav):
        wav = np.pad(wav, (0, max_wave_length - len(wav)), "constant")

    # Split the utterance into partials
    frames = wav_to_mel_spectrogram(wav)
    frames_batch = np.array([frames[s] for s in mel_slices])
    partial_embeds = embed_frames_batch(model, frames_batch)

    # Compute the utterance embedding from the partial embeddings
    raw_embed = np.mean(partial_embeds, axis=0)
    embed = raw_embed / np.linalg.norm(raw_embed, 2)

    if return_partials:
        return embed, partial_embeds, wave_slices
    return embed


def extract_ge2e_embeddings(filelist, model_path, output_dir):
    model = load_model(model_path)
    for filepath in tqdm(filelist):
        # Load audio file
        if not exists(filepath):
            print("file {} doesnt exist!".format(filepath))
            continue
        filename = basename(filepath)
        # Extract Embedding
        preprocessed_wav = preprocess_wav(filepath)
        file_embedding = extract_embed_utterance(model, preprocessed_wav)
        embedding = torch.tensor(file_embedding.reshape(-1).tolist())
        # Saving embedding
        output_filename = filename.split(".")[0] + ".pt"
        output_filepath = join(output_dir, output_filename)
        torch.save(embedding, output_filepath)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--base_dir', default='./')
    parser.add_argument('-i', '--input_dir', help='Wavs folder')
    parser.add_argument('-c', '--input_csv', help='Me   tadata filepath')
    parser.add_argument('--model_path', default='./checkpoints/ge2e/pretrained.pt', help='Model .pth filepath')
    parser.add_argument('-o', '--output_dir', default='output_embeddings', help='Name of csv file')
    args = parser.parse_args()

    output_dir = join(args.base_dir, args.output_dir)

    filelist = None
    if (args.input_dir != None):
        input_dir = join(args.base_dir, args.input_dir)
        filelist = glob(input_dir + '/*.wav')

    elif (args.input_csv != None):
        with open(join(args.base_dir, args.input_csv), encoding="utf-8") as f:
            content_file = f.readlines()
            filelist = [line.split(",")[0] for line in content_file]
    else:
        print("Error: args input_dir or input_csv are necessary!")
        exit()
    makedirs(output_dir, exist_ok=True)
    extract_ge2e_embeddings(filelist, args.model_path, output_dir)


if __name__ == "__main__":
    main()
