import argparse
import torch, torchaudio
from transformers import Wav2Vec2FeatureExtractor, WavLMModel
from os.path import join, exists, basename
from os import makedirs
from tqdm import tqdm
from glob import glob

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(model_name="wavlm-large"):
    model_path = None
    if (model_name == "wavlm-large"): # torch.shape: 1, t, 1024
        model_path = "microsoft/wavlm-large"
    elif (model_name == "wavlm-base-plus"): # torch.shape: 1, t, 768
        model_path = "microsoft/wavlm-base-plus"
    elif (model_name == "wavlm-base-plus-sv"): 
        model_path = "microsoft/wavlm-base-plus-sv"
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
    model = WavLMModel.from_pretrained(model_path)
    model.eval()
    model = model.to(device)
    return model, feature_extractor


def extract_wavlm_embeddings(filelist, output_dir, model_name):
    model, feature_extractor = load_model(model_name)
    for filepath in tqdm(filelist):
        # Load audio file
        if not exists(filepath):
            print("file {} doesnt exist!".format(filepath))
            continue
        filename = basename(filepath)
        audio_data, sr = torchaudio.load(filepath)
        audio_data = audio_data.squeeze().to(device)
        # Extract Embedding
        inputs = feature_extractor(
            audio_data, sampling_rate=16000, return_tensors="pt"
        )
        input_features = inputs.input_values
        input_features = input_features.to(device)
        file_embedding = model(input_features).last_hidden_state

        # Saving embedding
        output_filename = filename.split(".")[0] + ".pt"
        output_filepath = join(output_dir, output_filename)
        torch.save(file_embedding, output_filepath)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--base_dir', default='./')
    parser.add_argument('-i', '--input_dir', help='Wavs folder')
    parser.add_argument('-c', '--input_csv', help='Metadata filepath')
    parser.add_argument('-o', '--output_dir', default='output_embeddings', help='Name of csv file')
    parser.add_argument('-m', '--model_name', default="wavlm-large", help='Available Models: wavlm-large | wavlm-base-plus | wavlm-base-plus-sv')
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
    extract_wavlm_embeddings(filelist, output_dir, args.model_name)


if __name__ == "__main__":
    main()
