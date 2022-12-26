import argparse
import torchaudio
from compose import Compose
from pad_crop import PadCrop
from speech_change import SpeedChange
from time_shift import TimeShift
from trimming import Trimming
import torch
from os.path import join, exists, basename
from os import makedirs
from tqdm import tqdm
import random

MAX_VALUE = 32768

random.seed(1234)

def execute_transformations(input_filepath, output_filepath):
    waveform, sr = torchaudio.load(input_filepath)
    pad_crop_value = int(random.uniform(0.98, 1.05) *  waveform.shape[1])
    transforms = Compose([
        SpeedChange(factor_range=(0.85, 1.15), p=0.6),
        PadCrop(pad_crop_value, crop_position='random', pad_position='random', p=0.6),
        TimeShift(shift_factor=(10, 50), p=0.6),
        Trimming(top_db_range=(30, 60), p=0.6)
    ])
    transformed_waveform = transforms(waveform)
    #transformed_waveform = torch.tensor(transformed_waveform * MAX_VALUE, dtype=torch.int16)
    transformed_waveform = transformed_waveform * MAX_VALUE
    transformed_waveform = transformed_waveform.type(torch.int16)
    torchaudio.save(filepath=output_filepath, src=transformed_waveform, sample_rate=sr, encoding="PCM_S", bits_per_sample=16)


def create_values_bins():
    #bins = [i/3 for i in range(3,16,1)]
    #bins = [i/5 for i in range(5,26,1)]
    bins = [i/10 for i in range(10,51,1)]
    #bins = [i/4 for i in range(4,21,1)]
    return bins


def get_index_bin(score):
    bins_scores = create_values_bins()
    for index, value in enumerate(bins_scores):
        if score <= value:
            if abs(bins_scores[index]-score) < abs(bins_scores[index-1]-score):
                return bins_scores[index]
            else:
                return bins_scores[index-1]


def create_value_counts(filelist_scores):
    bins_scores = create_values_bins()
    value_counts = {}
    for filepath, score in filelist_scores:
        index = get_index_bin(score)
        value_counts[index] = value_counts.get(index, 0) + 1

    return value_counts


def create_weights(filelist_scores):
    value_counts = create_value_counts(filelist_scores)
    max_value = max(value_counts.values())
    filelist_scores_weights = []
    for filepath, score in filelist_scores:
        index = get_index_bin(score)
        #weight = max_value / value_counts[index]
        weight = value_counts[index]
        item = (filepath, score, weight)
        filelist_scores_weights.append(item)

    return filelist_scores_weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--base_dir', default='./')
    parser.add_argument('-i', '--input_dir', help='Input wavs folder')
    parser.add_argument('-c', '--input_csv', help='Input metadata filepath')
    parser.add_argument('-o', '--output_dir', default='output_data_aug', help='Output folder')
    parser.add_argument('-d', '--output_csv', default='output.csv   ', help='Output metadata filepath')
    args = parser.parse_args()

    output_dir = join(args.base_dir, args.output_dir)
    input_dir = join(args.base_dir, args.input_dir)

    input_csv = join(args.base_dir, args.input_csv)
    output_csv = join(args.base_dir, args.output_csv)

    with open(input_csv, encoding="utf-8") as f:
        content_file = f.readlines()[1:]
        # Create a list like [(filename, score)]
        filelist = [(line.split(",")[0],float(line.split(",")[1])) for line in content_file]

    filelist_scores_weights = create_weights(filelist)
    makedirs(output_dir, exist_ok=True)
    max_augmentations = 10
    # Creating output file header
    output_file = open(output_csv, 'w')
    separator = ","
    line = separator.join(['filepath', 'score'])
    output_file.write(line + '\n')

    for filepath, score, weight in tqdm(filelist_scores_weights):
        index = 0
        for index in range(min(max_augmentations, int(weight) -1 )):
            filename = "aug{}-{}".format(index, basename(filepath))
            in_filepath = join(input_dir, filepath + ".wav")
            out_filepath = join(output_dir, filename + ".wav")
            if not exists(in_filepath):
                print("File not found: {}".format(in_filepath))
                continue
            execute_transformations(in_filepath, out_filepath)

            # Write to output csv file
            line = separator.join([filename, str(score)])
            output_file.write(line + '\n')

    output_file.close()


if __name__ == "__main__":
    main()
