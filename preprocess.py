import numpy as np
import pandas as pd

from pathlib import Path
from tqdm import tqdm

import torchaudio
import os
import sys

dataframe = pd.read_csv('../DATASETS/NISQA_Corpus/NISQA_corpus_file.csv')

dataframe = dataframe[['db', 'filename_deg','mos']]
train = dataframe[((dataframe['db'] == 'NISQA_TRAIN_SIM') | (dataframe['db'] == 'NISQA_TRAIN_LIVE'))]
test = dataframe[((dataframe['db'] == 'NISQA_TEST_FOR') | (dataframe['db'] == 'NISQA_TEST_LIVETALK') | (dataframe['db'] == 'NISQA_TEST_P501'))]

train['path'] = '../DATASETS/NISQA_Corpus/' + train['db'] + '/deg/'  + train['filename_deg']
train = train.drop(['db', 'filename_deg'], axis=1)
train.columns = ['score', 'path']

test['path'] = '../DATASETS/NISQA_Corpus/' + test['db'] + '/deg/'  + test['filename_deg']
test = test.drop(['db', 'filename_deg'], axis=1)
test.columns = ['score', 'path']

train_df = []

for index, row in tqdm(train.iterrows(), total=train.shape[0]):
    filepath = row['path']
    score = row['score']
    try:
        # There are some broken files
        s = torchaudio.load(filepath)
        train_df.append({
            # "name": name,
            "path": filepath,
            "score": score
        })
    except Exception as e:
        print(str(filepath), e)
        pass

test_df = []

for index, row in tqdm(test.iterrows(), total=test.shape[0]):
    filepath = row['path']
    score = row['score']
    try:
        # There are some broken files
        s = torchaudio.load(filepath)
        test_df.append({
            # "name": name,
            "path": filepath,
            "score": score
        })
    except Exception as e:
        print(str(filepath), e)
        pass


save_path = "./dataset/"

train_df = pd.DataFrame(train_df)
test_df = pd.DataFrame(test_df)

train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

train_df.to_csv(f"{save_path}/train.csv", sep="\t", encoding="utf-8", index=False)
test_df.to_csv(f"{save_path}/test.csv", sep="\t", encoding="utf-8", index=False)


print(train_df.shape)
print(test_df.shape)