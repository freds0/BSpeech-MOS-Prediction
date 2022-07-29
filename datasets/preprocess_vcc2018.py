from vcc2018_dataset import VCC2018SpecDatasetLoad, custom_spec_collate_fn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchaudio
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np

df = pd.read_csv('../DATASETS/MOS/VCC2018_MOS_preprocessed/mos_list.txt', header=None)
df[0] = '../DATASETS/MOS/VCC2018_MOS_preprocessed/wav/' + df[0]
df.columns = ['filepath', 'score']

num_classes = len(df['score'].unique())

le = preprocessing.LabelEncoder()
le.fit(df['score'])
df['score'] = le.transform(df['score'])

data = []
wavs_folder = "../DATASETS/VCC2018_MOS_preprocessed/wav/"
for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    #filepath = os.path.join(wavs_folder, row['path'])
    filepath = row['filepath']
    score = row['score']
    try:
        # There are some broken files
        s = torchaudio.load(filepath)
        data.append({
            # "name": name,
            "filepath": filepath,
            "score": score
        })
    except Exception as e:
        #print(str(filepath), e)
        pass

data = pd.DataFrame(data)
train_data, test_data = train_test_split(data, test_size=0.10)

train_data.to_csv(f"./train.csv", sep=",", encoding="utf-8", index=False)
test_data.to_csv(f"./test.csv", sep=",", encoding="utf-8", index=False)

