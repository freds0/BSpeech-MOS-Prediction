import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from utils.logger import logger
from os.path import join
import pandas as pd
import numpy as np

class Wav2vecEmbeddingsDataset(Dataset):
    def __init__(self, filepaths: list, scores: list, reweight='none',
                 lds=False, lds_kernel='gaussian', lds_ks=5, lds_sigma=2):
        self.filepaths = filepaths
        self.scores = scores
        self.weights = self._prepare_weights(reweight=reweight, lds=lds, lds_kernel=lds_kernel, lds_ks=lds_ks, lds_sigma=lds_sigma)

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        filename = self.filepaths[idx]
        embedding = torch.load(filename)
        score = self.scores[idx]
        weight = np.asarray([self.weights[index]]).astype('float32') if self.weights is not None else np.asarray([np.float32(1.)])
        return embedding, score, weight

    def _prepare_weights(self, reweight, max_target=17, lds=False, lds_kernel='gaussian', lds_ks=5, lds_sigma=2):
        assert reweight in {'none', 'inverse', 'sqrt_inv'}
        assert reweight != 'none' if lds else True, \
            "Set reweight to \'sqrt_inv\' (default) or \'inverse\' when using LDS"

        value_dict = {x: 0 for x in range(max_target)}
        labels = self.scores 
        # mbr
        for label in labels:
            value_dict[min(max_target - 1, int(label))] += 1
        if reweight == 'sqrt_inv':
            value_dict = {k: np.sqrt(v) for k, v in value_dict.items()}
        elif reweight == 'inverse':
            value_dict = {k: np.clip(v, 5, 1000) for k, v in value_dict.items()}  # clip weights for inverse re-weight
        num_per_label = [value_dict[min(max_target - 1, int(label))] for label in labels]
        if not len(num_per_label) or reweight == 'none':
            return None
        print(f"Using re-weighting: [{reweight.upper()}]")

        if lds:
            lds_kernel_window = get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)
            print(f'Using LDS: [{lds_kernel.upper()}] ({lds_ks}/{lds_sigma})')
            smoothed_value = convolve1d(
                np.asarray([v for _, v in value_dict.items()]), weights=lds_kernel_window, mode='constant')
            num_per_label = [smoothed_value[min(max_target - 1, int(label))] for label in labels]

        weights = [np.float32(1 / x) for x in num_per_label]
        scaling = len(weights) / np.sum(weights)
        weights = [scaling * x for x in weights]
        return weights


def wav2vec_embedding_collate_fn(data):
    """
       data: is a list of tuples with (example, label, length)
             where 'example' is a tensor of arbitrary shape
             and label/length are scalars
    """
    features = []
    scores = []
    weights = []

    for feature, score, weight in data:
        features.append(feature)
        scores.append(score)
        weights.append(weight)

    features = pad_sequence([f.squeeze() for f in features], batch_first=True)
    scores = torch.tensor(np.array(scores))
    weights = torch.tensor(np.array(weights))
    return features, scores, weights


class WeightedWav2VecDataloader(DataLoader):
    def __init__(self, data_dir, metadata_file, val_metadata_file, emb_dir, train_batch_size, val_batch_size, shuffle=False):
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.shuffle = shuffle
        self.data_dir = data_dir
        self.emb_dir = emb_dir
        self.val_metadata = join(data_dir, val_metadata_file)

        train_data = pd.read_csv(join(data_dir, metadata_file))
        train_data['score'] = train_data['score'] / train_data['score'].max()
        train_scores = train_data['score'].to_list()
        train_data['filepath'] = str(self.data_dir + "/" + self.emb_dir + "/") + train_data['filepath'] + ".pt"
        train_filepaths = train_data['filepath'].to_list()

        logger.info("Dataset {} training files loaded".format(len(train_filepaths)))

        self.dataset = Wav2vecEmbeddingsDataset(train_filepaths, train_scores)
        super().__init__(dataset=self.dataset, batch_size=self.train_batch_size, shuffle=False, num_workers=0, collate_fn=wav2vec_embedding_collate_fn)


    def get_val_dataloader(self):
        val_data = pd.read_csv(self.val_metadata)
        val_data['score'] = val_data['score'] / val_data['score'].max()
        val_scores = val_data['score'].to_list()
        val_data['filepath'] = str(self.data_dir + "/" + self.emb_dir + "/") + val_data['filepath'] + ".pt"
        val_filepaths = val_data['filepath'].to_list()

        print("Dataset {} validating files loaded".format(len(val_filepaths)))
        self.val_dataset = Wav2vecEmbeddingsDataset(val_filepaths, val_scores)
        return DataLoader(dataset=self.val_dataset, batch_size=self.val_batch_size, shuffle=False, num_workers=0, collate_fn=wav2vec_embedding_collate_fn)
