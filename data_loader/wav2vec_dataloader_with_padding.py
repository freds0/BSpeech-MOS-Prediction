import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from utils.logger import logger
from os.path import join
import random

class Wav2vecEmbeddingsDatasetWithPadding(Dataset):
    def __init__(self, filepaths: list, scores: list, max_timestep):
        self.filepaths = filepaths
        self.scores = scores
        self.max_timestep = max_timestep
        self.seed = 1234

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        filename = self.filepaths[idx]
        embedding = torch.load(filename)
        
        ref_shape = [embedding.shape[0], self.max_timestep, embedding.shape[2]]
        embedding = self.pad(embedding, ref_shape)
        
        score = self.scores[idx]

        return embedding, score

    def read_list(filelist):
        f = open(filelist, 'r')
        Path=[]
        for line in f:
            Path=Path+[line[0:-1]]
        return Path

    def pad(self, array, reference_shape):
        result = torch.zeros(reference_shape)
        result[:array.shape[0], :array.shape[1], :array.shape[2]] = array
        return result


def wav2vec_collate_fn(data):
    """
       data: is a list of tuples with (example, label, length)
             where 'example' is a tensor of arbitrary shape
             and label/length are scalars
    """
    features = []
    scores = []
    for feature, score in data:
        features.append(feature)
        scores.append(score)

    features = pad_sequence([f.squeeze() for f in features], batch_first=True)
    features = features
    scores = torch.tensor(scores)
    return features, scores


class Wav2VecDataloaderWithPadding(DataLoader):
    def __init__(self, data_dir, metadata_file, val_metadata_file, emb_dir, train_batch_size, val_batch_size, shuffle=False):
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.shuffle = shuffle
        self.data_dir = data_dir
        self.emb_dir = emb_dir
        self.train_metadata = join(data_dir, metadata_file) 
        self.val_metadata   = join(data_dir, val_metadata_file)

        train_content = self._read_file_content(self.train_metadata, ignore_header=True)
        val_content = self._read_file_content(self.val_metadata, ignore_header=True)
        
        train_filepaths = [str(self.data_dir + "/" + self.emb_dir + "/" + line.split(",")[0] + ".pt") for line in train_content]
        val_filepaths = [str(self.data_dir + "/" + self.emb_dir + "/" + line.split(",")[0] + ".pt") for line in val_content]
        
        train_scores    = [float(line.split(",")[1]) for line in train_content]
        #val_scores    = [float(line.split(",")[1]) for line in file_content]
        
        logger.info("Dataset {} training files loaded".format(len(train_filepaths)))
        
        self.max_timestep = self.getmax_timestep(train_filepaths + val_filepaths)
        
        
        self.dataset = Wav2vecEmbeddingsDatasetWithPadding(train_filepaths, train_scores, self.max_timestep)
        super().__init__(dataset=self.dataset, batch_size=self.train_batch_size, shuffle=False, num_workers=0, collate_fn=wav2vec_collate_fn)

        
    def _read_file_content(self, filepath, ignore_header=False):
        with open(filepath, "r") as f:
            if ignore_header:
                content = f.readlines()[1:]
            else:
                content = f.readlines()
                
        return content
    

    def getmax_timestep(self, filepaths):
        random.seed(1234)
        random.shuffle(filepaths)
        for i, filename in enumerate(filepaths):
            feat = torch.load(filename)
            if i == 0:
                max_timestep = feat.shape[1]
            else:
                if feat.shape[1] > max_timestep:
                    max_timestep = feat.shape[1]
        return max_timestep    
    
    def get_val_dataloader(self):

        file_content = self._read_file_content(self.val_metadata, ignore_header=True)
        val_filepaths = [str(self.data_dir + "/" + self.emb_dir + "/" + line.split(",")[0] + ".pt") for line in file_content]
        val_scores    = [float(line.split(",")[1]) for line in file_content]

        print("Dataset {} validating files loaded".format(len(val_filepaths)))

        self.val_dataset = Wav2vecEmbeddingsDatasetWithPadding(val_filepaths, val_scores, self.max_timestep)
        return DataLoader(dataset=self.val_dataset, batch_size=self.val_batch_size, shuffle=False, num_workers=0, collate_fn=wav2vec_collate_fn)
