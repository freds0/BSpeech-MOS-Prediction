import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence 
from os.path import join
import pandas as pd
import random

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EmbeddingsDataset(Dataset):
    def __init__(self, filepaths: list, scores: list):
        self.filepaths = filepaths
        self.scores = scores

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        filename = self.filepaths[idx]
        embedding = torch.load(filename)#.transpose(2,1)
        embedding = embedding.mean(axis=-1)
        score = self.scores[idx]
        #return {"data": embedding, "score": score}
        return embedding, score


def embedding_collate_fn(data):
    """
       data: is a list of tuples with (example, label, length)
             where 'example' is a tensor of arbitrary shape
             and label/length are scalars
    """
    features = [torch.tensor(d['data']) for d in data] #(3)
    scores = torch.tensor([d['score']  for d in data])
    new_features = pad_sequence([f.T for f in features], batch_first=True).squeeze()

    return  {
        'data': new_features,
        'score': scores
    }

class EmbeddingsDataloader(DataLoader):
    def __init__(self, data_dir, metadata_file, val_metadata_file, emb_dir, train_batch_size, val_batch_size, shuffle=False, validation_split=0.1, training=True):
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.shuffle = shuffle
        self.data_dir = data_dir
        self.emb_dir = emb_dir
        #self.seed = 42
        self.val_metadata = join(data_dir, val_metadata_file)

        train_data = pd.read_csv(join(data_dir, metadata_file))
        train_data['score'] = train_data['score'] / train_data['score'].max()
        train_scores = train_data['score'].to_list()
        train_data['filepath'] = str(self.data_dir + "/" + self.emb_dir + "/") + train_data['filepath'] + ".pt"
        train_filepaths = train_data['filepath'].to_list()

        #random.seed(self.seed)

        #train_filepaths = train_filepaths[:int((len(train_filepaths) + 1) * validation_split)]
        #train_scores = train_scores[:int((len(train_scores) + 1) * validation_split)]

        print("Dataset {} training files loaded".format(len(train_filepaths)))
        #random.seed(self.seed)
        #test_filepaths = filepaths[int((len(filepaths) + 1) * validation_split):]
        #test_scores = filepaths[:int((len(scores) + 1) * validation_split)]

        #self.train_dataset = EmbeddingsDataset(train_filepaths, train_scores)
        #self.test_dataset = EmbeddingsDataset(test_filepaths, test_scores)

        self.dataset = EmbeddingsDataset(train_filepaths, train_scores)
        super().__init__(dataset=self.dataset, batch_size=self.train_batch_size, shuffle=False, num_workers=0)


    def get_val_dataloader(self):

        val_data = pd.read_csv(self.val_metadata)
        val_data['score'] = val_data['score'] / val_data['score'].max()
        val_scores = val_data['score'].to_list()
        val_data['filepath'] = str(self.data_dir + "/" + self.emb_dir + "/") + val_data['filepath'] + ".pt"
        val_filepaths = val_data['filepath'].to_list()

        print("Dataset {} validating files loaded".format(len(val_filepaths)))
        self.val_dataset = EmbeddingsDataset(val_filepaths, val_scores)
        return DataLoader(dataset=self.val_dataset, batch_size=self.val_batch_size, shuffle=False, num_workers=0)
#################################################################################################

class Wav2vecEmbeddingsDataset(Dataset):
    def __init__(self, filepaths: list, scores: list):
        self.filepaths = filepaths
        self.scores = scores

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        filename = self.filepaths[idx]
        embedding = torch.load(filename)#.transpose(2,1)
        #embedding = embedding.mean(axis=-1)
        score = self.scores[idx]
        #return {"data": embedding, "score": score}
        return embedding, score


def Wav2vecembedding_collate_fn(data):
    """
       data: is a list of tuples with (example, label, length)
             where 'example' is a tensor of arbitrary shape
             and label/length are scalars
    """
    features = []
    scores = []
    for feature, score in data:
        features.append(torch.tensor(feature))
        scores.append(score)

    features = pad_sequence([f.squeeze() for f in features], batch_first=True)#squeeze()
    features = torch.tensor(features)
    scores = torch.tensor(scores)
    return features, scores


class Wav2vecEmbeddingsDataloader(DataLoader):
    def __init__(self, data_dir, metadata_file, val_metadata_file, emb_dir, train_batch_size, val_batch_size, shuffle=False, validation_split=0.1, training=True):
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.shuffle = shuffle
        self.data_dir = data_dir
        self.emb_dir = emb_dir
        #self.seed = 42
        self.val_metadata = join(data_dir, val_metadata_file)

        train_data = pd.read_csv(join(data_dir, metadata_file))
        train_data['score'] = train_data['score'] / train_data['score'].max()
        train_scores = train_data['score'].to_list()
        train_data['filepath'] = str(self.data_dir + "/" + self.emb_dir + "/") + train_data['filepath'] + ".pt"
        train_filepaths = train_data['filepath'].to_list()

        #random.seed(self.seed)

        #train_filepaths = train_filepaths[:int((len(train_filepaths) + 1) * validation_split)]
        #train_scores = train_scores[:int((len(train_scores) + 1) * validation_split)]

        print("Dataset {} training files loaded".format(len(train_filepaths)))
        #random.seed(self.seed)
        #test_filepaths = filepaths[int((len(filepaths) + 1) * validation_split):]
        #test_scores = filepaths[:int((len(scores) + 1) * validation_split)]

        #self.train_dataset = EmbeddingsDataset(train_filepaths, train_scores)
        #self.test_dataset = EmbeddingsDataset(test_filepaths, test_scores)

        self.dataset = Wav2vecEmbeddingsDataset(train_filepaths, train_scores)
        super().__init__(dataset=self.dataset, batch_size=self.train_batch_size, shuffle=False, num_workers=0, collate_fn=Wav2vecembedding_collate_fn)


    def get_val_dataloader(self):

        val_data = pd.read_csv(self.val_metadata)
        val_data['score'] = val_data['score'] / val_data['score'].max()
        val_scores = val_data['score'].to_list()
        val_data['filepath'] = str(self.data_dir + "/" + self.emb_dir + "/") + val_data['filepath'] + ".pt"
        val_filepaths = val_data['filepath'].to_list()

        print("Dataset {} validating files loaded".format(len(val_filepaths)))
        self.val_dataset = Wav2vecEmbeddingsDataset(val_filepaths, val_scores)
        return DataLoader(dataset=self.val_dataset, batch_size=self.val_batch_size, shuffle=False, num_workers=0, collate_fn=Wav2vecembedding_collate_fn)


################################################################################################# 

class CNN2DEmbeddingsDataset(Dataset):
    def __init__(self, filepaths: list, scores: list):
        self.filepaths = filepaths
        self.scores = scores

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        filename = self.filepaths[idx]
        embedding = torch.load(filename)
        score = self.scores[idx]
        return embedding, score


def cnn2d_embedding_collate_fn(data):
    features = [torch.tensor(d[0]) for d in data]
    scores = torch.tensor([d[1] for d in data])
    new_features = pad_sequence([f.T for f in features], batch_first=True).squeeze().unsqueeze(1)
    return new_features, scores


class CNN2DEmbeddingsDataloader(DataLoader):
    def __init__(self, data_dir, metadata_file, val_metadata_file, emb_dir, train_batch_size, val_batch_size, shuffle=False, validation_split=0.1, training=True):
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.shuffle = shuffle
        self.data_dir = data_dir
        self.emb_dir = emb_dir
        #self.seed = 42
        self.val_metadata = join(data_dir, val_metadata_file)

        train_data = pd.read_csv(join(data_dir, metadata_file))
        train_data['score'] = train_data['score'] / train_data['score'].max()
        train_scores = train_data['score'].to_list()
        train_data['filepath'] = str(self.data_dir + "/" + self.emb_dir + "/") + train_data['filepath'] + ".pt"
        train_filepaths = train_data['filepath'].to_list()

        print("Dataset {} training files loaded".format(len(train_filepaths)))

        self.dataset = CNN2DEmbeddingsDataset(train_filepaths, train_scores)
        super().__init__(dataset=self.dataset, batch_size=self.train_batch_size, shuffle=False, num_workers=0, collate_fn=cnn2d_embedding_collate_fn)


    def get_val_dataloader(self):

        val_data = pd.read_csv(self.val_metadata)
        val_data['score'] = val_data['score'] / val_data['score'].max()
        val_scores = val_data['score'].to_list()
        val_data['filepath'] = str(self.data_dir + "/" + self.emb_dir + "/") + val_data['filepath'] + ".pt"
        val_filepaths = val_data['filepath'].to_list()

        print("Dataset {} validating files loaded".format(len(val_filepaths)))
        self.val_dataset = CNN2DEmbeddingsDataset(val_filepaths, val_scores)
        return DataLoader(dataset=self.val_dataset, batch_size=self.val_batch_size, shuffle=False, num_workers=0, collate_fn=cnn2d_embedding_collate_fn)

################################################################################################# 

class TitanetEmbeddingsDataset(Dataset):
    def __init__(self, filepaths: list, scores: list):
        self.filepaths = filepaths
        self.scores = scores

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        filename = self.filepaths[idx]
        embedding = torch.load(filename)#.transpose(2,1)
        #embedding = embedding.mean(axis=-1)
        score = self.scores[idx]
        #return {"data": embedding, "score": score}
        return embedding, score


class TitanetEmbeddingsDataloader(DataLoader):
    def __init__(self, data_dir, metadata_file, val_metadata_file, emb_dir, train_batch_size, val_batch_size, shuffle=False, validation_split=0.1, training=True):
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.shuffle = shuffle
        self.data_dir = data_dir
        self.emb_dir = emb_dir
        #self.seed = 42
        self.val_metadata = join(data_dir, val_metadata_file)

        train_data = pd.read_csv(join(data_dir, metadata_file))
        train_data['score'] = train_data['score'] / train_data['score'].max()
        train_scores = train_data['score'].to_list()
        train_data['filepath'] = str(self.data_dir + "/" + self.emb_dir + "/") + train_data['filepath'] + ".pt"
        train_filepaths = train_data['filepath'].to_list()

        #random.seed(self.seed)

        #train_filepaths = train_filepaths[:int((len(train_filepaths) + 1) * validation_split)]
        #train_scores = train_scores[:int((len(train_scores) + 1) * validation_split)]

        print("Dataset {} training files loaded".format(len(train_filepaths)))
        #random.seed(self.seed)
        #test_filepaths = filepaths[int((len(filepaths) + 1) * validation_split):]
        #test_scores = filepaths[:int((len(scores) + 1) * validation_split)]

        #self.train_dataset = EmbeddingsDataset(train_filepaths, train_scores)
        #self.test_dataset = EmbeddingsDataset(test_filepaths, test_scores)

        self.dataset = TitanetEmbeddingsDataset(train_filepaths, train_scores)
        super().__init__(dataset=self.dataset, batch_size=self.train_batch_size, shuffle=False, num_workers=0)


    def get_val_dataloader(self):

        val_data = pd.read_csv(self.val_metadata)
        val_data['score'] = val_data['score'] / val_data['score'].max()
        val_scores = val_data['score'].to_list()
        val_data['filepath'] = str(self.data_dir + "/" + self.emb_dir + "/") + val_data['filepath'] + ".pt"
        val_filepaths = val_data['filepath'].to_list()

        print("Dataset {} validating files loaded".format(len(val_filepaths)))
        self.val_dataset = TitanetEmbeddingsDataset(val_filepaths, val_scores)
        return DataLoader(dataset=self.val_dataset, batch_size=self.val_batch_size, shuffle=False, num_workers=0)

