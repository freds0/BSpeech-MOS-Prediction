import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import warnings
import pandas as pd
import numpy as np
import argparse
from datasets.vcc2018_dataset import VCC2018WavDatasetLoad, custom_spec_collate_fn
from models.wav2vec_model import Wav2Vec2ForSpeechClassification
from transformers import AutoConfig
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

warnings.filterwarnings("ignore")

def train(model, iterator, optimizer, criterion, scheduler, epoch=0):
    model.train()
    epoch_loss, accuracy, f1, recall, precision = 0, 0, 0, 0, 0

    total_steps = len(iterator)
    for i, batch in enumerate(iterator):
        data = batch['data'].to(device, dtype=torch.float32).unsqueeze(1)
        labels = batch['target'].to(device)

        optimizer.zero_grad()

        output = model(data)

        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        # calculate metrics
        output = F.softmax(output, 1)
        result = output.argmax(1)

        accuracy += accuracy_score(result.cpu(), labels.cpu())
        f1 += f1_score(result.cpu(), labels.cpu(), average='micro')
        recall += recall_score(result.cpu(), labels.cpu(), average='micro')
        precision += precision_score(result.cpu(), labels.cpu(), average='micro')

        if (i % 10 == 0):
            step_loss = "{:.5f}".format(epoch_loss / (i + 1))
            step_acc = "{:.5f}".format(accuracy / (i + 1))
            step_f1 = "{:.5f}".format(f1 / (i + 1))
            step_recall = "{:.5f}".format(recall / (i + 1))
            step_precision = "{:.5f}".format(precision / (i + 1))
            print(
                f"Train step {i} loss: {step_loss} acc: {step_acc} f1: {step_f1} recall {step_recall} precision: {step_precision}")
    # wandb.log({"loss_train": epoch_loss / (i+1), "accuracy_train": accuracy / (i+1), "f1_train": f1 / (i+1), "recall_train": recall / (i+1), "precision_train": precision / (i+1)})

    accuracy /= (i + 1)
    f1 /= (i + 1)
    epoch_loss /= (i + 1)
    recall /= (i + 1)
    precision /= (i + 1)

    return epoch_loss, accuracy, f1, recall, precision


def evaluate(model, iterator, criterion, epoch):
    model.eval()
    epoch_loss, accuracy, f1, recall, precision = 0, 0, 0, 0, 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            data = batch['data'].to(device, dtype=torch.float32).unsqueeze(1)
            batch_size = data.shape[0]

            labels = batch['target'].to(device)
            output = model(data)
            loss = criterion(output, labels)

            epoch_loss += loss.item()
            result = output.argmax(1)

            accuracy += accuracy_score(result.cpu(), labels.cpu())
            f1 += f1_score(result.cpu(), labels.cpu(), average='micro')
            recall += recall_score(result.cpu(), labels.cpu(), average='micro')
            precision += precision_score(result.cpu(), labels.cpu(), average='micro')

    # wandb.log({"loss_train": epoch_loss / (i+1), "accuracy_train": accuracy / (i+1), "f1_train": f1 / (i+1), "recall_train": recall / (i+1), "precision_train": precision / (i+1)})

    accuracy /= (i + 1)
    f1 /= (i + 1)
    epoch_loss /= (i + 1)
    recall /= (i + 1)
    precision /= (i + 1)

    return epoch_loss, accuracy, f1, recall, precision


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def execute_training(args):
    train_data = pd.read_csv(args.train_csv)
    test_data = pd.read_csv(args.test_csv)

    categories_train = train_data['score'].to_list()
    audio_names_train = train_data['filepath'].to_list()

    categories_test = test_data['score'].to_list()
    audio_names_test = test_data['filepath'].to_list()

    # get all unique categories
    num_classes = len(pd.concat([train_data, test_data])['score'].unique())
    #categories_types = np.sort(df['score'].unique())

    dataset_train = VCC2018WavDatasetLoad(audio_names_train, categories_train)
    loader_train = DataLoader(dataset_train, batch_size=32, shuffle=True, collate_fn=custom_spec_collate_fn)

    dataset_test = VCC2018WavDatasetLoad(audio_names_test, categories_test)
    loader_test = DataLoader(dataset_test, batch_size=8, shuffle=True, collate_fn=custom_spec_collate_fn)


    # wandb.init(project='emble_audio_classification', entity='miana')
    # config = wandb.config

    model_name_or_path = "facebook/wav2vec2-base-960h"
    pooling_mode = "mean"

    config = AutoConfig.from_pretrained(
        model_name_or_path,
        finetuning_task="wav2vec2_clf",
    )

    model = Wav2Vec2ForSpeechClassification(config).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-1)
    criterion = nn.CrossEntropyLoss()

    lambda2 = lambda epoch: epoch * 0.95
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,  lr_lambda=[lambda2])

    N_EPOCHS = 10

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    train_f1s, val_f1s = [], []

    best_valid_loss = float('inf')

    # wandb.watch(model)
    for epoch in range(N_EPOCHS):
        start_time = time.time()
        train_loss, train_accuracy, train_f1, train_recall, train_precision = train(model, loader_train, optimizer,
                                                                                    criterion, scheduler, epoch)
        val_loss, val_accuracy, val_f1, val_recall, val_precision = evaluate(model, loader_test, criterion, epoch)

        end_time = time.time()

        # fill data
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        train_f1s.append(train_f1)
        val_accuracies.append(val_accuracy)
        val_f1s.append(val_f1)

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        torch.save(model.state_dict(), 'best-val-model.pt')

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins} m {epoch_secs} s')
        print(
            f'\tTrain Loss: {train_loss}, accuracy: {train_accuracy}, f1 {train_f1}, recall {train_recall}, precision {train_precision}')
        print(
            f'\t Val. Loss: {val_loss}, accuracy: {val_accuracy}, f1 {val_f1}, recall {val_recall}, precision {val_precision}')


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--base_dir', default='./')
  parser.add_argument('--train_csv', default='data/train.csv')
  parser.add_argument('--test_csv', default='data/test.csv')
  parser.add_argument('--filepath_column', default='score')
  parser.add_argument('--target_column', default='filepath', help='Name of the directory where txt files will be saved')
  args = parser.parse_args()
  execute_training(args)

if __name__ == "__main__":
  main()
