# BRSpeechMOS Prediction

This repository is related to the paper "Evaluation of Speech Representations for MOS prediction". In this paper, we evaluate feature extraction models for predicting speech quality. We also propose a model architecture to compare embeddings of supervised learning and self-supervised learning models with embeddings of speaker verification models to predict the metric MOS. Our experiments were performed on the VCC2018 dataset and a Brazilian-Portuguese dataset called BRSpeechMOS, which was created for this work. The results show that the Whisper model is appropriate in all scenarios: with both the VCC2018 and BRSpeechMOS datasets. Among the supervised and self-supervised learning models using BRSpeechMOS, Whisper-Small achieved the best linear correlation of 0.6980, and the speaker verification model, SpeakerNet, had linear correlation of 0.6963. Using VCC2018, the best supervised and self-supervised learning model, Whisper-Large, achieved linear correlation of 0.7274, and the best model speaker verification, TitaNet, achieved a linear correlation of 0.6933. Although the results of the speaker verification models are slightly lower, the SpeakerNet model has only 5M parameters, making it suitable for real-time applications, and the TitaNet model produces an embedding of size 192, the smallest among all the evaluated models. 


## Download Dataset

- [Google Drive](https://drive.google.com/file/d/1_nSqIzFzvDdmIaJB6uxjhJ20pnZJA7ru/view?usp=drive_link)
- [One Drive](https://ufmtbr-my.sharepoint.com/:u:/g/personal/frederico_oliveira_ufmt_br/ERGXsp4t10ZOmNpGUx2Z_moBWMuinZU8lztW26OogD17-A?e=2Zv0QN)


## Requirements

To install the requirements, create a virtual environment in Conda:

```bash
$ conda create env -n brspeech-mos-prediction python=3.9 pip
$ conda activate brspeech-mos-prediction
$ pip install -r requirements.txt
```

Alternatively, load the virtual environment from the environment.yml file.

```bash
$ conda env create -f environment.yml
```

## Feature Extraction

To perform the extraction of embeddings, use the scripts available in the extract_emb folder.

## Training

To run the training, the first step is to create a .json file used for configuring the training parameters. Examples of this file can be found in the configs folder.

To execute the training, run the following command

```bash
$ python train.py -c configs/config_model.json
```

## Testing

To evaluate your model, run the following command:

```bash
$ python test.py -c configs/config_model.json
```

## Citation

```
@inproceedings{brspeech-mos,
  title={Evaluation of Speech Representations for MOS prediction},
  author={Oliveira, Frederico S. and Casanova, Edresson and Junior, Arnaldo Candido and Gris, Lucas R. S. and Soares, Anderson S. and Galvão, Arlindo},
  booktitle={International Conference on Text, Speech, and Dialogue},
  year={2023},
  organization={Springer}
}
```
