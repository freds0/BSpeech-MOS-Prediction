# Extract Embeddings

This folder has the scripts needed to extract the embeddings used in this work.

## Models from HuggingFace

To extract the embeddings using the Hubert, Wav2Vec or Whisper models, it is necessary to install the following requirements:

```bash
$ pip install transformers tqdm 
```

It is also necessary to install the Pytorch framework:

```bash
$ pip install torch torchaudio
```

Alternatively, you can create a conda env using the available yml file:

```bash
$ conda env create -f huggingface_environment.yml
```

Then just activate the virtual environment:

```bash
conda activate emb_hf
```

## Models from NeMo

To extract the embeddings using the SpeakerNet or TitaNet models, it is necessary to install the following requirements:

```bash
apt-get update && apt-get install -y libsndfile1 ffmpeg
pip install Cython tqdm
pip install nemo_toolkit['all']
```
It is also necessary to install the Pytorch framework:

```bash
$ pip install torch torchaudio
```

Alternatively, you can create a conda env using the available yml file:

```bash
$ conda env create -f nemo_environment.yml
```

Then just activate the virtual environment:

```bash
conda activate emb_nemo
```


## GE2E Model

To extract the embeddings using the GE2E model, you need to download the source code available from the following github repository:

```bash
$ git clone https://github.com/CorentinJ/Real-Time-Voice-Cloning.git
```
Download the encoder checkpoint:

```bash
$ wget https://github.com/Edresson/Real-Time-Voice-Cloning/releases/download/checkpoints/pretrained.zip
$ unzip pretrained.zip
```

It is necessary to install the following requirements:

```bash
$ python -m pip install umap-learn visdom webrtcvad librosa>=0.5.1 matplotlib>=2.0.2 numpy>=1.14.0  scipy>=1.0.0  tqdm sounddevice Unidecode inflect multiprocess numba
```

It is also necessary to install the Pytorch framework:

```bash
$ pip install torch
```

Alternatively, you can create a conda env using the available yml file:

```bash
$ conda env create -f ge2e_environment.yml
```

Then just activate the virtual environment:

```bash
conda activate emb_ge2e
```

## CLOVA Model

the CLOVA model was used from the coqui-TTS repository. To use it, you need clone coqui-TTS and install it locally.

```bash
$ git clone https://github.com/freds0/YourTTS/
```

It is also necessary to install the Pytorch framework:

```bash
$ conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

Install coqui repository requirements:

```bash
$ cd YourTTS
$ pip install -e .
```

Alternatively, you can create a conda env using the available yml file:

```bash
$ conda env create -f yourtts_environment.yml
```

Then just activate the virtual environment:

```bash
conda activate yourtts
```
