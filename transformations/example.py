import torchaudio
import random
from compose import Compose
from pad_crop import PadCrop
from speech_change import SpeedChange
from time_shift import TimeShift
from trimming import Trimming


if __name__ == '__main__':
    audio_filepath = "/home/fred/Projetos/DATASETS/MOS/BRSPEECH_MOS_DATASET/data/vits/vits_2961-0013.wav"
    waveform, sr = torchaudio.load(audio_filepath)

    pad_crop_value = int(random.uniform(0.95, 1.05)*  waveform.shape[1])

    transforms = Compose([
        SpeedChange(factor_range=(0.98, 1.02), p=1.0),
        PadCrop(pad_crop_value, crop_position='random', pad_position='random', p=1.0),
        TimeShift(shift_factor=(10, 50), p=1.0),
        Trimming(top_db_range=(30, 60), p=1.0)
    ])

    transformed_waveform = transforms(waveform)

    torchaudio.save(filepath="transformed_waveform.wav", src=transformed_waveform, sample_rate=sr)
    


