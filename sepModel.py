#importing libraries
import pathlib as pl
import soundfile as sf
import torchaudio as ta
import numpy as np
import torch as tch
import librosa

#Creates array of audio files that will be used for model training
root = pl.Path("/media/bilal/HardDrive2/musdb18/wav_train/")
tracks = list(root.glob("*"))

audio, sr = sf.read(tracks[0])

