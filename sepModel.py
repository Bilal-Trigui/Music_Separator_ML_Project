#importing libraries
import pathlib as pl
import soundfile as sf
import torchaudio as ta
import numpy as np
import torch
import librosa
from dataset import dataSet as ds

#Creates array of audio files that will be used for model training
train_root = pl.Path("/media/bilal/HardDrive2/musdb18hq/train/")
val_root = pl.Path("/media/bilal/HardDrive2/musdb18hq/validation/")
test_root = pl.Path("/media/bilal/HardDrive2/musdb18hq/test/")

trackset = Dataset(train_root)

