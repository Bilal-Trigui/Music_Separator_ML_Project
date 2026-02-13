#importing libraries
import pathlib as pl
import soundfile as sf
import torchaudio as ta
import numpy as np
import torch
from torch.utils.data import DataSet, DataLoader
import librosa
from dataset import dataSet as ds

#Creates array of audio files that will be used for model training
train_root = pl.Path("/media/bilal/HardDrive2/musdb18hq/train/")
val_root = pl.Path("/media/bilal/HardDrive2/musdb18hq/validation/")
test_root = pl.Path("/media/bilal/HardDrive2/musdb18hq/test/")

#wrapping dataset into a dataloader
trackset = ds(train_root)
track_loader = DataLoader(trackset, batch_size=4, shuffle=True, num_workers=2)


