#importing libraries
import pathlib as pl
import soundfile as sf
import torchaudio as ta
import numpy as np
import torch
import librosa

#Creates array of audio files that will be used for model training
root = pl.Path("/media/bilal/HardDrive2/musdb18hq/train/")
tracks = list(root.glob("*"))



