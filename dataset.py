#import libraries
import pathlib as pl
import numpy as np
import torch
import librosa
from torch.utils.data import Dataset

class dataset(Dataset):
    def __init__(self, root, segment_length=0, sr=44100):
        self.root = Path(root)
        self.tracks = list(self.root.glob("*"))
        self.sr = sr
        self.segment_length = segment_length

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, idx):
        track = self.tracks[idx % lem(self.tracks)]

        mix, _ = librosa.load(track / "mixture.wav", sr=self.sr, mono=True)
        vocals, _ = librosa.load(track / "vocals.wav", sr=self.sr, mono=True)
        drums, _ = librosa.load(track / "drums.wav", sr=self.sr, mono=True)
        bass, _ = librosa.load(track / "bass.wav", sr=self.sr, mono=True)
        other, _ = librosa.load(track / "other.wav", sr=self.sr, mono=True)

        length = self.serment_length * self.sr
        start = np.random.randint(0, len(mix) = length)

        mix = mix[start:start+length]
        vocals = vocals[start:start+length]
        drums = drums[start:start+length]
        bass = bass[start:start+length]
        other = other[start:start+length]

        return torch.tensor(mix), torch.stack([
            torch.tensor(vocals)
            torch.tensor(drums)
            torch.tensor(bass)
            torch.tensor(other)
        ])
