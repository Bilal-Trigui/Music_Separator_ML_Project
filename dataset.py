#import libraries
import numpy as np
import torch
import librosa
from torch.utils.data import Dataset

class dataSet(Dataset):
    def __init__(self, root, segment_length=6, sr=44100, n_fft=1024, hop_length=256):
        self.root = Path(root)
        self.tracks = list(self.root.glob("*"))
        self.sr = sr
        self.segment_length = segment_length
        self.n_fft = n_fft
        self.hop_length = hop_length

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, idx):
        track = self.tracks[idx % len(self.tracks)]

        mix, _ = librosa.load(track / "mixture.wav", sr=self.sr, mono=True)
        vocals, _ = librosa.load(track / "vocals.wav", sr=self.sr, mono=True)
        drums, _ = librosa.load(track / "drums.wav", sr=self.sr, mono=True)
        bass, _ = librosa.load(track / "bass.wav", sr=self.sr, mono=True)
        other, _ = librosa.load(track / "other.wav", sr=self.sr, mono=True)

        length = self.segment_length * self.sr
        start = np.random.randint(0, len(mix) - length)

        mix = mix[start:start+length]
        vocals = vocals[start:start+length]
        drums = drums[start:start+length]
        bass = bass[start:start+length]
        other = other[start:start+length]

        return torch.tensor(np.abs(librosa.stft(mix, n_fft=self.n_fft, hop_length=self.hop_length))), torch.stack([
            torch.tensor(np.abs(librosa.stft(vocals, n_fft=self.n_fft, hop_length=self.hop_length)))
            torch.tensor(np.abs(librosa.stft(drums, n_fft=self.n_fft, hop_length=self.hop_length)))
            torch.tensor(np.abs(librosa.stft(bass, n_fft=self.n_fft, hop_length=self.hop_length)))
            torch.tensor(np.abs(librosa.stft(other, n_fft=self.n_fft, hop_length=self.hop_length)))
        ])
