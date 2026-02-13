#import libraries
import numpy as np
import torch
import librosa
from torch.utils.data import Dataset
from pathlib import Path

class dataSet(Dataset):
    def __init__(self, root, segment_length=6, sr=44100, n_fft=1024, hop_length=256):

        #initiallizes data to dafaults and gets files from file directory for training
        self.root = Path(root)
        self.tracks = list(self.root.glob("*"))
        self.sr = sr
        self.segment_length = segment_length
        self.n_fft = n_fft
        self.hop_length = hop_length

    def __len__(self):

        #returns total length of the dataset
        return len(self.tracks)

    def __getitem__(self, idx):

        #gets trackj to turn into tensors
        track = self.tracks[idx % len(self.tracks)]

        #loads the mix and all individual stems into librosa
        mix, _ = librosa.load(track / "mixture.wav", sr=self.sr, mono=True)
        vocals, _ = librosa.load(track / "vocals.wav", sr=self.sr, mono=True)
        drums, _ = librosa.load(track / "drums.wav", sr=self.sr, mono=True)
        bass, _ = librosa.load(track / "bass.wav", sr=self.sr, mono=True)
        other, _ = librosa.load(track / "other.wav", sr=self.sr, mono=True)

        #defines length for the training window of the track
        length = int(self.segment_length * self.sr)
        
        if len(mix) <= length:
            start = 0
        else:
            start = np.random.randint(0, len(mix) - length)

        #sets training window for each track to convert to spectrogram data
        mix = mix[start:start+length]
        vocals = vocals[start:start+length]
        drums = drums[start:start+length]
        bass = bass[start:start+length]
        other = other[start:start+length]

        #creates tensors for all the audio tracks using logorithimic spectrogram data
        mix_log = torch.tensor(librosa.amplitude_to_db(np.abs(librosa.stft(mix, n_fft=self.n_fft, hop_length=self.hop_length))), dtype=torch.float32)

        stem_logs = torch.stack([
            torch.tensor(librosa.amplitude_to_db(np.abs(librosa.stft(vocals, n_fft=self.n_fft, hop_length=self.hop_length))), dtype=torch.float32),
            torch.tensor(librosa.amplitude_to_db(np.abs(librosa.stft(drums, n_fft=self.n_fft, hop_length=self.hop_length))), dtype=torch.float32),
            torch.tensor(librosa.amplitude_to_db(np.abs(librosa.stft(bass, n_fft=self.n_fft, hop_length=self.hop_length))), dtype=torch.float32),
            torch.tensor(librosa.amplitude_to_db(np.abs(librosa.stft(other, n_fft=self.n_fft, hop_length=self.hop_length))), dtype=torch.float32),
        ])

        return mix_log, stem_logs