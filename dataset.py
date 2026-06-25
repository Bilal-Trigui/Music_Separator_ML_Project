#import libraries
import numpy as np
import torch
import librosa
from torch.utils.data import Dataset
from pathlib import Path

def to_normalized_db(audio, n_fft, hop_length):

    #converts audio to a dB-scale spectrogram using a fixed absolute floor
    #(top_db=None) instead of librosa's default, which floors each call
    #relative to its own peak and so puts mix/vocals/drums/bass/other on
    #inconsistent scales depending on how loud each one happens to be
    db = librosa.amplitude_to_db(np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)), top_db=None)
    db = np.clip(db, -80, 0)

    #rescales the fixed [-80, 0] dB range to [-1, 1]
    db = db / 40 + 1
    return torch.tensor(db, dtype=torch.float32)

def from_normalized_db(db_norm):

    #inverts to_normalized_db: [-1, 1] -> [-80, 0] dB -> linear amplitude
    db = (np.clip(db_norm, -1, 1) - 1) * 40
    return librosa.db_to_amplitude(db, ref=1.0)

class dataSet(Dataset):
    def __init__(self, root, segment_length=6, sr=44100, n_fft=1024, hop_length=256):

        #initiallizes data to dafaults and gets files from file directory for training
        self.root = Path(root)
        self.tracks = list(self.root.glob("*"))
        if not self.tracks:
            raise FileNotFoundError(
                f"No tracks found in '{self.root}' - check that the drive is mounted "
                f"and the path is correct."
            )
        self.sr = sr
        self.segment_length = segment_length
        self.n_fft = n_fft
        self.hop_length = hop_length

    def __len__(self):

        #returns total length of the dataset
        return len(self.tracks)

    def _to_normalized_db(self, audio):
        return to_normalized_db(audio, self.n_fft, self.hop_length)

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

        #creates tensors for all the audio tracks using normalized logorithimic spectrogram data
        mix_log = self._to_normalized_db(mix).unsqueeze(0)

        stem_logs = torch.stack([
            self._to_normalized_db(vocals),
            self._to_normalized_db(drums),
            self._to_normalized_db(bass),
            self._to_normalized_db(other),
        ])

        return mix_log, stem_logs