#import libraries
import argparse
import pathlib as pl
import subprocess as sp

import numpy as np
import torch
import librosa
import soundfile as sf

from dataset import to_normalized_db, from_normalized_db
from sepModel import seperatorModel as sm

STEM_NAMES = ["vocals", "drums", "bass", "other"]

#separates a mixture file into its stems and saves each as an mp3
def separate(mixture_path, model_path, output_dir, n_fft=1024, hop_length=256, sr=44100):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    mix_audio, _ = librosa.load(mixture_path, sr=sr, mono=True)

    #keeps the mixture's phase since the model only predicts magnitude
    mix_complex = librosa.stft(mix_audio, n_fft=n_fft, hop_length=hop_length)
    mix_phase = np.angle(mix_complex)

    mix_log = to_normalized_db(mix_audio, n_fft, hop_length).unsqueeze(0).unsqueeze(0).to(device)

    model = sm().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    with torch.no_grad():
        pred = model(mix_log)

    pred = pred.squeeze(0).cpu().numpy()  # (4, F, T)

    #the model's internal pooling/upsampling can shrink dimensions slightly,
    #so crop the mixture phase down to match the predicted spectrogram size
    phase = mix_phase[:pred.shape[1], :pred.shape[2]]

    output_dir = pl.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for stem_idx, name in enumerate(STEM_NAMES):
        amplitude = from_normalized_db(pred[stem_idx])
        complex_spec = amplitude * np.exp(1j * phase)
        waveform = librosa.istft(complex_spec, hop_length=hop_length)

        wav_path = output_dir / f"{name}.wav"
        mp3_path = output_dir / f"{name}.mp3"
        sf.write(wav_path, waveform, sr)

        sp.run(["ffmpeg", "-y", "-i", str(wav_path), str(mp3_path)], check=True)
        wav_path.unlink()

    print(f"Saved stems to {output_dir}")

#function calls
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Separate a song into vocals/drums/bass/other stems")
    parser.add_argument("song_path", help="path to the input song file, e.g. a .mp3")
    parser.add_argument("--model-path", default="separator.pth", help="path to the trained model checkpoint")
    parser.add_argument("--output-dir", default="~/InferenceResults/", help="directory to save the output stems (defaults to <song name>_stems)")
    args = parser.parse_args()

    song_path = pl.Path(args.song_path)
    output_dir = args.output_dir or f"{song_path.stem}_stems"

    separate(
        mixture_path=song_path,
        model_path=args.model_path,
        output_dir=output_dir,
    )
