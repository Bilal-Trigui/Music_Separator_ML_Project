# Music Separator ML Project

This is a machine learning project that attempts to separate a song into four different audio stems:

* Vocals
* Drums
* Bass
* Other

The project is mainly a learning exercise for understanding how machine learning can be applied to audio.

## How It Works

The project takes a song and converts the audio into a **spectrogram**, which gives the model a way to represent the frequencies in the audio over time.

A **U-Net-style convolutional neural network** then attempts to predict the individual stems from the song.

The basic process is:

```text
Song
 ↓
Spectrogram
 ↓
Neural Network
 ↓
Vocals / Drums / Bass / Other
 ↓
Audio Files
```

## Dataset

The project uses the **MUSDB18** dataset for training.

Each song contains:

* `mixture.wav`
* `vocals.wav`
* `drums.wav`
* `bass.wav`
* `other.wav`

The model uses the mixture as its input and the individual stems as the targets it is trying to learn.

## Project Files

* `dataset.py` - Loads and prepares the audio data for training.
* `sepModel.py` - Contains the neural network used for separation.
* `musicSeparator.py` - Handles training the model.
* `inference.py` - Uses a trained model to separate a new song.
* `mp4TOwav.py` - Converts MP4 files to WAV.
* `requirements.txt` - Lists the Python packages needed for the project.

## Technologies

* Python
* PyTorch
* Librosa
* NumPy
* SciPy
* SoundFile
* FFmpeg
* CUDA

## Current Status

This project is still **in development**.

The basic training and inference pipeline has been created, but there is still a lot that can be improved, especially the quality of the separated audio.

Some areas I plan to work on include:

* Improving the model architecture
* Improving separation quality
* Improving the training process
* Adding better evaluation methods
* Experimenting with different approaches to audio processing

## Goal

The main goal of this project is to learn more about **machine learning, PyTorch, neural networks, and audio processing** by building a music separation system from the ground up.
