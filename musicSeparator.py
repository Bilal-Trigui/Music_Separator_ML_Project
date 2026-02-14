#importing libraries
import pathlib as pl
import soundfile as sf
import numpy as np

import torch
import torchaudio as ta
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataSet, DataLoader


import librosa
from dataset import dataSet as ds
from sepModel import seperatorModel as sm

#Creates array of audio files that will be used for model training
train_root = pl.Path("/media/bilal/HardDrive2/musdb18hq/train/")
val_root = pl.Path("/media/bilal/HardDrive2/musdb18hq/validation/")
test_root = pl.Path("/media/bilal/HardDrive2/musdb18hq/test/")

#wrapping dataset into a dataloader
training_set = ds(train_root)
train_loader = DataLoader(training_set, batch_size=4, shuffle=True, num_workers=2)

#initalizes device to run the training, model to be trained, and the opitmizer for the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = sm().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

#loop through the training data and lowers loss with every iteration
for epoch in range(10):
    total_loss = 0
    for mix, sttems in train_loader:
        mix, stems = mix.to(device), stems.to(device)

        pred = model(mix)
        loss = F.mse_loss(pred, stems)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), "separator.pth ")