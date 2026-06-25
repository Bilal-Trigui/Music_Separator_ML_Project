#importing libraries
import pathlib as pl
import time
import soundfile as sf
import numpy as np

import torch
import torchaudio as ta
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


import librosa
from dataset import dataSet as ds
from sepModel import seperatorModel as sm

#Creates array of audio files that will be used for model training
train_root = pl.Path("/media/bilal/HardDrive2/musdb18hq/train")
val_root = pl.Path("/media/bilal/HardDrive2/musdb18hq/validation")
test_root = pl.Path("/media/bilal/HardDrive2/musdb18hq/test")

#wrapping dataset into a dataloader
training_set = ds(train_root)
train_loader = DataLoader(training_set, batch_size=4, shuffle=True, num_workers=2)

#initalizes device to run the training, model to be trained, and the opitmizer for the model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
model = sm().to(device)
optimizer = optim.Adam(model.parameters(), lr=10**-3.5)

num_epochs = 500
checkpoint_every = 10
checkpoint_dir = pl.Path("checkpoints")
checkpoint_dir.mkdir(exist_ok=True)

#loop through the training data and lowers loss with every iteration
for epoch in range(num_epochs):
    start_time = time.time()
    total_loss = 0
    for mix, stems in train_loader:
        mix, stems = mix.to(device), stems.to(device)

        pred = model(mix)
        stems_cropped = stems[:, :, :pred.shape[2], :pred.shape[3]]
        loss = F.l1_loss(pred, stems_cropped)

        #optimizes the cost of training by recalculating the gradient vector
        #and moving in the direction of the gradient vector
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    elapsed = time.time() - start_time
    print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f} | {elapsed:.1f}s")

    #periodic safety snapshot so a long run isn't all-or-nothing
    if (epoch + 1) % checkpoint_every == 0:
        torch.save(model.state_dict(), checkpoint_dir / f"separator_epoch{epoch+1}.pth")

torch.save(model.state_dict(), "separator.pth")