import torch.nn as nn
import torch

class seperatorModel(nn.Module):
    def __init__(self):
        super().__init__()
        #goes through three levels of encoding in order to give the model the long range context
        #required for understanding the big picture of the spectrogram
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        #takes whatever was encoded and starts upscaling the image back up to have the long range
        #context as a large image that will be combined with the finer details from earlier to
        #create even more accurate training
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.ReLU()
        )
        self.dec4_conv = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),  # 128 + 128 from skip
            nn.ReLU()
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.ReLU()
        )
        self.dec3_conv = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),  # 64 + 64 from skip
            nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.ReLU()
        )
        self.dec2_conv = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),   # 32 + 32 from skip
            nn.ReLU()
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(32, 4, 2, stride=2),
            nn.ReLU()
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        d4 = self.dec4(e4)
        e3 = e3[:, :, :d4.shape[2], :d4.shape[3]]  # crop e3 to match d4
        d4 = torch.cat([d4, e3], dim=1)
        d4 = self.dec4_conv(d4)

        d3 = self.dec3(d4)
        e2 = e2[:, :, :d3.shape[2], :d3.shape[3]]  # crop e2 to match d3
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.dec3_conv(d3)

        d2 = self.dec2(d3)
        e1 = e1[:, :, :d2.shape[2], :d2.shape[3]]  # crop e1 to match d2
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2_conv(d2)

        return self.dec1(d2)