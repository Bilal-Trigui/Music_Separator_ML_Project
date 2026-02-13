import torch.nn as nn

class seperatorModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        #goes through three levels of encoding in order to give the model the long range context
        #required for understanding the big picture of the spectrogram
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, paddings=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, paddings=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.enc3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, paddings=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        #takes wahtever was encoded and starts upscaling the image back up to have the long range
        #context as a large image that will be combined with the finer details from earlier to
        #create even more accurate training
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.ReLU()
        )

        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 2, stride=2),
            nn.ReLU()
        )

        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(16, 4, 2, stride=2),
            nn.ReLU()
        )

    def forward(self, x):


        x = x.unsqueeze(1)

        #encodes spectrogram data
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        #decodes spectrogram data with the fine detail data from earlier
        d3 = self.dec3(e3) + e2
        d2 = self.dec2(d3) + e1
        output = self.dec1(d2)

        return output



