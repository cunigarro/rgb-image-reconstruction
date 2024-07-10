from torch import nn
import torch.nn.functional as F

class RGBToHyperSpectralNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ELU(),
            nn.ConvTranspose2d(64, output_channels, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.upsample(x)
        x = F.pad(x, (0, 0, 1, 1))
        return x
