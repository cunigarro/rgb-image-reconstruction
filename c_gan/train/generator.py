import torch
import torch.nn as nn

class RGBToNIRGenerator(nn.Module):
    def __init__(self):
        super(RGBToNIRGenerator, self).__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # 256 -> 128
            nn.ReLU(inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 128 -> 64
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # 64 -> 32
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1), # 32 -> 16
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        # Decoder with skip connections
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1), # 16 -> 32
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1), # (256 from d1 + 256 from enc3) -> 64
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1),  # (128 from d2 + 128 from enc2) -> 128
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2, padding=1),   # (64 from d3 + 64 from enc1) -> 256
            nn.Tanh()
        )

    def forward(self, x):
        e1 = self.enc1(x)   # [B, 64, 128, 128]
        e2 = self.enc2(e1)  # [B, 128, 64, 64]
        e3 = self.enc3(e2)  # [B, 256, 32, 32]
        e4 = self.enc4(e3)  # [B, 512, 16, 16]

        d1 = self.dec1(e4)              # [B, 256, 32, 32]
        d1 = torch.cat([d1, e3], dim=1)   # Concatenate skip connection: [B, 512, 32, 32]

        d2 = self.dec2(d1)              # [B, 128, 64, 64]
        d2 = torch.cat([d2, e2], dim=1)   # [B, 256, 64, 64]

        d3 = self.dec3(d2)              # [B, 64, 128, 128]
        d3 = torch.cat([d3, e1], dim=1)   # [B, 128, 128, 128]

        d4 = self.dec4(d3)              # [B, 1, 256, 256]
        return d4
