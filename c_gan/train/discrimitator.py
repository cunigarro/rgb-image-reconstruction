import torch
import torch.nn as nn

class NIRDiscriminator(nn.Module):
    def __init__(self):
        super(NIRDiscriminator, self).__init__()
        # A PatchGAN discriminator that outputs a patch map of validity scores
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),  # [B, 64, 128, 128]
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # [B, 128, 64, 64]
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # [B, 256, 32, 32]
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1), # [B, 512, 31, 31]
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)    # [B, 1, 30, 30]
        )

    def forward(self, img):
        return torch.sigmoid(self.model(img))
