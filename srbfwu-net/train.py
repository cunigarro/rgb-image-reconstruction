import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class SRBFWU_Net(nn.Module):
    def __init__(self, in_channels=3, num_bases=8, num_bands=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.num_bases = num_bases
        self.num_bands = num_bands

        # Encoder
        self.encoder = nn.ModuleList()
        for feat in features:
            self.encoder.append(ConvBlock(in_channels, feat))
            in_channels = feat

        # Bottleneck
        self.bottleneck = ConvBlock(features[-1], features[-1] * 2)

        # Decoder
        self.decoder = nn.ModuleList()
        for feat in reversed(features):
            self.decoder.append(nn.ConvTranspose2d(feat * 2, feat, kernel_size=2, stride=2))
            self.decoder.append(ConvBlock(feat * 2, feat))

        # Final layer to predict per-pixel weights
        self.weight_map = nn.Conv2d(features[0], num_bases, kernel_size=1)

        # Learnable basis functions
        self.spectral_bases = nn.Parameter(torch.randn(num_bases, num_bands))

    def forward(self, x):
        skips = []
        for down in self.encoder:
            x = down(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)

        x = self.bottleneck(x)
        skips = skips[::-1]

        for i in range(0, len(self.decoder), 2):
            x = self.decoder[i](x)
            skip = skips[i // 2]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])
            x = torch.cat((skip, x), dim=1)
            x = self.decoder[i + 1](x)

        weights = self.weight_map(x)  # [B, num_bases, H, W]
        weights = weights.permute(0, 2, 3, 1)  # [B, H, W, num_bases]
        output = torch.matmul(weights, self.spectral_bases)  # [B, H, W, num_bands]
        output = output.permute(0, 3, 1, 2)  # [B, num_bands, H, W]
        return output
