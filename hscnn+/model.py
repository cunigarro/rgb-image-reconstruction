import torch
import torch.nn as nn

class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        channels = in_channels
        for _ in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.Conv2d(channels, growth_rate, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ))
            channels += growth_rate

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, dim=1))
            features.append(out)
        return torch.cat(features, dim=1)

class HSCNN_D_NIR_RE(nn.Module):
    def __init__(self, growth_rate=16, num_layers=4):
        super(HSCNN_D_NIR_RE, self).__init__()
        self.init_conv = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.dense_block1 = DenseBlock(32, growth_rate, num_layers)
        self.dense_block2 = DenseBlock(32 + growth_rate * num_layers, growth_rate, num_layers)
        self.final_conv = nn.Conv2d(32 + 2 * growth_rate * num_layers, 2, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.init_conv(x)
        x = self.dense_block1(x)
        x = self.dense_block2(x)
        x = self.final_conv(x)
        return x  # Output shape: (batch, 2, H, W)
