import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.conv(x)
        return torch.cat([x, out], dim=1)  # Concatenate input and output

class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        layers = []
        channels = in_channels
        for _ in range(num_layers):
            layers.append(DenseLayer(channels, growth_rate))
            channels += growth_rate
        self.block = nn.Sequential(*layers)
        self.out_channels = channels  # Needed for transition layer

    def forward(self, x):
        return self.block(x)

class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.transition = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.transition(x)

class HSCNN_D(nn.Module):
    def __init__(self, in_channels=3, out_channels=31, growth_rate=24, block_config=[4, 4, 4]):
        """
        in_channels: typically 3 (RGB)
        out_channels: number of spectral bands (e.g., 31)
        growth_rate: channels added per layer inside a dense block
        block_config: list indicating number of layers per dense block
        """
        super(HSCNN_D, self).__init__()
        self.growth_rate = growth_rate

        # Initial convolution
        num_channels = 2 * growth_rate
        self.init_conv = nn.Conv2d(in_channels, num_channels, kernel_size=3, padding=1)

        # Dense blocks and transition layers
        self.blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()

        for i, num_layers in enumerate(block_config):
            block = DenseBlock(num_channels, growth_rate, num_layers)
            self.blocks.append(block)
            num_channels = block.out_channels

            if i != len(block_config) - 1:
                transition = TransitionLayer(num_channels, num_channels // 2)
                self.transitions.append(transition)
                num_channels = num_channels // 2

        # Final convolution to reduce to out_channels
        self.final_conv = nn.Conv2d(num_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.init_conv(x)
        for i, block in enumerate(self.blocks):
            out = block(out)
            if i < len(self.transitions):
                out = self.transitions[i](out)
        out = self.final_conv(out)
        return out

model = HSCNN_D(in_channels=3, out_channels=31)  # Reconstrucción hiperespectral
input_rgb = torch.randn(1, 3, 256, 256)  # Imagen RGB simulada
output_hs = model(input_rgb)
print(output_hs.shape)  # [1, 31, 256, 256]
