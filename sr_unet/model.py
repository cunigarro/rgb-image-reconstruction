import torch
import torch.nn as nn

class SRUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(SRUNet, self).__init__()
        # Encoder: convoluciones 3x3 sin padding (reducción de dimensiones)
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0)
        # Decoder: convoluciones transpuestas 3x3 sin padding (aumentan dimensiones)
        self.tconv1 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=0)
        self.tconv2 = nn.ConvTranspose2d(256, 64, kernel_size=3, stride=1, padding=0)
        self.tconv3 = nn.ConvTranspose2d(192, 32, kernel_size=3, stride=1, padding=0)
        self.tconv4 = nn.ConvTranspose2d(96, 32, kernel_size=3, stride=1, padding=0)
        self.tconv5 = nn.ConvTranspose2d(32, out_channels, kernel_size=3, stride=1, padding=0)
        # Activación ReLU (inplace) para usar después de cada conv/convT
        self.relu = nn.ReLU(inplace=True)
        # Activación de salida (según paper, e.g. ReLU para restringir valores válidos de NIR)
        self.output_act = nn.ReLU(inplace=True)

    def forward(self, x):
        # Encoder (downsamping con convs 3x3 sin padding + ReLU)
        d1 = self.relu(self.conv1(x))   # 3->32 canales, dim reduce a (H-2)x(W-2)
        d2 = self.relu(self.conv2(d1))  # 32->64 canales, reduce 2px más
        d3 = self.relu(self.conv3(d2))  # 64->128 canales
        d4 = self.relu(self.conv4(d3))  # 128->128 canales (dim reducida cada vez)
        d5 = self.relu(self.conv5(d4))  # 128->128 canales (bottleneck)
        # Decoder (upsampling con convT 3x3 sin padding + ReLU, concatenando skips)
        u1 = self.relu(self.tconv1(d5))         # 128->128 canales, dim +2px
        u1 = torch.cat([u1, d4], dim=1)         # skip connection con d4 (concat canal): 128+128=256 canales
        u2 = self.relu(self.tconv2(u1))         # 256->64 canales, dim +2px
        u2 = torch.cat([u2, d3], dim=1)         # concat con d3: 64+128=192 canales
        u3 = self.relu(self.tconv3(u2))         # 192->32 canales, dim +2px
        u3 = torch.cat([u3, d2], dim=1)         # concat con d2: 32+64=96 canales
        u4 = self.relu(self.tconv4(u3))         # 96->32 canales, dim +2px
        # *Skip* de nivel superior omitido: no concatenamos d1
        out = self.tconv5(u4)                   # 32->1 canal (salida NIR), dim +2px recuperando tamaño original
        out = self.output_act(out)              # activación de salida (p.ej. ReLU para valores >= 0)
        return out
