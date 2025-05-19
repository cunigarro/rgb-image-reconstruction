import torch
import torch.nn as nn
import torch.nn.functional as F

class SRBFWU_Net(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, n_basis=10, base_filters=64):
        super(SRBFWU_Net, self).__init__()
        # Bloques de convolución (dos conv 3x3 + ReLU) para el encoder
        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
        # Encoder: cinco niveles (entrada + 4 downsamplings)
        self.enc1 = conv_block(in_channels, base_filters)        # Nivel 1
        self.enc2 = conv_block(base_filters, base_filters*2)     # Nivel 2 (1/2 resolución)
        self.enc3 = conv_block(base_filters*2, base_filters*4)   # Nivel 3 (1/4 resolución)
        self.enc4 = conv_block(base_filters*4, base_filters*8)   # Nivel 4 (1/8 resolución)
        self.enc5 = conv_block(base_filters*8, base_filters*16)  # Nivel 5 (1/16 res., capa más profunda)
        # Bloques de convolución para el decoder (con skip connections)
        def conv_block_up(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
        self.dec4 = conv_block_up(base_filters*16 + base_filters*8, base_filters*8)   # Fusiona nivel 5 con skip de nivel 4
        self.dec3 = conv_block_up(base_filters*8  + base_filters*4, base_filters*4)   # Fusiona resultado nivel 4 con skip nivel 3
        self.dec2 = conv_block_up(base_filters*4  + base_filters*2, base_filters*2)   # Fusiona resultado nivel 3 con skip nivel 2
        self.dec1 = conv_block_up(base_filters*2  + base_filters,   base_filters)     # Fusiona resultado nivel 2 con skip nivel 1
        # Capa final de la U-Net: predice n mapas de pesos (n_basis)
        self.final_conv = nn.Conv2d(base_filters, n_basis, kernel_size=1)
        # Parámetro aprendible de funciones base espectrales: matriz n_basis x out_channels
        # (Para salida NIR, out_channels=1, así que basis tendrá shape [10 x 1])
        self.basis = nn.Parameter(torch.randn(n_basis, out_channels))

    def forward(self, x):
        # --- Encoder (contracción) ---
        e1 = self.enc1(x)
        # Downsampling lineal (bilinear) en lugar de MaxPool
        d1 = F.interpolate(e1, scale_factor=0.5, mode='bilinear', align_corners=False)
        e2 = self.enc2(d1)
        d2 = F.interpolate(e2, scale_factor=0.5, mode='bilinear', align_corners=False)
        e3 = self.enc3(d2)
        d3 = F.interpolate(e3, scale_factor=0.5, mode='bilinear', align_corners=False)
        e4 = self.enc4(d3)
        d4 = F.interpolate(e4, scale_factor=0.5, mode='bilinear', align_corners=False)
        e5 = self.enc5(d4)
        # --- Decoder (expansión) con skip connections ---
        # Upsample lineal y concatenación con feature map del encoder correspondiente (skip)
        u4 = F.interpolate(e5, size=e4.shape[2:], mode='bilinear', align_corners=False)
        u4 = torch.cat([u4, e4], dim=1)        # concat sin recortar
        u4 = self.dec4(u4)
        u3 = F.interpolate(u4, size=e3.shape[2:], mode='bilinear', align_corners=False)
        u3 = torch.cat([u3, e3], dim=1)
        u3 = self.dec3(u3)
        u2 = F.interpolate(u3, size=e2.shape[2:], mode='bilinear', align_corners=False)
        u2 = torch.cat([u2, e2], dim=1)
        u2 = self.dec2(u2)
        u1 = F.interpolate(u2, size=e1.shape[2:], mode='bilinear', align_corners=False)
        u1 = torch.cat([u1, e1], dim=1)
        u1 = self.dec1(u1)
        # Mapa de pesos predicho (10 canales de peso por píxel)
        weight_map = self.final_conv(u1)  # shape: [B, n_basis, H, W]
        # --- Reconstrucción espectral usando funciones base ---
        # Reorganizar weight_map a [B, H*W, n_basis] para multiplicar por la matriz de bases
        B, n_basis, H, W = weight_map.shape
        weight_map_flat = weight_map.permute(0, 2, 3, 1).reshape(B, H*W, n_basis)
        # Combinar pesos con las funciones base aprendidas (matmul): resultado [B, H*W, out_channels]
        spectral_flat = weight_map_flat @ self.basis  # self.basis es [n_basis, out_channels]
        # Volver a forma de imagen [B, out_channels, H, W]
        spectral_out = spectral_flat.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return spectral_out
