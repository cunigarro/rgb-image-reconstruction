
import torch
import torch.nn as nn
import torch.nn.functional as F

# Modelo SRAWAN (Adaptive Weighted Attention Network con prioridad CSS)
# Basado en: "Adaptive Weighted Attention Network with Camera Spectral Sensitivity Prior for Spectral Reconstruction from RGB Images"
# CVPRW 2020, Jiaojiao Li et al.
# Adaptado para predecir una sola banda (NIR) a partir de imagen RGB + sensibilidad espectral de cámara (CSS).

class ConvLayer(nn.Module):
    """
    Capa de convolución 2D con padding reflectante para evitar artefactos en bordes.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(ConvLayer, self).__init__()
        padding = int(dilation * (kernel_size - 1) / 2)
        self.pad = nn.ReflectionPad2d(padding)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                               stride=1, padding=0, dilation=dilation, bias=False)
    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        return x

class AWCA(nn.Module):
    """
    Módulo de Atención de Canal Ponderada Adaptativa (Adaptive Weighted Channel Attention).
    Recalibra las características por canal utilizando una combinación ponderada de la activación espacial.
    """
    def __init__(self, channels, reduction=16):
        super(AWCA, self).__init__()
        # Conv 1x1 para generar un mapa de atención espacial para pooling adaptativo
        self.conv_mask = nn.Conv2d(channels, 1, kernel_size=1, bias=False)
        self.softmax = nn.Softmax(dim=2)
        # MLP para atención de canal: reducción de dimensionalidad y posterior expansión (usando PReLU y Sigmoide)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.PReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, h, w = x.size()
        # Calcula mapa de atención espacial (softmax sobre mapa conv_mask)
        # Conv 1x1 produce máscara de tamaño 1xHxW que luego aplanamos a 1xN
        mask = self.conv_mask(x).view(b, 1, -1)   # (b, 1, H*W)
        mask = self.softmax(mask)                # Distribución de pesos espaciales (b, 1, H*W)
        # Pooling ponderado: combina características espaciales de x usando la máscara
        x_flat = x.view(b, c, -1)                # (b, c, H*W)
        mask_t = mask.permute(0, 2, 1)           # (b, H*W, 1)
        # Producto matricial: cada canal se reduce vía suma ponderada por mask
        y = torch.bmm(x_flat, mask_t).view(b, c) # (b, c)
        # Pasa por la MLP de atención de canal para obtener pesos de atención por canal
        y = self.fc(y).view(b, c, 1, 1)          # (b, c, 1, 1)
        # Escala las características de entrada por los pesos calculados por canal
        return x * y

class NonLocalBlock2D(nn.Module):
    """
    Bloque No-Local 2D (auto-atención espacial) para capturar dependencias de largo alcance en la imagen.
    Implementación estándar de un bloque no-local (Wang et al., 2018).
    """
    def __init__(self, in_channels, reduction=8, use_bn=False):
        super(NonLocalBlock2D, self).__init__()
        inter_channels = max(1, in_channels // reduction)
        # Transformaciones theta, phi, g para calcular atención
        self.theta = nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False)
        self.phi   = nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False)
        self.g     = nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False)
        # Transformación para la salida
        if use_bn:
            self.W = nn.Sequential(
                nn.Conv2d(inter_channels, in_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = nn.Conv2d(inter_channels, in_channels, kernel_size=1, bias=False)
            nn.init.constant_(self.W.weight, 0)
    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        # Proyecciones theta, phi, g
        theta_x = self.theta(x).view(B, -1, H*W)   # (B, inter_channels, N)
        phi_x   = self.phi(x).view(B, -1, H*W)     # (B, inter_channels, N)
        g_x     = self.g(x).view(B, -1, H*W)       # (B, inter_channels, N)
        # Calcula matriz de afinidad (similitud) a = theta^T * phi
        theta_x = theta_x.permute(0, 2, 1)         # (B, N, inter_channels)
        affinity = torch.bmm(theta_x, phi_x)       # (B, N, N)
        affinity = F.softmax(affinity, dim=2)      # normaliza coeficientes
        # Aplica atención: y = g * a^T
        y = torch.bmm(g_x, affinity.permute(0, 2, 1))  # (B, inter_channels, N)
        y = y.view(B, -1, H, W)                    # (B, inter_channels, H, W)
        # Proyección de vuelta a C canales y conexión residual
        out = self.W(y) + x
        return out

class PSNL(nn.Module):
    """
    Módulo No-Local de Segundo Orden a nivel de Parche (Patch-level Second-order Non-Local).
    Divide la característica en 4 cuadrantes y aplica un bloque no-local en cada uno,
    para capturar atención espacial con menor costo computacional.
    """
    def __init__(self, channels):
        super(PSNL, self).__init__()
        self.non_local = NonLocalBlock2D(channels)
    def forward(self, x):
        B, C, H, W = x.shape
        half_H = H // 2
        half_W = W // 2
        # Divide el mapa de características en cuatro partes
        out = torch.zeros_like(x)
        out[:, :, 0:half_H, 0:half_W] = self.non_local(x[:, :, 0:half_H, 0:half_W])   # cuadrante sup-izq
        out[:, :, 0:half_H, half_W:W] = self.non_local(x[:, :, 0:half_H, half_W:W])   # cuadrante sup-der
        out[:, :, half_H:H, 0:half_W] = self.non_local(x[:, :, half_H:H, 0:half_W])   # cuadrante inf-izq
        out[:, :, half_H:H, half_W:W] = self.non_local(x[:, :, half_H:H, half_W:W])   # cuadrante inf-der
        return out

class DRAB(nn.Module):
    """
    Bloque de Atención Residual Dual (Dual Residual Attention Block).
    Contiene:
      - Un módulo residual fundamental (dos conv 3x3 con conexión residual corta).
      - Un módulo de atención de canal (AWCA).
      - Convoluciones en par: una de kernel grande (p. ej. 5x5) seguida de otra de kernel pequeño (p. ej. 3x3).
      - Conexión residual larga (entrada del bloque a salida) y conexiones residuales cortas internas.
    """
    def __init__(self, channels, res_channels=None, k1_size=5, k2_size=3):
        super(DRAB, self).__init__()
        if res_channels is None:
            res_channels = channels
        # Módulo residual fundamental
        self.conv1 = ConvLayer(channels, channels, kernel_size=3)
        self.prelu1 = nn.PReLU()
        self.conv2 = ConvLayer(channels, channels, kernel_size=3)
        self.prelu2 = nn.PReLU()
        # Convolución de kernel grande (T^l_1)
        self.conv_large = ConvLayer(channels, res_channels, kernel_size=k1_size)
        self.prelu_large = nn.PReLU()
        # Atención de canal adaptativa
        self.channel_att = AWCA(res_channels)
        # Convolución de kernel pequeño (T^l_2)
        self.conv_small = ConvLayer(res_channels, channels, kernel_size=k2_size)
        self.prelu_small = nn.PReLU()
    def forward(self, x, res):
        identity = x  # guardar entrada para salto largo
        # Bloque residual fundamental (residual corto interno)
        out = self.conv1(x)
        out = self.prelu1(out)
        out = self.conv2(out)
        out = out + identity  # skip corto dentro del bloque
        out = self.prelu2(out)
        # T^l_1: conv de kernel grande + añade característica residual acumulada
        out = self.conv_large(out)
        out = out + res       # fusión jerárquica con característica residual previa
        out = self.prelu_large(out)
        # Actualiza la característica residual para el siguiente bloque
        res_out = out
        # Atención de canal (AWCA)
        out = self.channel_att(out)
        # T^l_2: conv de kernel pequeño + añade skip largo desde la entrada original
        out = self.conv_small(out)
        out = out + identity  # conexión residual larga del bloque
        out = self.prelu_small(out)
        return out, res_out

class SRAWAN(nn.Module):
    """
    Implementación del modelo SRAWAN en PyTorch.
    Este modelo toma una imagen RGB y la sensibilidad espectral de la cámara (CSS)
    para reconstruir una banda espectral NIR.
    Incluye atención de canal (AWCA) y atención espacial (PSNL) con fusión residual jerárquica entre bloques.
    """
    def __init__(self, in_channels=3, out_channels=1, mid_channels=200, num_blocks=8, use_css=True, use_spatial_att=True):
        super(SRAWAN, self).__init__()
        self.use_css = use_css
        # Si se usa CSS, se espera concatenar 3 mapas constantes (uno por canal RGB) al input
        self.use_spatial_att = use_spatial_att
        effective_in = in_channels + (3 if use_css else 0)
        # Extracción de características superficiales (shallow features)
        self.input_conv = ConvLayer(effective_in, mid_channels, kernel_size=3)
        self.input_prelu = nn.PReLU()
        self.head_conv = ConvLayer(mid_channels, mid_channels, kernel_size=3)
        # Bloques de atención residual dual apilados
        self.blocks = nn.ModuleList([DRAB(mid_channels, res_channels=mid_channels, k1_size=5, k2_size=3)
                                     for _ in range(num_blocks)])
        # Convolución de salida de backbone antes de añadir conexión global
        self.tail_conv = ConvLayer(mid_channels, mid_channels, kernel_size=3)
        # Capa de salida (una conv final) después de PReLU
        self.output_prelu = nn.PReLU()
        self.output_conv = ConvLayer(mid_channels, out_channels, kernel_size=3)
        # Módulo de atención espacial al final (PSNL)
        self.spatial_att = PSNL(out_channels)
    def forward(self, x, css=None):
        # x: tensor de entrada RGB de tamaño (B, 3, H, W)
        # css: tensor de sensibilidad espectral de cámara (CSS) de tamaño (B, 3) o (B, 3, 1, 1)
        if self.use_css:
            assert css is not None, "Se requiere el vector CSS cuando use_css=True"
            B, _, H, W = x.shape
            # Expande el vector CSS a mapas constantes del mismo tamaño espacial que x
            css_map = css.view(B, 3, 1, 1).expand(B, 3, H, W)
            # Concatena la imagen RGB con los mapas de CSS (entrada de tamaño 6 canales)
            x_in = torch.cat([x, css_map], dim=1)
        else:
            x_in = x
        # Extracción de características iniciales (shallow)
        out = self.input_conv(x_in)
        out = self.input_prelu(out)
        out = self.head_conv(out)
        # Guarda la característica para conexión residual global y la inicial para res
        residual_global = out
        res = out
        # Pasa por los bloques DRAB secuencialmente
        for block in self.blocks:
            out, res = block(out, res)
        # Convolución final del backbone y añade conexión residual global (salto directo)
        out = self.tail_conv(out)
        out = out + residual_global
        out = self.output_prelu(out)
        # Proyección a la banda NIR de salida
        out = self.output_conv(out)
        # Atención espacial sobre la salida (PSNL)
        if self.use_spatial_att:
            out = self.spatial_att(out)
        return out
