import torch
import torch.nn as nn
import torch.optim as optim
from generator import RGBToNIRGenerator
from discrimitator import NIRDiscriminator

# Hiperparámetros
lr = 0.0002
b1 = 0.5
b2 = 0.999
n_epochs = 100
batch_size = 64

# Preparar los datos: necesitas un dataset que tenga imágenes RGB y su correspondiente en 850nm
# Aquí se deben agregar transformaciones y DataLoader para cargar los datos

# Inicializar generador y discriminador
generator = RGBToNIRGenerator()
discriminator = NIRDiscriminator()

# Optimizadores
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

# Pérdida de entropía cruzada binaria y de consistencia
adversarial_loss = nn.BCELoss()
consistency_loss = nn.L1Loss()

for epoch in range(n_epochs):
    for i, (imgs_rgb, imgs_nir) in enumerate(dataloader):

        batch_size = imgs_rgb.size(0)

        # Etiquetas reales y falsas
        valid = torch.ones(batch_size, 1, requires_grad=False)
        fake = torch.zeros(batch_size, 1, requires_grad=False)

        # Convertir imágenes y etiquetas a tensores
        real_imgs = imgs_nir  # Imágenes en 850nm
        rgb_imgs = imgs_rgb   # Imágenes RGB

        # Entrenamiento del Generador
        optimizer_G.zero_grad()

        # Generar imágenes en 850nm a partir de imágenes RGB
        gen_imgs = generator(rgb_imgs)

        # Pérdida del generador (adversarial y de consistencia)
        g_loss_adv = adversarial_loss(discriminator(gen_imgs), valid)
        g_loss_consistency = consistency_loss(gen_imgs, real_imgs)

        g_loss = g_loss_adv + g_loss_consistency

        g_loss.backward()
        optimizer_G.step()

        # Entrenamiento del Discriminador
        optimizer_D.zero_grad()

        # Pérdida para imágenes reales y falsas
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)

        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        # Imprimir el progreso
        print(
            f"[Epoch {epoch}/{n_epochs}] [Batch {i}/{len(dataloader)}] "
            f"[D loss: {d_loss.item()}] [G loss: {g_loss.item()}]"
        )
