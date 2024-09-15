import torch
import torch.nn as nn
import torch.optim as optim

from generator import RGBToNIRGenerator
from discrimitator import NIRDiscriminator
from dataset import RGBNIRDataset
from torch.utils.data import DataLoader
from torchvision import transforms

# Hiperparámetros
lr = 0.0002
b1 = 0.5
b2 = 0.999
n_epochs = 100
batch_size = 64
dataset_dir = './dataset'
rgb_dir = f'{dataset_dir}/rgb_images'
nir_dir = f'{dataset_dir}nir_images'

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

dataset = RGBNIRDataset(rgb_dir, nir_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

generator = RGBToNIRGenerator()
discriminator = NIRDiscriminator()

optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

adversarial_loss = nn.BCELoss()
consistency_loss = nn.L1Loss()

for epoch in range(n_epochs):
    for i, (imgs_rgb, imgs_nir) in enumerate(dataloader):

        batch_size = imgs_rgb.size(0)

        valid = torch.ones(batch_size, 1, requires_grad=False)
        fake = torch.zeros(batch_size, 1, requires_grad=False)

        real_imgs = imgs_nir
        rgb_imgs = imgs_rgb

        optimizer_G.zero_grad()

        gen_imgs = generator(rgb_imgs)

        g_loss_adv = adversarial_loss(discriminator(gen_imgs), valid)
        g_loss_consistency = consistency_loss(gen_imgs, real_imgs)

        g_loss = g_loss_adv + g_loss_consistency

        g_loss.backward()
        optimizer_G.step()

        optimizer_D.zero_grad()

        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)

        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            f"[Epoch {epoch}/{n_epochs}] [Batch {i}/{len(dataloader)}] "
            f"[D loss: {d_loss.item()}] [G loss: {g_loss.item()}]"
        )
