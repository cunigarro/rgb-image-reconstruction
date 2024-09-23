import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from generator import RGBToNIRGenerator
from discrimitator import NIRDiscriminator
from dataset import RGBNIRDataset
from torch.utils.data import DataLoader
from torchvision import transforms

g_losses = []
d_losses = []

def main():
    lr = 0.0002 # Learning rate
    b1 = 0.5 # Betas (b1, b2) for the Adam optimizer: values used to control momentum in gradient updates
    b2 = 0.999
    n_epochs = 100
    batch_size = 64
    dataset_dir = './dataset_registered'
    rgb_dir = f'{dataset_dir}/rgb_images'
    nir_dir = f'{dataset_dir}/nir_images'

    transform_rgb = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    transform_nir = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    dataset = RGBNIRDataset(rgb_dir, nir_dir, transform_rgb=transform_rgb, transform_nir=transform_nir)
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

            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())

            print(
                f"[Epoch {epoch}/{n_epochs}] [Batch {i}/{len(dataloader)}] "
                f"[D loss: {d_loss.item()}] [G loss: {g_loss.item()}]"
            )

    torch.save(generator.state_dict(), 'generator.pth')
    torch.save(discriminator.state_dict(), 'discriminator.pth')

    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(g_losses, label="G Loss")
    plt.plot(d_losses, label="D Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
