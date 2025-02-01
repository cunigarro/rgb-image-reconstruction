import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

from c_gan.train.dataset import RGBNIRDataset
from c_gan.train.discrimitator import NIRDiscriminator
from c_gan.train.generator import RGBToNIRGenerator

def calculate_msrae(pred, target):
    return torch.mean(torch.abs(pred - target) / (torch.abs(target) + 1e-6))

def calculate_rmse(pred, target):
    return torch.sqrt(torch.mean((pred - target) ** 2))

def main():
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lr = 0.0002
    n_epochs = 100
    batch_size = 64
    dataset_dir = './dataset'
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

    generator = RGBToNIRGenerator().to(device)
    discriminator = NIRDiscriminator().to(device)

    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    # Learning rate schedulers reduce the LR every 30 epochs by half
    scheduler_G = optim.lr_scheduler.StepLR(optimizer_G, step_size=30, gamma=0.5)
    scheduler_D = optim.lr_scheduler.StepLR(optimizer_D, step_size=30, gamma=0.5)

    adversarial_loss = nn.BCELoss().to(device)
    consistency_loss = nn.L1Loss().to(device)

    g_losses = []
    d_losses = []
    msrae_list = []
    rmse_list = []

    for epoch in range(n_epochs):
        for i, (imgs_rgb, imgs_nir) in enumerate(dataloader):
            imgs_rgb = imgs_rgb.to(device)
            imgs_nir = imgs_nir.to(device)
            current_batch = imgs_rgb.size(0)

            # Create target patch labels matching the discriminator output size (e.g., 30x30)
            valid = torch.ones(current_batch, 1, 30, 30, device=device)
            fake = torch.zeros(current_batch, 1, 30, 30, device=device)

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()

            gen_imgs = generator(imgs_rgb)
            g_loss_adv = adversarial_loss(discriminator(gen_imgs), valid)
            g_loss_consistency = consistency_loss(gen_imgs, imgs_nir)
            g_loss = g_loss_adv + g_loss_consistency
            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            real_loss = adversarial_loss(discriminator(imgs_nir), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())

            msrae = calculate_msrae(gen_imgs, imgs_nir)
            rmse = calculate_rmse(gen_imgs, imgs_nir)
            msrae_list.append(msrae.item())
            rmse_list.append(rmse.item())

            print(
                f"[Epoch {epoch+1}/{n_epochs}] [Batch {i+1}/{len(dataloader)}] "
                f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}] "
                f"[MSRAE: {msrae.item():.4f}] [RMSE: {rmse.item():.4f}]"
            )

        # Update learning rate
        scheduler_G.step()
        scheduler_D.step()

    torch.save(generator.state_dict(), 'generator.pth')
    torch.save(discriminator.state_dict(), 'discriminator.pth')

    # Plot training progress
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(g_losses, label="G Loss")
    plt.plot(d_losses, label="D Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    plt.figure(figsize=(10,5))
    plt.title("MSRAE and RMSE During Training")
    plt.plot(msrae_list, label="MSRAE")
    plt.plot(rmse_list, label="RMSE")
    plt.xlabel("Iterations")
    plt.ylabel("Error")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
