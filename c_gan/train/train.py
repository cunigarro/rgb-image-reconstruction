import asyncio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from datetime import datetime
from zoneinfo import ZoneInfo
from torch.cuda.amp import autocast, GradScaler

from c_gan.train.dataset import RGBNIRDatasetS3
from c_gan.train.discrimitator import NIRDiscriminator
from c_gan.train.generator import RGBToNIRGenerator
from utils.list_s3_files import list_s3_files
from utils.metrics import compute_metrics
from telegram import Bot
from config.settings import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

async def notify():
    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    await bot.send_message(
        chat_id=TELEGRAM_CHAT_ID,
        text=f"✅ Entrenamiento SRAWAN finalizado a las {datetime.now(ZoneInfo('America/Bogota')).strftime('%H:%M:%S')} (hora Colombia)."
    )

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Configuración
    lr = 0.0002
    n_epochs = 100
    batch_size = 1  # reducido para ahorro de memoria
    bucket_name = 'dataset-rgb-nir-01'
    rgb_keys = list_s3_files(bucket_name, 'rgb_images/')
    nir_keys = list_s3_files(bucket_name, 'nir_images/')

    # Obtener hora de Colombia
    colombia_tz = ZoneInfo("America/Bogota")
    now = datetime.now(colombia_tz)
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    log_path = f"training_log_cgan_{timestamp}.txt"

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

    dataset = RGBNIRDatasetS3(
        bucket_name=bucket_name,
        rgb_keys=rgb_keys,
        nir_keys=nir_keys,
        transform_rgb=transform_rgb,
        transform_nir=transform_nir
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    generator = RGBToNIRGenerator().to(device)
    discriminator = NIRDiscriminator().to(device)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=30, gamma=0.5)
    scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=30, gamma=0.5)

    adversarial_loss = nn.BCEWithLogitsLoss().to(device)
    consistency_loss = nn.L1Loss().to(device)
    scaler = GradScaler()

    g_losses, d_losses = [], []

    with open(log_path, "w") as log_file:
        log_file.write(f"Inicio de entrenamiento: {now}\n\n")
        start_time = datetime.now(colombia_tz)

        for epoch in range(n_epochs):
            for i, (imgs_rgb, imgs_nir) in enumerate(dataloader):
                imgs_rgb, imgs_nir = imgs_rgb.to(device), imgs_nir.to(device)
                valid = torch.ones(imgs_rgb.size(0), 1, 30, 30, device=device)
                fake = torch.zeros(imgs_rgb.size(0), 1, 30, 30, device=device)

                # Generador
                optimizer_G.zero_grad()
                with autocast():
                    gen_imgs = generator(imgs_rgb)
                    g_loss_adv = adversarial_loss(discriminator(gen_imgs), valid)
                    g_loss_consistency = consistency_loss(gen_imgs, imgs_nir)
                    g_loss = g_loss_adv + g_loss_consistency
                scaler.scale(g_loss).backward()
                scaler.step(optimizer_G)
                scaler.update()

                # Discriminador
                optimizer_D.zero_grad()
                with autocast():
                    real_loss = adversarial_loss(discriminator(imgs_nir), valid)
                    fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
                    d_loss = (real_loss + fake_loss) / 2
                scaler.scale(d_loss).backward()
                scaler.step(optimizer_D)
                scaler.update()

                g_losses.append(g_loss.item())
                d_losses.append(d_loss.item())

                log_line = (f"[Epoch {epoch+1}/{n_epochs}] [Batch {i+1}/{len(dataloader)}] "
                            f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")
                print(log_line)
                log_file.write(log_line + "\n")

            scheduler_G.step()
            scheduler_D.step()
            torch.cuda.empty_cache()

        end_time = datetime.now(colombia_tz)
        log_file.write(f"\nFin del entrenamiento: {end_time}\n")
        log_file.write(f"Duración total: {end_time - start_time}\n\n")

        # Calcular métricas finales con compute_metrics
        with torch.no_grad():
            mrae, rmse, sam = compute_metrics(generator, dataloader, device)
            log_file.write("\nMétricas finales:\n")
            log_file.write(f"MRAE: {mrae:.5f}\n")
            log_file.write(f"RMSE: {rmse:.5f}\n")
            log_file.write(f"SAM:  {sam:.5f}\n")

    torch.save(generator.state_dict(), f'generator_{timestamp}.pth')
    torch.save(discriminator.state_dict(), f'discriminator_{timestamp}.pth')
    asyncio.run(notify())

if __name__ == '__main__':
    main()
