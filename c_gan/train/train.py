import asyncio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
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
        text=f"✅ Entrenamiento CGAN finalizado a las {datetime.now(ZoneInfo('America/Bogota')).strftime('%H:%M:%S')} (hora Colombia)."
    )

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Configuración
    lr = 0.0002
    n_epochs = 50
    batch_size = 4
    bucket_name = 'dataset-rgb-nir-01'
    img_size = (256, 256)

    # Hora
    colombia_tz = ZoneInfo("America/Bogota")
    timestamp = datetime.now(colombia_tz).strftime("%Y%m%d_%H%M%S")
    log_path = f"training_log_cgan_{timestamp}.txt"

    # Transforms
    transform_rgb = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    transform_nir = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # Dataset completo
    rgb_keys = list_s3_files(bucket_name, 'rgb_images/')
    nir_keys = list_s3_files(bucket_name, 'nir_images/')
    full_dataset = RGBNIRDatasetS3(
        bucket_name=bucket_name,
        rgb_keys=rgb_keys,
        nir_keys=nir_keys,
        transform_rgb=transform_rgb,
        transform_nir=transform_nir
    )
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Dataset test
    test_rgb_keys = list_s3_files(bucket_name, 'rgb_images_test/')
    test_nir_keys = list_s3_files(bucket_name, 'nir_images_test/')
    test_dataset = RGBNIRDatasetS3(
        bucket_name=bucket_name,
        rgb_keys=test_rgb_keys,
        nir_keys=test_nir_keys,
        transform_rgb=transform_rgb,
        transform_nir=transform_nir
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Modelos
    generator = RGBToNIRGenerator().to(device)
    discriminator = NIRDiscriminator().to(device)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=30, gamma=0.5)
    scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=30, gamma=0.5)

    adversarial_loss = nn.BCEWithLogitsLoss().to(device)
    consistency_loss = nn.L1Loss().to(device)
    scaler = GradScaler()

    with open(log_path, "w") as log_file:
        start_time = datetime.now(colombia_tz)
        log_file.write(f"Inicio de entrenamiento: {start_time}\n\n")

        for epoch in range(n_epochs):
            generator.train()
            discriminator.train()
            g_loss_epoch, d_loss_epoch = 0.0, 0.0
            num_batches = len(train_loader)

            for i, (imgs_rgb, imgs_nir) in enumerate(train_loader, start=1):
                imgs_rgb, imgs_nir = imgs_rgb.to(device), imgs_nir.to(device)
                valid = torch.ones((imgs_rgb.size(0), 1, 30, 30), device=device)
                fake = torch.zeros((imgs_rgb.size(0), 1, 30, 30), device=device)

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

                g_loss_epoch += g_loss.item()
                d_loss_epoch += d_loss.item()

                log_line = (f"[Epoch {epoch+1}/{n_epochs}] [Batch {i}/{num_batches}] "
                            f"[D loss: {d_loss.item():.5f}] [G loss: {g_loss.item():.5f}]")
                print(log_line)
                log_file.write(log_line + "\n")

            avg_g_loss = g_loss_epoch / num_batches
            avg_d_loss = d_loss_epoch / num_batches
            scheduler_G.step()
            scheduler_D.step()

            # Validación
            generator.eval()
            val_loss_total = 0.0
            with torch.no_grad():
                for imgs_rgb, imgs_nir in val_loader:
                    imgs_rgb, imgs_nir = imgs_rgb.to(device), imgs_nir.to(device)
                    gen_imgs = generator(imgs_rgb)
                    val_loss_total += consistency_loss(gen_imgs, imgs_nir).item()
            avg_val_loss = val_loss_total / len(val_loader)

            summary_line = (f"=> Epoch {epoch+1} Promedios => "
                            f"G_loss: {avg_g_loss:.5f}, D_loss: {avg_d_loss:.5f}, "
                            f"Val_loss: {avg_val_loss:.5f}")
            print(summary_line)
            log_file.write(summary_line + "\n\n")
            torch.cuda.empty_cache()

        # Final
        end_time = datetime.now(colombia_tz)
        log_file.write(f"\nFin del entrenamiento: {end_time}\n")
        log_file.write(f"Duración total: {end_time - start_time}\n\n")

        # Test set
        generator.eval()
        with torch.no_grad():
            mrae, rmse, mae, psnr = compute_metrics(generator, test_loader, device)
            log_file.write("Métricas en Test Set:\n")
            log_file.write(f"MRAE: {mrae:.5f}\n")
            log_file.write(f"RMSE: {rmse:.5f}\n")
            log_file.write(f"MAE:  {mae:.5f}\n")
            log_file.write(f"PSNR: {psnr:.2f} dB\n")

    torch.save(generator.state_dict(), f'generator_{timestamp}.pth')
    torch.save(discriminator.state_dict(), f'discriminator_{timestamp}.pth')
    asyncio.run(notify())

if __name__ == '__main__':
    main()
