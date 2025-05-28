import asyncio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime
from zoneinfo import ZoneInfo
from torch.cuda.amp import autocast, GradScaler

from utils.dataset import SequoiaDatasetNIR_S3
from utils.list_s3_files import list_s3_files
from utils.metrics import compute_metrics
from telegram import Bot
from config.settings import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from sr_unet.model import SRUNet

# Configuración
bucket_name = 'dataset-rgb-nir-01'

# Dataset completo (300 imágenes)
rgb_keys = list_s3_files(bucket_name, 'rgb_images/')
nir_keys = list_s3_files(bucket_name, 'nir_images/')
full_dataset = SequoiaDatasetNIR_S3(bucket_name, rgb_keys, nir_keys, img_size=(256, 256))
print(f"Total imágenes en dataset: {len(full_dataset)}")

# Dividir en train y val
from torch.utils.data import random_split
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# Dataset de prueba (60 imágenes)
test_rgb_keys = list_s3_files(bucket_name, 'rgb_images_test/')
test_nir_keys = list_s3_files(bucket_name, 'nir_images_test/')
test_dataset = SequoiaDatasetNIR_S3(bucket_name, test_rgb_keys, test_nir_keys, img_size=(256, 256))
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Modelo y demás componentes (igual)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SRUNet(in_channels=3, out_channels=1).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scaler = GradScaler()

# Logging
colombia_zone = ZoneInfo("America/Bogota")
timestamp = datetime.now(colombia_zone).strftime("%Y%m%d_%H%M%S")
log_path = f"training_log_srunet_{timestamp}.txt"

with open(log_path, "w") as log_file:
    start_time = datetime.now(colombia_zone)
    log_file.write(f"Entrenamiento iniciado: {start_time}\n\n")

    for epoch in range(50):
        model.train()
        running_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader, start=1):
            if batch_idx > 300: break

            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            log_line = f"[Epoch {epoch+1}/50] [Batch {batch_idx}/300] [Loss: {loss.item():.5f}]"
            print(log_line)
            log_file.write(log_line + "\n")

        avg_train_loss = running_loss / min(300, len(train_loader))
        log_file.write(f"=> Epoch {epoch+1} Promedio Train: {avg_train_loss:.5f}\n")

        # Evaluación en validación
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        log_file.write(f"=> Epoch {epoch+1} Promedio Val: {avg_val_loss:.5f}\n\n")

        torch.cuda.empty_cache()

    end_time = datetime.now(colombia_zone)
    log_file.write(f"\nEntrenamiento finalizado: {end_time}\n")
    log_file.write(f"Duración total: {end_time - start_time}\n\n")

    # Evaluación en test
    model.eval()
    with torch.no_grad():
        mrae, rmse, sam = compute_metrics(model, test_loader, device)
        log_file.write("Métricas en Test Set:\n")
        log_file.write(f"MRAE: {mrae:.5f}\n")
        log_file.write(f"RMSE: {rmse:.5f}\n")
        log_file.write(f"SAM:  {sam:.5f}\n")

    # Guardar modelo
    torch.save(model.state_dict(), f"srunet_d_inference_{timestamp}.pth")

# Telegram notification
async def notify():
    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    await bot.send_message(
        chat_id=TELEGRAM_CHAT_ID,
        text=f"✅ Entrenamiento SRUNET finalizado a las {datetime.now(ZoneInfo('America/Bogota')).strftime('%H:%M:%S')} (hora Colombia)."
    )

asyncio.run(notify())
