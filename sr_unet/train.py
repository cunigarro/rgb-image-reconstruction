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
rgb_keys = list_s3_files(bucket_name, 'rgb_images/')
nir_keys = list_s3_files(bucket_name, 'nir_images/')

# Dataset y Dataloader
dataset = SequoiaDatasetNIR_S3(bucket_name, rgb_keys, nir_keys, img_size=(256, 256))
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
print(f"Total imágenes en dataset: {len(dataset)}")

# Dispositivo y modelo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SRUNet(in_channels=3, out_channels=1).to(device)

# Loss, optimizer, scaler
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scaler = GradScaler()

# Hora Colombia
colombia_zone = ZoneInfo("America/Bogota")
colombia_now = datetime.now(colombia_zone)
timestamp = colombia_now.strftime("%Y%m%d_%H%M%S")
log_path = f"training_log_srunet_{timestamp}.txt"

with open(log_path, "w") as log_file:
    start_time = datetime.now(colombia_zone)
    log_file.write(f"Entrenamiento iniciado: {start_time}\n\n")

    for epoch in range(50):
        running_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(dataloader, start=1):
            if batch_idx > 300:
                break

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

        avg_loss = running_loss / min(300, len(dataloader))
        log_file.write(f"=> Epoch {epoch+1} promedio: {avg_loss:.5f}\n\n")
        torch.cuda.empty_cache()

    end_time = datetime.now(colombia_zone)
    log_file.write(f"\nEntrenamiento finalizado: {end_time}\n")
    log_file.write(f"Duración total: {end_time - start_time}\n\n")

    with torch.no_grad():
        mrae, rmse, sam = compute_metrics(model, dataloader, device)
        log_file.write("Métricas finales:\n")
        log_file.write(f"MRAE: {mrae:.5f}\n")
        log_file.write(f"RMSE: {rmse:.5f}\n")
        log_file.write(f"SAM:  {sam:.5f}\n")

    # Guardar el modelo entrenado
    torch.save(model.state_dict(), f"srunet_d_inference_{timestamp}.pth")

# Notificación por Telegram
async def notify():
    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    await bot.send_message(
        chat_id=TELEGRAM_CHAT_ID,
        text=f"✅ Entrenamiento SRUNET finalizado a las {datetime.now(ZoneInfo('America/Bogota')).strftime('%H:%M:%S')} (hora Colombia)."
    )

asyncio.run(notify())
