import asyncio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.dataset import SequoiaDatasetNIR_S3
from utils.list_s3_files import list_s3_files
from utils.metrics import compute_metrics
from telegram import Bot
from config.settings import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from sr_unet.model import SRUNet
from datetime import datetime
from torch.cuda.amp import autocast, GradScaler

# Configuración
bucket_name = 'dataset-rgb-nir-01'
rgb_keys = list_s3_files(bucket_name, 'rgb_images/')
nir_keys = list_s3_files(bucket_name, 'nir_images/')

# Dataset y Dataloader (ajustar tamaño si necesitas reducir uso de memoria)
dataset = SequoiaDatasetNIR_S3(bucket_name, rgb_keys, nir_keys, img_size=(256, 256))
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Dispositivo y modelo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SRUNet(in_channels=3, out_channels=1).to(device)

# Loss, optimizer, scaler
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scaler = GradScaler()

# Logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = f"training_log_srunet_{timestamp}.txt"

with open(log_path, "w") as log_file:
    start_time = datetime.now()
    log_file.write(f"Entrenamiento iniciado: {start_time}\n\n")

    for epoch in range(50):
        running_loss = 0.0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        log_line = f"Epoch {epoch+1}, Loss: {avg_loss:.5f}"
        print(log_line)
        log_file.write(log_line + "\n")

        torch.cuda.empty_cache()

    end_time = datetime.now()
    log_file.write(f"\nEntrenamiento finalizado: {end_time}\n")
    log_file.write(f"Duración total: {end_time - start_time}\n\n")

    with torch.no_grad():
        mrae, rmse, sam = compute_metrics(model, dataloader, device)
        log_file.write("Métricas finales:\n")
        log_file.write(f"MRAE: {mrae:.5f}\n")
        log_file.write(f"RMSE: {rmse:.5f}\n")
        log_file.write(f"SAM:  {sam:.5f}\n")

# Notificación por Telegram
async def notify():
    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text="✅ Entrenamiento SRUNET finalizado.")

asyncio.run(notify())
