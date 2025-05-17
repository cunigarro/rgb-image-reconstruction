import asyncio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime
from utils.dataset import SequoiaDatasetNIR_S3
from utils.list_s3_files import list_s3_files
from utils.metrics import compute_metrics
from telegram import Bot
from config.settings import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from sr_bfwu.model import SRBFWU_Net

# Configuración
bucket_name = 'dataset-rgb-nir-01'
rgb_keys = list_s3_files(bucket_name, 'rgb_images/')
nir_keys = list_s3_files(bucket_name, 'nir_images/')

# Dataset y Dataloader
dataset = SequoiaDatasetNIR_S3(bucket_name, rgb_keys, nir_keys)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Modelo y entrenamiento
model = SRBFWU_Net(in_channels=3, num_bands=1).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Timestamp para archivos
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = f"training_log_srbfwu_{timestamp}.txt"

with open(log_path, "w") as log_file:
    start_time = datetime.now()
    log_file.write(f"Entrenamiento iniciado: {start_time}\n\n")

    # Loop de entrenamiento
    for epoch in range(50):
        running_loss = 0.0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        log_line = f"Epoch {epoch+1}, Loss: {avg_loss:.5f}"
        print(log_line)
        log_file.write(log_line + "\n")

    end_time = datetime.now()
    log_file.write(f"\nEntrenamiento finalizado: {end_time}\n")
    log_file.write(f"Duración total: {end_time - start_time}\n\n")

    # Métricas
    mrae, rmse, sam = compute_metrics(model, dataloader, device)
    log_file.write("Métricas finales:\n")
    log_file.write(f"MRAE: {mrae:.5f}\n")
    log_file.write(f"RMSE: {rmse:.5f}\n")
    log_file.write(f"SAM: {sam:.5f}\n")

# Notificación Telegram
async def notify():
    bot_token = TELEGRAM_BOT_TOKEN
    chat_id = TELEGRAM_CHAT_ID
    bot = Bot(token=bot_token)
    await bot.send_message(chat_id=chat_id, text="✅ Entrenamiento SRBFWU finalizado.")

asyncio.run(notify())
