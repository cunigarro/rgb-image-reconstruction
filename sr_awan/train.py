import asyncio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.dataset import SequoiaDatasetNIR_S3
from utils.list_s3_files import list_s3_files
from utils.metrics import compute_metrics
from telegram import Bot
from config.settings import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from sr_awan.model import SRAWAN

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
model = SRAWAN(in_channels=3, out_channels=1).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

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

    print(f"Epoch {epoch+1}, Loss: {running_loss / len(dataloader):.5f}")

# Métricas
compute_metrics(model, dataloader, device)

# Notificación Telegram
async def notify():
    bot_token = TELEGRAM_BOT_TOKEN
    chat_id = TELEGRAM_CHAT_ID
    bot = Bot(token=bot_token)
    await bot.send_message(chat_id=chat_id, text="✅ Entrenamiento SRAWAN finalizado.")

asyncio.run(notify())
