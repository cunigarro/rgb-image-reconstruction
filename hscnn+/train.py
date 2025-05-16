import asyncio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import SequoiaDatasetNIR_S3
from model import HSCNN_D_NIR
from metrics import compute_metrics
import boto3
from telegram import Bot

def list_s3_files(bucket_name, prefix):
    s3 = boto3.client('s3')
    keys = []
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        if 'Contents' in page:
            for obj in page['Contents']:
                if obj['Key'].endswith('.jpg'):
                    keys.append(obj['Key'])
    return sorted(keys)

bucket_name = 'dataset-rgb-nir-01'
rgb_keys = list_s3_files(bucket_name, 'rgb_images/')
nir_keys = list_s3_files(bucket_name, 'nir_images/')

dataset = SequoiaDatasetNIR_S3(bucket_name, rgb_keys, nir_keys)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HSCNN_D_NIR().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

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

async def notify():
    bot_token = '7248407303:AAEwITYB3KgY4Eff11Jhgyq5c8tC3bHVDkk'
    chat_id = '6411041440'
    bot = Bot(token=bot_token)
    await bot.send_message(chat_id=chat_id, text="✅ Entrenamiento HSCNN-D finalizado.")

compute_metrics(model, dataloader, device)
asyncio.run(notify())
