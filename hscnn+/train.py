import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import SequoiaDatasetNIR
from model import HSCNN_D_NIR_RE

# Paths de tus datos
rgb_dir = './dataset_registered/rgb_images'
target_dir = './dataset_registered/nir_images'  # NPY files with shape (H, W, 2)

dataset = SequoiaDatasetNIR(rgb_dir, target_dir)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HSCNN_D_NIR_RE().to(device)
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
