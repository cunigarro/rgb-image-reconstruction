from torch import nn, optim, no_grad, save
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

from architecture import RGBToHyperSpectralNet
from train_dataset import RGBToHyperSpectralDataset

# Configuration
dataset_dir = './dataset'
epochs = 13

train_transforms = transforms.Compose([
    transforms.ToTensor()
])

rgb_dir = f'{dataset_dir}/train_rgb'
hyperspectral_dir = f'{dataset_dir}/train_spectral'
dataset_train = RGBToHyperSpectralDataset(rgb_dir, hyperspectral_dir, transform=train_transforms)
dataloader_train = DataLoader(dataset_train, shuffle=True, batch_size=16)

val_rgb_dir = f'{dataset_dir}/valid_rgb'
val_hyperspectral_dir = f'{dataset_dir}/valid_spectral'
dataset_val = RGBToHyperSpectralDataset(val_rgb_dir, val_hyperspectral_dir, transform=train_transforms)
dataloader_val = DataLoader(dataset_val, batch_size=16)

input_channels = 4
output_channels = 31
net = RGBToHyperSpectralNet(input_channels, output_channels)

criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

train_losses = []
val_losses = []

for epoch in range(epochs):
    net.train()
    for rgb_images, hyperspectral_images in dataloader_train:
        optimizer.zero_grad()
        outputs = net(rgb_images)
        loss = criterion(outputs, hyperspectral_images)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    if 'dataset_02' in dataset_dir:
        print(f'Epoch [{epoch+1}/{epochs}], Training Loss: {np.mean(train_losses):.4f}')

    if 'dataset_01' in dataset_dir:
        net.eval()
        with no_grad():
            for rgb_images_val, hyperspectral_images_val in dataloader_val:
                outputs_val = net(rgb_images_val)
                loss_val = criterion(outputs_val, hyperspectral_images_val)
                val_losses.append(loss_val.item())

        print(f'Epoch [{epoch+1}/{epochs}], Training Loss: {np.mean(train_losses):.4f}, Validation Loss: {np.mean(val_losses):.4f}')

model_path = './model_weights.pth'
save(net.state_dict(), model_path)
print(f"Model weights saved to {model_path}")

print("Training complete")

plt.figure(figsize=(epochs, round(epochs / 2)))
plt.plot(train_losses, label='Training Loss')
if 'dataset_01' in dataset_dir:
    plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
if 'dataset_01' in dataset_dir:
    plt.title('Training and Validation Loss')
else:
    plt.title('Training')
plt.legend()
plt.grid(True)
plt.show()
