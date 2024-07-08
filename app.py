from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from scipy.io import loadmat
import h5py
import cv2
import os
import glob
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

class RGBToHyperSpectralDataset(Dataset):
    def __init__(self, rgb_dir, hyperspectral_dir, transform=None):
        self.rgb_files = sorted(glob.glob(os.path.join(rgb_dir, '*.jpg')))
        self.hyperspectral_files = sorted(glob.glob(os.path.join(hyperspectral_dir, '*.mat')))
        self.transform = transform

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        rgb_path = self.rgb_files[idx]
        hyperspectral_path = self.hyperspectral_files[idx]

        # Load RGB image
        rgb_image = cv2.imread(rgb_path)
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        rgb_image = rgb_image.astype(np.float32) / 255.0

        # Load hyperspectral image
        with h5py.File(hyperspectral_path, 'r') as f:
            hyperspectral_image = np.array(f['cube'], dtype=np.float32)
            hyperspectral_image = np.transpose(hyperspectral_image, (2, 1, 0))

        if self.transform:
            rgb_image = self.transform(rgb_image)
            hyperspectral_image = self.transform(hyperspectral_image)

        # plt.imshow(hyperspectral_image[:,15,:].numpy())

        return rgb_image, hyperspectral_image

class RGBToHyperSpectralNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ELU(),
            nn.ConvTranspose2d(64, output_channels, kernel_size=2, stride=2),
            nn.Sigmoid()  # To ensure the output is in the range [0, 1]
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.upsample(x)
        x = F.pad(x, (0, 0, 1, 1))
        return x


# Define the transformations
train_transforms = transforms.Compose([
    transforms.ToTensor()
])

# Create the dataset and dataloader
rgb_dir = './dataset/Train_RGB'
hyperspectral_dir = './dataset/Train_Spectral'
dataset_train = RGBToHyperSpectralDataset(rgb_dir, hyperspectral_dir, transform=train_transforms)
dataloader_train = DataLoader(dataset_train, shuffle=True, batch_size=16)

# Initialize the network, loss function and optimizer
input_channels = 3  # RGB channels
output_channels = 31  # Number of hyperspectral channels (adjust as needed)
net = RGBToHyperSpectralNet(input_channels, output_channels)

criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# Train the model
for epoch in range(10):
    for rgb_images, hyperspectral_images in dataloader_train:
        optimizer.zero_grad()
        outputs = net(rgb_images)
        loss = criterion(outputs, hyperspectral_images)
        loss.backward()
        optimizer.step()
        print(f'Epoch [{epoch+1}/10], Loss: {loss.item():.4f}')

print("Training complete")
