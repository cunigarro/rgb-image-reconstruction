import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

hyperspectral_dir = './dataset/Train_Spectral'
hyperspectral_path = os.path.join(hyperspectral_dir, 'ARAD_1K_0001.mat')

with h5py.File(hyperspectral_path, 'r') as f:
    hyperspectral_image = np.array(f['cube'], dtype=np.float32)

train_transforms = transforms.Compose([
    transforms.ToTensor()
])

hyperspectral_image = train_transforms(hyperspectral_image)

plt.imshow(hyperspectral_image[:,30,:].numpy())

print(hyperspectral_image)
