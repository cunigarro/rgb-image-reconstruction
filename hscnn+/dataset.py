import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np

class SequoiaDatasetNIR(Dataset):
    def __init__(self, rgb_dir, nir_dir, img_size=(394, 394), transform=None):
        self.rgb_paths = sorted([os.path.join(rgb_dir, f) for f in os.listdir(rgb_dir) if f.endswith('.jpg')])
        self.nir_paths = sorted([os.path.join(nir_dir, f) for f in os.listdir(nir_dir) if f.endswith('.jpg')])
        self.img_size = img_size
        self.transform = transform

        assert len(self.rgb_paths) == len(self.nir_paths), "Mismatch in number of RGB and NIR images"

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, idx):
        # RGB image (resize to img_size)
        rgb_image = cv2.imread(self.rgb_paths[idx])
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        rgb_image = cv2.resize(rgb_image, self.img_size) / 255.0
        rgb_image = np.transpose(rgb_image, (2, 0, 1)).astype(np.float32)

        # NIR image (grayscale, resize to img_size)
        nir_image = cv2.imread(self.nir_paths[idx], cv2.IMREAD_GRAYSCALE)
        nir_image = cv2.resize(nir_image, self.img_size) / 255.0
        nir_image = np.expand_dims(nir_image, axis=0).astype(np.float32)  # Shape: (1, H, W)

        return torch.tensor(rgb_image), torch.tensor(nir_image)
