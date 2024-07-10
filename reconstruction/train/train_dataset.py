from torch.utils.data import Dataset
import h5py
import cv2
import os
import glob
import numpy as np

class RGBToHyperSpectralDataset(Dataset):
    def __init__(self, rgb_dir, hyperspectral_dir, transform=None):
        self.rgb_files = sorted(glob.glob(os.path.join(rgb_dir, '*.jpg')))
        self.rgb_files = [f for f in self.rgb_files if not f.endswith('_nir.jpg')]
        self.nir_files = sorted(glob.glob(os.path.join(rgb_dir, '*_nir.jpg')))
        self.hyperspectral_files = sorted(glob.glob(os.path.join(hyperspectral_dir, '*.mat')))
        self.transform = transform

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        rgb_path = self.rgb_files[idx]
        nir_path = self.nir_files[idx]
        hyperspectral_path = self.hyperspectral_files[idx]

        rgb_image = cv2.imread(rgb_path)
        nir_image = cv2.imread(nir_path)
        rgb_image = np.dstack((rgb_image, nir_image[:, :, 0]))
        rgb_image = rgb_image.astype(np.float32) / 255.0

        with h5py.File(hyperspectral_path, 'r') as f:
            hyperspectral_image = np.array(f['cube'], dtype=np.float32)
            hyperspectral_image = np.transpose(hyperspectral_image, (2, 1, 0))

        if self.transform:
            rgb_image = self.transform(rgb_image)
            hyperspectral_image = self.transform(hyperspectral_image)

        return rgb_image, hyperspectral_image
