import os

from PIL import Image
from torch.utils.data import Dataset

class RGBNIRDataset(Dataset):
    def __init__(self, rgb_dir, nir_dir, transform=None):
        self.rgb_dir = rgb_dir
        self.nir_dir = nir_dir
        self.transform = transform
        self.rgb_images = sorted(os.listdir(rgb_dir))
        self.nir_images = sorted(os.listdir(nir_dir))

    def __len__(self):
        return len(self.rgb_images)

    def __getitem__(self, idx):
        # Cargar las imágenes RGB y NIR correspondientes
        rgb_path = os.path.join(self.rgb_dir, self.rgb_images[idx])
        nir_path = os.path.join(self.nir_dir, self.nir_images[idx])

        rgb_image = Image.open(rgb_path).convert("RGB")
        nir_image = Image.open(nir_path).convert("L")  # 'L' para cargar la imagen NIR como monocromática

        if self.transform:
            rgb_image = self.transform(rgb_image)
            nir_image = self.transform(nir_image)

        return rgb_image, nir_image
