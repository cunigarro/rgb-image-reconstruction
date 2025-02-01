import os
from PIL import Image
from torch.utils.data import Dataset

class RGBNIRDataset(Dataset):
    def __init__(self, rgb_dir, nir_dir, transform_rgb=None, transform_nir=None):
        self.rgb_dir = rgb_dir
        self.nir_dir = nir_dir
        self.transform_rgb = transform_rgb
        self.transform_nir = transform_nir
        self.rgb_images = sorted(os.listdir(rgb_dir))
        self.nir_images = sorted(os.listdir(nir_dir))
        if len(self.rgb_images) != len(self.nir_images):
            raise ValueError("Number of RGB and NIR images do not match!")

    def __len__(self):
        return len(self.rgb_images)

    def __getitem__(self, idx):
        rgb_path = os.path.join(self.rgb_dir, self.rgb_images[idx])
        nir_path = os.path.join(self.nir_dir, self.nir_images[idx])
        rgb_image = Image.open(rgb_path).convert("RGB")
        nir_image = Image.open(nir_path).convert("L")
        if self.transform_rgb:
            rgb_image = self.transform_rgb(rgb_image)
        if self.transform_nir:
            nir_image = self.transform_nir(nir_image)
        return rgb_image, nir_image
