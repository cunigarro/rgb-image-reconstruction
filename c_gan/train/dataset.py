import boto3
from PIL import Image
from torch.utils.data import Dataset
from io import BytesIO

class RGBNIRDatasetS3(Dataset):
    def __init__(self, bucket_name, rgb_keys, nir_keys, transform_rgb=None, transform_nir=None):
        self.bucket_name = bucket_name
        self.rgb_keys = sorted(rgb_keys)
        self.nir_keys = sorted(nir_keys)
        self.transform_rgb = transform_rgb
        self.transform_nir = transform_nir
        self.s3 = boto3.client('s3')

        if len(self.rgb_keys) != len(self.nir_keys):
            raise ValueError("Number of RGB and NIR images do not match!")

    def __len__(self):
        return len(self.rgb_keys)

    def _load_image_from_s3(self, key, mode):
        response = self.s3.get_object(Bucket=self.bucket_name, Key=key)
        img_data = response['Body'].read()
        img = Image.open(BytesIO(img_data)).convert(mode)
        return img

    def __getitem__(self, idx):
        rgb_image = self._load_image_from_s3(self.rgb_keys[idx], mode="RGB")
        nir_image = self._load_image_from_s3(self.nir_keys[idx], mode="L")

        if self.transform_rgb:
            rgb_image = self.transform_rgb(rgb_image)
        if self.transform_nir:
            nir_image = self.transform_nir(nir_image)

        return rgb_image, nir_image
