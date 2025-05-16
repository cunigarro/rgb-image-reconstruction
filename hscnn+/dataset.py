import boto3
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
from io import BytesIO

class SequoiaDatasetNIR_S3(Dataset):
    def __init__(self, bucket_name, rgb_keys, nir_keys, img_size=(394, 394), transform=None):
        self.s3 = boto3.client('s3')
        self.bucket = bucket_name
        self.rgb_keys = rgb_keys
        self.nir_keys = nir_keys
        self.img_size = img_size
        self.transform = transform

        assert len(self.rgb_keys) == len(self.nir_keys), "Mismatch in number of RGB and NIR images"

    def __len__(self):
        return len(self.rgb_keys)

    def read_image_from_s3(self, key, grayscale=False):
        response = self.s3.get_object(Bucket=self.bucket, Key=key)
        img_bytes = response['Body'].read()
        img_array = np.frombuffer(img_bytes, np.uint8)
        flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
        img = cv2.imdecode(img_array, flag)
        return img

    def __getitem__(self, idx):
        # RGB image from S3
        rgb_image = self.read_image_from_s3(self.rgb_keys[idx], grayscale=False)
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        rgb_image = cv2.resize(rgb_image, self.img_size) / 255.0
        rgb_image = np.transpose(rgb_image, (2, 0, 1)).astype(np.float32)

        # NIR image from S3 (grayscale)
        nir_image = self.read_image_from_s3(self.nir_keys[idx], grayscale=True)
        nir_image = cv2.resize(nir_image, self.img_size) / 255.0
        nir_image = np.expand_dims(nir_image, axis=0).astype(np.float32)

        return torch.tensor(rgb_image), torch.tensor(nir_image)
