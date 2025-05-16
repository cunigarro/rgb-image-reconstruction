import boto3
import cv2
import numpy as np
from io import BytesIO

s3 = boto3.client('s3')

def read_image_from_s3(bucket, key):
    response = s3.get_object(Bucket=bucket, Key=key)
    img_bytes = response['Body'].read()
    img_array = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img
