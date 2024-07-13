import numpy as np
import pandas as pd
import cv2
import scipy.io as sio

def load_spectral_response(csv_path):
    df = pd.read_csv(csv_path)
    wavelengths = df.values[:,0].astype(float)
    response_matrix = df.values[:, 1:].astype(float)
    return wavelengths, response_matrix

def rgb_to_hyperspectral(rgb_image, response_matrix):
    h, w, c = rgb_image.shape
    rgb_flat = rgb_image.reshape(-1, 3).astype(float)

    hyperspectral_flat = np.dot(rgb_flat, response_matrix.T)

    num_bands = response_matrix.shape[0]
    hyperspectral_image = hyperspectral_flat.reshape(h, w, num_bands)

    return hyperspectral_image

rgb_image_path = './dataset/Valid_RGB/ARAD_1K_0901.jpg'
rgb_image = cv2.imread(rgb_image_path)

csv_path = './spectral_response/spectral_response_similar_to_xn9/1_xiaomi_12s_rear_ultrawide_camera.csv'
wavelengths, response_matrix = load_spectral_response(csv_path)

hyperspectral_image = rgb_to_hyperspectral(rgb_image, response_matrix)

output_mat_file = './hyperspectral_image.mat'
sio.savemat(output_mat_file, {'hyperspectral_image': hyperspectral_image, 'wavelengths': wavelengths})

print(f"Hyperspectral image saved to {output_mat_file}")
