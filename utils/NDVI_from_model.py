import cv2
import numpy as np

def calculate_ndvi(rgb_path, nir_path):
    rgb_image = cv2.imread(rgb_path)
    nir_image = cv2.imread(nir_path, cv2.IMREAD_GRAYSCALE)

    nir_height, nir_width = nir_image.shape

    rgb_image_resized = cv2.resize(rgb_image, (nir_width, nir_height))

    red_image = rgb_image_resized[:, :, 2]

    nir_image = nir_image.astype(np.float32)
    red_image = red_image.astype(np.float32)

    ndvi = (nir_image - red_image) / (nir_image + red_image + 1e-10)

    ndvi_normalized = cv2.normalize(ndvi, None, 0, 1, cv2.NORM_MINMAX)
    colored_ndvi = cv2.applyColorMap((ndvi_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)

    cv2.imshow('NDVI Image', colored_ndvi)
    cv2.imwrite('ndvi_image_from_model.jpg', colored_ndvi)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

rgb_image_path = './rgb_image.jpg'
nir_image_path = './generated_nir_image.jpg'

calculate_ndvi(rgb_image_path, nir_image_path)
