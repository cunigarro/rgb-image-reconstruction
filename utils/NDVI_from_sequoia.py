import cv2
import numpy as np

nir_image = cv2.imread('./nir_image.jpg', cv2.IMREAD_GRAYSCALE)
red_image = cv2.imread('./red_image.jpg', cv2.IMREAD_GRAYSCALE)

nir_image = nir_image.astype(np.float32)
red_image = red_image.astype(np.float32)

ndvi = (nir_image - red_image) / (nir_image + red_image + 1e-10)

ndvi_normalized = cv2.normalize(ndvi, None, 0, 1, cv2.NORM_MINMAX)

colored_ndvi = cv2.applyColorMap((ndvi_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)

cv2.imshow('NDVI Image', colored_ndvi)
cv2.imwrite('ndvi_image_from_sequoia.png', colored_ndvi)
cv2.waitKey(0)
cv2.destroyAllWindows()
