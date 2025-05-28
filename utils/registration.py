import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import os

# Cargar imágenes
rgb_path = "rgb.jpg"
nir_path = "nir.jpg"

# Crear máscaras más estrictas
rgb = cv2.imread(rgb_path)
nir = cv2.imread(nir_path)

# Convertir a escala de grises
rgb_gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
nir_gray = cv2.cvtColor(nir, cv2.COLOR_BGR2GRAY)

# SIFT y alineación con homografía
sift = cv2.SIFT_create()
kps1, desc1 = sift.detectAndCompute(rgb_gray, None)
kps2, desc2 = sift.detectAndCompute(nir_gray, None)
bf = cv2.BFMatcher()
matches = bf.knnMatch(desc1, desc2, k=2)
good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

if len(good_matches) >= 4:
    src_pts = np.float32([kps1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kps2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)
    aligned_rgb = cv2.warpPerspective(rgb, H, (nir.shape[1], nir.shape[0]))
else:
    aligned_rgb = rgb

# Máscaras de píxeles válidos
valid_rgb = (cv2.cvtColor(aligned_rgb, cv2.COLOR_BGR2GRAY) > 10).astype(np.uint8)
valid_nir = (nir_gray > 10).astype(np.uint8)
valid_mask = cv2.bitwise_and(valid_rgb, valid_nir)

# Erosionar máscara para asegurar que no haya bordes negros
kernel = np.ones((5, 5), np.uint8)
valid_mask_eroded = cv2.erode(valid_mask, kernel, iterations=5)

# Encontrar región cuadrada más grande dentro de la máscara erosionada
ys, xs = np.where(valid_mask_eroded == 1)
x_min, x_max = xs.min(), xs.max()
y_min, y_max = ys.min(), ys.max()
side = min(x_max - x_min, y_max - y_min)
x_center = (x_min + x_max) // 2
y_center = (y_min + y_max) // 2
half = side // 2
x_sq = max(x_center - half, 0)
y_sq = max(y_center - half, 0)
x_sq = min(x_sq, aligned_rgb.shape[1] - side)
y_sq = min(y_sq, aligned_rgb.shape[0] - side)

# Recortar imágenes
final_rgb_strict = aligned_rgb[y_sq:y_sq+side, x_sq:x_sq+side]
final_nir_strict = nir[y_sq:y_sq+side, x_sq:x_sq+side]

# Guardar imágenes finales sin bordes
final_rgb_strict_path = "final_rgb_strict.jpg"
final_nir_strict_path = "final_nir_strict.jpg"
cv2.imwrite(final_rgb_strict_path, final_rgb_strict)
cv2.imwrite(final_nir_strict_path, final_nir_strict)

(final_rgb_strict_path, final_nir_strict_path)
