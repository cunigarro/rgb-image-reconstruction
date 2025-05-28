import cv2
import numpy as np
import os

def procesar_par(rgb_path, nir_path, out_rgb_dir, out_nir_dir, nombre_salida):
    rgb = cv2.imread(rgb_path)
    nir = cv2.imread(nir_path)
    if rgb is None or nir is None:
        print(f"Error cargando: {rgb_path} o {nir_path}")
        return

    rgb_gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    nir_gray = cv2.cvtColor(nir, cv2.COLOR_BGR2GRAY)

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

    valid_rgb = (cv2.cvtColor(aligned_rgb, cv2.COLOR_BGR2GRAY) > 10).astype(np.uint8)
    valid_nir = (nir_gray > 10).astype(np.uint8)
    valid_mask = cv2.bitwise_and(valid_rgb, valid_nir)

    kernel = np.ones((5, 5), np.uint8)
    valid_mask_eroded = cv2.erode(valid_mask, kernel, iterations=5)

    ys, xs = np.where(valid_mask_eroded == 1)
    if len(xs) == 0 or len(ys) == 0:
        print(f"No se encontró región válida en {nombre_salida}")
        return

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

    final_rgb = aligned_rgb[y_sq:y_sq+side, x_sq:x_sq+side]
    final_nir = nir[y_sq:y_sq+side, x_sq:x_sq+side]

    cv2.imwrite(os.path.join(out_rgb_dir, f"{nombre_salida}.jpg"), final_rgb)
    cv2.imwrite(os.path.join(out_nir_dir, f"{nombre_salida}.jpg"), final_nir)

def procesar_masivamente(rgb_dir, nir_dir, out_rgb_dir, out_nir_dir):
    os.makedirs(out_rgb_dir, exist_ok=True)
    os.makedirs(out_nir_dir, exist_ok=True)

    nombres = sorted(os.listdir(rgb_dir))
    for nombre in nombres:
        rgb_path = os.path.join(rgb_dir, nombre)
        nir_path = os.path.join(nir_dir, nombre)  # mismo nombre esperado
        nombre_base = os.path.splitext(nombre)[0]
        procesar_par(rgb_path, nir_path, out_rgb_dir, out_nir_dir, nombre_base)

# USO:
procesar_masivamente("dataset_02_test/rgb_images", "dataset_02_test/nir_images", "dataset_02_registered_test/rgb_images", "dataset_02_registered_test/nir_images")
