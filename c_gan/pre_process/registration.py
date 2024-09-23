import cv2
import numpy as np
import os

def register_images(reference_img, target_img):
    """
    Registra la imagen 'target_img' a la imagen 'reference_img' usando SIFT y homografía.

    :param reference_img: Imagen de referencia (RGB)
    :param target_img: Imagen a registrar (NIR)
    :return: Imagen registrada (NIR)
    """

    ref_gray = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    keypoints_ref, descriptors_ref = sift.detectAndCompute(ref_gray, None)
    keypoints_target, descriptors_target = sift.detectAndCompute(target_gray, None)

    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors_ref, descriptors_target, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    if len(good_matches) > 10:
        ref_pts = np.float32([keypoints_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        target_pts = np.float32([keypoints_target[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        homography_matrix, mask = cv2.findHomography(target_pts, ref_pts, cv2.RANSAC, 5.0)

        height, width, channels = reference_img.shape
        registered_img = cv2.warpPerspective(target_img, homography_matrix, (width, height))

        return registered_img
    else:
        print("No se encontraron suficientes emparejamientos.")
        return None

def register_image_pairs(rgb_images_dir, nir_images_dir, output_dir):
    """
    Registra pares de imágenes RGB y NIR de dos directorios.

    :param rgb_images_dir: Directorio con las imágenes RGB (de referencia)
    :param nir_images_dir: Directorio con las imágenes NIR (a registrar)
    :param output_dir: Directorio donde se guardarán las imágenes NIR registradas
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    rgb_images_files = sorted(os.listdir(rgb_images_dir))
    nir_images_files = sorted(os.listdir(nir_images_dir))

    for rgb_file, nir_file in zip(rgb_images_files, nir_images_files):
        rgb_image_path = os.path.join(rgb_images_dir, rgb_file)
        nir_image_path = os.path.join(nir_images_dir, nir_file)

        rgb_image = cv2.imread(rgb_image_path)
        nir_image = cv2.imread(nir_image_path)

        if rgb_image is None or nir_image is None:
            print(f"No se pudo cargar alguna de las imágenes: {rgb_file}, {nir_file}")
            continue

        registered_nir_img = register_images(rgb_image, nir_image)

        if registered_nir_img is not None:
            output_image_path = os.path.join(output_dir, nir_file)
            cv2.imwrite(output_image_path, registered_nir_img)
            print(f"Imagen NIR registrada guardada en: {output_image_path}")
        else:
            print(f"No se pudo registrar la imagen: {nir_file}")

def main():
    rgb_images_dir = './dataset/rgb_images'
    nir_images_dir = './dataset/nir_images'
    output_dir = './registered_nir_images'

    register_image_pairs(rgb_images_dir, nir_images_dir, output_dir)

if __name__ == "__main__":
    main()
