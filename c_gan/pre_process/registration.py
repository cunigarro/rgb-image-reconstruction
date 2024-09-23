import cv2
import numpy as np
import os

def register_images(reference_img, target_img):
    """
    Registra la imagen 'target_img' a la imagen 'reference_img' usando SIFT y homografía.

    Luego recorta ambas imágenes registradas al centro, con un tamaño fijo de 416x416 píxeles.

    :param reference_img: Imagen de referencia (RGB)
    :param target_img: Imagen a registrar (NIR)
    :return: Imagenes registradas y recortadas (RGB y NIR)
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

        ref_cropped, registered_cropped = crop_center(reference_img, registered_img, 416, 416)

        return ref_cropped, registered_cropped
    else:
        print("No se encontraron suficientes emparejamientos.")
        return None, None

def crop_center(ref_img, target_img, crop_height, crop_width):
    """
    Recorta ambas imágenes al centro con el tamaño especificado.

    :param ref_img: Imagen de referencia
    :param target_img: Imagen objetivo registrada
    :param crop_height: Altura deseada del recorte
    :param crop_width: Anchura deseada del recorte
    :return: Imágenes recortadas
    """
    ref_height, ref_width, _ = ref_img.shape
    target_height, target_width = target_img.shape[:2]

    ref_x = (ref_width - crop_width) // 2
    ref_y = (ref_height - crop_height) // 2
    target_x = (target_width - crop_width) // 2
    target_y = (target_height - crop_height) // 2

    ref_cropped = ref_img[ref_y:ref_y + crop_height, ref_x:ref_x + crop_width]
    target_cropped = target_img[target_y:target_y + crop_height, target_x:target_x + crop_width]

    return ref_cropped, target_cropped

def register_image_pairs(rgb_images_dir, nir_images_dir, output_dir):
    """
    Registra pares de imágenes RGB y NIR de dos directorios, luego recorta las imágenes registradas a 416x416 píxeles.

    :param rgb_images_dir: Directorio con las imágenes RGB (de referencia)
    :param nir_images_dir: Directorio con las imágenes NIR (a registrar)
    :param output_dir: Directorio donde se guardarán las imágenes registradas y recortadas
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

        ref_cropped, registered_cropped = register_images(rgb_image, nir_image)

        if ref_cropped is not None and registered_cropped is not None:
            output_ref_image_path = os.path.join(output_dir, f'ref_{rgb_file}')
            output_nir_image_path = os.path.join(output_dir, f'registered_{nir_file}')
            cv2.imwrite(output_ref_image_path, ref_cropped)
            cv2.imwrite(output_nir_image_path, registered_cropped)
            print(f"Imágenes registradas y recortadas guardadas en: {output_ref_image_path}, {output_nir_image_path}")
        else:
            print(f"No se pudo registrar la imagen: {nir_file}")

def main():
    rgb_images_dir = './dataset/rgb_images'
    nir_images_dir = './dataset/nir_images'
    output_dir = './registered_cropped_images'

    register_image_pairs(rgb_images_dir, nir_images_dir, output_dir)

if __name__ == "__main__":
    main()
