import cv2
import numpy as np
import os

def register_images(reference_img, target_img):
    ref_gray = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    keypoints_ref, descriptors_ref = sift.detectAndCompute(ref_gray, None)
    keypoints_target, descriptors_target = sift.detectAndCompute(target_gray, None)

    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors_target, descriptors_ref, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    if len(good_matches) > 10:
        target_pts = np.float32([keypoints_target[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        ref_pts = np.float32([keypoints_ref[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        homography_matrix, mask = cv2.findHomography(target_pts, ref_pts, cv2.RANSAC, 5.0)

        height, width, channels = reference_img.shape
        registered_img = cv2.warpPerspective(target_img, homography_matrix, (width, height))

        ref_cropped, registered_cropped = crop_to_overlap_square(reference_img, registered_img)

        return ref_cropped, registered_cropped
    else:
        print("No se encontraron suficientes emparejamientos.")
        return None, None

def crop_to_overlap_square(ref_img, target_img):
    ref_mask = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY) > 0
    target_mask = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY) > 0

    combined_mask = ref_mask & target_mask

    x, y, w, h = cv2.boundingRect(combined_mask.astype(np.uint8))

    side_length = min(w, h)

    center_x = x + w // 2
    center_y = y + h // 2

    half_side = side_length // 2
    x_start = center_x - half_side
    y_start = center_y - half_side

    ref_cropped = ref_img[y_start:y_start+side_length, x_start:x_start+side_length]
    target_cropped = target_img[y_start:y_start+side_length, x_start:x_start+side_length]

    return ref_cropped, target_cropped

def register_image_pairs(nir_images_dir, rgb_images_dir, output_dir):
    nir_output_dir = os.path.join(output_dir, 'nir_images')
    rgb_output_dir = os.path.join(output_dir, 'rgb_images')

    if not os.path.exists(nir_output_dir):
        os.makedirs(nir_output_dir)
    if not os.path.exists(rgb_output_dir):
        os.makedirs(rgb_output_dir)

    nir_images_files = sorted(os.listdir(nir_images_dir))
    rgb_images_files = sorted(os.listdir(rgb_images_dir))

    for nir_file, rgb_file in zip(nir_images_files, rgb_images_files):
        nir_image_path = os.path.join(nir_images_dir, nir_file)
        rgb_image_path = os.path.join(rgb_images_dir, rgb_file)

        nir_image = cv2.imread(nir_image_path)
        rgb_image = cv2.imread(rgb_image_path)

        if nir_image is None or rgb_image is None:
            print(f"No se pudo cargar alguna de las imágenes: {nir_file}, {rgb_file}")
            continue

        ref_cropped, registered_cropped = register_images(nir_image, rgb_image)

        if ref_cropped is not None and registered_cropped is not None:
            output_nir_image_path = os.path.join(nir_output_dir, nir_file)
            output_rgb_image_path = os.path.join(rgb_output_dir, rgb_file)
            cv2.imwrite(output_nir_image_path, ref_cropped)
            cv2.imwrite(output_rgb_image_path, registered_cropped)
            print(f"Imágenes registradas y recortadas guardadas en: {output_nir_image_path}, {output_rgb_image_path}")
        else:
            print(f"No se pudo registrar la imagen: {rgb_file}")

def main():
    nir_images_dir = './dataset/nir_images'
    rgb_images_dir = './dataset/rgb_images'
    output_dir = './dataset_registered'

    register_image_pairs(nir_images_dir, rgb_images_dir, output_dir)

if __name__ == "__main__":
    main()
