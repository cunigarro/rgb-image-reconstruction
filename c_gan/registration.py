import cv2
import numpy as np

def register_images(reference_img, target_img):
    """
    Registra la imagen 'target_img' a la imagen 'reference_img' usando SIFT y homografía.

    :param reference_img: Imagen de referencia (por ejemplo, canal NIR)
    :param target_img: Imagen a registrar (por ejemplo, canal RGB o Red)
    :return: Imagen registrada
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

nir_img = cv2.imread('nir_image.png')
red_img = cv2.imread('red_image.png'

registered_red_img = register_images(nir_img, red_img)

if registered_red_img is not None:
    cv2.imshow('Imagen Registrada', registered_red_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('registered_red_image.png', registered_red_img)
