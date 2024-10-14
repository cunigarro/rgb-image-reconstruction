import cv2
import numpy as np
from PIL import Image, ImageOps

class ProyectividadOpenCV():

    error_reproyeccion = 4

    def __init__(self):
        pass

    def estabilizador_imagen(self, imagen_base, imagen_a_estabilizar, radio=0.75, error_reproyeccion=4.0,
                             coincidencias=False):

        (kpsBase, featuresBase) = self.obtener_puntos_interes(imagen_base)
        (kpsAdicional, featuresAdicional) = self.obtener_puntos_interes(imagen_a_estabilizar)

        M = self.encontrar_coincidencias(imagen_base, imagen_a_estabilizar, kpsBase, kpsAdicional, featuresBase,
                                         featuresAdicional, radio)

        if M is None:
            print("pocas coincidencias")
            return None

        if len(M) > 4:
            (H, status) = self.encontrar_H_RANSAC_Estable(M, kpsBase, kpsAdicional, error_reproyeccion)
            estabilizada = cv2.warpPerspective(imagen_base, H, (imagen_base.shape[1], imagen_base.shape[0]))
            return estabilizada
        print("sin coincidencias")
        return None

    def img_alignment_sequoia(self, img_RGB, stb_NIR, img_RED):

        stb_RGB = self.estabilizador_imagen(img_RGB, stb_NIR)
        stb_RED = self.estabilizador_imagen(img_RED, stb_NIR)

        return stb_RGB, stb_NIR, stb_RED

    def obtener_puntos_interes(self, imagen):

        descriptor = cv2.SIFT_create()
        (kps, features) = descriptor.detectAndCompute(imagen, None)

        return kps, features

    def encontrar_coincidencias(self, img1, img2, kpsA, kpsB, featuresA, featuresB, ratio):

        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []

        for m in rawMatches:
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        return matches

    def encontrar_H_RANSAC_Estable(self, matches, kpsA, kpsB, reprojThresh):

        if len(matches) > 4:
            ptsA = np.float32([kpsA[i].pt for (_, i) in matches])
            ptsB = np.float32([kpsB[i].pt for (i, _) in matches])

            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)

            return (H, status)

        return None

def main():
    width = 700
    height = 500

    example_2 = ProyectividadOpenCV()

    img_RGB_PIL = Image.open('c_gan/pre_process/sequoia_images/img_RGB.JPG')
    img_RGB_PIL = ImageOps.exif_transpose(img_RGB_PIL)
    img_RGB_PIL = np.asarray(img_RGB_PIL)

    img_RED = img_RGB_PIL[:,:,0]

    img_RGB = cv2.imread("c_gan/pre_process/sequoia_images/img_RGB.JPG", 0)
    img_NIR = cv2.imread("c_gan/pre_process/sequoia_images/img_NIR.TIF", 0)

    img_NULL = np.zeros(width*height, dtype=np.uint8).reshape(width,height)

    img_RGB = cv2.resize(img_RGB, (width, height), interpolation=cv2.INTER_LINEAR)
    img_RED = cv2.resize(img_RED, (width, height), interpolation=cv2.INTER_LINEAR)
    img_NIR = cv2.resize(img_NIR, (width, height), interpolation=cv2.INTER_LINEAR)
    img_NULL = cv2.resize(img_NULL, (width, height), interpolation=cv2.INTER_LINEAR)

    merged_fix_bad = cv2.merge((img_RGB, img_NIR, img_NULL))

    stb_RGB, stb_NIR, stb_RED = example_2.img_alignment_sequoia(img_RGB, img_NIR, img_RED)

    mask_NIR = (stb_NIR > 0).astype(np.uint8)
    mask_RED = (stb_RED > 0).astype(np.uint8)
    mask_RGB = (stb_RGB > 0).astype(np.uint8)

    mask_intersection = cv2.bitwise_and(mask_RED, mask_NIR, mask_RGB)

    contours, _ = cv2.findContours(mask_intersection, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    x, y, w, h = cv2.boundingRect(contours[0])

    cropped_RED = stb_RED[y:y+h, x:x+w]
    cropped_NIR = stb_NIR[y:y+h, x:x+w]
    cropped_RGB = stb_RGB[y:y+h, x:x+w]

    cv2.imwrite('cropped_RED.jpg', cropped_RED)
    cv2.imwrite('cropped_NIR.jpg', cropped_NIR)
    cv2.imwrite('cropped_RGB.jpg', cropped_RGB)

    merged_fix_stb = cv2.merge((stb_RGB, stb_NIR, img_NULL))

    print("La primera imagen que se genera simplemente superpone las imágenes sin alinear \n Cerrar la ventana para continuar \n")
    cv2.imshow('frame', merged_fix_bad)
    cv2.waitKey(0)

    print("La siguiente imagen si se encuentra debidamente alineada. Cerrar la ventana para terminar")
    cv2.imshow('frame', merged_fix_stb)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
