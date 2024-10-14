import cv2
import numpy as np

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

    def img_alignment_sequoia(self, img_RGB, img_GRE, img_base_NIR, img_RED, img_REG, width, height):

        b_RGB = cv2.resize(img_RGB, (width, height), interpolation=cv2.INTER_LINEAR)
        b_GRE = cv2.resize(img_GRE, (width, height), interpolation=cv2.INTER_LINEAR)
        base_NIR = cv2.resize(img_base_NIR, (width, height), interpolation=cv2.INTER_LINEAR)
        b_RED = cv2.resize(img_RED, (width, height), interpolation=cv2.INTER_LINEAR)
        b_REG = cv2.resize(img_REG, (width, height), interpolation=cv2.INTER_LINEAR)

        stb_GRE = self.estabilizador_imagen(b_GRE, base_NIR)
        stb_RGB = self.estabilizador_imagen(b_RGB, base_NIR)
        stb_RED = self.estabilizador_imagen(b_RED, base_NIR)
        stb_REG = self.estabilizador_imagen(b_REG, base_NIR)

        return stb_RGB, stb_GRE, base_NIR, stb_RED, stb_REG

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
    print("Comenzado proceso con imágenes de prueba propias \n")
    print("Por favor ingrese ancho y alto deseado para las imágenes \npor ejemplo 700,500. De no asignarlos, los valores por defecto serán 700X500 \n \n")
    print("Ancho=")
    width_str = input()
    print("Alto=")
    height_str = input()

    width = int(width_str)
    height = int(height_str)

    example_2 = ProyectividadOpenCV()

    img_RGB = cv2.imread("c_gan/pre_process/sequoia_images/img_RGB.JPG", 0)
    img_GRE = cv2.imread("c_gan/pre_process/sequoia_images/img_GRE.TIF", 0)
    img_NIR = cv2.imread("c_gan/pre_process/sequoia_images/img_NIR.TIF", 0)
    img_RED = cv2.imread("c_gan/pre_process/sequoia_images/img_RED.TIF", 0)
    img_REG = cv2.imread("c_gan/pre_process/sequoia_images/img_REG.TIF", 0)

    merged_fix_bad = cv2.merge((img_GRE, img_RED, img_NIR))
    merged_fix_bad = cv2.resize(merged_fix_bad, (width, height), interpolation=cv2.INTER_LINEAR)

    stb_RGB, stb_GRE, stb_NIR, stb_RED, stb_REG = example_2.img_alignment_sequoia(img_RGB, img_GRE, img_NIR,
                                                                                    img_RED, img_REG, width, height)

    mask_GRE = (stb_GRE > 0).astype(np.uint8)
    mask_REG = (stb_REG > 0).astype(np.uint8)
    mask_NIR = (stb_NIR > 0).astype(np.uint8)

    mask_intersection = cv2.bitwise_and(mask_GRE, mask_REG)
    mask_intersection = cv2.bitwise_and(mask_intersection, mask_NIR)

    contours, _ = cv2.findContours(mask_intersection, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    x, y, w, h = cv2.boundingRect(contours[0])

    cropped_GRE = stb_GRE[y:y+h, x:x+w]
    cropped_REG = stb_REG[y:y+h, x:x+w]
    cropped_NIR = stb_NIR[y:y+h, x:x+w]

    cv2.imwrite('cropped_GRE.jpg', cropped_GRE)
    cv2.imwrite('cropped_REG.jpg', cropped_REG)
    cv2.imwrite('cropped_NIR.jpg', cropped_NIR)

    merged_fix_stb = cv2.merge((stb_GRE, stb_RED, stb_NIR))

    print("La primera imagen que se genera simplemente superpone las imágenes sin alinear \n Cerrar la ventana para continuar \n")
    cv2.imshow('frame', merged_fix_bad)
    cv2.waitKey(0)

    print("La siguiente imagen si se encuentra debidamente alineada. Cerrar la ventana para terminar")
    cv2.imshow('frame', merged_fix_stb)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
