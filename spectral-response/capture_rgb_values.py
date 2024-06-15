import cv2
import numpy as np

# Cargar la imagen
image_path = 'color_checker_image.jpg'
image = cv2.imread(image_path)

# Mostrar la imagen para identificar manualmente las regiones de los parches de color
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Definir las coordenadas de los parches de color manualmente (ejemplo)
# Aquí se deben ajustar las coordenadas a la ubicación real de los parches en tu imagen
patches_coordinates = [
    # (x, y, width, height)
    (50, 50, 20, 20),
    (100, 50, 20, 20),
    (150, 50, 20, 20),
    # Agrega más coordenadas según sea necesario
]

# Extraer los valores RGB de los parches de color
captured_rgb = []

for (x, y, width, height) in patches_coordinates:
    patch = image[y:y+height, x:x+width]
    average_color = patch.mean(axis=(0, 1))
    captured_rgb.append(average_color)

# Convertir a un array numpy
captured_rgb = np.array(captured_rgb)

# Guardar los valores RGB capturados
np.savetxt('captured_rgb.txt', captured_rgb, fmt='%d')

print("Captured RGB values:", captured_rgb)
