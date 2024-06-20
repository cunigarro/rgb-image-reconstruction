import numpy as np
import h5py
import matplotlib.pyplot as plt

# Cargar el archivo .mat usando h5py
file_path = './dataset/Valid_spectral/ARAD_1K_0901.mat'
with h5py.File(file_path, 'r') as file:
    data_cube = np.array(file['cube'])

# data_cube tiene la forma (altura, ancho, bandas)
height, width, bands = data_cube.shape

# Función para mostrar una banda específica
def mostrar_banda(data_cube, banda):
    plt.imshow(data_cube[:, :, banda], cmap='gray')
    plt.title(f'Banda {banda}')
    plt.colorbar()
    plt.show()

# Mostrar algunas bandas específicas
mostrar_banda(data_cube, 0)  # Mostrar la primera banda
mostrar_banda(data_cube, bands // 2)  # Mostrar la banda del medio
mostrar_banda(data_cube, bands - 1)  # Mostrar la última banda

# Generar una imagen RGB aproximada usando 3 bandas específicas (por ejemplo: R: 30, G: 20, B: 10)
def generar_imagen_rgb(data_cube, banda_r, banda_g, banda_b):
    rgb_image = np.zeros((height, width, 3))
    rgb_image[:, :, 0] = data_cube[:, :, banda_r]
    rgb_image[:, :, 1] = data_cube[:, :, banda_g]
    rgb_image[:, :, 2] = data_cube[:, :, banda_b]

    # Normalizar la imagen para que los valores estén entre 0 y 1
    rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())

    plt.imshow(rgb_image)
    plt.title('Imagen RGB aproximada')
    plt.show()

# Generar y mostrar la imagen RGB
generar_imagen_rgb(data_cube, 30, 20, 10)
