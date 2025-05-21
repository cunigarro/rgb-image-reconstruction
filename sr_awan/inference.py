import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import imageio.v3 as iio

from sr_awan.model import SRAWAN

# Rutas
weights_path = './srawan_d_inference_20250520_155825.pth'
image_path = './rgb_image.jpg'

# Cargar modelo
model = SRAWAN(use_css=False, use_spatial_att=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load(weights_path, map_location=device))
model.eval()

# Transformación (sin resize ni normalización)
transform = transforms.ToTensor()

# Cargar imagen RGB
image_rgb = Image.open(image_path).convert('RGB')
image_rgb_tensor = transform(image_rgb).unsqueeze(0)

# Inferencia
with torch.no_grad():
    nir_generated_tensor = model(image_rgb_tensor)

# Convertir de [-1, 1] a [0, 1]
nir_generated_tensor = nir_generated_tensor.squeeze(0) * 0.5 + 0.5

# A NumPy array
nir_array = nir_generated_tensor.squeeze().cpu().numpy()

# Mostrar en plot
plt.imshow(nir_array, cmap='gray')
plt.title("Generated NIR")
plt.axis('off')
plt.show()

# Guardar como imagen visible (jpg) usando imsave
plt.imsave('generated_nir_image.jpg', nir_array, cmap='gray')

# Guardar como TIFF sin pérdida, en float32 (alta precisión)
iio.imwrite('generated_nir_image.tif', nir_array.astype(np.float32))

# (Opcional) Guardar como TIFF en uint8 (imagen visual de 0–255)
# iio.imwrite('generated_nir_image_uint8.tif', (nir_array * 255).astype(np.uint8))
