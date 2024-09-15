import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from train.generator import RGBToNIRGenerator

# Ruta al archivo de pesos guardados
generator_weights_path = './generator.pth'

generator = RGBToNIRGenerator()
generator.load_state_dict(torch.load(generator_weights_path))
generator.eval()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalización
])

image_path = './rgb_image.jpg'
image_rgb = Image.open(image_path).convert('RGB')

image_rgb_tensor = transform(image_rgb).unsqueeze(0)

with torch.no_grad():
    nir_generated_tensor = generator(image_rgb_tensor)

nir_generated_tensor = nir_generated_tensor.squeeze(0) * 0.5 + 0.5  # Convertir el tensor a rango [0,1]

nir_generated_image = transforms.ToPILImage()(nir_generated_tensor)

nir_generated_image.show()

nir_generated_image.save('generated_nir_image.png')

plt.imshow(nir_generated_image, cmap='gray')
plt.show()
