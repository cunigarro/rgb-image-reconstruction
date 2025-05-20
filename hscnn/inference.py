import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from hscnn.model import HSCNN_D_NIR

weights_path = './hscnn_d_inference_20250520_142157.pth'

model = HSCNN_D_NIR()
model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
model.eval()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

image_path = './rgb_image.jpg'
image_rgb = Image.open(image_path).convert('RGB')

image_rgb_tensor = transform(image_rgb).unsqueeze(0)

with torch.no_grad():
    nir_generated_tensor = model(image_rgb_tensor)

nir_generated_tensor = nir_generated_tensor.squeeze(0) * 0.5 + 0.5

nir_generated_image = transforms.ToPILImage()(nir_generated_tensor)

nir_generated_image.show()

nir_generated_image.save('generated_nir_image.jpg')

plt.imshow(nir_generated_image, cmap='gray')
plt.show()
