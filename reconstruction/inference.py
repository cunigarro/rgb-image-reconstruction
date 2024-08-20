import torch
import scipy.io as sio
from PIL import Image
import numpy as np
from train.architecture import RGBToHyperSpectralNet

input_channels = 4
output_channels = 31
model = RGBToHyperSpectralNet(input_channels, output_channels)

model_path = './model_weights_03.pth'
model.load_state_dict(torch.load(model_path))

model.eval()

def infer_and_save(rgb_image, model, output_path):
    model.eval()

    rgb_image = torch.tensor(rgb_image).unsqueeze(0).float()

    with torch.no_grad():
        hyperspectral_image = model(rgb_image)

    hyperspectral_image = hyperspectral_image.squeeze(0).numpy()

    sio.savemat(output_path, {'hyperspectral_image': hyperspectral_image})
    print(f'Hyperspectral image saved to {output_path}')

def load_image_from_file(file_path):
    img = Image.open(file_path)
    img = img.convert('RGB')
    img = img.resize((256, 256))
    img = np.array(img)
    img = np.transpose(img, (2, 0, 1))
    return img

file_path = './input_image.jpeg'
rgb_image = load_image_from_file(file_path)

output_path = './output_hyperspectral_image.mat'
infer_and_save(rgb_image, model, output_path)
