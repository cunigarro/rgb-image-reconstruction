import torch
import scipy.io as sio
from train.architecture import RGBToHyperSpectralNet

input_channels = 3
output_channels = 31
model = RGBToHyperSpectralNet(input_channels, output_channels)

model_path = './model_weights.pth'
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

output_path = './output_hyperspectral_image.mat'
