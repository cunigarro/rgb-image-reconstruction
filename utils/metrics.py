import torch
import torch.nn.functional as F
import numpy as np
import math

def compute_metrics(model, dataloader, device):
    model.eval()
    mrae_list = []
    rmse_list = []
    mae_list = []
    psnr_list = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            # Flatten: (batch, 1, H, W) ➝ (batch, H*W)
            preds_flat = outputs.squeeze(1).view(outputs.size(0), -1)
            targets_flat = targets.squeeze(1).view(targets.size(0), -1)

            # MRAE: Mean Relative Absolute Error
            mrae = torch.abs(preds_flat - targets_flat) / (torch.abs(targets_flat) + 1e-6)
            mrae = mrae.mean(dim=1)
            mrae_list.append(mrae.cpu().numpy())

            # RMSE: Root Mean Square Error
            rmse = torch.sqrt(F.mse_loss(preds_flat, targets_flat, reduction='none'))
            rmse = rmse.mean(dim=1)
            rmse_list.append(rmse.cpu().numpy())

            # MAE: Mean Absolute Error
            mae = torch.abs(preds_flat - targets_flat).mean(dim=1)
            mae_list.append(mae.cpu().numpy())

            # PSNR: Peak Signal-to-Noise Ratio (por imagen)
            mse = F.mse_loss(preds_flat, targets_flat, reduction='none').mean(dim=1)
            psnr = 20 * torch.log10(torch.tensor(1.0)) - 10 * torch.log10(mse + 1e-8)
            psnr_list.append(psnr.cpu().numpy())

    # Calcular promedios
    mrae_mean = np.concatenate(mrae_list).mean()
    rmse_mean = np.concatenate(rmse_list).mean()
    mae_mean = np.concatenate(mae_list).mean()
    psnr_mean = np.concatenate(psnr_list).mean()

    print(f"Final Metrics on Dataset:")
    print(f"  ➤ MRAE: {mrae_mean:.5f}")
    print(f"  ➤ RMSE: {rmse_mean:.5f}")
    print(f"  ➤ MAE:  {mae_mean:.5f}")
    print(f"  ➤ PSNR: {psnr_mean:.2f} dB")

    return mrae_mean, rmse_mean, mae_mean, psnr_mean
