import torch
import torch.nn.functional as F
import numpy as np
import math

def compute_metrics(model, dataloader, device, nir_threshold=0.05):
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
            preds_flat = outputs.squeeze(1).view(-1)
            targets_flat = targets.squeeze(1).view(-1)

            # Máscara para ignorar regiones con poco valor NIR
            mask = targets_flat > nir_threshold
            if mask.sum() == 0:
                continue  # Evita división por cero si no hay píxeles válidos

            # Aplicar máscara
            preds_valid = preds_flat[mask]
            targets_valid = targets_flat[mask]

            # MRAE
            mrae = torch.abs(preds_valid - targets_valid) / (torch.abs(targets_valid) + 1e-6)
            mrae_list.append(mrae.mean().item())

            # RMSE
            rmse = torch.sqrt(F.mse_loss(preds_valid, targets_valid, reduction='mean'))
            rmse_list.append(rmse.item())

            # MAE: Mean Absolute Error
            mae = torch.abs(preds_valid - targets_valid).mean()
            mae_list.append(mae.item())

            # PSNR: Peak Signal-to-Noise Ratio
            mse = F.mse_loss(preds_valid, targets_valid, reduction='mean')
            psnr = 20 * torch.log10(torch.tensor(1.0)) - 10 * torch.log10(mse + 1e-8)
            psnr_list.append(psnr.item())

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
