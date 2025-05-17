import torch
import torch.nn.functional as F
import numpy as np

def compute_metrics(model, dataloader, device):
    model.eval()
    mrae_list = []
    rmse_list = []
    sam_list = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            # Flatten para métricas: (batch, H, W) ➔ (batch, H*W)
            preds_flat = outputs.squeeze(1).view(outputs.size(0), -1)
            targets_flat = targets.squeeze(1).view(targets.size(0), -1)

            # MRAE: Mean Relative Absolute Error
            mrae = torch.abs(preds_flat - targets_flat) / (targets_flat + 1e-6)
            mrae = mrae.mean(dim=1)  # mean por imagen
            mrae_list.append(mrae.cpu().numpy())

            # RMSE: Root Mean Square Error
            rmse = torch.sqrt(F.mse_loss(preds_flat, targets_flat, reduction='none'))
            rmse = rmse.mean(dim=1)  # mean por imagen
            rmse_list.append(rmse.cpu().numpy())

            # SAM: Spectral Angle Mapper (por pixel)
            dot_product = (preds_flat * targets_flat).sum(dim=1)
            norm_pred = torch.norm(preds_flat, p=2, dim=1)
            norm_target = torch.norm(targets_flat, p=2, dim=1)
            sam = torch.acos(torch.clamp(dot_product / (norm_pred * norm_target + 1e-6), -1.0, 1.0))
            sam_list.append(sam.cpu().numpy())

    # Promediar sobre todo el dataset
    mrae_mean = np.concatenate(mrae_list).mean()
    rmse_mean = np.concatenate(rmse_list).mean()
    sam_mean = np.degrees(np.concatenate(sam_list).mean())  # en grados

    print(f"Final Metrics on Dataset:")
    print(f"  ➤ MRAE: {mrae_mean:.5f}")
    print(f"  ➤ RMSE: {rmse_mean:.5f}")
    print(f"  ➤ SAM (deg): {sam_mean:.5f}")

    return mrae_mean, rmse_mean, sam_mean
