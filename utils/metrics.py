import torch
import torch.nn.functional as F
import numpy as np

def compute_metrics(model, dataloader, device, nir_threshold=0.05):
    model.eval()
    mrae_list = []
    rmse_list = []
    sam_list = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

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

            # SAM: en imágenes monocromáticas, el SAM no tiene sentido (ángulo entre vectores de dimensión 1)
            # Se puede omitir o usar 0
            sam_list.append(0.0)

    # Promedio sobre todas las imágenes
    mrae_mean = np.mean(mrae_list)
    rmse_mean = np.mean(rmse_list)
    sam_mean = 0.0  # omitido si output tiene solo una banda

    print(f"Final Metrics on Dataset (threshold > {nir_threshold}):")
    print(f"  ➤ MRAE: {mrae_mean:.5f}")
    print(f"  ➤ RMSE: {rmse_mean:.5f}")
    print(f"  ➤ SAM (deg): {sam_mean:.5f}")

    return mrae_mean, rmse_mean, sam_mean
