import torch


def recon_loss(images: torch.Tensor,
               X_coarse: torch.Tensor,
               X_recon: torch.Tensor,
               masks: torch.Tensor) -> torch.Tensor:

    masks_flat = masks.view(masks.shape[0], -1)
    masks_flat_reshaped = masks_flat.mean(1).view(-1, 1, 1, 1)

    return 1.2 * torch.mean(torch.abs(images - X_recon) * masks / masks_flat_reshaped) + \
           1.2 * torch.mean(torch.abs(images - X_recon) * (1 - masks) / (1 - masks_flat_reshaped)) + \
           1.2 * torch.mean(torch.abs(images - X_coarse) * masks / masks_flat_reshaped) + \
           1.2 * torch.mean(torch.abs(images - X_coarse) * (1 - masks) / (1 - masks_flat_reshaped))
