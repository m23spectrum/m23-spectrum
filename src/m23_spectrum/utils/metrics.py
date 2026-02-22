"""Image quality metrics."""

from typing import Optional, Tuple, Dict
import math

import numpy as np
import torch
import torch.nn.functional as F


def calculate_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    border: int = 0,
    max_val: float = 1.0,
    y_channel: bool = True,
) -> float:
    """
    Calculate PSNR (Peak Signal-to-Noise Ratio).

    Args:
        pred: Predicted image tensor (B, C, H, W) or (C, H, W)
        target: Target image tensor
        border: Border pixels to ignore
        max_val: Maximum pixel value
        y_channel: Whether to compute on Y channel only (for RGB)

    Returns:
        PSNR value in dB
    """
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)

    if border > 0:
        pred = pred[:, :, border:-border, border:-border]
        target = target[:, :, border:-border, border:-border]

    if y_channel and pred.shape[1] == 3:
        # Convert RGB to Y (BT.601)
        pred_y = rgb_to_y(pred)
        target_y = rgb_to_y(target)
        mse = F.mse_loss(pred_y, target_y).item()
    else:
        mse = F.mse_loss(pred, target).item()

    if mse < 1e-10:
        return float("inf")

    psnr = -10 * math.log10(mse / (max_val ** 2))
    return psnr


def calculate_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    border: int = 0,
    window_size: int = 11,
    channel: int = 1,
) -> float:
    """
    Calculate SSIM (Structural Similarity Index).

    Args:
        pred: Predicted image tensor (B, C, H, W)
        target: Target image tensor
        border: Border pixels to ignore
        window_size: Size of Gaussian window
        channel: Number of channels

    Returns:
        SSIM value (0 to 1)
    """
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)

    if border > 0:
        pred = pred[:, :, border:-border, border:-border]
        target = target[:, :, border:-border, border:-border]

    if pred.shape[1] == 3:
        # Use Y channel for SSIM
        pred = rgb_to_y(pred)
        target = rgb_to_y(target)
        channel = 1

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Create Gaussian window
    sigma = 1.5
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    window = g.outer(g)
    window = window / window.sum()
    window = window.view(1, 1, window_size, window_size).expand(channel, 1, -1, -1)
    window = window.to(pred.device, pred.dtype)

    mu1 = F.conv2d(pred, window, stride=1, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(target, window, stride=1, padding=window_size // 2, groups=channel)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred ** 2, window, stride=1, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(target ** 2, window, stride=1, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, stride=1, padding=window_size // 2, groups=channel) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean().item()


def calculate_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    border: int = 0,
) -> Dict[str, float]:
    """
    Calculate multiple image quality metrics.

    Args:
        pred: Predicted image tensor (B, C, H, W)
        target: Target image tensor
        border: Border pixels to ignore

    Returns:
        Dictionary with PSNR, SSIM, and other metrics
    """
    return {
        "psnr": calculate_psnr(pred, target, border=border),
        "ssim": calculate_ssim(pred, target, border=border),
    }


def rgb_to_y(tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert RGB tensor to Y (luminance) channel.

    Uses BT.601 coefficients:
    Y = 0.299 * R + 0.587 * G + 0.114 * B

    Args:
        tensor: RGB tensor (B, 3, H, W)

    Returns:
        Y channel tensor (B, 1, H, W)
    """
    weights = torch.tensor([0.299, 0.587, 0.114], device=tensor.device, dtype=tensor.dtype)
    weights = weights.view(1, 3, 1, 1)
    return (tensor * weights).sum(dim=1, keepdim=True)


def calculate_lpips(
    pred: torch.Tensor,
    target: torch.Tensor,
    net: str = "vgg",
) -> float:
    """
    Calculate LPIPS (Learned Perceptual Image Patch Similarity).

    Requires lpips package: pip install lpips

    Args:
        pred: Predicted image tensor (B, C, H, W) in [-1, 1]
        target: Target image tensor
        net: Network type ('vgg', 'alex', 'squeeze')

    Returns:
        LPIPS value (lower is better)
    """
    try:
        import lpips
        loss_fn = lpips.LPIPS(net=net)
        loss_fn = loss_fn.to(pred.device)

        # LPIPS expects images in [-1, 1]
        pred_scaled = pred * 2 - 1
        target_scaled = target * 2 - 1

        with torch.no_grad():
            lpips_val = loss_fn(pred_scaled, target_scaled).item()

        return lpips_val

    except ImportError:
        raise ImportError(
            "LPIPS requires the lpips package. "
            "Install with: pip install lpips"
        )


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
