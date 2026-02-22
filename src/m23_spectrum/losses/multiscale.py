"""
Multi-Scale Loss Functions
==========================

Multi-scale losses for better gradient flow and detail preservation:
- MultiScaleLoss: Combines losses at different scales
- MultiScaleFreqLoss: Frequency loss at multiple scales

These typically add +0.1-0.2 dB PSNR.
"""

from typing import List, Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleLoss(nn.Module):
    """
    Multi-Scale Loss.

    Computes loss at multiple scales for better gradient flow.
    Helps capture both fine details and overall structure.

    Args:
        num_scales: Number of scales to use (default: 4)
        scale_factor: Downsampling factor between scales (default: 0.5)
        base_loss: Base loss function (default: L1)
        weights: Weights for each scale (default: [1, 0.8, 0.6, 0.4])

    Example:
        >>> criterion = MultiScaleLoss(num_scales=4)
        >>> loss = criterion(sr_output, hr_target)
    """

    def __init__(
        self,
        num_scales: int = 4,
        scale_factor: float = 0.5,
        base_loss: str = "l1",
        weights: Optional[List[float]] = None,
    ):
        super().__init__()

        self.num_scales = num_scales
        self.scale_factor = scale_factor

        if base_loss == "l1":
            self.loss_fn = F.l1_loss
        elif base_loss == "l2":
            self.loss_fn = F.mse_loss
        elif base_loss == "charbonnier":
            eps = 1e-3
            self.loss_fn = lambda x, y: torch.mean(torch.sqrt((x - y) ** 2 + eps ** 2))
        else:
            raise ValueError(f"Unknown base loss: {base_loss}")

        if weights is None:
            weights = [1.0 * (scale_factor ** i) for i in range(num_scales)]
        self.weights = weights

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute multi-scale loss.

        Args:
            pred: Predicted image (B, C, H, W)
            target: Target image (B, C, H, W)

        Returns:
            Weighted sum of losses at each scale
        """
        total_loss = 0.0

        for i in range(self.num_scales):
            scale = self.scale_factor ** i

            if scale != 1.0:
                h, w = pred.shape[2], pred.shape[3]
                new_h, new_w = int(h * scale), int(w * scale)

                pred_scaled = F.interpolate(
                    pred, size=(new_h, new_w),
                    mode='bicubic', align_corners=False
                )
                target_scaled = F.interpolate(
                    target, size=(new_h, new_w),
                    mode='bicubic', align_corners=False
                )
            else:
                pred_scaled = pred
                target_scaled = target

            loss = self.loss_fn(pred_scaled, target_scaled)
            total_loss += self.weights[i] * loss

        return total_loss


class MultiScaleFreqLoss(nn.Module):
    """
    Multi-Scale Frequency Loss.

    Computes frequency domain loss at multiple scales for better
    preservation of both low and high frequency details.

    Args:
        num_scales: Number of scales
        scale_factor: Downsampling factor between scales
        freq_weight: Weight for frequency component

    Example:
        >>> criterion = MultiScaleFreqLoss(num_scales=3)
        >>> loss = criterion(sr_output, hr_target)
    """

    def __init__(
        self,
        num_scales: int = 3,
        scale_factor: float = 0.5,
        freq_weight: float = 0.1,
    ):
        super().__init__()

        self.num_scales = num_scales
        self.scale_factor = scale_factor
        self.freq_weight = freq_weight

    def _freq_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute frequency domain loss."""
        pred_fft = torch.fft.rfft2(pred, norm="ortho")
        target_fft = torch.fft.rfft2(target, norm="ortho")

        amp_loss = F.l1_loss(torch.abs(pred_fft), torch.abs(target_fft))

        phase_diff = torch.angle(pred_fft) - torch.angle(target_fft)
        phase_loss = (1 - torch.cos(phase_diff)).mean()

        return amp_loss + 0.1 * phase_loss

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute multi-scale frequency loss.

        Args:
            pred: Predicted image (B, C, H, W)
            target: Target image (B, C, H, W)

        Returns:
            Multi-scale frequency loss
        """
        total_loss = 0.0
        weight_sum = 0.0

        for i in range(self.num_scales):
            scale = self.scale_factor ** i
            weight = 1.0 / (i + 1)

            if scale != 1.0:
                h, w = pred.shape[2], pred.shape[3]
                new_h, new_w = int(h * scale), int(w * scale)

                pred_scaled = F.interpolate(
                    pred, size=(new_h, new_w),
                    mode='bicubic', align_corners=False
                )
                target_scaled = F.interpolate(
                    target, size=(new_h, new_w),
                    mode='bicubic', align_corners=False
                )
            else:
                pred_scaled = pred
                target_scaled = target

            # Spatial loss
            spatial_loss = F.l1_loss(pred_scaled, target_scaled)

            # Frequency loss
            freq_loss = self._freq_loss(pred_scaled, target_scaled)

            total_loss += weight * (spatial_loss + self.freq_weight * freq_loss)
            weight_sum += weight

        return total_loss / weight_sum


class PyramidLoss(nn.Module):
    """
    Laplacian Pyramid Loss.

    Uses Laplacian pyramid decomposition for multi-scale edge-aware loss.

    Args:
        num_levels: Number of pyramid levels
        loss_weight: Weight for the loss
    """

    def __init__(
        self,
        num_levels: int = 4,
        loss_weight: float = 1.0,
    ):
        super().__init__()

        self.num_levels = num_levels
        self.loss_weight = loss_weight

        # Gaussian kernel for pyramid construction
        kernel = self._gaussian_kernel(5, 1.4)
        self.register_buffer("kernel", kernel)

    def _gaussian_kernel(self, size: int, sigma: float) -> torch.Tensor:
        """Create Gaussian kernel."""
        x = torch.arange(size) - size // 2
        kernel_1d = torch.exp(-x ** 2 / (2 * sigma ** 2))
        kernel_2d = kernel_1d.view(-1, 1) @ kernel_1d.view(1, -1)
        return kernel_2d / kernel_2d.sum()

    def _pyramid_down(self, img: torch.Tensor) -> torch.Tensor:
        """Downsample image."""
        B, C, H, W = img.shape
        img = img.view(B * C, 1, H, W)
        img = F.conv2d(img, self.kernel.unsqueeze(0).unsqueeze(0), padding=2)
        img = img[:, :, ::2, ::2]
        return img.view(B, C, H // 2, W // 2)

    def _pyramid_up(self, img: torch.Tensor, target_size: tuple) -> torch.Tensor:
        """Upsample image."""
        B, C, H, W = img.shape
        img_up = F.interpolate(img, size=target_size, mode='bilinear', align_corners=False)
        return img_up

    def _build_pyramid(self, img: torch.Tensor) -> List[torch.Tensor]:
        """Build Laplacian pyramid."""
        pyramid = []
        current = img

        for _ in range(self.num_levels):
            down = self._pyramid_down(current)
            up = self._pyramid_up(down, current.shape[2:])
            laplacian = current - up
            pyramid.append(laplacian)
            current = down

        pyramid.append(current)  # Residual
        return pyramid

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Laplacian pyramid loss.

        Args:
            pred: Predicted image
            target: Target image

        Returns:
            Pyramid loss
        """
        pred_pyramid = self._build_pyramid(pred)
        target_pyramid = self._build_pyramid(target)

        loss = 0.0
        for i, (pred_level, target_level) in enumerate(zip(pred_pyramid, target_pyramid)):
            weight = 1.0 / (i + 1)
            loss += weight * F.l1_loss(pred_level, target_level)

        return self.loss_weight * loss
