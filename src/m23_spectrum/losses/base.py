"""Base Loss Functions for Super-Resolution."""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (Robust L1)."""

    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps2 = eps ** 2

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        return torch.mean(torch.sqrt(diff ** 2 + self.eps2))


class FrequencyLoss(nn.Module):
    """Frequency Domain Loss using FFT."""

    def __init__(self, loss_weight: float = 0.1):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_fft = torch.fft.rfft2(pred, norm="ortho")
        target_fft = torch.fft.rfft2(target, norm="ortho")

        amp_loss = F.l1_loss(torch.abs(pred_fft), torch.abs(target_fft))
        phase_loss = (1 - torch.cos(
            torch.angle(pred_fft) - torch.angle(target_fft)
        )).mean()

        return self.loss_weight * (amp_loss + 0.1 * phase_loss)


class CombinedSRLoss(nn.Module):
    """Combined Loss: Charbonnier + Frequency."""

    def __init__(self, freq_weight: float = 0.05, eps: float = 1e-3):
        super().__init__()
        self.charb = CharbonnierLoss(eps=eps)
        self.freq = FrequencyLoss(loss_weight=freq_weight)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.charb(pred, target) + self.freq(pred, target)
