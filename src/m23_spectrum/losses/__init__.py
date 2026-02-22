"""Advanced loss functions for super-resolution."""

from .base import CharbonnierLoss, CombinedSRLoss, FrequencyLoss
from .perceptual import LPIPSLoss, PerceptualLoss
from .gan import GANLoss, GANLossType
from .multiscale import MultiScaleLoss, MultiScaleFreqLoss

__all__ = [
    # Base losses
    "CharbonnierLoss",
    "CombinedSRLoss",
    "FrequencyLoss",
    # Perceptual losses
    "LPIPSLoss",
    "PerceptualLoss",
    # GAN losses
    "GANLoss",
    "GANLossType",
    # Multi-scale losses
    "MultiScaleLoss",
    "MultiScaleFreqLoss",
]
