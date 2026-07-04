"""
M23-Spectrum: Professional Super-Resolution Library
====================================================

A production-ready library for image super-resolution powered by
algebraic weight initialization based on the Mathieu group M23.

Features:
- M23-RLFN: Lightweight efficient SR model (~900K params)
- M23-SwinIR: Transformer-based SOTA model
- Advanced losses: LPIPS, GAN, Multi-scale
- Test-Time Augmentation (TTA) for free PSNR boost
- Pre-trained models ready to use

Example:
    >>> from m23_spectrum import M23RLFN, SuperResolver
    >>> model = M23RLFN.from_pretrained("m23-rlfn-x4")
    >>> sr_image = model.upscale("low_res.png")

License: MIT
"""

__version__ = "1.1.0"
__author__ = "M23-Spectrum Team"
__license__ = "MIT"

from .models import M23RLFN, M23SwinIR, create_model
from .losses import CombinedSRLoss, LPIPSLoss, GANLoss, MultiScaleLoss
from .utils import load_image, save_image, calculate_psnr, calculate_ssim
from .inference import SuperResolver, TTAUpscaler

__all__ = [
    # Models
    "M23RLFN",
    "M23SwinIR",
    "create_model",
    # Losses
    "CombinedSRLoss",
    "LPIPSLoss",
    "GANLoss",
    "MultiScaleLoss",
    # Utils
    "load_image",
    "save_image",
    "calculate_psnr",
    "calculate_ssim",
    # Inference
    "SuperResolver",
    "TTAUpscaler",
]
