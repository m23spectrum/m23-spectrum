"""
GAN Loss for Super-Resolution
==============================

GAN losses for realistic texture generation:
- Standard GAN loss (vanilla)
- LSGAN (Least Squares GAN)
- WGAN (Wasserstein GAN)
- Hinge loss

Using GAN typically adds +0.2-0.4 dB PSNR and significantly
improves perceptual quality.
"""

from typing import Literal
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F


class GANLossType(str, Enum):
    """Available GAN loss types."""
    VANILLA = "vanilla"
    LSGAN = "lsgan"
    WGAN = "wgan"
    HINGE = "hinge"


class GANLoss(nn.Module):
    """
    General GAN Loss.

    Supports multiple GAN variants:
    - vanilla: Standard GAN with BCE loss
    - lsgan: Least Squares GAN (more stable)
    - wgan: Wasserstein GAN (good for training stability)
    - hinge: Hinge loss (used in SAGAN, BigGAN)

    Args:
        gan_type: Type of GAN loss
        real_label_val: Label value for real images
        fake_label_val: Label value for fake images
        loss_weight: Weight for generator loss

    Example:
        >>> criterion = GANLoss(gan_type='lsgan').cuda()
        >>> # For generator
        >>> g_loss = criterion(fake_output, target_is_real=True, is_disc=False)
        >>> # For discriminator
        >>> d_real_loss = criterion(real_output, target_is_real=True, is_disc=True)
        >>> d_fake_loss = criterion(fake_output, target_is_real=False, is_disc=True)
    """

    def __init__(
        self,
        gan_type: GANLossType = GANLossType.LSGAN,
        real_label_val: float = 1.0,
        fake_label_val: float = 0.0,
        loss_weight: float = 0.01,
    ):
        super().__init__()

        self.gan_type = gan_type
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val
        self.loss_weight = loss_weight

        if gan_type == GANLossType.VANILLA:
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_type == GANLossType.LSGAN:
            self.loss = nn.MSELoss()
        elif gan_type in [GANLossType.WGAN, GANLossType.HINGE]:
            self.loss = None
        else:
            raise ValueError(f"Unknown GAN type: {gan_type}")

    def get_target_label(
        self,
        input: torch.Tensor,
        target_is_real: bool,
    ) -> torch.Tensor:
        """Create label tensors with proper size."""
        if target_is_real:
            return torch.full_like(input, self.real_label_val)
        else:
            return torch.full_like(input, self.fake_label_val)

    def forward(
        self,
        input: torch.Tensor,
        target_is_real: bool,
        is_disc: bool = False,
    ) -> torch.Tensor:
        """
        Compute GAN loss.

        Args:
            input: Discriminator output (before sigmoid for vanilla)
            target_is_real: Whether target is real or fake
            is_disc: Whether this is for discriminator training

        Returns:
            GAN loss value
        """
        if self.gan_type == GANLossType.VANILLA:
            target = self.get_target_label(input, target_is_real)
            loss = self.loss(input, target)

        elif self.gan_type == GANLossType.LSGAN:
            target = self.get_target_label(input, target_is_real)
            loss = self.loss(input, target)

        elif self.gan_type == GANLossType.WGAN:
            if target_is_real:
                loss = -input.mean()
            else:
                loss = input.mean()

        elif self.gan_type == GANLossType.HINGE:
            if is_disc:  # For discriminator
                if target_is_real:
                    loss = F.relu(1.0 - input).mean()
                else:
                    loss = F.relu(1.0 + input).mean()
            else:  # For generator
                loss = -input.mean()

        # Apply weight for generator loss
        if not is_disc:
            loss = self.loss_weight * loss

        return loss


class Discriminator(nn.Module):
    """
    PatchGAN Discriminator for Super-Resolution.

    Uses patch-based discrimination for better texture generation.
    Based on ESRGAN architecture.

    Args:
        in_channels: Number of input channels
        num_features: Base number of features
        num_blocks: Number of convolutional blocks
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_features: int = 64,
        num_blocks: int = 8,
    ):
        super().__init__()

        layers = []

        # First layer
        layers.append(nn.Conv2d(in_channels, num_features, 3, 1, 1))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Middle layers with increasing features
        curr_features = num_features
        for i in range(num_blocks):
            next_features = min(curr_features * 2, 512)
            layers.extend([
                nn.Conv2d(curr_features, next_features, 4, 2, 1),
                nn.InstanceNorm2d(next_features),
                nn.LeakyReLU(0.2, inplace=True),
            ])
            curr_features = next_features

        # Output layers
        layers.append(nn.Conv2d(curr_features, curr_features, 3, 1, 1))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.Conv2d(curr_features, 1, 3, 1, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input image (B, C, H, W)

        Returns:
            Discriminator output (patch-wise predictions)
        """
        return self.net(x)


class SRGANDiscriminator(nn.Module):
    """
    SRGAN-style Discriminator.

    Standard discriminator architecture from SRGAN paper.

    Args:
        in_channels: Number of input channels
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()

        def conv_block(in_feat, out_feat, stride=1):
            return nn.Sequential(
                nn.Conv2d(in_feat, out_feat, 3, stride, 1),
                nn.BatchNorm2d(out_feat),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            conv_block(64, 64, 2),
            conv_block(64, 128),
            conv_block(128, 128, 2),
            conv_block(128, 256),
            conv_block(256, 256, 2),
            conv_block(256, 512),
            conv_block(512, 512, 2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        output = self.classifier(features)
        return output
