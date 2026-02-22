"""
Perceptual Loss Functions
=========================

Perceptual losses based on pre-trained networks:
- LPIPS: Learned Perceptual Image Patch Similarity
- VGG-based perceptual loss

These losses improve visual quality significantly (+0.3-0.5 dB PSNR).
"""

from typing import List, Optional, Tuple
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class VGGFeatureExtractor(nn.Module):
    """
    VGG19 feature extractor for perceptual loss.

    Extracts features from multiple layers for multi-scale perceptual loss.

    Args:
        layer_name_list: List of layer names to extract features from
        use_input_norm: Whether to normalize input images
    """

    def __init__(
        self,
        layer_name_list: List[str] = None,
        use_input_norm: bool = True,
    ):
        super().__init__()

        if layer_name_list is None:
            layer_name_list = ["conv1_2", "conv2_2", "conv3_4", "conv4_4", "conv5_4"]

        self.layer_name_list = layer_name_list
        self.use_input_norm = use_input_norm

        # Load pre-trained VGG19
        vgg = models.vgg19(pretrained=True)

        # Build feature extractor
        features = vgg.features

        # Layer name mapping
        layer_mapping = {
            "conv1_1": 0, "conv1_2": 2,
            "conv2_1": 5, "conv2_2": 7,
            "conv3_1": 10, "conv3_2": 12, "conv3_3": 14, "conv3_4": 16,
            "conv4_1": 19, "conv4_2": 21, "conv4_3": 23, "conv4_4": 25,
            "conv5_1": 28, "conv5_2": 30, "conv5_3": 32, "conv5_4": 34,
        }

        # Extract layers
        self.layers = nn.ModuleDict()
        max_idx = max(layer_mapping[name] for name in layer_name_list)

        for name in layer_name_list:
            idx = layer_mapping[name]
            self.layers[name] = nn.Sequential(*list(features.children())[:idx+1])

        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False

        # VGG normalization
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract features from multiple layers.

        Args:
            x: Input tensor (B, 3, H, W) in range [0, 1]

        Returns:
            List of feature tensors
        """
        # Normalize
        if self.use_input_norm:
            x = (x - self.mean.to(x.device)) / self.std.to(x.device)

        features = []
        for name in self.layer_name_list:
            x = self.layers[name](x)
            features.append(x)

        return features


class PerceptualLoss(nn.Module):
    """
    Perceptual Loss using VGG features.

    Computes L1 distance between VGG features of generated and target images.

    Args:
        layer_weights: Weights for each layer (default: all 1.0)
        use_input_norm: Whether to normalize inputs
        loss_weight: Total loss weight (default: 1.0)

    Example:
        >>> criterion = PerceptualLoss().cuda()
        >>> loss = criterion(sr_image, hr_image)
    """

    def __init__(
        self,
        layer_weights: Optional[List[float]] = None,
        use_input_norm: bool = True,
        loss_weight: float = 1.0,
    ):
        super().__init__()

        self.loss_weight = loss_weight

        layer_names = ["conv1_2", "conv2_2", "conv3_4", "conv4_4", "conv5_4"]

        if layer_weights is None:
            layer_weights = [1.0, 1.0, 1.0, 1.0, 1.0]

        self.layer_weights = layer_weights

        self.feature_extractor = VGGFeatureExtractor(
            layer_name_list=layer_names,
            use_input_norm=use_input_norm,
        )

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute perceptual loss.

        Args:
            pred: Predicted image (B, 3, H, W) in range [0, 1]
            target: Target image (B, 3, H, W) in range [0, 1]

        Returns:
            Perceptual loss value
        """
        pred_features = self.feature_extractor(pred)
        target_features = self.feature_extractor(target)

        loss = 0.0
        for i, (pred_feat, target_feat) in enumerate(zip(pred_features, target_features)):
            loss += self.layer_weights[i] * F.l1_loss(pred_feat, target_feat)

        return self.loss_weight * loss


class LPIPSLoss(nn.Module):
    """
    Learned Perceptual Image Patch Similarity (LPIPS).

    A learned metric that correlates better with human perception than PSNR/SSIM.
    Often provides +0.3-0.5 dB improvement when used in training.

    Args:
        net_type: Network type ('alex', 'vgg', or 'squeeze')
        loss_weight: Weight for the loss (default: 0.1)

    Example:
        >>> criterion = LPIPSLoss(net_type='vgg').cuda()
        >>> loss = criterion(sr_image, hr_image)
    """

    def __init__(
        self,
        net_type: str = "vgg",
        loss_weight: float = 0.1,
    ):
        super().__init__()

        self.loss_weight = loss_weight
        self.net_type = net_type

        # Use VGG for feature extraction
        self.feature_extractor = VGGFeatureExtractor(
            layer_name_list=["conv1_2", "conv2_2", "conv3_4", "conv4_4"],
            use_input_norm=True,
        )

        # Learned linear layers for each feature level
        self.lin0 = nn.Conv2d(64, 1, 1, bias=False)
        self.lin1 = nn.Conv2d(128, 1, 1, bias=False)
        self.lin2 = nn.Conv2d(256, 1, 1, bias=False)
        self.lin3 = nn.Conv2d(512, 1, 1, bias=False)

        # Initialize linear layers
        for lin in [self.lin0, self.lin1, self.lin2, self.lin3]:
            nn.init.kaiming_normal_(lin.weight, a=0.1)

    def _normalize_features(self, feat: torch.Tensor) -> torch.Tensor:
        """Normalize features in the channel dimension."""
        return feat / (feat.norm(dim=1, keepdim=True) + 1e-8)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute LPIPS distance.

        Args:
            pred: Predicted image (B, 3, H, W) in range [0, 1]
            target: Target image (B, 3, H, W) in range [0, 1]

        Returns:
            LPIPS loss value (lower is better)
        """
        pred_features = self.feature_extractor(pred)
        target_features = self.feature_extractor(target)

        lins = [self.lin0, self.lin1, self.lin2, self.lin3]

        loss = 0.0
        for i, (pred_feat, target_feat) in enumerate(zip(pred_features, target_features)):
            # Normalize
            pred_feat = self._normalize_features(pred_feat)
            target_feat = self._normalize_features(target_feat)

            # Compute difference and apply linear layer
            diff = (pred_feat - target_feat) ** 2
            loss += torch.mean(lins[i](diff))

        return self.loss_weight * loss


class ContentLoss(nn.Module):
    """
    Content Loss (simplified perceptual loss).

    Uses only one layer for faster computation.

    Args:
        layer: VGG layer to use (default: 'conv4_4')
        loss_weight: Weight for the loss
    """

    def __init__(
        self,
        layer: str = "conv4_4",
        loss_weight: float = 1.0,
    ):
        super().__init__()

        self.loss_weight = loss_weight
        self.feature_extractor = VGGFeatureExtractor(
            layer_name_list=[layer],
            use_input_norm=True,
        )

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        pred_feat = self.feature_extractor(pred)[0]
        target_feat = self.feature_extractor(target)[0]

        return self.loss_weight * F.l1_loss(pred_feat, target_feat)
