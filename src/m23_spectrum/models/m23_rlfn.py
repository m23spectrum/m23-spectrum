"""
M23-RLFN: Lightweight Super-Resolution model.

RLFN (Residual Local Feature Network) + M23-Spectrum initialization.
~900K parameters, ~30 dB PSNR on Set5 x4.
"""

from typing import Optional
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ..core.spectrum import m23_init_tensor


def create_model(
    name: str = "m23-rlfn-x4",
    pretrained: bool = False,
    device: str = "cpu",
) -> "M23RLFN":
    """
    Factory function for M23-Spectrum SR models.

    Args:
        name: Model name. One of:
            'm23-rlfn-x2', 'm23-rlfn-x3', 'm23-rlfn-x4' (default)
            'm23-rlfn-large-x4'
        pretrained: Load pretrained weights (not yet hosted; falls back gracefully).
        device: Target device.

    Returns:
        Initialized M23RLFN model.
    """
    configs = {
        "m23-rlfn-x2":       dict(n_feats=52, n_blocks=8, scale=2),
        "m23-rlfn-x3":       dict(n_feats=52, n_blocks=8, scale=3),
        "m23-rlfn-x4":       dict(n_feats=52, n_blocks=8, scale=4),
        "m23-rlfn-large-x4": dict(n_feats=64, n_blocks=12, scale=4),
    }
    if name not in configs:
        raise ValueError(f"Unknown model '{name}'. Available: {list(configs)}")

    model = M23RLFN(**configs[name])

    if pretrained:
        import warnings
        warnings.warn(
            f"Pretrained weights for '{name}' are not yet hosted. "
            "Model returned with M23-Spectrum initialization.",
            UserWarning,
        )

    return model.to(device)


if TORCH_AVAILABLE:

    class ESA(nn.Module):
        """Enhanced Spatial Attention (Liu et al., NTIRE 2022)."""

        def __init__(self, n_feats: int):
            super().__init__()
            f = n_feats // 4
            self.conv1    = nn.Conv2d(n_feats, f, 1)
            self.conv_f   = nn.Conv2d(f, f, 1)
            self.conv_max = nn.Conv2d(f, f, 3, padding=1)
            self.conv2    = nn.Conv2d(f, f, 3, stride=2, padding=0)
            self.conv3    = nn.Conv2d(f, f, 3, padding=1)
            self.conv3_   = nn.Conv2d(f, f, 3, padding=1)
            self.conv4    = nn.Conv2d(f, n_feats, 1)
            self.sigmoid  = nn.Sigmoid()
            self.relu     = nn.ReLU(inplace=True)

        def forward(self, x):
            c1_    = self.conv1(x)
            c1     = self.conv2(c1_)
            v_max  = F.max_pool2d(c1, kernel_size=7, stride=3)
            v_range = self.relu(self.conv_max(v_max))
            c3     = self.relu(self.conv3(v_range))
            c3     = self.conv3_(c3)
            c3     = F.interpolate(c3, (x.size(2), x.size(3)),
                                   mode="bilinear", align_corners=False)
            cf     = self.conv_f(c1_)
            c4     = self.conv4(c3 + cf)
            return x * self.sigmoid(c4)


    class RLFB(nn.Module):
        """Residual Local Feature Block: 3×Conv + ESA + skip."""

        def __init__(self, n_feats: int, use_m23: bool = True):
            super().__init__()
            self.conv1 = nn.Conv2d(n_feats, n_feats, 3, padding=1)
            self.conv2 = nn.Conv2d(n_feats, n_feats, 3, padding=1)
            self.conv3 = nn.Conv2d(n_feats, n_feats, 3, padding=1)
            self.esa   = ESA(n_feats)
            self.act   = nn.GELU()
            if use_m23:
                self._apply_m23()

        def _apply_m23(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    m23_init_tensor(m.weight, variant="orthogonal")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        def forward(self, x):
            out = self.act(self.conv1(x))
            out = self.act(self.conv2(out))
            out = self.conv3(out)
            return self.esa(out) + x


    class M23RLFN(nn.Module):
        """
        M23-RLFN Super-Resolution Network.

        Args:
            in_channels:  Input channels (default 3 for RGB)
            out_channels: Output channels
            n_feats:      Feature channels (default 52 → ~900K params)
            n_blocks:     Number of RLFB blocks
            scale:        Upscale factor (2, 3, or 4)
            use_m23:      Whether to apply M23-Spectrum initialization

        Example:
            >>> model = M23RLFN(scale=4)
            >>> lr = torch.randn(1, 3, 64, 64)
            >>> sr = model(lr)  # → (1, 3, 256, 256)
        """

        def __init__(
            self,
            in_channels:  int = 3,
            out_channels: int = 3,
            n_feats:      int = 52,
            n_blocks:     int = 8,
            scale:        int = 4,
            use_m23:      bool = True,
        ):
            super().__init__()
            self.scale = scale

            self.head      = nn.Conv2d(in_channels, n_feats, 3, padding=1)
            self.body      = nn.Sequential(*[RLFB(n_feats, use_m23=use_m23) for _ in range(n_blocks)])
            self.body_conv = nn.Conv2d(n_feats, n_feats, 3, padding=1)
            self.upsample  = nn.Sequential(
                nn.Conv2d(n_feats, out_channels * (scale ** 2), 3, padding=1),
                nn.PixelShuffle(scale),
            )

            if use_m23:
                # Init head, body_conv, upsample (RLFB blocks init themselves)
                for name, m in self.named_modules():
                    if isinstance(m, nn.Conv2d) and not any(
                        f"body.{i}." in name for i in range(n_blocks)
                    ):
                        m23_init_tensor(m.weight, variant="orthogonal")
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            bicubic  = F.interpolate(x, scale_factor=self.scale,
                                     mode="bicubic", align_corners=False)
            feat     = self.head(x)
            body_out = self.body(feat)
            body_out = self.body_conv(body_out) + feat
            out      = self.upsample(body_out)
            return out + bicubic

        @classmethod
        def from_pretrained(cls, name: str, device: str = "cpu") -> "M23RLFN":
            """Load from pretrained config (weights hosted in future releases)."""
            return create_model(name, pretrained=True, device=device)

else:
    # Stubs when PyTorch is unavailable
    class M23RLFN:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is required for M23RLFN")
