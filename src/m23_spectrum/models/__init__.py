"""Neural network models for M23-Spectrum."""

from .m23_rlfn import M23RLFN, create_model

try:
    from .m23_swinir import M23SwinIR
except ImportError:
    M23SwinIR = None  # timm/swinir optional

__all__ = ["M23RLFN", "M23SwinIR", "create_model"]
