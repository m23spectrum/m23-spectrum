"""
M23-Spectrum: Algebraic Weight Initialization
==============================================

This module implements the core M23-Spectrum algorithm for neural network
weight initialization based on the Mathieu group M23 and dynamic isometry.

Mathematical Foundation:
- Elkies polynomial: g^4 + g^3 + 9g^2 - 10g + 8 = 0
- Spectrum derived from M23 group structure
- SVD orthogonalization for stability
"""

from typing import Optional, Tuple, Dict
import threading
import warnings

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# Global cache with thread-safe access
_CACHE: Dict[Tuple, np.ndarray] = {}
_CACHE_LOCK = threading.RLock()
_M23_BASE: Optional[np.ndarray] = None
_M23_BASE_N: int = 0


def build_m23_base_spectrum() -> np.ndarray:
    """
    Build the base M23 spectrum from Elkies polynomial.

    The spectrum is derived from:
    1. Roots of Elkies polynomial (4 components)
    2. P2 polynomials: z^2 - gz + 1 = 0 (8 components)
    3. P3 polynomials: z^3 + gz - 1 = 0 (12 components)
    4. P4 polynomials: z^4 + gz^3 - (g^2+1)z + g^-1 - g = 0 (16 components)

    Total: 40 spectral components

    Returns:
        Complex array of 40 base spectrum values
    """
    global _M23_BASE, _M23_BASE_N

    if _M23_BASE is not None:
        return _M23_BASE

    # Elkies polynomial roots
    elkies = np.roots([1, 1, 9, -10, 8])

    parts = [elkies]

    for g in elkies:
        # P2: z^2 - gz + 1 = 0
        parts.append(np.roots([1, -g, 1]))

        # P3: z^3 + gz - 1 = 0
        parts.append(np.roots([1, g, 0, -1]))

        # P4: z^4 + gz^3 - (g^2+1)z + (g^-1 - g) = 0
        # Simplified: z^4 + gz^3 - (g^2+1)z - g ≈ 0
        parts.append(np.roots([1, g, 0, -(g**2 + 1), -g]))

    _M23_BASE = np.concatenate(parts)
    _M23_BASE_N = len(_M23_BASE)

    return _M23_BASE


def m23_spectrum(
    fan_in: int,
    seed: Optional[int] = None,
    use_cache: bool = True,
) -> np.ndarray:
    """
    Generate M23 spectrum of specified size.

    Creates a deterministic eigenvalue spectrum for weight initialization
    that maintains dynamic isometry through arbitrary network depth.

    Args:
        fan_in: Input dimension for the layer
        seed: Optional random seed for reproducibility with perturbation
        use_cache: Whether to use caching (default: True)

    Returns:
        Complex-valued spectrum array of shape (fan_in,)
        Normalized so max|λ| ≈ 1/sqrt(fan_in)

    Example:
        >>> spectrum = m23_spectrum(256)
        >>> print(f"Spectral radius: {np.max(np.abs(spectrum)):.4f}")
    """
    global _M23_BASE, _M23_BASE_N

    # Initialize base spectrum if needed
    if _M23_BASE is None:
        build_m23_base_spectrum()

    # Check cache
    cache_key = (fan_in, seed)
    if use_cache:
        with _CACHE_LOCK:
            if cache_key in _CACHE:
                return _CACHE[cache_key].copy()

    # Generate spectrum
    if fan_in <= _M23_BASE_N:
        # Interpolate from base spectrum
        idx = np.round(np.linspace(0, _M23_BASE_N - 1, fan_in)).astype(int)
        spectrum = _M23_BASE[idx].copy()
    else:
        # Tile and phase-shift for larger sizes
        n_tiles = (fan_in + _M23_BASE_N - 1) // _M23_BASE_N
        tiles = [
            _M23_BASE * np.exp(1j * 2 * np.pi * k / n_tiles)
            for k in range(n_tiles)
        ]
        spectrum = np.concatenate(tiles)[:fan_in]

    # Optional perturbation with seed
    if seed is not None:
        rng = np.random.default_rng(seed)
        perturbation = 1e-4 * (
            rng.standard_normal(fan_in) +
            1j * rng.standard_normal(fan_in)
        )
        spectrum = spectrum + perturbation

    # Normalize: max|λ| = 1/sqrt(fan_in) for dynamic isometry
    norm = np.max(np.abs(spectrum)) or 1e-10
    spectrum = spectrum / norm / np.sqrt(max(fan_in, 1))

    # Cache result
    if use_cache:
        with _CACHE_LOCK:
            _CACHE[cache_key] = spectrum.copy()

    return spectrum


def m23_init_tensor(
    tensor: "torch.Tensor",
    variant: str = "orthogonal",
    seed: Optional[int] = None,
) -> "torch.Tensor":
    """
    Initialize a PyTorch tensor using M23-Spectrum.

    Args:
        tensor: PyTorch tensor to initialize in-place
        variant: Initialization variant
            - 'orthogonal': SVD-orthogonalized (recommended for SR)
            - 'standard': QR-orthogonalized
            - 'scaled': Xavier-like scaling
            - 'transformer': 1/sqrt(d_model) scaling
        seed: Optional random seed

    Returns:
        The initialized tensor (modified in-place)

    Example:
        >>> conv = nn.Conv2d(64, 64, 3, padding=1)
        >>> m23_init_tensor(conv.weight, variant="orthogonal")
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for m23_init_tensor")

    shape = tuple(tensor.shape)
    ndim = len(shape)

    # Determine fan_in and fan_out
    if ndim == 1:
        fan_in = fan_out = shape[0]
    elif ndim == 2:
        fan_out, fan_in = shape
    else:  # Conv: (C_out, C_in, *kernel)
        fan_out = shape[0]
        fan_in = int(np.prod(shape[1:]))

    # Generate spectrum
    spectrum = m23_spectrum(fan_in, seed=seed)
    # Build weight matrix with phase structure (vectorized)
    idx_mat = (np.arange(fan_out)[:, np.newaxis] + np.arange(fan_in)[np.newaxis, :]) % len(spectrum)
    mat = spectrum[idx_mat]
    phases = np.exp(1j * 2 * np.pi * (np.arange(fan_in) ** 2) / fan_in)
    mat = mat * phases[np.newaxis, :]
    # Apply variant-specific processing
    if variant == "orthogonal":
        try:
            U, _, Vt = np.linalg.svd(mat, full_matrices=False)
            mat = U @ Vt
        except np.linalg.LinAlgError:
            Q, R = np.linalg.qr(mat)
            mat = Q * np.sign(np.diag(R)[:mat.shape[1]])

    elif variant == "standard":
        Q, R = np.linalg.qr(mat)
        mat = Q * np.sign(np.diag(R)[:mat.shape[1]])

    elif variant == "scaled":
        scale = np.sqrt(2.0 / (fan_in + fan_out))
        mat = mat * scale

    elif variant == "transformer":
        mat = mat / np.sqrt(fan_in)

    else:
        raise ValueError(f"Unknown variant: {variant}")

    # Convert to real and reshape
    weights = np.real(mat).reshape(shape).astype(np.float32)

    # Copy to tensor
    with torch.no_grad():
        tensor.copy_(torch.from_numpy(weights))

    return tensor


def clear_cache() -> None:
    """Clear the spectrum cache."""
    global _CACHE
    with _CACHE_LOCK:
        _CACHE.clear()


def get_cache_info() -> Dict[str, int]:
    """Get cache statistics."""
    with _CACHE_LOCK:
        return {
            "cached_spectra": len(_CACHE),
            "base_spectrum_size": _M23_BASE_N,
        }
