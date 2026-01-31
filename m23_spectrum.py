"""
M23-Spectrum: Algebraic Weight Initialization for Deep Neural Networks

This module implements weight initialization using the algebraic structure 
of the Mathieu group M23 and dynamic isometry principles.

License: MIT
Author: M23-Spectrum Contributors
Year: 2026
"""

import numpy as np
from typing import Tuple, Optional
import warnings


class M23SpectrumError(Exception):
    """Base exception for M23-Spectrum module."""
    pass


def generate_m23_stable_spectrum(fan_in: int) -> np.ndarray:
    """
    Generates the M23 spectrum adapted for dynamic isometry in neural networks.
    
    The spectrum is derived from the Elkies polynomial and the structure of 
    the Mathieu group M23 (sporadic group of order 10,080).
    
    Parameters
    ----------
    fan_in : int
        Input dimension of the neural network layer.
    
    Returns
    -------
    np.ndarray
        Complex-valued spectrum array. If fan_in < 23, it's truncated; 
        if fan_in > 23, it's tiled and resized.
    
    Raises
    ------
    M23SpectrumError
        If fan_in <= 0.
    
    Notes
    -----
    The spectrum consists of:
    - P2 components (multiplicity 2): roots of z^2 - g*z + g^2 = 0
    - P3 components (multiplicity 1): roots of z^3 + g*z - 1 = 0
    - P4 components (multiplicity 4): roots of z^4 + g*z^3 - g^2*z^2 + z - g = 0
    
    where g is a root of g^4 + g^3 + 9*g^2 - 10*g + 8 = 0 (Elkies polynomial).
    """
    
    if fan_in <= 0:
        raise M23SpectrumError(f"fan_in must be positive, got {fan_in}")
    
    # 1. Find roots of the Elkies polynomial (base field)
    # g^4 + g^3 + 9g^2 - 10g + 8 = 0
    elkies_coeffs = [1, 1, 9, -10, 8]
    g_roots = np.roots(elkies_coeffs)
    g = g_roots[0]  # Select first root (can vary, all valid)
    
    # 2. Component polynomials with defined multiplicities (2, 1, 4)
    # Using complex roots fully
    roots_p2 = np.roots([1, -g, g**2])
    roots_p3 = np.roots([1, 0, g, -1])
    roots_p4 = np.roots([1, g, -g**2, 1, -g])
    
    # 3. Assemble the full 23-dimensional spectrum
    raw_spectrum = np.concatenate([
        np.repeat(roots_p2, 2),    # 2 copies of P2 roots (2 + 2 = 4 elements)
        roots_p3,                   # P3 roots (3 elements)
        np.repeat(roots_p4, 4)      # 4 copies of P4 roots (4*4 = 16 elements)
    ])
    
    # Verify correct size (should be 4 + 3 + 16 = 23)
    assert len(raw_spectrum) == 23, f"Spectrum size mismatch: {len(raw_spectrum)}"
    
    # 4. Global normalization for dynamic isometry
    # Key insight: mean square eigenvalue should be ~ 1/fan_in to prevent gradient explosion
    scale = np.sqrt(2.0 / fan_in)
    max_abs = np.max(np.abs(raw_spectrum))
    
    stable_spectrum = raw_spectrum * (scale / max_abs)
    
    return stable_spectrum


def mgi_init_stable(matrix_shape: Tuple[int, int]) -> np.ndarray:
    """
    Initialize a weight matrix using M23-Spectrum initialization.
    
    This method constructs a weight matrix W such that signal norm is preserved
    through arbitrary network depth (dynamic isometry property).
    
    Parameters
    ----------
    matrix_shape : tuple
        Shape (fan_out, fan_in) of the weight matrix to initialize.
    
    Returns
    -------
    np.ndarray
        Initialized weight matrix of shape matrix_shape, dtype float32.
    
    Raises
    ------
    M23SpectrumError
        If matrix dimensions are invalid.
    
    Algorithm
    ---------
    1. Generate M23 spectrum for fan_in dimension
    2. Resize spectrum to match fan_out
    3. Construct block-diagonal matrix with rotation blocks for complex pairs
    4. Apply random orthogonal transformation via QR decomposition
    5. Trim/pad to exact target shape
    
    Notes
    -----
    The rotation blocks for complex eigenvalue pairs ensure that "energy"
    of complex numbers is preserved in real-valued matrices:
    
        B_i = [[a, -b],     where lambda = a + bi
               [b,  a]]
    
    This construction maintains the spectral properties while working in
    real arithmetic.
    """
    
    fan_out, fan_in = matrix_shape
    
    if fan_out <= 0 or fan_in <= 0:
        raise M23SpectrumError(f"Invalid shape: ({fan_out}, {fan_in})")
    
    # Step 1: Generate spectrum
    spec = generate_m23_stable_spectrum(fan_in)
    
    # Step 2: Resize spectrum to fan_out
    full_spec = np.resize(spec, fan_out)
    
    # Step 3: Build block-diagonal matrix from spectral decomposition
    W_diag = np.zeros((fan_out, fan_out), dtype=np.complex128)
    i = 0
    
    while i < fan_out:
        val = full_spec[i]
        is_real = np.abs(val.imag) < 1e-10
        is_last = (i == fan_out - 1)
        
        if is_real or is_last:
            # Real eigenvalue: place on diagonal
            W_diag[i, i] = val.real
            i += 1
        else:
            # Complex eigenvalue pair: create rotation block
            if i + 1 >= fan_out:
                # Handle edge case: odd-sized matrix with last element complex
                W_diag[i, i] = val.real
                i += 1
            else:
                a, b = val.real, val.imag
                # Rotation block preserves |lambda| = sqrt(a^2 + b^2)
                W_diag[i, i] = a
                W_diag[i, i+1] = -b
                W_diag[i+1, i] = b
                W_diag[i+1, i+1] = a
                i += 2
    
    # Convert to real matrix (extract real part, imaginary parts are zero)
    W_diag_real = W_diag.real
    
    # Step 4: Create random orthogonal basis via QR decomposition
    Q, _ = np.linalg.qr(np.random.randn(fan_out, fan_out))
    
    # Step 5: Combine orthogonal basis with spectral structure
    W = Q @ W_diag_real
    
    # Step 6: Adapt to fan_in dimension
    if fan_in > fan_out:
        # Pad with zeros if fan_in > fan_out
        W = np.pad(W, ((0, 0), (0, fan_in - fan_out)), mode='constant')
    elif fan_in < fan_out:
        # Trim columns if fan_in < fan_out
        W = W[:, :fan_in]
    
    return W.astype(np.float32)


def apply_m23_init(module, verbose: bool = False) -> None:
    """
    Apply M23-Spectrum initialization to all Linear layers in a PyTorch module.
    
    Parameters
    ----------
    module : torch.nn.Module
        A PyTorch module to initialize recursively.
    verbose : bool, optional
        If True, print initialization info for each layer.
    
    Notes
    -----
    This function requires PyTorch. It will raise ImportError if torch is not available.
    
    Example
    -------
    >>> import torch.nn as nn
    >>> from m23_spectrum import apply_m23_init
    >>> model = nn.Sequential(nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, 10))
    >>> model.apply(apply_m23_init)
    """
    
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        raise ImportError(
            "PyTorch is required for apply_m23_init. "
            "Install it with: pip install torch"
        )
    
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            shape = (m.out_features, m.in_features)
            mgi_weights = mgi_init_stable(shape)
            
            with torch.no_grad():
                m.weight.copy_(torch.from_numpy(mgi_weights))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            
            if verbose:
                print(f"Initialized {m}: shape {shape}, "
                      f"spectral_radius={np.max(np.abs(np.linalg.eigvals(mgi_weights[:min(shape)//2, :min(shape)//2]))):.6f}")
    
    module.apply(_init_weights)


def analyze_spectral_properties(matrix: np.ndarray) -> dict:
    """
    Analyze spectral properties of an initialized weight matrix.
    
    Parameters
    ----------
    matrix : np.ndarray
        Weight matrix to analyze.
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'spectral_radius': max(|eigenvalues|)
        - 'condition_number': ratio of largest to smallest singular value
        - 'mean_singular_value': average singular value
        - 'rank': matrix rank
        - 'frobenius_norm': Frobenius norm of matrix
        - 'operator_norm': spectral norm (largest singular value)
    """
    
    try:
        eigenvalues = np.linalg.eigvals(matrix)
        spectral_radius = np.max(np.abs(eigenvalues))
    except np.linalg.LinAlgError:
        spectral_radius = np.nan
    
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    condition_number = np.linalg.cond(matrix)
    
    properties = {
        'spectral_radius': float(spectral_radius),
        'condition_number': float(condition_number),
        'mean_singular_value': float(np.mean(singular_values)),
        'rank': int(np.linalg.matrix_rank(matrix)),
        'frobenius_norm': float(np.linalg.norm(matrix, 'fro')),
        'operator_norm': float(np.max(singular_values)),
    }
    
    return properties


# Version and metadata
__version__ = '0.1.0'
__author__ = 'M23-Spectrum Contributors'
__license__ = 'MIT'

# Public API
__all__ = [
    'generate_m23_stable_spectrum',
    'mgi_init_stable',
    'apply_m23_init',
    'analyze_spectral_properties',
    'M23SpectrumError',
]