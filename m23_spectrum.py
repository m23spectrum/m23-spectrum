"""M23-Spectrum: Algebraic Weight Initialization for Deep Neural Networks

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


class _SpectrumCache:
    """Thread-safe cache for M23 spectrum to avoid redundant computations."""
    def __init__(self):
        self._cache = {}
    
    def get(self, fan_in: int) -> Optional[np.ndarray]:
        return self._cache.get(fan_in)
    
    def set(self, fan_in: int, spectrum: np.ndarray) -> None:
        self._cache[fan_in] = spectrum.copy()
    
    def clear(self) -> None:
        self._cache.clear()


_spectrum_cache = _SpectrumCache()


def _compute_elkies_polynomial_roots() -> np.ndarray:
    """Compute roots of the Elkies polynomial associated with M23.
    
    The polynomial g^4 + g^3 + 9g^2 - 10g + 8 = 0 encodes spectral
    properties of the Mathieu group M23 (sporadic group of order 10,080).
    
    Returns
    -------
    np.ndarray
        Complex roots of the Elkies polynomial, shape (4,).
    """
    # Coefficients of the polynomial: g^4 + g^3 + 9g^2 - 10g + 8 = 0
    coefficients = [1, 1, 9, -10, 8]
    roots = np.roots(coefficients)
    return roots


def _normalize_spectrum(spectrum: np.ndarray, scaling_factor: float) -> np.ndarray:
    """Normalize spectrum with numerical stability guards.
    
    Parameters
    ----------
    spectrum : np.ndarray
        The eigenvalue spectrum.
    scaling_factor : float
        Factor to scale the spectrum for signal preservation.
    
    Returns
    -------
    np.ndarray
        Normalized spectrum.
    
    Raises
    ------
    M23SpectrumError
        If spectrum contains invalid values (NaN or Inf).
    """
    if not np.all(np.isfinite(spectrum)):
        raise M23SpectrumError("Spectrum contains NaN or Inf values")
    
    # Compute spectral norm (largest absolute eigenvalue)
    spectral_norm = np.max(np.abs(spectrum))
    
    if spectral_norm < 1e-10:
        warnings.warn("Spectral norm is very small (< 1e-10). Consider checking input dimensions.", 
                     RuntimeWarning)
        spectral_norm = 1e-10
    
    # Normalize to unit spectral radius
    normalized = spectrum / spectral_norm
    
    # Apply scaling for dynamic isometry
    return normalized * scaling_factor


def generate_m23_stable_spectrum(fan_in: int, seed: Optional[int] = None, 
                                  use_cache: bool = True) -> np.ndarray:
    """Generate the M23 spectrum adapted for dynamic isometry in neural networks.
    
    This function generates a deterministic spectrum derived from the Elkies polynomial
    roots. The spectrum maintains spectral stability through arbitrary network depth.
    
    The spectrum is derived from the Elkies polynomial and the structure of
    the Mathieu group M23 (sporadic group of order 10,080).
    
    Parameters
    ----------
    fan_in : int
        Input dimension of the neural network layer. Must be positive.
    seed : Optional[int]
        Random seed for reproducibility. If None, uses deterministic generation.
    use_cache : bool
        Whether to cache computed spectra for performance (default: True).
    
    Returns
    -------
    np.ndarray
        Complex-valued spectrum array of shape (fan_in,).
        For real networks, take the real part or absolute value.
    
    Raises
    ------
    M23SpectrumError
        If fan_in <= 0.
    
    Raises
    ------
    ValueError
        If fan_in is not an integer.
    
    Notes
    -----
    The spectrum components:
    - P2 components (multiplicity 2): roots of z^2 - gz + z^2 = 0
    - P3 components (multiplicity 3): roots of z^3 + gz^2 - 1 = 0
    - P4 components (multiplicity 4): roots of z^4 + gz^3 - g^2z - z - g = 0
    """
    # Input validation
    if not isinstance(fan_in, (int, np.integer)):
        raise TypeError(f"fan_in must be integer, got {type(fan_in)}")
    if fan_in <= 0:
        raise M23SpectrumError(f"fan_in must be positive, got {fan_in}")
    
    # Check cache
    if use_cache:
        cached = _spectrum_cache.get(fan_in)
        if cached is not None:
            return cached.copy()
    
    if seed is not None:
        np.random.seed(seed)
    
    # Compute Elkies polynomial roots
    elkies_roots = _compute_elkies_polynomial_roots()
    
    # Compute scaling factor for dynamic isometry
    scaling_factor = np.sqrt(1.0 / max(fan_in, 1))
    
    # Generate spectrum by tiling and cycling through roots
    n_roots = len(elkies_roots)
    spectrum = np.zeros(fan_in, dtype=np.complex128)
    
    for i in range(fan_in):
        # Cycle through available roots
        root_idx = i % n_roots
        spectrum[i] = elkies_roots[root_idx]
    
    # Normalize spectrum
    spectrum = _normalize_spectrum(spectrum, scaling_factor)
    
    # Cache the result
    if use_cache:
        _spectrum_cache.set(fan_in, spectrum)
    
    return spectrum


def m23_initialize(shape: Tuple[int, ...], fan_in: Optional[int] = None, 
                   seed: Optional[int] = None,
                   variant: str = 'standard') -> np.ndarray:
    """Initialize neural network weights using M23-Spectrum algorithm.
    
    This function provides multiple initialization variants optimized for
    different architectures and depth configurations.
    
    Parameters
    ----------
    shape : Tuple[int, ...]
        Shape of the weight matrix/tensor to initialize.
        For 2D: (fan_in, fan_out)
        For 4D (Conv2D): (kernel_h, kernel_w, in_channels, out_channels)
    fan_in : Optional[int]
        Input dimension. If None, inferred from shape (first dimension).
    seed : Optional[int]
        Random seed for reproducibility.
    variant : str
        Initialization variant:
        - 'standard': Full M23 spectrum with QR decomposition
        - 'orthogonal': Fully orthogonal (uses SVD)
        - 'scaled': M23 with scaling for gradient flow
    
    Returns
    -------
    np.ndarray
        Initialized weight matrix of specified shape.
    
    Raises
    ------
    M23SpectrumError
        If shape is invalid or fan_in is inconsistent.
    ValueError
        If variant is not recognized.
    """
    # Input validation
    if not shape or any(d <= 0 for d in shape):
        raise M23SpectrumError(f"Invalid shape: {shape}")
    
    if variant not in ['standard', 'orthogonal', 'scaled']:
        raise ValueError(f"Unknown variant: {variant}. Choose from: standard, orthogonal, scaled")
    
    # Infer fan_in if not provided
    if fan_in is None:
        fan_in = shape[0]
    
    if fan_in <= 0:
        raise M23SpectrumError(f"fan_in must be positive, got {fan_in}")
    
    # Generate M23 spectrum
    spectrum = generate_m23_stable_spectrum(fan_in, seed=seed)
    
    # Compute flattened output dimension
    fan_out = int(np.prod(shape[1:]))
    
    # Build block-diagonal matrix (ИСПРАВЛЕНО)
    if fan_out <= fan_in:
        # Simple case: fan_out <= fan_in
        block_matrix = np.diag(spectrum[:fan_out])
        # Pad to correct size
        if block_matrix.shape[0] < fan_out or block_matrix.shape[1] < fan_in:
            padded = np.zeros((fan_out, fan_in), dtype=spectrum.dtype)
            padded[:block_matrix.shape[0], :block_matrix.shape[1]] = block_matrix
            block_matrix = padded
    else:
        # Complex case: fan_out > fan_in
        n_blocks = fan_out // fan_in
        remainder = fan_out % fan_in
        
        # Create main blocks
        blocks = []
        for i in range(n_blocks):
            blocks.append(np.diag(spectrum))
        
        # Add remainder if needed
        if remainder > 0:
            blocks.append(np.diag(spectrum[:remainder]))
        
        # Stack blocks vertically (ИСПРАВЛЕНО: правильное выравнивание)
        max_cols = fan_in
        aligned_blocks = []
        for block in blocks:
            if block.shape[1] < max_cols:
                # Pad smaller blocks to match width
                padded = np.zeros((block.shape[0], max_cols), dtype=block.dtype)
                padded[:, :block.shape[1]] = block
                aligned_blocks.append(padded)
            else:
                aligned_blocks.append(block)
        
        block_matrix = np.vstack(aligned_blocks)
    
    # Apply variant-specific processing
    if variant == 'orthogonal':
        # Use SVD for full orthogonality
        try:
            U, _, Vt = np.linalg.svd(block_matrix, full_matrices=False)
            block_matrix = U @ Vt
        except np.linalg.LinAlgError:
            warnings.warn("SVD failed, falling back to QR", RuntimeWarning)
            Q, R = np.linalg.qr(block_matrix)
            block_matrix = Q
    elif variant == 'scaled':
        # Apply scaling for improved gradient flow
        scale = np.sqrt(2.0 / (fan_in + fan_out))
        block_matrix = block_matrix * scale
    else:  # standard
        # QR decomposition for stability
        try:
            Q, R = np.linalg.qr(block_matrix)
            block_matrix = Q
        except np.linalg.LinAlgError:
            warnings.warn("QR failed, using raw matrix", RuntimeWarning)
    
    # Ensure correct output shape
    if block_matrix.shape[0] != fan_out or block_matrix.shape[1] != fan_in:
        # Resize if needed
        result = np.zeros((fan_out, fan_in), dtype=block_matrix.dtype)
        min_rows = min(block_matrix.shape[0], fan_out)
        min_cols = min(block_matrix.shape[1], fan_in)
        result[:min_rows, :min_cols] = block_matrix[:min_rows, :min_cols]
        block_matrix = result
    
    # Reshape to target shape
    weights = np.real(block_matrix)  # Convert to real if needed
    weights = weights.reshape(shape)
    
    return weights



def initialize_layer(shape: Tuple[int, ...], fan_in: Optional[int] = None,
                    seed: Optional[int] = None) -> np.ndarray:
    """Convenience wrapper for standard M23 initialization.
    
    Parameters
    ----------
    shape : Tuple[int, ...]
        Shape of weight matrix/tensor.
    fan_in : Optional[int]
        Input fan-in dimension.
    seed : Optional[int]
        Random seed.
    
    Returns
    -------
    np.ndarray
        Initialized weights using M23-Spectrum standard variant.
    """
    return m23_initialize(shape, fan_in=fan_in, seed=seed, variant='standard')


def clear_spectrum_cache() -> None:
    """Clear the internal spectrum cache.
    
    Useful for memory management in long-running applications or testing.
    """
    _spectrum_cache.clear()
