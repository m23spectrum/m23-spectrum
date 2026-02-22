"""
Example 1: Basic M23-Spectrum Initialization

This example demonstrates how to initialize a simple weight matrix
using M23-Spectrum and analyze its spectral properties.
"""

import numpy as np

# Import from standalone module
from m23_spectrum import (
    generate_m23_stable_spectrum,
    m23_initialize,
    clear_spectrum_cache,
)

print("=" * 60)
print("Example 1: Basic M23-Spectrum Initialization")
print("=" * 60)

# Generate M23 spectrum for a 256-dimensional input
fan_in = 256
spectrum = generate_m23_stable_spectrum(fan_in)
print(f"\nGenerated spectrum for fan_in={fan_in}")
print(f"  Spectrum size: {spectrum.size}")
print(f"  Spectral radius: {np.max(np.abs(spectrum)):.6f}")
print(f"  Mean value: {np.mean(spectrum):.6f}")
print(f"  Spectrum dtype: {spectrum.dtype}")

# Initialize a 512x256 weight matrix with different variants
print("\n" + "-" * 60)
print("Weight Matrix Initialization Examples")
print("-" * 60)

for variant in ["standard", "orthogonal", "scaled"]:
    weights = m23_initialize((512, 256), variant=variant)
    print(f"\nVariant: {variant}")
    print(f"  Shape: {weights.shape}")
    print(f"  Mean: {np.mean(weights):.6f}")
    print(f"  Std:  {np.std(weights):.6f}")
    print(f"  Min:  {np.min(weights):.6f}")
    print(f"  Max:  {np.max(weights):.6f}")

    # Analyze spectral properties
    if weights.shape[0] == weights.shape[1]:
        eigenvalues = np.linalg.eigvals(weights)
        spectral_radius = np.max(np.abs(eigenvalues))
        print(f"  Spectral radius: {spectral_radius:.6f}")

    singular_values = np.linalg.svd(weights, compute_uv=False)
    print(f"  Condition number: {singular_values[0] / singular_values[-1]:.4f}")
    print(f"  Mean singular value: {np.mean(singular_values):.6f}")

# Test Conv2D-like initialization (4D tensor)
print("\n" + "-" * 60)
print("Conv2D Weight Initialization")
print("-" * 60)

conv_shape = (64, 32, 3, 3)  # (out_channels, in_channels, kernel_h, kernel_w)
conv_weights = m23_initialize(conv_shape, variant="orthogonal")
print(f"Conv2D shape: {conv_shape}")
print(f"  Mean: {np.mean(conv_weights):.6f}")
print(f"  Std:  {np.std(conv_weights):.6f}")

# Demonstrate caching
print("\n" + "-" * 60)
print("Spectrum Caching Demo")
print("-" * 60)

import time

# First call (may be cached)
start = time.perf_counter()
spectrum1 = generate_m23_stable_spectrum(1024)
time1 = time.perf_counter() - start

# Second call (should hit cache)
start = time.perf_counter()
spectrum2 = generate_m23_stable_spectrum(1024)
time2 = time.perf_counter() - start

print(f"First call:  {time1*1000:.3f} ms")
print(f"Second call: {time2*1000:.3f} ms (cached)")
print(f"Speedup: {time1/time2:.1f}x")

# Clear cache
clear_spectrum_cache()
print("Cache cleared.")

print("\n" + "=" * 60)
print("Example 1 Complete!")
print("=" * 60)
