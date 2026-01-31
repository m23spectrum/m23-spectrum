"""
Example 1: Basic M23-Spectrum Initialization

This example demonstrates how to initialize a simple weight matrix
using M23-Spectrum and analyze its spectral properties.
"""

import numpy as np
from m23_spectrum import generate_m23_stable_spectrum, mgi_init_stable, analyze_spectral_properties

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

# Initialize a 512x256 weight matrix
weights = mgi_init_stable((512, 256))
print(f"\nInitialized weight matrix shape: {weights.shape}")

# Analyze spectral properties
props = analyze_spectral_properties(weights[:256, :256])
print(f"\nSpectral Properties:")
print(f"  Spectral radius: {props['spectral_radius']:.6f}")
print(f"  Condition number: {props['condition_number']:.4f}")
print(f"  Mean singular value: {props['mean_singular_value']:.6f}")
print(f"  Frobenius norm: {props['frobenius_norm']:.4f}")
print(f"  Operator norm: {props['operator_norm']:.6f}")

print("\n" + "=" * 60)
print("Example 1 Complete!")
print("=" * 60)