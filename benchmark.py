"""
M23-Spectrum Benchmark Script
==============================

Compares M23-Spectrum initialization against standard methods:
- Xavier/Glorot uniform and normal
- He/Kaiming uniform and normal
- Orthogonal (scipy)
- M23-Spectrum variants

Metrics:
- Spectral radius at initialization
- Gradient flow through deep networks
- Condition number
- Activation variance preservation

Run: python benchmark.py
"""

import time
import argparse
from typing import Dict, List, Tuple, Callable
from dataclasses import dataclass

import numpy as np


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    name: str
    spectral_radius: float
    condition_number: float
    activation_variance: float
    gradient_variance: float
    init_time_ms: float


def xavier_uniform(shape: Tuple[int, ...]) -> np.ndarray:
    """Xavier/Glorot uniform initialization."""
    fan_in = shape[0]
    fan_out = shape[1] if len(shape) > 1 else shape[0]
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, shape).astype(np.float32)


def xavier_normal(shape: Tuple[int, ...]) -> np.ndarray:
    """Xavier/Glorot normal initialization."""
    fan_in = shape[0]
    fan_out = shape[1] if len(shape) > 1 else shape[0]
    std = np.sqrt(2.0 / (fan_in + fan_out))
    return np.random.normal(0, std, shape).astype(np.float32)


def he_uniform(shape: Tuple[int, ...]) -> np.ndarray:
    """He/Kaiming uniform initialization."""
    fan_in = shape[0]
    limit = np.sqrt(6.0 / fan_in)
    return np.random.uniform(-limit, limit, shape).astype(np.float32)


def he_normal(shape: Tuple[int, ...]) -> np.ndarray:
    """He/Kaiming normal initialization."""
    fan_in = shape[0]
    std = np.sqrt(2.0 / fan_in)
    return np.random.normal(0, std, shape).astype(np.float32)


def orthogonal_np(shape: Tuple[int, ...]) -> np.ndarray:
    """Orthogonal initialization using QR decomposition."""
    fan_in = shape[0]
    fan_out = shape[1] if len(shape) > 1 else shape[0]
    flat_shape = (fan_out, fan_in)
    random_matrix = np.random.normal(0, 1, flat_shape).astype(np.float32)
    q, r = np.linalg.qr(random_matrix)
    # Ensure uniform distribution of singular values
    d = np.diag(r)
    ph = np.sign(d)
    q = q * ph
    return q.reshape(shape)


# Import M23-Spectrum methods
from m23_spectrum import m23_initialize


def create_m23_variant(variant: str) -> Callable:
    """Create M23 initializer for specific variant."""
    def m23_init(shape: Tuple[int, ...]) -> np.ndarray:
        return m23_initialize(shape, variant=variant).astype(np.float32)
    return m23_init


def compute_spectral_radius(matrix: np.ndarray) -> float:
    """Compute spectral radius (largest absolute eigenvalue)."""
    if matrix.shape[0] == matrix.shape[1]:
        eigenvalues = np.linalg.eigvals(matrix)
        return float(np.max(np.abs(eigenvalues)))
    else:
        # For non-square, use singular values
        s = np.linalg.svd(matrix, compute_uv=False)
        return float(s[0])


def compute_condition_number(matrix: np.ndarray) -> float:
    """Compute condition number."""
    s = np.linalg.svd(matrix, compute_uv=False)
    return float(s[0] / max(s[-1], 1e-10))


def simulate_forward_pass(
    weights: List[np.ndarray],
    input_dim: int,
    batch_size: int = 32,
) -> Tuple[float, float]:
    """
    Simulate forward pass through layers and compute activation statistics.

    Returns:
        Tuple of (final_activation_variance, activation_variance_ratio)
    """
    # Start with normalized input
    x = np.random.randn(batch_size, input_dim).astype(np.float32)
    x = x / np.linalg.norm(x, axis=1, keepdims=True)

    activation_variances = []

    for W in weights:
        # Simple linear forward pass
        if W.ndim == 2:
            x = x @ W.T
        elif W.ndim == 4:
            # Simulate conv by flattening
            fan_out, fan_in = W.shape[0], int(np.prod(W.shape[1:]))
            x = x @ W.reshape(fan_out, fan_in).T

        activation_variances.append(np.var(x))

        # ReLU-like nonlinearity (keep positive, zero negative)
        x = np.maximum(0, x)

        # Normalize to prevent explosion/vanishing in simulation
        norm = np.linalg.norm(x, axis=1, keepdims=True)
        norm = np.maximum(norm, 1e-8)
        x = x / norm

    return activation_variances[-1] if activation_variances else 0.0


def simulate_backward_pass(
    weights: List[np.ndarray],
    output_dim: int,
    batch_size: int = 32,
) -> float:
    """
    Simulate backward pass and compute gradient variance.

    Returns:
        Gradient variance at input
    """
    # Start gradient from output
    grad = np.random.randn(batch_size, output_dim).astype(np.float32)
    grad = grad / np.linalg.norm(grad, axis=1, keepdims=True)

    for W in reversed(weights):
        if W.ndim == 2:
            grad = grad @ W  # Backprop through linear
        elif W.ndim == 4:
            fan_out, fan_in = W.shape[0], int(np.prod(W.shape[1:]))
            grad = grad @ W.reshape(fan_out, fan_in)

        # Simulate ReLU gradient (binary mask)
        mask = (np.random.randn(*grad.shape) > 0).astype(np.float32)
        grad = grad * mask

        # Normalize
        norm = np.linalg.norm(grad, axis=1, keepdims=True)
        norm = np.maximum(norm, 1e-8)
        grad = grad / norm

    return float(np.var(grad))


def run_benchmark(
    initializer: Callable,
    name: str,
    n_layers: int,
    hidden_dim: int,
) -> BenchmarkResult:
    """Run benchmark for a single initializer."""

    # Time the initialization
    start = time.perf_counter()

    weights = []
    for i in range(n_layers):
        W = initializer((hidden_dim, hidden_dim))
        weights.append(W)

    init_time = (time.perf_counter() - start) * 1000  # ms

    # Compute metrics
    # Spectral radius of first layer
    spectral_radius = compute_spectral_radius(weights[0])

    # Condition number of first layer
    condition_number = compute_condition_number(weights[0])

    # Activation variance through network
    activation_var = simulate_forward_pass(weights, hidden_dim)

    # Gradient variance
    gradient_var = simulate_backward_pass(weights, hidden_dim)

    return BenchmarkResult(
        name=name,
        spectral_radius=spectral_radius,
        condition_number=condition_number,
        activation_variance=activation_var,
        gradient_variance=gradient_var,
        init_time_ms=init_time,
    )


def print_results(results: List[BenchmarkResult], n_layers: int, hidden_dim: int):
    """Print benchmark results in a formatted table."""

    print("\n" + "=" * 80)
    print(f"  M23-Spectrum Benchmark: {n_layers} layers, hidden_dim={hidden_dim}")
    print("=" * 80)

    # Header
    print(f"\n{'Method':<20} {'Spectral R':>12} {'Condition':>12} "
          f"{'Act Var':>12} {'Grad Var':>12} {'Time (ms)':>10}")
    print("-" * 80)

    # Results
    for r in results:
        print(f"{r.name:<20} {r.spectral_radius:>12.4f} {r.condition_number:>12.2f} "
              f"{r.activation_variance:>12.6f} {r.gradient_variance:>12.6f} {r.init_time_ms:>10.2f}")

    print("-" * 80)

    # Find best for each metric
    best_spectral = min(results, key=lambda r: abs(r.spectral_radius - 1.0))
    best_condition = min(results, key=lambda r: r.condition_number)
    best_time = min(results, key=lambda r: r.init_time_ms)

    print(f"\nBest spectral radius ≈ 1.0:  {best_spectral.name} ({best_spectral.spectral_radius:.4f})")
    print(f"Best condition number:       {best_condition.name} ({best_condition.condition_number:.2f})")
    print(f"Fastest initialization:      {best_time.name} ({best_time.init_time_ms:.2f} ms)")


def main():
    parser = argparse.ArgumentParser(description="M23-Spectrum Benchmark")
    parser.add_argument("--layers", type=int, default=24, help="Number of layers")
    parser.add_argument("--dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Set seed for reproducibility
    np.random.seed(args.seed)

    # Define initializers to benchmark
    initializers = [
        ("Xavier Uniform", xavier_uniform),
        ("Xavier Normal", xavier_normal),
        ("He Uniform", he_uniform),
        ("He Normal", he_normal),
        ("Orthogonal (QR)", orthogonal_np),
        ("M23-Standard", create_m23_variant("standard")),
        ("M23-Orthogonal", create_m23_variant("orthogonal")),
        ("M23-Scaled", create_m23_variant("scaled")),
    ]

    results = []
    for name, init_fn in initializers:
        print(f"Running {name}...", end="\r")
        result = run_benchmark(init_fn, name, args.layers, args.dim)
        results.append(result)

    print_results(results, args.layers, args.dim)

    # Additional deep network test
    print("\n" + "=" * 80)
    print("  Deep Network Gradient Flow Test (50 layers)")
    print("=" * 80)

    np.random.seed(args.seed)
    deep_results = []
    for name, init_fn in [("He Normal", he_normal), ("Orthogonal", orthogonal_np),
                           ("M23-Orthogonal", create_m23_variant("orthogonal"))]:
        result = run_benchmark(init_fn, name, 50, args.dim)
        deep_results.append(result)

    print(f"\n{'Method':<20} {'Activation Var':>15} {'Gradient Var':>15}")
    print("-" * 55)
    for r in deep_results:
        print(f"{r.name:<20} {r.activation_variance:>15.6f} {r.gradient_variance:>15.6f}")


if __name__ == "__main__":
    main()
