"""
M23-Spectrum Visualization Script
=================================

Generates visualizations of the M23 spectrum:
- Complex plane plot of eigenvalues
- Spectral distribution histogram
- Comparison with random initialization

Run: python visualize_spectrum.py
"""

import argparse
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not installed. Install with: pip install matplotlib")

from m23_spectrum import (
    generate_m23_stable_spectrum,
    m23_initialize,
    _compute_elkies_polynomial_roots,
)


def plot_complex_spectrum(
    spectrum: np.ndarray,
    title: str = "M23 Spectrum",
    save_path: str = None,
):
    """Plot spectrum in complex plane."""
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib required for plotting")
        return

    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot eigenvalues
    real_parts = spectrum.real
    imag_parts = spectrum.imag

    # Color by magnitude
    magnitudes = np.abs(spectrum)
    scatter = ax.scatter(real_parts, imag_parts, c=magnitudes, cmap='viridis',
                         alpha=0.7, s=30, edgecolors='white', linewidth=0.5)

    # Add unit circle for reference
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'r--', alpha=0.5, label='Unit circle')

    # Add axes
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)

    # Labels and title
    ax.set_xlabel('Real Part', fontsize=12)
    ax.set_ylabel('Imaginary Part', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Magnitude', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close()


def plot_elkies_roots(save_path: str = None):
    """Plot Elkies polynomial roots."""
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib required for plotting")
        return

    roots = _compute_elkies_polynomial_roots()

    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot roots
    ax.scatter(roots.real, roots.imag, c='red', s=200, marker='*',
               edgecolors='black', linewidth=1, zorder=5, label='Elkies roots')

    # Add unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'b--', alpha=0.5, label='Unit circle')

    # Add axes
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)

    # Annotate each root
    for i, r in enumerate(roots):
        ax.annotate(f'g{i+1}', (r.real, r.imag), xytext=(10, 10),
                    textcoords='offset points', fontsize=10)

    ax.set_xlabel('Real Part', fontsize=12)
    ax.set_ylabel('Imaginary Part', fontsize=12)
    ax.set_title('Elkies Polynomial Roots: $g^4 + g^3 + 9g^2 - 10g + 8 = 0$', fontsize=12)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close()


def plot_spectrum_histogram(
    spectrum: np.ndarray,
    title: str = "M23 Spectrum Distribution",
    save_path: str = None,
):
    """Plot histogram of spectrum magnitudes."""
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib required for plotting")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    magnitudes = np.abs(spectrum)

    # Magnitude histogram
    ax = axes[0]
    ax.hist(magnitudes, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    ax.set_xlabel('Magnitude', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Magnitude Distribution', fontsize=12)
    ax.grid(True, alpha=0.3)

    # Add statistics
    stats_text = f'Mean: {np.mean(magnitudes):.4f}\nStd: {np.std(magnitudes):.4f}\nMax: {np.max(magnitudes):.4f}'
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Phase histogram
    ax = axes[1]
    phases = np.angle(spectrum)
    ax.hist(phases, bins=50, color='coral', edgecolor='white', alpha=0.8)
    ax.set_xlabel('Phase (radians)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Phase Distribution', fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close()


def plot_singular_values_comparison(
    shapes: list,
    save_path: str = None,
):
    """Compare singular value distributions across initialization methods."""
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib required for plotting")
        return

    fig, axes = plt.subplots(1, len(shapes), figsize=(5*len(shapes), 5))
    if len(shapes) == 1:
        axes = [axes]

    for ax, shape in zip(axes, shapes):
        # Generate weights with different methods
        m23_ortho = m23_initialize(shape, variant="orthogonal")
        m23_std = m23_initialize(shape, variant="standard")
        random_ortho = np.linalg.qr(np.random.randn(*shape))[0]

        # Compute singular values
        sv_m23_ortho = np.linalg.svd(m23_ortho, compute_uv=False)
        sv_m23_std = np.linalg.svd(m23_std, compute_uv=False)
        sv_random = np.linalg.svd(random_ortho, compute_uv=False)

        # Plot
        ax.plot(sv_m23_ortho, 'b-', label='M23-Orthogonal', linewidth=2)
        ax.plot(sv_m23_std, 'g--', label='M23-Standard', linewidth=2)
        ax.plot(sv_random, 'r:', label='Random QR', linewidth=2)

        ax.set_xlabel('Index', fontsize=12)
        ax.set_ylabel('Singular Value', fontsize=12)
        ax.set_title(f'Shape: {shape}', fontsize=12)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Singular Value Distribution Comparison', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="M23-Spectrum Visualization")
    parser.add_argument("--output", "-o", default="figures", help="Output directory")
    parser.add_argument("--fan-in", type=int, default=256, help="Fan-in for spectrum")
    args = parser.parse_args()

    if not MATPLOTLIB_AVAILABLE:
        print("\nError: matplotlib is required for visualization.")
        print("Install with: pip install matplotlib")
        return

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating visualizations in {output_dir}/...")

    # 1. Elkies polynomial roots
    print("\n1. Plotting Elkies polynomial roots...")
    plot_elkies_roots(output_dir / "elkies_roots.png")

    # 2. M23 spectrum in complex plane
    print("\n2. Plotting M23 spectrum in complex plane...")
    spectrum = generate_m23_stable_spectrum(args.fan_in)
    plot_complex_spectrum(spectrum, f"M23 Spectrum (fan_in={args.fan_in})",
                          output_dir / f"m23_spectrum_fan{args.fan_in}.png")

    # 3. Spectrum histogram
    print("\n3. Plotting spectrum histogram...")
    plot_spectrum_histogram(spectrum, f"M23 Spectrum Distribution (fan_in={args.fan_in})",
                            output_dir / f"m23_spectrum_hist_fan{args.fan_in}.png")

    # 4. Singular value comparison
    print("\n4. Plotting singular value comparison...")
    plot_singular_values_comparison(
        [(128, 64), (256, 256), (512, 128)],
        output_dir / "singular_values_comparison.png"
    )

    print("\n" + "=" * 50)
    print(f"All visualizations saved to {output_dir}/")
    print("=" * 50)


if __name__ == "__main__":
    main()
