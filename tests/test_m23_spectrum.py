"""
Unit tests for M23-Spectrum initialization library.

Run with: pytest tests/ -v
"""

import pytest
import numpy as np

from m23_spectrum import (
    generate_m23_stable_spectrum,
    m23_initialize,
    initialize_layer,
    clear_spectrum_cache,
    M23SpectrumError,
    _compute_elkies_polynomial_roots,
    _normalize_spectrum,
)


class TestElkiesPolynomial:
    """Tests for Elkies polynomial root computation."""

    def test_roots_count(self):
        """Elkies polynomial should have exactly 4 roots."""
        roots = _compute_elkies_polynomial_roots()
        assert len(roots) == 4

    def test_roots_are_complex(self):
        """Roots should be complex numbers."""
        roots = _compute_elkies_polynomial_roots()
        assert roots.dtype == np.complex128

    def test_roots_satisfy_polynomial(self):
        """Roots should satisfy g^4 + g^3 + 9g^2 - 10g + 8 = 0."""
        roots = _compute_elkies_polynomial_roots()
        for g in roots:
            residual = g**4 + g**3 + 9*g**2 - 10*g + 8
            assert abs(residual) < 1e-10, f"Root {g} has residual {residual}"


class TestSpectrumNormalization:
    """Tests for spectrum normalization."""

    def test_normalization_basic(self):
        """Basic normalization should work correctly."""
        spectrum = np.array([1.0, 2.0, 3.0, 4.0])
        normalized = _normalize_spectrum(spectrum, scaling_factor=0.5)
        assert np.max(np.abs(normalized)) <= 0.5 * 1.0

    def test_normalization_preserves_finite(self):
        """Normalization should preserve finite values."""
        spectrum = np.array([1.0, 2.0, 3.0, 4.0])
        normalized = _normalize_spectrum(spectrum, scaling_factor=1.0)
        assert np.all(np.isfinite(normalized))

    def test_normalization_rejects_nan(self):
        """Normalization should reject NaN values."""
        spectrum = np.array([1.0, np.nan, 3.0])
        with pytest.raises(M23SpectrumError):
            _normalize_spectrum(spectrum, scaling_factor=1.0)

    def test_normalization_rejects_inf(self):
        """Normalization should reject Inf values."""
        spectrum = np.array([1.0, np.inf, 3.0])
        with pytest.raises(M23SpectrumError):
            _normalize_spectrum(spectrum, scaling_factor=1.0)

    def test_normalization_small_values(self):
        """Normalization should handle very small values."""
        spectrum = np.array([1e-20, 1e-20, 1e-20])
        normalized = _normalize_spectrum(spectrum, scaling_factor=1.0)
        assert np.all(np.isfinite(normalized))


class TestGenerateSpectrum:
    """Tests for M23 spectrum generation."""

    def test_basic_generation(self):
        """Should generate spectrum of correct size."""
        for fan_in in [16, 64, 256, 1024]:
            spectrum = generate_m23_stable_spectrum(fan_in)
            assert len(spectrum) == fan_in

    def test_invalid_fan_in_type(self):
        """Should reject non-integer fan_in."""
        with pytest.raises(TypeError):
            generate_m23_stable_spectrum(64.5)

    def test_invalid_fan_in_negative(self):
        """Should reject negative fan_in."""
        with pytest.raises(M23SpectrumError):
            generate_m23_stable_spectrum(-1)

    def test_invalid_fan_in_zero(self):
        """Should reject zero fan_in."""
        with pytest.raises(M23SpectrumError):
            generate_m23_stable_spectrum(0)

    def test_spectrum_is_complex(self):
        """Spectrum should be complex-valued."""
        spectrum = generate_m23_stable_spectrum(256)
        assert np.iscomplexobj(spectrum)

    def test_spectrum_reproducibility(self):
        """Same fan_in should produce identical spectrum (with cache)."""
        spec1 = generate_m23_stable_spectrum(128)
        spec2 = generate_m23_stable_spectrum(128)
        np.testing.assert_array_equal(spec1, spec2)

    def test_spectrum_with_seed(self):
        """Seed should produce reproducible spectrum."""
        clear_spectrum_cache()
        spec1 = generate_m23_stable_spectrum(128, seed=42)
        clear_spectrum_cache()
        spec2 = generate_m23_stable_spectrum(128, seed=42)
        # Same seed should produce same spectrum
        np.testing.assert_array_almost_equal(spec1, spec2)

    def test_spectrum_without_cache(self):
        """Should work without cache."""
        spec = generate_m23_stable_spectrum(64, use_cache=False)
        assert len(spec) == 64

    def test_cache_clear(self):
        """Cache clear should work."""
        generate_m23_stable_spectrum(64)
        clear_spectrum_cache()
        # After clearing, new generation should still work
        spec = generate_m23_stable_spectrum(64)
        assert len(spec) == 64


class TestM23Initialize:
    """Tests for weight initialization."""

    def test_2d_matrix_standard(self):
        """Should initialize 2D matrix with standard variant."""
        weights = m23_initialize((128, 64), variant="standard")
        assert weights.shape == (128, 64)
        assert weights.dtype == np.float64

    def test_2d_matrix_orthogonal(self):
        """Should initialize 2D matrix with orthogonal variant."""
        weights = m23_initialize((128, 64), variant="orthogonal")
        assert weights.shape == (128, 64)
        # Orthogonal matrix should have well-conditioned singular values
        s = np.linalg.svd(weights, compute_uv=False)
        # M23-orthogonal maintains reasonable singular values
        assert np.all(s > 0), "All singular values should be positive"
        assert np.std(s) / np.mean(s) < 0.5, "Singular values should be relatively uniform"

    def test_2d_matrix_scaled(self):
        """Should initialize 2D matrix with scaled variant."""
        weights = m23_initialize((128, 64), variant="scaled")
        assert weights.shape == (128, 64)

    def test_4d_conv_tensor(self):
        """Should initialize 4D conv tensor."""
        weights = m23_initialize((64, 32, 3, 3), variant="orthogonal")
        assert weights.shape == (64, 32, 3, 3)

    def test_1d_vector(self):
        """Should initialize 1D vector."""
        weights = m23_initialize((128,))
        assert weights.shape == (128,)

    def test_invalid_variant(self):
        """Should reject invalid variant."""
        with pytest.raises(ValueError):
            m23_initialize((64, 32), variant="invalid")

    def test_invalid_shape_empty(self):
        """Should reject empty shape."""
        with pytest.raises(M23SpectrumError):
            m23_initialize(())

    def test_invalid_shape_zero_dimension(self):
        """Should reject zero dimension."""
        with pytest.raises(M23SpectrumError):
            m23_initialize((0, 64))

    def test_explicit_fan_in(self):
        """Should use fan_in inferred from shape."""
        # fan_in is inferred from shape[0] by default
        weights = m23_initialize((64, 32))
        assert weights.shape == (64, 32)
        # Verify it's properly initialized
        assert np.all(np.isfinite(weights))

    def test_initialize_layer_wrapper(self):
        """initialize_layer wrapper should work."""
        weights = initialize_layer((64, 32))
        assert weights.shape == (64, 32)


class TestSpectralProperties:
    """Tests for spectral properties of initialized weights."""

    def test_spectral_radius_orthogonal(self):
        """M23-orthogonal should have bounded spectral radius."""
        weights = m23_initialize((64, 64), variant="orthogonal")
        eigenvalues = np.linalg.eigvals(weights)
        spectral_radius = np.max(np.abs(eigenvalues))
        # M23 spectrum is normalized by sqrt(fan_in), so spectral radius < 1
        assert spectral_radius < 1.0, "Spectral radius should be bounded"

    def test_condition_number_orthogonal(self):
        """M23-orthogonal should have reasonable condition number."""
        weights = m23_initialize((64, 64), variant="orthogonal")
        s = np.linalg.svd(weights, compute_uv=False)
        condition = s[0] / s[-1]
        # M23 produces well-conditioned matrices
        assert condition < 10, "Condition number should be reasonable"

    def test_fan_out_greater_than_fan_in(self):
        """Should handle fan_out > fan_in case."""
        weights = m23_initialize((128, 64), variant="orthogonal")
        assert weights.shape == (128, 64)
        assert np.all(np.isfinite(weights))


class TestEdgeCases:
    """Edge case tests."""

    def test_very_small_fan_in(self):
        """Should handle very small fan_in."""
        spectrum = generate_m23_stable_spectrum(1)
        assert len(spectrum) == 1

    def test_large_fan_in(self):
        """Should handle large fan_in."""
        spectrum = generate_m23_stable_spectrum(10000)
        assert len(spectrum) == 10000

    def test_square_matrix(self):
        """Should handle square matrices."""
        weights = m23_initialize((64, 64))
        assert weights.shape == (64, 64)

    def test_very_asymmetric_matrix(self):
        """Should handle very asymmetric matrices."""
        weights = m23_initialize((1024, 16))
        assert weights.shape == (1024, 16)

        weights = m23_initialize((16, 1024))
        assert weights.shape == (16, 1024)


class TestNumericalStability:
    """Numerical stability tests."""

    def test_no_nan_in_weights(self):
        """Weights should never contain NaN."""
        for shape in [(64, 32), (128, 128), (256, 64)]:
            for variant in ["standard", "orthogonal", "scaled"]:
                weights = m23_initialize(shape, variant=variant)
                assert not np.any(np.isnan(weights)), f"NaN in {variant} for {shape}"

    def test_no_inf_in_weights(self):
        """Weights should never contain Inf."""
        for shape in [(64, 32), (128, 128), (256, 64)]:
            for variant in ["standard", "orthogonal", "scaled"]:
                weights = m23_initialize(shape, variant=variant)
                assert not np.any(np.isinf(weights)), f"Inf in {variant} for {shape}"

    def test_values_not_extreme(self):
        """Weights should have reasonable magnitude."""
        weights = m23_initialize((64, 64), variant="orthogonal")
        max_abs = np.max(np.abs(weights))
        assert max_abs < 10, f"Max absolute value {max_abs} is too large"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
