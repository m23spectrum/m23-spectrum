# Changelog

All notable changes to M23-Spectrum will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive test suite (`tests/test_m23_spectrum.py`)
- Benchmark script for comparing initialization methods
- Visualization script for spectrum analysis
- GitHub Actions CI/CD pipeline
- `pyproject.toml` for modern Python packaging
- `requirements.txt` for dependency management
- `CONTRIBUTING.md` for contributors guide

### Changed
- Fixed `example_basic.py` with correct imports
- Renamed `GITIGNORE` to `.gitignore`
- Improved code documentation and type hints

## [0.3.0] - 2026-02-18

### Added
- M23-RLFN architecture for super-resolution
- RLFB (Residual Local Feature Block) from NTIRE 2022
- ESA (Enhanced Spatial Attention) module
- CharbonnierLoss for robust SR training
- FrequencyLoss for high-frequency detail preservation
- CombinedSRLoss (Charbonnier + Frequency)
- WarmStartScheduler for multi-stage learning rate
- SRTrainer with AMP support
- Complete DIV2K training pipeline (`train_div2k.py`)

### Changed
- Enhanced M23 spectrum generation with phase structure
- Improved orthogonal initialization via SVD
- Better spectral normalization

## [0.2.0] - 2025-12-01

### Added
- Thread-safe spectrum caching with RLock
- Multiple initialization variants: `standard`, `orthogonal`, `scaled`
- Support for Conv2D weight tensors (4D)
- Comprehensive error handling and validation

### Changed
- Refactored spectrum generation algorithm
- Improved numerical stability

## [0.1.0] - 2025-09-15

### Added
- Initial release
- Basic M23 spectrum generation from Elkies polynomial
- `generate_m23_stable_spectrum()` function
- `m23_initialize()` for weight matrix initialization
- `initialize_layer()` convenience wrapper
- Basic documentation and examples

---

## Version History Summary

| Version | Date | Key Changes |
|---------|------|-------------|
| 0.3.0 | 2026-02-18 | SR Engine v3.0, M23-RLFN, training pipeline |
| 0.2.0 | 2025-12-01 | Caching, variants, Conv2D support |
| 0.1.0 | 2025-09-15 | Initial release |
