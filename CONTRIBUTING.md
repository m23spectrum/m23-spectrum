# Contributing to M23-Spectrum

Thank you for your interest in contributing to M23-Spectrum! This document provides guidelines and instructions for contributing.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip or uv package manager

### Development Setup

1. **Fork and clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/m23-spectrum.git
   cd m23-spectrum
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies:**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Install PyTorch (optional, for SR Engine):**
   ```bash
   pip install torch torchvision
   ```

## Development Guidelines

### Code Style

- Follow PEP 8 style guidelines
- Use Black for code formatting (line length: 100)
- Use Ruff for linting
- Add type hints to all public functions

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=m23_spectrum --cov-report=html

# Run specific test file
pytest tests/test_m23_spectrum.py -v
```

### Code Formatting

```bash
# Format code
black .

# Check linting
ruff check .

# Type checking
mypy m23_spectrum.py
```

## Pull Request Process

1. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes:**
   - Write clean, documented code
   - Add tests for new functionality
   - Ensure all tests pass

3. **Commit your changes:**
   ```bash
   git add .
   git commit -m "feat: description of your changes"
   ```

   Use conventional commit messages:
   - `feat:` for new features
   - `fix:` for bug fixes
   - `docs:` for documentation changes
   - `test:` for test additions/modifications
   - `refactor:` for code refactoring
   - `chore:` for maintenance tasks

4. **Push and create PR:**
   ```bash
   git push origin feature/your-feature-name
   ```

5. **PR Requirements:**
   - All CI checks must pass
   - Code coverage should not decrease
   - New features need tests
   - Documentation updates if needed

## Adding New Features

### New Initialization Variants

To add a new initialization variant:

1. Add the variant name to the allowed list in `m23_initialize()`
2. Implement the variant logic
3. Add tests in `tests/test_m23_spectrum.py`
4. Update documentation in README.md

### New Loss Functions

To add a new loss function for SR:

1. Add the class to `m23_sr_engine.py`
2. Inherit from `nn.Module`
3. Add tests and documentation

## Reporting Issues

When reporting issues, please include:

- Python version
- NumPy/PyTorch versions
- Minimal reproducible example
- Expected vs actual behavior

## Questions?

Feel free to open an issue for questions or discussions about the project.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
