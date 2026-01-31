"""
Setup script for M23-Spectrum package
"""

from setuptools import setup, find_packages

with open("README_M23.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="m23-spectrum",
    version="0.1.0",
    author="M23-Spectrum Contributors",
    author_email="contact@m23spectrum.dev",
    description="Algebraic weight initialization for deep neural networks using M23 spectrum",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/m23-spectrum",
    packages=find_packages(),
    py_modules=["m23_spectrum"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
        "scipy>=1.5.0",
    ],
    extras_require={
        "torch": ["torch>=1.9.0"],
        "tensorflow": ["tensorflow>=2.4.0"],
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.10.0",
            "black>=21.0",
            "flake8>=3.9.0",
            "mypy>=0.900",
        ],
    },
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/m23-spectrum/issues",
        "Documentation": "https://github.com/yourusername/m23-spectrum",
        "Source Code": "https://github.com/yourusername/m23-spectrum",
    },
    keywords=[
        "neural-networks",
        "weight-initialization",
        "deep-learning",
        "dynamic-isometry",
        "mathieu-groups",
        "super-resolution",
        "frame-generation",
        "transformer",
    ],
)