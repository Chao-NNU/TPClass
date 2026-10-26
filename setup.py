"""Setup script for TPClass.

Note: This file is kept for backward compatibility. The primary build
configuration is in pyproject.toml. New tooling should use pyproject.toml.
"""
import os

from setuptools import find_packages, setup

# Read long description from README
this_dir = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_dir, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="tpclass-bio",
    version="1.0.0",
    description="TPClass: A Two-Phase Predictor for Membrane Transporter Annotation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="TPClass Development Team",
    license="MIT",
    packages=find_packages(exclude=["tests*", "docs*", "examples*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.40.0",
        "sentencepiece>=0.2.0",
        "numpy>=1.24",
        "pandas>=2.0",
        "scipy>=1.10",
        "scikit-learn>=1.3",
        "xgboost>=2.0",
        "einops>=0.7",
        "biopython>=1.81",
        "joblib>=1.3",
        "matplotlib>=3.7",
    ],
    extras_require={
        "dev": ["pytest>=7.0", "pytest-cov>=4.0", "black>=23.0", "flake8>=6.0"],
        "gpu": ["faiss-gpu-cu12"],
        "esm": ["fair-esm>=2.0.0"],
        "tools": ["s4pred"],
    },
    entry_points={
        "console_scripts": [
            "tpclass=tpclass.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    include_package_data=True,
)
