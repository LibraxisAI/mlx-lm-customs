#!/usr/bin/env python3
# Copyright © 2023-2025 Apple Inc.

from setuptools import setup, find_packages

setup(
    name="mlx-mergekit",
    version="0.1.0",
    description="MergeKit for MLX - Advanced model merging techniques for Apple Silicon",
    long_description=open("mergekit/readme.md").read(),
    long_description_content_type="text/markdown",
    author="Szowesgad",
    author_email="szowesgad@example.com",
    url="https://github.com/Szowesgad/mlx-lm",
    packages=find_packages(),
    include_package_data=True,
    scripts=["mergekit/bin/mergekit"],
    install_requires=[
        "mlx>=0.3.0",
        "mlx-lm>=0.2.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
        "huggingface_hub>=0.16.0",
        "hf-transfer>=0.1.3",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "psutil>=5.9.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
)
