"""
setup.py - SecureLens Package Installation
Run: pip install -e .
"""

from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="securelens",
    version="1.0.0",
    author="Amartya",
    description="Privacy-preserving medical image diagnostics using Fully Homomorphic Encryption",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/securelens",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security :: Cryptography",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=9.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "flake8>=6.0",
            "pylint>=2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "securelens=app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "client": ["templates/*", "static/css/*", "static/js/*"],
        "cloud_server": ["models/*"],
        "data": ["chest_xray/**/*"],
    },
    keywords="homomorphic-encryption privacy machine-learning medical-imaging chest-xray",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/securelens/issues",
        "Source": "https://github.com/yourusername/securelens",
        "Documentation": "https://github.com/yourusername/securelens/blob/main/README.md",
    },
)
