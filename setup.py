#!/usr/bin/env python3
"""
Setup script for German Credit Card Risk Analysis Project
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="german-credit-analysis",
    version="1.0.0",
    author="gundamking",
    author_email="latias001@gmail.com",
    description="A comprehensive machine learning analysis of German Credit Card dataset for credit risk assessment",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/gundamking/german-credit-analysis",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.7",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "myst-parser>=0.15",
        ],
    },
    entry_points={
        "console_scripts": [
            "german-credit-analysis=analysis:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.csv", "*.data", "*.doc", "*.pdf"],
    },
    keywords="machine-learning, credit-risk, data-analysis, finance, scikit-learn",
    project_urls={
        "Bug Reports": "https://github.com/gundamking/german-credit-analysis/issues",
        "Source": "https://github.com/gundamking/german-credit-analysis",
        "Documentation": "https://github.com/gundamking/german-credit-analysis#readme",
    },
) 