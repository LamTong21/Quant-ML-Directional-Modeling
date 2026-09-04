from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="emerging-equities-directional-ml",
    version="1.0.0",
    author="Lam Tong",
    description="Exogenous Context, Cross-Fold Consensus, and Leak-Free Directional Modeling in Emerging Equities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LamTong21/Emerging-Equities-Directional-ML",
    packages=find_packages(include=["src", "src.*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    python_requires=">=3.10",
    install_requires=[
        "vnstock>=4.0.7",
        "numpy>=1.24.0,<2.3.0",
        "pandas>=2.2.0",
        "scipy>=1.11.0",
        "statsmodels>=0.14.0",
        "scikit-learn>=1.4.0",
        "xgboost>=2.0.0",
        "optuna>=3.6.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
        "PyYAML>=6.0.0",
    ],
    extras_require={
        "dev": ["pytest>=8.0.0", "black", "ruff"],
    },
)