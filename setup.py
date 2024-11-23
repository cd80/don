from setuptools import setup, find_packages

setup(
    name="bitcoin_trading_rl",
    version="0.1.0",
    description="Deep Reinforcement Learning for Bitcoin Trading",
    author="Don Cline",
    packages=find_packages(),
    install_requires=[
        "tensorflow>=2.8.0",
        "torch>=1.10.0",
        "transformers>=4.15.0",
        "ray[rllib]>=1.9.0",
        "optuna>=2.10.0",
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "pyarrow>=6.0.0",
        "h5py>=3.6.0",
        "dask>=2021.12.0",
        "ta>=0.7.0",
        "ta-lib>=0.4.24",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.2",
        "plotly>=5.4.0",
        "dash>=2.0.0",
        "python-binance>=1.0.15",
        "ccxt>=1.60.0",
        "requests>=2.26.0",
        "aiohttp>=3.8.0",
        "joblib>=1.1.0",
        "multiprocess>=0.70.12",
        "distributed>=2021.12.0",
        "pytest>=6.2.5",
        "hypothesis>=6.30.0",
        "wandb>=0.12.0",
        "tensorboard>=2.7.0",
        "python-json-logger>=2.0.2",
        "tqdm>=4.62.0",
        "python-dotenv>=0.19.0",
        "pyyaml>=6.0",
        "shap>=0.40.0",
        "lime>=0.2.0"
    ],
    extras_require={
        'dev': [
            'pytest>=6.2.5',
            'pytest-cov>=2.12.1',
            'flake8>=3.9.0',
            'black>=21.5b2',
            'isort>=5.9.1',
            'mypy>=0.910',
            'pre-commit>=2.15.0'
        ]
    },
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'btrl=src.main:main',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
