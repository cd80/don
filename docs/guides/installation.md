# Installation Guide

This guide walks you through the process of setting up the Bitcoin Trading RL project for development or production use.

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional, but recommended)
- Docker (optional)
- Git

## System Requirements

- **CPU**: 4+ cores recommended
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: 20GB minimum
- **GPU**: NVIDIA GPU with 8GB+ VRAM (optional)

## Installation Methods

Choose one of the following installation methods:

### Method 1: Using Make (Recommended for Development)

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/bitcoin_trading_rl.git
   cd bitcoin_trading_rl
   ```

2. Initialize the project:

   ```bash
   make init
   ```

   This command will:

   - Set up a virtual environment
   - Install dependencies
   - Set up pre-commit hooks
   - Initialize the development environment

3. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```

### Method 2: Using Docker (Recommended for Production)

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/bitcoin_trading_rl.git
   cd bitcoin_trading_rl
   ```

2. Build and run Docker containers:
   ```bash
   make docker-build
   make docker-run
   ```

### Method 3: Manual Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/bitcoin_trading_rl.git
   cd bitcoin_trading_rl
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

1. Copy the example configuration:

   ```bash
   cp configs/config.example.yaml configs/config.yaml
   ```

2. Edit the configuration file:

   ```yaml
   # configs/config.yaml

   # API Keys
   sentiment:
     news:
       apis:
         - name: "newsapi"
           key: "${NEWSAPI_KEY}"
     social:
       twitter:
         api_key: "${TWITTER_API_KEY}"
         api_secret: "${TWITTER_API_SECRET}"

   # Hardware Settings
   model:
     hardware:
       device: "auto" # auto, cuda, cpu
       num_gpus: -1 # -1 for all available
   ```

3. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

## Verification

1. Run tests:

   ```bash
   make test
   ```

2. Run linting checks:

   ```bash
   make lint
   ```

3. Check documentation:
   ```bash
   make docs-serve
   # Visit http://localhost:8000
   ```

## Common Issues

### CUDA Installation

If you're using GPU acceleration:

1. Install CUDA Toolkit:

   ```bash
   # On Ubuntu
   sudo apt install nvidia-cuda-toolkit

   # On macOS with M1/M2
   # GPU acceleration is handled by Metal, no CUDA needed
   ```

2. Verify CUDA installation:
   ```bash
   nvidia-smi
   ```

### Dependencies Issues

If you encounter dependency issues:

1. Update pip:

   ```bash
   python -m pip install --upgrade pip
   ```

2. Clear pip cache:

   ```bash
   pip cache purge
   ```

3. Reinstall dependencies:
   ```bash
   pip install -r requirements.txt --no-cache-dir
   ```

### Docker Issues

If Docker containers fail to start:

1. Check Docker service:

   ```bash
   sudo systemctl status docker
   ```

2. Check container logs:
   ```bash
   docker-compose logs
   ```

## Development Setup

For development work:

1. Install development dependencies:

   ```bash
   make dev-setup
   ```

2. Set up pre-commit hooks:

   ```bash
   make hooks
   ```

3. Start development services:
   ```bash
   make tensorboard  # Monitor training
   make jupyter     # Run notebooks
   ```

## Production Setup

For production deployment:

1. Build production Docker image:

   ```bash
   make ci-build
   ```

2. Set up monitoring:

   ```bash
   # Configure monitoring in configs/config.yaml
   monitoring:
     enabled: true
     interval: 60
     metrics:
       - cpu_usage
       - memory_usage
       - gpu_usage
   ```

3. Deploy:
   ```bash
   make docker-run
   ```

## Next Steps

- Follow the [Quick Start Guide](quickstart.md)
- Read about [Feature Engineering](feature-engineering.md)
- Learn about [Sentiment Analysis](sentiment-analysis.md)
- Explore [Model Architecture](model-architecture.md)

## Support

If you encounter any issues:

1. Check the [Common Issues](#common-issues) section
2. Search existing [GitHub Issues](https://github.com/yourusername/bitcoin_trading_rl/issues)
3. Create a new issue with:
   - System information
   - Error messages
   - Steps to reproduce
   - Expected vs actual behavior
