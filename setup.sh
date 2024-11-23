#!/bin/bash

# Exit on error
set -e

echo "Setting up Bitcoin Trading RL project environment..."

# Create and activate virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -e ".[dev]"

# Create necessary directories if they don't exist
echo "Creating project directories..."
mkdir -p data/{raw,processed,external}
mkdir -p results/{logs,checkpoints,evaluation}
mkdir -p notebooks

# Install pre-commit hooks
echo "Setting up pre-commit hooks..."
pre-commit install

# Initialize git if not already initialized
if [ ! -d ".git" ]; then
    echo "Initializing git repository..."
    git init
    
    # Create .gitignore
    echo "Creating .gitignore..."
    cat > .gitignore << EOL
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
ENV/

# IDE
.idea/
.vscode/
*.swp
*.swo

# Project specific
data/raw/*
data/processed/*
data/external/*
results/logs/*
results/checkpoints/*
results/evaluation/*
!data/raw/.gitkeep
!data/processed/.gitkeep
!data/external/.gitkeep
!results/logs/.gitkeep
!results/checkpoints/.gitkeep
!results/evaluation/.gitkeep

# Jupyter Notebook
.ipynb_checkpoints
*.ipynb

# Environment variables
.env

# Logs
*.log
EOL
    
    # Create empty .gitkeep files to track empty directories
    touch data/raw/.gitkeep
    touch data/processed/.gitkeep
    touch data/external/.gitkeep
    touch results/logs/.gitkeep
    touch results/checkpoints/.gitkeep
    touch results/evaluation/.gitkeep
    
    # Initial commit
    git add .
    git commit -m "Initial commit"
fi

# Make script executable
chmod +x setup.sh

echo "Setup completed successfully!"
echo "To activate the virtual environment, run: source venv/bin/activate"
echo "To start training, run: python src/main.py"
