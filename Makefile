# Makefile for Bitcoin Trading RL project

.PHONY: setup clean test lint format docker-build docker-run docker-stop help status update-status progress docs ci-check ci-test ci-docs ci-security ci-build

# Variables
PYTHON := python3
PIP := pip3
DOCKER := docker
DOCKER_COMPOSE := docker-compose
PROJECT_NAME := bitcoin_trading_rl

help:
	@echo "Available commands:"
	@echo "Development:"
	@echo "  make setup         - Set up development environment"
	@echo "  make clean         - Clean up generated files"
	@echo "  make test          - Run tests"
	@echo "  make lint          - Run linting checks"
	@echo "  make format        - Format code"
	@echo "Docker:"
	@echo "  make docker-build  - Build Docker containers"
	@echo "  make docker-run    - Run Docker containers"
	@echo "  make docker-stop   - Stop Docker containers"
	@echo "Project Status:"
	@echo "  make status        - Show project status"
	@echo "  make update-status - Update project status"
	@echo "  make progress      - Generate progress report"
	@echo "Documentation:"
	@echo "  make docs          - Generate documentation"
	@echo "  make docs-serve    - Serve documentation locally"
	@echo "  make docs-deploy   - Deploy documentation"
	@echo "CI/CD:"
	@echo "  make ci-check      - Run all CI checks"
	@echo "  make ci-test       - Run CI tests"
	@echo "  make ci-docs       - Build documentation in CI"
	@echo "  make ci-security   - Run security checks"
	@echo "  make ci-build      - Build for production"
	@echo "Help:"
	@echo "  make help          - Show this help message"

setup:
	@echo "Setting up development environment..."
	chmod +x setup.sh
	./setup.sh

clean:
	@echo "Cleaning up generated files..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name "build" -exec rm -rf {} +
	find . -type d -name "dist" -exec rm -rf {} +
	rm -rf site/

test:
	@echo "Running tests..."
	pytest tests/ -v --cov=src --cov-report=term-missing --asyncio-mode=strict

lint:
	@echo "Running linting checks..."
	flake8 src/ tests/
	mypy src/ tests/
	pylint src/ tests/
	bandit -r src/

format:
	@echo "Formatting code..."
	black src/ tests/
	isort src/ tests/

docker-build:
	@echo "Building Docker containers..."
	$(DOCKER_COMPOSE) build

docker-run:
	@echo "Running Docker containers..."
	$(DOCKER_COMPOSE) up -d

docker-stop:
	@echo "Stopping Docker containers..."
	$(DOCKER_COMPOSE) down

# Project status and progress tracking
status:
	@echo "Current project status:"
	@cat PROJECT_STATUS.md

update-status:
	@echo "Updating project status..."
	./scripts/update_status.py

progress:
	@echo "Generating progress report..."
	./scripts/update_status.py
	@cat results/progress_report.txt

# Documentation
docs:
	@echo "Building documentation..."
	mkdocs build

docs-serve:
	@echo "Serving documentation locally..."
	mkdocs serve

docs-deploy:
	@echo "Deploying documentation..."
	mkdocs gh-deploy --force

# CI/CD Commands
ci-check: ci-test ci-docs ci-security ci-build
	@echo "All CI checks completed"

ci-test:
	@echo "Running CI tests..."
	pytest tests/ -v --cov=src --cov-report=xml --asyncio-mode=strict
	coverage report

ci-docs:
	@echo "Building documentation in CI..."
	mkdocs build --strict

ci-security:
	@echo "Running security checks..."
	bandit -r src/ -ll
	safety check
	pip-audit

ci-build:
	@echo "Building for production..."
	python setup.py sdist bdist_wheel
	$(DOCKER) build -t $(PROJECT_NAME):latest .

# Development commands
dev-setup: setup
	@echo "Installing development dependencies..."
	$(PIP) install -e ".[dev]"
	pre-commit install

dev-clean: clean
	@echo "Cleaning development files..."
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/

dev-test: test
	@echo "Running tests with coverage report..."
	pytest tests/ -v --cov=src --cov-report=html

dev-lint: lint
	@echo "Running additional development linting..."
	black --check src/ tests/
	isort --check-only src/ tests/

# Data management commands
data-download:
	@echo "Downloading data..."
	$(PYTHON) src/main.py --mode download

data-process:
	@echo "Processing data..."
	$(PYTHON) src/main.py --mode process

# Training commands
train:
	@echo "Starting model training..."
	$(PYTHON) src/main.py --mode train

evaluate:
	@echo "Evaluating model..."
	$(PYTHON) src/main.py --mode evaluate

# Monitoring commands
tensorboard:
	@echo "Starting TensorBoard..."
	tensorboard --logdir results/logs

jupyter:
	@echo "Starting Jupyter Notebook..."
	jupyter notebook --ip=0.0.0.0 --port=8888

# Environment management
create-env:
	@echo "Creating virtual environment..."
	$(PYTHON) -m venv venv

activate-env:
	@echo "To activate the virtual environment, run:"
	@echo "source venv/bin/activate"

update-deps:
	@echo "Updating dependencies..."
	$(PIP) install -U pip
	$(PIP) install -e ".[dev]"

# Git hooks
hooks:
	@echo "Setting up git hooks..."
	pre-commit install

# Project initialization
init: setup dev-setup hooks
	@echo "Project initialized successfully"

# Default target
all: setup test lint format
