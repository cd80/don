# Don Trading Framework CLI

## Overview
The Don trading framework provides a comprehensive command-line interface (CLI) for managing Bitcoin futures trading operations. This document describes the available commands and their usage.

## Commands

### Setup
```bash
don setup --all
```
Check and validate the project setup:
- Verifies configuration file completeness
- Validates Binance API keys
- Tests PostgreSQL database connection
- Initializes database tables

Options:
- `--all`: Perform complete setup validation

### Data Collection
```bash
don collect [action]
```
Manage data collection processes for market data:
- `start`: Begin collecting market data
- `stop`: Stop the data collection process
- `resume`: Resume a stopped collection process

Examples:
```bash
don collect start   # Start data collection
don collect stop    # Stop data collection
don collect resume  # Resume data collection
```

### Feature Calculation
```bash
don feature --all
```
Calculate and aggregate technical indicators:
- Processes collected market data
- Calculates technical indicators
- Aggregates results into database tables

Options:
- `--all`: Calculate all available features

### Training
```bash
don train --start
```
Manage reinforcement learning model training:
- Starts the training process
- Launches dashboard webserver in background
- Provides real-time training metrics

Options:
- `--start`: Begin training and launch dashboard

## Error Handling
The CLI provides rich logging output with color-coded messages:
- Green: Success messages
- Yellow: Warnings
- Red: Error messages
- Cyan: Information messages

## Process Management
Background processes (data collection and training) are managed automatically:
- PID tracking for process management
- Graceful shutdown handling
- Process status monitoring
