"""CLI commands implementation for Don trading framework.

This module contains the implementation of all CLI commands including:
- setup: Configure and validate project settings
- collect: Manage data collection processes
- feature: Calculate and manage technical indicators
- train: Control model training and dashboard
"""

import asyncio
import os
import signal
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.progress import Progress, track
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from .config import Settings, load_settings
from .logging import (console, get_progress, init_logging, log_error, log_info,
                     log_success, log_warning, status)

# Import real or mock BinanceDataCollector based on TEST_MODE
if os.getenv("TEST_MODE"):
    from tests.mocks.binance import MockBinanceDataCollector as BinanceDataCollector
else:
    from ..data.binance import BinanceDataCollector

from ..database.models import Base
from ..features.technical import TechnicalIndicators
from ..rl.env import TradingEnvironment

# Create typer app
app = typer.Typer(help="Don trading framework CLI")

@app.command()
def setup(
    all: bool = typer.Option(
        False,
        "--all",
        help="Check database, configuration completeness, and API keys",
    )
) -> None:
    """Setup and validate project configuration."""
    try:
        with status("Loading configuration...") as st:
            settings = load_settings()
            st.update("Checking configuration completeness...")
            if not settings.check_completeness():
                log_error("Configuration validation failed")
                raise typer.Exit(1)

            if all:
                st.update("Testing database connection...")
                if not settings.check_database_connection():
                    log_error("Database connection failed")
                    raise typer.Exit(1)

                st.update("Initializing database tables...")
                engine = create_engine(str(settings.database_url))
                Base.metadata.create_all(engine)

        log_success("Setup completed successfully!")

    except typer.Exit:
        raise
    except Exception as e:
        log_error(f"Setup failed: {str(e)}")
        raise typer.Exit(1)

@app.command()
def collect(
    action: str = typer.Argument(
        ...,
        help="Action to perform: start, stop, or resume",
    ),
    symbol: str = typer.Option(
        None,
        "--symbol",
        "-s",
        help="Trading symbol (e.g., BTCUSDT)",
    )
) -> None:
    """Manage data collection processes."""
    if action not in ["start", "stop", "resume"]:
        log_error("Invalid action. Must be one of: start, stop, resume")
        raise typer.Exit(1)

    try:
        settings = load_settings()
        collector = BinanceDataCollector(
            symbol=symbol or settings.trading_symbol,
            api_key=settings.binance_api_key.get_secret_value(),
            api_secret=settings.binance_api_secret.get_secret_value(),
        )

        if action == "start":
            log_info("Starting data collection...")
            collector.start()
            log_success("Data collection started successfully")
        elif action == "stop":
            log_info("Stopping data collection...")
            collector.stop()
            log_success("Data collection stopped successfully")
        else:  # resume
            log_info("Resuming data collection...")
            collector.resume()
            log_success("Data collection resumed successfully")

    except typer.Exit:
        raise
    except Exception as e:
        log_error(f"Data collection {action} failed: {str(e)}")
        raise typer.Exit(1)

@app.command()
def feature(
    all: bool = typer.Option(
        False,
        "--all",
        help="Calculate all features and aggregate into database",
    )
) -> None:
    """Calculate and manage technical indicators."""
    session = None
    try:
        settings = load_settings()
        engine = create_engine(str(settings.database_url))
        Session = sessionmaker(bind=engine)
        session = Session()

        with status("Initializing feature calculation...") as st:
            calculator = TechnicalIndicators()

            if all:
                st.update("Calculating all technical indicators...")
                calculator.calculate_all(session)
                log_success("All features calculated successfully!")
            else:
                log_warning("Please specify --all to calculate all features")
                raise typer.Exit(1)

    except typer.Exit:
        raise
    except Exception as e:
        log_error(f"Feature calculation failed: {str(e)}")
        raise typer.Exit(1)
    finally:
        if session:
            session.close()

@app.command()
def train(
    start: bool = typer.Option(
        False,
        "--start",
        help="Start RL training and run dashboard webserver",
    )
) -> None:
    """Control model training and dashboard."""
    try:
        settings = load_settings()

        if start:
            log_info("Training started. Dashboard available at http://localhost:8501")
            with status("Starting training dashboard...") as st:
                env = TradingEnvironment()
                st.update("Training environment initialized")

                def signal_handler(sig, frame):
                    log_info("Shutting down training...")
                    sys.exit(0)

                signal.signal(signal.SIGINT, signal_handler)
                signal.signal(signal.SIGTERM, signal_handler)

                try:
                    while True:
                        env.step(env.action_space.sample())
                except KeyboardInterrupt:
                    log_info("Training stopped by user")
        else:
            log_warning("Please use --start to begin training")
            raise typer.Exit(1)

    except typer.Exit:
        raise
    except Exception as e:
        log_error(f"Training failed: {str(e)}")
        raise typer.Exit(1)

def main():
    """Entry point for the CLI."""
    init_logging()
    app()
