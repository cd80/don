"""CLI commands implementation for Don trading framework.

This module contains the implementation of all CLI commands including:
- setup: Configure and validate project settings
- collect: Manage data collection processes
- feature: Calculate and manage technical indicators
- train: Control model training and dashboard
"""

import os
import sys
import time
import signal
import subprocess
import multiprocessing
from typing import Optional

import typer
from rich.progress import Progress, track
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
import pandas as pd

from .config import Settings, load_settings
from .logging import (console, get_progress, init_logging, log_error, log_info,
                     log_success, log_warning, status)

# Import real or mock BinanceDataCollector based on TEST_MODE
if os.getenv("TEST_MODE"):
    from tests.mocks.binance import MockBinanceDataCollector as BinanceDataCollector
else:
    from ..data.binance import BinanceDataCollector

from ..database.models import Base, MarketData, TechnicalFeatures
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
    if not all:
        log_warning("Please specify --all to calculate all features")
        raise typer.Exit(1)

    session = None
    try:
        settings = load_settings()
        engine = create_engine(str(settings.database_url))
        Session = sessionmaker(bind=engine)
        session = Session()

        with status("Initializing feature calculation...") as st:
            # Get OHLCV data from database
            st.update("Fetching market data...")
            data = pd.read_sql_query(
                f"SELECT timestamp, open, high, low, close, volume FROM market_data WHERE symbol = '{settings.trading_symbol}' ORDER BY timestamp",
                session.bind,
                parse_dates=['timestamp']
            )

            if data.empty:
                log_warning("No market data found in database")
                raise typer.Exit(1)

            # Set timestamp as index after ensuring it's datetime
            data.set_index('timestamp', inplace=True)

            calculator = TechnicalIndicators()
            st.update("Calculating all technical indicators...")
            result = calculator.calculate_all(data)

            # Save results back to database
            st.update("Saving calculated features...")
            for timestamp, row in result.iterrows():
                feature = TechnicalFeatures(
                    timestamp=pd.Timestamp(timestamp),  # Ensure timestamp is properly converted
                    symbol=settings.trading_symbol,
                    sma_20=row['sma_20'] if pd.notna(row['sma_20']) else None,
                    rsi=row['rsi'] if pd.notna(row['rsi']) else None,
                    macd=row['macd'] if pd.notna(row['macd']) else None,
                    macd_signal=row['macd_signal'] if pd.notna(row['macd_signal']) else None,
                    macd_hist=row['macd_hist'] if pd.notna(row['macd_hist']) else None,
                    bb_upper=row['bb_upper'] if pd.notna(row['bb_upper']) else None,
                    bb_middle=row['bb_middle'] if pd.notna(row['bb_middle']) else None,
                    bb_lower=row['bb_lower'] if pd.notna(row['bb_lower']) else None,
                    obv=row['obv'] if pd.notna(row['obv']) else None,
                    vwap=row['vwap'] if pd.notna(row['vwap']) else None,
                    stoch_k=row['stoch_k'] if pd.notna(row['stoch_k']) else None,
                    stoch_d=row['stoch_d'] if pd.notna(row['stoch_d']) else None,
                    adx=row['adx'] if pd.notna(row['adx']) else None,
                    plus_di=row['plus_di'] if pd.notna(row['plus_di']) else None,
                    minus_di=row['minus_di'] if pd.notna(row['minus_di']) else None
                )
                session.add(feature)
            session.commit()
            log_success("All features calculated successfully!")

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
    start: bool = typer.Option(False, "--start", help="Start training and dashboard"),
):
    """Start RL training and run dashboard webserver in background."""
    if not start:
        log_error("Please use --start flag to start training")
        raise typer.Exit(1)

    try:
        # Set up signal handler for graceful shutdown
        def signal_handler(signum, frame):
            log_warning("\nTraining stopped by user")
            if 'dashboard_process' in globals():
                dashboard_process.terminate()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Load settings and initialize environment
        settings = load_settings()
        env = TradingEnvironment()

        # Start dashboard in background
        global dashboard_process
        dashboard_process = multiprocessing.Process(
            target=lambda: subprocess.run(
                ["streamlit", "run",
                 os.path.join(os.path.dirname(__file__), "..", "dashboard", "app.py"),
                 "--server.port", str(settings.dashboard_port),
                 "--server.address", settings.dashboard_host],
                check=True
            )
        )
        dashboard_process.start()
        log_success(f"Dashboard started at http://{settings.dashboard_host}:{settings.dashboard_port}")

        # Start training
        log_info("Training started")
        while True:
            # Placeholder training loop
            state = env.reset()
            action = env.action_space.sample()
            state, reward, done, info = env.step(action)
            time.sleep(1)  # Prevent CPU overload in test mode

    except KeyboardInterrupt:
        log_warning("\nTraining stopped by user")
        if 'dashboard_process' in globals():
            dashboard_process.terminate()
        raise typer.Exit(0)
    except Exception as e:
        log_error(f"Training failed: {str(e)}")
        if 'dashboard_process' in globals():
            dashboard_process.terminate()
        raise typer.Exit(1)

def main():
    """Entry point for the CLI."""
    init_logging()
    app()
