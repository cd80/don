"""Rich logging configuration for Don trading framework.

This module sets up rich logging handlers and provides logging utilities
for enhanced console output and progress tracking.
"""

import logging
import sys
from typing import Any, Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.status import Status
from rich.theme import Theme

THEME = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "red bold",
    "success": "green bold",
})

console = Console(theme=THEME)

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(
        console=console,
        show_path=False,
        enable_link_path=False,
        rich_tracebacks=True,
        tracebacks_show_locals=True,
    )]
)

logger = logging.getLogger("don")

def get_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    )

def status(message: str) -> Status:
    return console.status(message, spinner="dots")

def log_info(message: str, **kwargs: Any) -> None:
    console.print(f"[info]{message}[/info]", **kwargs)

def log_success(message: str, **kwargs: Any) -> None:
    console.print(f"[success]{message}[/success]", **kwargs)

def log_warning(message: str, **kwargs: Any) -> None:
    console.print(f"[warning]{message}[/warning]", **kwargs)

def log_error(message: str, **kwargs: Any) -> None:
    console.print(f"[error]{message}[/error]", **kwargs)

def init_logging(debug: bool = False) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logger.setLevel(level)

    for handler in logger.handlers:
        handler.setLevel(level)
