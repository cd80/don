"""Entry point for Don trading framework CLI.

This module serves as the entry point for the CLI commands,
making them accessible through the 'don' command.
"""

from .cli.commands import app

if __name__ == "__main__":
    app()
