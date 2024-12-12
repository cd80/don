"""Test cases for Don trading framework rich logging functionality."""

import pytest
from unittest.mock import Mock, patch

from don.cli.logging import (console, get_progress, init_logging, log_error,
                           log_info, log_success, log_warning, status)

def test_log_info():
    """Test info logging with rich formatting."""
    with patch('don.cli.logging.console.print') as mock_print:
        log_info("Test message")
        mock_print.assert_called_once_with("[info]Test message[/info]")

def test_log_success():
    """Test success logging with rich formatting."""
    with patch('don.cli.logging.console.print') as mock_print:
        log_success("Test message")
        mock_print.assert_called_once_with("[success]Test message[/success]")

def test_log_warning():
    """Test warning logging with rich formatting."""
    with patch('don.cli.logging.console.print') as mock_print:
        log_warning("Test message")
        mock_print.assert_called_once_with("[warning]Test message[/warning]")

def test_log_error():
    """Test error logging with rich formatting."""
    with patch('don.cli.logging.console.print') as mock_print:
        log_error("Test message")
        mock_print.assert_called_once_with("[error]Test message[/error]")

def test_get_progress():
    """Test progress bar creation."""
    progress = get_progress()
    assert progress is not None
    assert hasattr(progress, 'add_task')

def test_status():
    """Test status message creation."""
    with patch('don.cli.logging.console.status') as mock_status:
        status("Test status")
        mock_status.assert_called_once_with("Test status", spinner="dots")

def test_init_logging():
    """Test logging initialization."""
    with patch('don.cli.logging.logger') as mock_logger:
        init_logging(debug=True)
        mock_logger.setLevel.assert_called_once()
