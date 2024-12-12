"""Process management for Don trading framework.

This module provides process management functionality for running
data collection and training processes in the background.
"""

import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Optional

import psutil
from rich.console import Console

from ..cli.logging import log_error, log_info, log_success, log_warning

console = Console()

class ProcessManager:
    """Manage background processes for data collection and training."""

    def __init__(self, name: str, pid_dir: Path = Path("/tmp/don")):
        """Initialize process manager.

        Args:
            name: Process name for identification
            pid_dir: Directory to store PID files
        """
        self.name = name
        self.pid_dir = pid_dir
        self.pid_file = self.pid_dir / f"{name}.pid"
        self.pid_dir.mkdir(parents=True, exist_ok=True)

    def _read_pid(self) -> Optional[int]:
        """Read PID from file if it exists."""
        try:
            if self.pid_file.exists():
                return int(self.pid_file.read_text().strip())
            return None
        except (ValueError, IOError) as e:
            log_error(f"Error reading PID file: {str(e)}")
            return None

    def _write_pid(self, pid: int) -> None:
        """Write PID to file."""
        try:
            self.pid_file.write_text(str(pid))
        except IOError as e:
            log_error(f"Error writing PID file: {str(e)}")
            raise

    def _remove_pid(self) -> None:
        """Remove PID file."""
        try:
            if self.pid_file.exists():
                self.pid_file.unlink()
        except IOError as e:
            log_error(f"Error removing PID file: {str(e)}")

    def is_running(self) -> bool:
        """Check if process is running."""
        pid = self._read_pid()
        if pid is None:
            return False

        try:
            process = psutil.Process(pid)
            return process.is_running() and process.name() == self.name
        except psutil.NoSuchProcess:
            self._remove_pid()
            return False

    def start(self, command: list[str], env: Optional[dict] = None) -> None:
        """Start a new background process.

        Args:
            command: Command to run as list of strings
            env: Optional environment variables
        """
        if self.is_running():
            log_warning(f"Process {self.name} is already running")
            return

        try:
            process = subprocess.Popen(
                command,
                env={**os.environ, **(env or {})},
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
            self._write_pid(process.pid)
            log_success(f"Started {self.name} process (PID: {process.pid})")

        except subprocess.SubprocessError as e:
            log_error(f"Failed to start {self.name} process: {str(e)}")
            raise

    def stop(self, timeout: int = 5) -> None:
        """Stop the running process.

        Args:
            timeout: Seconds to wait for graceful shutdown
        """
        pid = self._read_pid()
        if pid is None:
            log_warning(f"No PID file found for {self.name}")
            return

        try:
            process = psutil.Process(pid)
            if not process.is_running():
                log_warning(f"Process {self.name} is not running")
                self._remove_pid()
                return

            # Send SIGTERM for graceful shutdown
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=timeout)
            except psutil.TimeoutExpired:
                # Force kill if timeout expires
                process.kill()
                log_warning(f"Force killed {self.name} process")
            else:
                log_success(f"Stopped {self.name} process")

        except psutil.NoSuchProcess:
            log_warning(f"Process {self.name} not found")
        finally:
            self._remove_pid()

    def resume(self, command: list[str], env: Optional[dict] = None) -> None:
        """Resume a stopped process.

        Args:
            command: Command to run as list of strings
            env: Optional environment variables
        """
        if self.is_running():
            log_warning(f"Process {self.name} is already running")
            return

        self.start(command, env)
        log_success(f"Resumed {self.name} process")
