"""East Python CLI - Command-line interface for running East IR programs.

Usage:
    east-py run [--runtime PKG]... [--input FILE]... [--output FILE] <ir_file>
    east-py version
"""

from east_py_cli.cli import main

__version__ = "0.1.0"

__all__ = ["main", "__version__"]
