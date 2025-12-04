"""Tests for console platform functions."""

from east_py_std.console import console_error_impl, console_log_impl, console_write_impl


def test_console_log(capsys):
    """Test console_log writes to stdout with newline."""
    console_log_impl("Hello, World!")
    captured = capsys.readouterr()
    assert captured.out == "Hello, World!\n"
    assert captured.err == ""


def test_console_error(capsys):
    """Test console_error writes to stderr with newline."""
    console_error_impl("Error message")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "Error message\n"


def test_console_write(capsys):
    """Test console_write writes to stdout without newline."""
    console_write_impl("No newline")
    captured = capsys.readouterr()
    assert captured.out == "No newline"
    assert captured.err == ""


def test_console_write_multiple(capsys):
    """Test multiple console_write calls."""
    console_write_impl("Part 1 ")
    console_write_impl("Part 2")
    captured = capsys.readouterr()
    assert captured.out == "Part 1 Part 2"
