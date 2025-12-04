"""Tests for filesystem platform functions."""

import os
import tempfile

import pytest

from east_py_std.fs import (
    fs_append_file_impl,
    fs_create_directory_impl,
    fs_delete_file_impl,
    fs_exists_impl,
    fs_is_directory_impl,
    fs_is_file_impl,
    fs_read_directory_impl,
    fs_read_file_bytes_impl,
    fs_read_file_impl,
    fs_write_file_bytes_impl,
    fs_write_file_impl,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


def test_fs_write_and_read_file(temp_dir):
    """Test writing and reading a file."""
    file_path = os.path.join(temp_dir, "test.txt")
    content = "Hello, World!"

    fs_write_file_impl(file_path, content)
    result = fs_read_file_impl(file_path)

    assert result == content


def test_fs_write_file_overwrites(temp_dir):
    """Test writing to existing file overwrites."""
    file_path = os.path.join(temp_dir, "test.txt")

    fs_write_file_impl(file_path, "First")
    fs_write_file_impl(file_path, "Second")
    result = fs_read_file_impl(file_path)

    assert result == "Second"


def test_fs_append_file(temp_dir):
    """Test appending to a file."""
    file_path = os.path.join(temp_dir, "test.txt")

    fs_write_file_impl(file_path, "Line 1\n")
    fs_append_file_impl(file_path, "Line 2\n")
    result = fs_read_file_impl(file_path)

    assert result == "Line 1\nLine 2\n"


def test_fs_append_file_creates(temp_dir):
    """Test append creates file if it doesn't exist."""
    file_path = os.path.join(temp_dir, "test.txt")

    fs_append_file_impl(file_path, "Content")
    result = fs_read_file_impl(file_path)

    assert result == "Content"


def test_fs_delete_file(temp_dir):
    """Test deleting a file."""
    file_path = os.path.join(temp_dir, "test.txt")

    fs_write_file_impl(file_path, "Content")
    assert fs_exists_impl(file_path)

    fs_delete_file_impl(file_path)
    assert not fs_exists_impl(file_path)


def test_fs_exists_file(temp_dir):
    """Test fs_exists returns True for existing file."""
    file_path = os.path.join(temp_dir, "test.txt")

    assert not fs_exists_impl(file_path)

    fs_write_file_impl(file_path, "Content")
    assert fs_exists_impl(file_path)


def test_fs_exists_directory(temp_dir):
    """Test fs_exists returns True for existing directory."""
    assert fs_exists_impl(temp_dir)


def test_fs_is_file(temp_dir):
    """Test fs_is_file distinguishes files."""
    file_path = os.path.join(temp_dir, "test.txt")
    fs_write_file_impl(file_path, "Content")

    assert fs_is_file_impl(file_path)
    assert not fs_is_file_impl(temp_dir)
    assert not fs_is_file_impl("/nonexistent")


def test_fs_is_directory(temp_dir):
    """Test fs_is_directory distinguishes directories."""
    file_path = os.path.join(temp_dir, "test.txt")
    fs_write_file_impl(file_path, "Content")

    assert fs_is_directory_impl(temp_dir)
    assert not fs_is_directory_impl(file_path)
    assert not fs_is_directory_impl("/nonexistent")


def test_fs_create_directory(temp_dir):
    """Test creating a directory."""
    dir_path = os.path.join(temp_dir, "newdir")

    fs_create_directory_impl(dir_path)
    assert fs_is_directory_impl(dir_path)


def test_fs_create_directory_recursive(temp_dir):
    """Test creating nested directories."""
    dir_path = os.path.join(temp_dir, "a", "b", "c")

    fs_create_directory_impl(dir_path)
    assert fs_is_directory_impl(dir_path)


def test_fs_create_directory_exists(temp_dir):
    """Test creating directory that already exists."""
    dir_path = os.path.join(temp_dir, "newdir")

    fs_create_directory_impl(dir_path)
    fs_create_directory_impl(dir_path)  # Should not raise
    assert fs_is_directory_impl(dir_path)


def test_fs_read_directory(temp_dir):
    """Test reading directory contents."""
    # Create some files
    fs_write_file_impl(os.path.join(temp_dir, "file1.txt"), "")
    fs_write_file_impl(os.path.join(temp_dir, "file2.txt"), "")
    fs_create_directory_impl(os.path.join(temp_dir, "subdir"))

    result = fs_read_directory_impl(temp_dir)

    assert isinstance(result, list) or hasattr(result, "__iter__")
    entries = list(result)
    assert "file1.txt" in entries
    assert "file2.txt" in entries
    assert "subdir" in entries


def test_fs_read_directory_empty(temp_dir):
    """Test reading empty directory."""
    empty_dir = os.path.join(temp_dir, "empty")
    fs_create_directory_impl(empty_dir)

    result = fs_read_directory_impl(empty_dir)
    assert list(result) == []


def test_fs_write_and_read_bytes(temp_dir):
    """Test writing and reading binary data."""
    file_path = os.path.join(temp_dir, "test.bin")
    content = b"\x00\x01\x02\x03\xff"

    fs_write_file_bytes_impl(file_path, content)
    result = fs_read_file_bytes_impl(file_path)

    assert result == content


def test_fs_write_bytes_overwrites(temp_dir):
    """Test writing bytes to existing file overwrites."""
    file_path = os.path.join(temp_dir, "test.bin")

    fs_write_file_bytes_impl(file_path, b"First")
    fs_write_file_bytes_impl(file_path, b"Second")
    result = fs_read_file_bytes_impl(file_path)

    assert result == b"Second"


def test_fs_read_file_unicode(temp_dir):
    """Test reading and writing unicode content."""
    file_path = os.path.join(temp_dir, "test.txt")
    content = "Hello 世界 🌍"

    fs_write_file_impl(file_path, content)
    result = fs_read_file_impl(file_path)

    assert result == content
