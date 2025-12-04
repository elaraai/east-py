"""Tests for path platform functions."""

import os

from east.types.types import StringType
from east.types.values import EastArray

from east_py_std.path import (
    path_basename_impl,
    path_dirname_impl,
    path_extname_impl,
    path_join_impl,
    path_resolve_impl,
)


def test_path_join():
    """Test path_join combines segments."""
    segments = EastArray(StringType, ["dir", "subdir", "file.txt"])
    result = path_join_impl(segments)
    expected = os.path.join("dir", "subdir", "file.txt")
    assert result == expected


def test_path_join_single():
    """Test path_join with single segment."""
    segments = EastArray(StringType, ["file.txt"])
    result = path_join_impl(segments)
    assert result == "file.txt"


def test_path_join_empty():
    """Test path_join with empty array."""
    segments = EastArray(StringType, [])
    result = path_join_impl(segments)
    assert result == "."


def test_path_resolve():
    """Test path_resolve creates absolute path."""
    result = path_resolve_impl("file.txt")
    assert os.path.isabs(result)
    assert result.endswith("file.txt")


def test_path_resolve_absolute():
    """Test path_resolve with already absolute path."""
    abs_path = "C:\\test\\file.txt" if os.name == "nt" else "/test/file.txt"

    result = path_resolve_impl(abs_path)
    assert os.path.isabs(result)


def test_path_dirname():
    """Test path_dirname extracts directory."""
    result = path_dirname_impl("/home/user/documents/file.txt")
    assert result == "/home/user/documents"


def test_path_dirname_no_directory():
    """Test path_dirname with just filename."""
    result = path_dirname_impl("file.txt")
    assert result == ""


def test_path_dirname_trailing_slash():
    """Test path_dirname with trailing slash."""
    result = path_dirname_impl("/home/user/documents/")
    assert result == "/home/user/documents"


def test_path_basename():
    """Test path_basename extracts filename."""
    result = path_basename_impl("/home/user/documents/file.txt")
    assert result == "file.txt"


def test_path_basename_no_directory():
    """Test path_basename with just filename."""
    result = path_basename_impl("file.txt")
    assert result == "file.txt"


def test_path_basename_directory():
    """Test path_basename with directory."""
    result = path_basename_impl("/home/user/documents")
    assert result == "documents"


def test_path_extname():
    """Test path_extname extracts extension."""
    result = path_extname_impl("/home/user/documents/file.txt")
    assert result == ".txt"


def test_path_extname_no_extension():
    """Test path_extname with no extension."""
    result = path_extname_impl("/home/user/documents/README")
    assert result == ""


def test_path_extname_multiple_dots():
    """Test path_extname with multiple dots."""
    result = path_extname_impl("file.tar.gz")
    assert result == ".gz"  # Only returns last extension


def test_path_extname_hidden_file():
    """Test path_extname with hidden file."""
    result = path_extname_impl(".bashrc")
    assert result == ""  # No extension
