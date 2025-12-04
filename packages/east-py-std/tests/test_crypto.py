"""Tests for crypto platform functions."""

import re

import pytest

from east_py_std.crypto import (
    crypto_hash_sha256_bytes_impl,
    crypto_hash_sha256_impl,
    crypto_random_bytes_impl,
    crypto_uuid_impl,
)


def test_crypto_random_bytes():
    """Test crypto_random_bytes generates correct length."""
    result = crypto_random_bytes_impl(16)
    assert isinstance(result, bytes)
    assert len(result) == 16


def test_crypto_random_bytes_zero_length():
    """Test crypto_random_bytes with zero length."""
    result = crypto_random_bytes_impl(0)
    assert result == b""


def test_crypto_random_bytes_negative():
    """Test crypto_random_bytes rejects negative length."""
    with pytest.raises(ValueError, match="non-negative"):
        crypto_random_bytes_impl(-1)


def test_crypto_random_bytes_uniqueness():
    """Test crypto_random_bytes generates unique values."""
    bytes1 = crypto_random_bytes_impl(32)
    bytes2 = crypto_random_bytes_impl(32)
    assert bytes1 != bytes2  # Extremely unlikely to be equal


def test_crypto_hash_sha256():
    """Test SHA-256 hash of string."""
    result = crypto_hash_sha256_impl("hello")
    # Known SHA-256 hash of "hello"
    expected = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    assert result == expected


def test_crypto_hash_sha256_empty():
    """Test SHA-256 hash of empty string."""
    result = crypto_hash_sha256_impl("")
    # Known SHA-256 hash of empty string
    expected = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    assert result == expected


def test_crypto_hash_sha256_unicode():
    """Test SHA-256 hash with unicode characters."""
    result = crypto_hash_sha256_impl("Hello 世界")
    assert isinstance(result, str)
    assert len(result) == 64  # SHA-256 produces 64 hex characters
    assert re.match(r"^[0-9a-f]{64}$", result)


def test_crypto_hash_sha256_bytes():
    """Test SHA-256 hash of binary data."""
    result = crypto_hash_sha256_bytes_impl(b"hello")
    assert isinstance(result, bytes)
    assert len(result) == 32  # SHA-256 produces 32 bytes
    # Should match the hex version
    assert result.hex() == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"


def test_crypto_hash_sha256_bytes_empty():
    """Test SHA-256 hash of empty bytes."""
    result = crypto_hash_sha256_bytes_impl(b"")
    assert len(result) == 32
    assert result.hex() == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def test_crypto_uuid():
    """Test UUID generation."""
    result = crypto_uuid_impl()
    assert isinstance(result, str)
    assert len(result) == 36  # UUID format: 8-4-4-4-12
    # Check UUID v4 format
    assert re.match(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$", result
    )


def test_crypto_uuid_uniqueness():
    """Test UUIDs are unique."""
    uuid1 = crypto_uuid_impl()
    uuid2 = crypto_uuid_impl()
    assert uuid1 != uuid2
