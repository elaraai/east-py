"""East primitive types.

This module provides Python implementations of East's primitive types:
- Null: Unit type (singleton value)
- Boolean: Python bool
- Integer: Python int (arbitrary precision)
- Float: Python float (IEEE 754 double)
- String: Python str (UTF-8)
- Blob: Immutable binary data
- DateTime: UTC-aware datetime
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any


class Null:
    """East's canonical unit type.

    Represents the absence of a value, analogous to None but with
    distinct type identity for East's type system.
    """

    _instance: Null | None = None

    def __new__(cls) -> Null:
        """Ensure Null is a singleton."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        """Return East text format representation."""
        return "null"

    def __str__(self) -> str:
        """Return East text format representation."""
        return "null"

    def __eq__(self, other: object) -> bool:
        """Null equals only itself."""
        return isinstance(other, Null)

    def __hash__(self) -> int:
        """Hash for use in sets/dicts."""
        return hash(None)

    def __lt__(self, other: object) -> bool:
        """Null is not less than anything (including itself)."""
        if not isinstance(other, Null):
            return NotImplemented
        return False

    def __le__(self, other: object) -> bool:
        """Null is less than or equal to itself."""
        if not isinstance(other, Null):
            return NotImplemented
        return True

    def __gt__(self, other: object) -> bool:
        """Null is not greater than anything."""
        if not isinstance(other, Null):
            return NotImplemented
        return False

    def __ge__(self, other: object) -> bool:
        """Null is greater than or equal to itself."""
        if not isinstance(other, Null):
            return NotImplemented
        return True


# Singleton instance
null = Null()


class Blob:
    """Immutable container for binary data.

    Wraps Python bytes to provide an immutable, indexable container
    with East-specific formatting (hexadecimal).
    """

    __slots__ = ("_data",)

    def __init__(self, data: bytes | list[int] | Blob):
        """Create a Blob from bytes, list of integers, or another Blob."""
        if isinstance(data, Blob):
            self._data = data._data
        elif isinstance(data, bytes):
            self._data = data
        elif isinstance(data, list):
            self._data = bytes(data)
        else:
            raise TypeError(f"Cannot create Blob from {type(data)}")

    @property
    def data(self) -> bytes:
        """Access underlying bytes (immutable)."""
        return self._data

    def __len__(self) -> int:
        """Return number of bytes."""
        return len(self._data)

    def __getitem__(self, index: int | slice) -> int | Blob:
        """Get byte at index or slice of bytes."""
        if isinstance(index, slice):
            return Blob(self._data[index])
        return self._data[index]

    def __eq__(self, other: object) -> bool:
        """Blobs are equal if their bytes are equal."""
        if not isinstance(other, Blob):
            return NotImplemented
        return self._data == other._data

    def __lt__(self, other: object) -> bool:
        """Lexicographic comparison of bytes."""
        if not isinstance(other, Blob):
            return NotImplemented
        return self._data < other._data

    def __hash__(self) -> int:
        """Hash based on bytes."""
        return hash(self._data)

    def __repr__(self) -> str:
        """Return East hexadecimal format."""
        if len(self._data) == 0:
            return "0x"
        # Limit display for very large blobs
        if len(self._data) > 256:
            hex_str = self._data[:256].hex()
            return f"0x{hex_str}..."
        return f"0x{self._data.hex()}"

    def __str__(self) -> str:
        """Return East hexadecimal format."""
        return repr(self)


def ensure_utc_datetime(dt: datetime) -> datetime:
    """Ensure datetime is UTC-aware.

    Args:
        dt: Datetime to convert

    Returns:
        UTC-aware datetime

    If the datetime is naive, it's assumed to be UTC.
    If it has a different timezone, it's converted to UTC.
    """
    if dt.tzinfo is None:
        # Naive datetime - assume UTC
        return dt.replace(tzinfo=UTC)
    if dt.tzinfo != UTC:
        # Different timezone - convert to UTC
        return dt.astimezone(UTC)
    # Already UTC
    return dt


def validate_east_value(value: Any, expected_type: str) -> None:
    """Validate that a Python value matches an expected East primitive type.

    Args:
        value: The value to validate
        expected_type: Expected type name (e.g., "Boolean", "Integer", "Float", "String")

    Raises:
        TypeError: If value doesn't match expected type
    """
    type_checks = {
        "Null": lambda v: isinstance(v, Null),
        "Boolean": lambda v: isinstance(v, bool),
        "Integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
        "Float": lambda v: isinstance(v, float),
        "String": lambda v: isinstance(v, str),
        "Blob": lambda v: isinstance(v, Blob),
        "DateTime": lambda v: isinstance(v, datetime),
    }

    if expected_type not in type_checks:
        raise ValueError(f"Unknown East primitive type: {expected_type}")

    if not type_checks[expected_type](value):
        actual_type = type(value).__name__
        raise TypeError(f"Expected {expected_type}, got {actual_type}")


__all__ = [
    "Null",
    "null",
    "Blob",
    "ensure_utc_datetime",
    "validate_east_value",
]
