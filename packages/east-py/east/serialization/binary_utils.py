#
# Copyright (c) 2025 Elara AI Pty Ltd
# Licensed under the Business Source License 1.1. See LICENSE.md for details.
#
"""Binary utilities for Beast format serialization.

Provides BufferWriter for encoding and read functions for decoding.
Implements "twiddled" encoding for integers and floats to preserve byte-ordering.
"""

from __future__ import annotations

import struct


class BufferWriter:
    """Managed bytearray with auto-growth for binary encoding."""

    def __init__(self, initial_capacity: int = 16384):  # 16KB default
        self._buffer = bytearray(initial_capacity)
        self._offset = 0

    def _ensure_capacity(self, needed: int) -> None:
        """Ensure buffer has capacity for needed bytes."""
        required = self._offset + needed
        if required <= len(self._buffer):
            return  # Sufficient capacity

        # Exponential growth: min 2x, max +1GB per resize
        doubled = len(self._buffer) * 2
        max_growth = len(self._buffer) + 1024 * 1024 * 1024
        new_size = max(min(doubled, max_growth), required)

        # Grow buffer
        self._buffer.extend(bytes(new_size - len(self._buffer)))

    def write_uint8(self, value: int) -> None:
        """Write single unsigned byte."""
        self._ensure_capacity(1)
        self._buffer[self._offset] = value & 0xFF
        self._offset += 1

    def write_int64_twiddled(self, value: int) -> None:
        """Write 64-bit integer with sign-bit flip for byte-ordering.

        Positive values become > 0x8000_0000_0000_0000
        Negative values become < 0x8000_0000_0000_0000
        This ensures memcmp ordering matches numeric ordering.
        """
        self._ensure_capacity(8)

        # Flip sign bit for byte-ordering
        # XOR with sign bit (0x8000_0000_0000_0000)
        twiddled = value ^ -(2**63)

        # Write as big-endian signed
        struct.pack_into(">q", self._buffer, self._offset, twiddled)
        self._offset += 8

    def write_float64_twiddled(self, value: float) -> None:
        """Write 64-bit float with bit-twiddling for total ordering.

        This ensures memcmp(encoded_a, encoded_b) matches float comparison,
        including proper handling of NaN, infinities, and signed zeros.
        """
        self._ensure_capacity(8)

        # Convert float to bit pattern
        bits = struct.unpack(">Q", struct.pack(">d", value))[0]

        # Bit-twiddling for total ordering
        if bits < 0x8000_0000_0000_0000:
            # Positive float (sign bit = 0) - flip sign bit
            # Maps: 0.0 -> 0x8000..., +inf -> 0xFFF0..., NaN -> 0xFFF8...
            bits = bits ^ 0x8000_0000_0000_0000
        else:
            # Negative float (sign bit = 1) - flip all bits
            # Maps: -0.0 -> 0x7FFF..., -inf -> 0x000F..., -smallest -> 0x7FFF...
            bits = (~bits) & 0xFFFFFFFFFFFFFFFF

        # Write as big-endian
        struct.pack_into(">Q", self._buffer, self._offset, bits)
        self._offset += 8

    def write_string_utf8_null(self, s: str) -> None:
        """Write east_null-terminated UTF-8 string."""
        utf8_bytes = s.encode("utf-8")
        self._ensure_capacity(len(utf8_bytes) + 1)
        self._buffer[self._offset : self._offset + len(utf8_bytes)] = utf8_bytes
        self._offset += len(utf8_bytes)
        self._buffer[self._offset] = 0  # EastNull terminator
        self._offset += 1

    def write_bytes(self, data: bytes) -> None:
        """Write raw bytes."""
        self._ensure_capacity(len(data))
        self._buffer[self._offset : self._offset + len(data)] = data
        self._offset += len(data)

    def write_varint(self, value: int) -> None:
        """Write unsigned integer as varint (variable-length encoding).

        Uses 7 bits per byte with continuation bit in MSB.
        """
        if value < 0:
            raise ValueError(f"write_varint requires non-negative value, got {value}")

        while value >= 0x80:
            self._ensure_capacity(1)
            self._buffer[self._offset] = (value & 0x7F) | 0x80
            self._offset += 1
            value >>= 7

        self._ensure_capacity(1)
        self._buffer[self._offset] = value & 0x7F
        self._offset += 1

    def write_zigzag(self, value: int) -> None:
        """Write signed integer as zigzag-encoded varint.

        Zigzag encoding maps signed integers to unsigned:
        0 -> 0, -1 -> 1, 1 -> 2, -2 -> 3, 2 -> 4, ...
        """
        # Zigzag encode: (n << 1) ^ (n >> 63)
        zigzag = (value << 1) ^ (value >> 63)
        self.write_varint(zigzag)

    def write_float64_le(self, value: float) -> None:
        """Write 64-bit float in little-endian byte order."""
        self._ensure_capacity(8)
        struct.pack_into("<d", self._buffer, self._offset, value)
        self._offset += 8

    def write_string_utf8_varint(self, s: str) -> None:
        """Write UTF-8 string with varint length prefix."""
        utf8_bytes = s.encode("utf-8")
        self.write_varint(len(utf8_bytes))
        self._ensure_capacity(len(utf8_bytes))
        self._buffer[self._offset : self._offset + len(utf8_bytes)] = utf8_bytes
        self._offset += len(utf8_bytes)

    @property
    def size(self) -> int:
        """Current size of written data."""
        return self._offset

    @property
    def current_offset(self) -> int:
        """Current offset in the buffer."""
        return self._offset

    def to_bytes(self) -> bytes:
        """Extract current buffer contents."""
        return bytes(self._buffer[: self._offset])


def read_int64_twiddled(buffer: bytes, offset: int) -> tuple[int, int]:
    """Read twiddled 64-bit integer.

    Returns:
        Tuple of (value, new_offset)
    """
    if offset + 8 > len(buffer):
        raise ValueError(f"Buffer underflow reading int64 at offset {offset}")

    # Read as big-endian signed
    twiddled = struct.unpack_from(">q", buffer, offset)[0]

    # Reverse the twiddling (XOR with sign bit again)
    value = twiddled ^ -(2**63)

    return (value, offset + 8)


def read_float64_twiddled(buffer: bytes, offset: int) -> tuple[float, int]:
    """Read twiddled 64-bit float.

    Returns:
        Tuple of (value, new_offset)
    """
    if offset + 8 > len(buffer):
        raise ValueError(f"Buffer underflow reading float64 at offset {offset}")

    # Read bit pattern
    bits = struct.unpack_from(">Q", buffer, offset)[0]

    # Reverse the twiddling
    if bits >= 0x8000_0000_0000_0000:
        # Was positive - reverse sign bit flip
        bits = bits ^ 0x8000_0000_0000_0000
    else:
        # Was negative - reverse bit inversion
        bits = (~bits) & 0xFFFFFFFFFFFFFFFF

    # Convert bits back to float
    value = struct.unpack(">d", struct.pack(">Q", bits))[0]

    return (value, offset + 8)


def read_string_utf8_null(buffer: bytes, offset: int) -> tuple[str, int]:
    """Read east_null-terminated UTF-8 string.

    Returns:
        Tuple of (string, new_offset)
    """
    # Find east_null terminator
    null_pos = buffer.find(b"\x00", offset)
    if null_pos == -1:
        raise ValueError(f"Missing east_null terminator for string starting at offset {offset}")

    # Extract UTF-8 bytes
    utf8_bytes = buffer[offset:null_pos]
    s = utf8_bytes.decode("utf-8")

    return (s, null_pos + 1)  # Skip past east_null terminator


def read_varint(buffer: bytes, offset: int) -> tuple[int, int]:
    """Read varint (variable-length unsigned integer).

    Returns:
        Tuple of (value, new_offset)
    """
    result = 0
    shift = 0

    while offset < len(buffer):
        byte = buffer[offset]
        offset += 1

        result |= (byte & 0x7F) << shift

        if (byte & 0x80) == 0:
            return (result, offset)

        shift += 7

    raise ValueError(f"Buffer underflow reading varint at offset {offset}")


def read_zigzag(buffer: bytes, offset: int) -> tuple[int, int]:
    """Read zigzag-encoded varint (variable-length signed integer).

    Returns:
        Tuple of (value, new_offset)
    """
    zigzag, new_offset = read_varint(buffer, offset)

    # Zigzag decode: (n >>> 1) ^ -(n & 1)
    value = (zigzag >> 1) ^ (-(zigzag & 1))

    return (value, new_offset)


def read_float64_le(buffer: bytes, offset: int) -> tuple[float, int]:
    """Read 64-bit float in little-endian byte order.

    Returns:
        Tuple of (value, new_offset)
    """
    if offset + 8 > len(buffer):
        raise ValueError(f"Buffer underflow reading float64 at offset {offset}")

    value = struct.unpack_from("<d", buffer, offset)[0]

    return (value, offset + 8)


def read_string_utf8_varint(buffer: bytes, offset: int) -> tuple[str, int]:
    """Read UTF-8 string with varint length prefix.

    Returns:
        Tuple of (string, new_offset)
    """
    length, new_offset = read_varint(buffer, offset)

    if new_offset + length > len(buffer):
        raise ValueError(f"Buffer underflow reading string at offset {offset}, length {length}")

    utf8_bytes = buffer[new_offset : new_offset + length]
    s = utf8_bytes.decode("utf-8")

    return (s, new_offset + length)
