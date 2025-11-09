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
        """Write null-terminated UTF-8 string."""
        utf8_bytes = s.encode("utf-8")
        self._ensure_capacity(len(utf8_bytes) + 1)
        self._buffer[self._offset : self._offset + len(utf8_bytes)] = utf8_bytes
        self._offset += len(utf8_bytes)
        self._buffer[self._offset] = 0  # Null terminator
        self._offset += 1

    def write_bytes(self, data: bytes) -> None:
        """Write raw bytes."""
        self._ensure_capacity(len(data))
        self._buffer[self._offset : self._offset + len(data)] = data
        self._offset += len(data)

    @property
    def size(self) -> int:
        """Current size of written data."""
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
    """Read null-terminated UTF-8 string.

    Returns:
        Tuple of (string, new_offset)
    """
    # Find null terminator
    null_pos = buffer.find(b"\x00", offset)
    if null_pos == -1:
        raise ValueError(f"Missing null terminator for string starting at offset {offset}")

    # Extract UTF-8 bytes
    utf8_bytes = buffer[offset:null_pos]
    s = utf8_bytes.decode("utf-8")

    return (s, null_pos + 1)  # Skip past null terminator
