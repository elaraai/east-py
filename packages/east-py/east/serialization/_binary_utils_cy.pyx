#
# Copyright (c) 2025 Elara AI Pty Ltd
# Licensed under the Business Source License 1.1. See LICENSE.md for details.
#
"""Cython-accelerated binary read functions for Beast format deserialization.

Drop-in replacements for the pure-Python functions in binary_utils.py.
Uses typed memoryviews and cdef locals for tight inner loops.
"""

import struct

cpdef tuple read_varint(const unsigned char[:] buffer, int offset):
    """Read varint (variable-length unsigned integer).

    Returns:
        Tuple of (value, new_offset)
    """
    cdef unsigned long long result = 0
    cdef int shift = 0
    cdef int buf_len = buffer.shape[0]
    cdef unsigned char byte

    while offset < buf_len:
        byte = buffer[offset]
        offset += 1
        result |= (<unsigned long long>(byte & 0x7F)) << shift
        if byte < 0x80:
            return (result, offset)
        shift += 7

    raise ValueError(f"Buffer underflow reading varint at offset {offset}")


cpdef tuple read_zigzag(const unsigned char[:] buffer, int offset):
    """Read zigzag-encoded varint (variable-length signed integer).

    Returns:
        Tuple of (value, new_offset)
    """
    cdef unsigned long long result = 0
    cdef int shift = 0
    cdef int buf_len = buffer.shape[0]
    cdef unsigned char byte
    cdef long long value

    while offset < buf_len:
        byte = buffer[offset]
        offset += 1
        result |= (<unsigned long long>(byte & 0x7F)) << shift
        if byte < 0x80:
            # Zigzag decode: (n >>> 1) ^ -(n & 1)
            value = <long long>(result >> 1) ^ -<long long>(result & 1)
            return (value, offset)
        shift += 7

    raise ValueError(f"Buffer underflow reading zigzag at offset {offset}")


cpdef tuple read_float64_le(const unsigned char[:] buffer, int offset):
    """Read 64-bit float in little-endian byte order.

    Returns:
        Tuple of (value, new_offset)
    """
    if offset + 8 > buffer.shape[0]:
        raise ValueError(f"Buffer underflow reading float64 at offset {offset}")

    cdef double value = struct.unpack_from("<d", buffer, offset)[0]
    return (value, offset + 8)


cpdef tuple read_string_utf8_varint(const unsigned char[:] buffer, int offset):
    """Read UTF-8 string with varint length prefix.

    Returns:
        Tuple of (string, new_offset)
    """
    cdef unsigned long long length = 0
    cdef int shift = 0
    cdef int buf_len = buffer.shape[0]
    cdef unsigned char byte
    cdef int end

    # Inline varint reading for the length prefix
    while offset < buf_len:
        byte = buffer[offset]
        offset += 1
        length |= (<unsigned long long>(byte & 0x7F)) << shift
        if byte < 0x80:
            break
        shift += 7
    else:
        raise ValueError(f"Buffer underflow reading string length at offset {offset}")

    end = offset + <int>length
    if end > buf_len:
        raise ValueError(f"Buffer underflow reading string, length {length}")

    cdef str s = bytes(buffer[offset:end]).decode("utf-8")
    return (s, end)
