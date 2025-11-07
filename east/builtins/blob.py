"""Blob builtin functions."""

from east.builtins.registry import register_builtin
from east.types.primitives import Blob


def blob_length(b: Blob) -> int:
    """Get length of blob.

    Args:
        b: Blob

    Returns:
        Number of bytes in blob
    """
    return len(b.data)


def blob_get(b: Blob, index: int) -> int:
    """Get byte at index.

    Args:
        b: Blob
        index: Byte index (0-based)

    Returns:
        Byte value (0-255)

    Raises:
        IndexError: If index out of bounds
    """
    return b.data[index]


def blob_set(b: Blob, index: int, value: int) -> None:
    """Set byte at index (mutation).

    Args:
        b: Blob
        index: Byte index (0-based)
        value: Byte value (0-255)

    Raises:
        IndexError: If index out of bounds
        ValueError: If value not in 0-255
    """
    if not 0 <= value <= 255:
        raise ValueError(f"Byte value must be 0-255, got {value}")
    # Blobs are mutable via bytearray
    # Access internal _data attribute directly for mutation
    if isinstance(b._data, bytes):  # type: ignore
        # Convert to bytearray for mutation
        b._data = bytearray(b._data)  # type: ignore
    b._data[index] = value  # type: ignore


def blob_create(size: int) -> Blob:
    """Create blob of given size filled with zeros.

    Args:
        size: Number of bytes

    Returns:
        New blob of given size filled with zeros
    """
    return Blob(bytearray(size))


def blob_slice(b: Blob, start: int, end: int) -> Blob:
    """Get blob slice.

    Args:
        b: Blob
        start: Start index (inclusive)
        end: End index (exclusive)

    Returns:
        New blob with slice
    """
    return Blob(b.data[start:end])


def blob_concat(a: Blob, b: Blob) -> Blob:
    """Concatenate two blobs.

    Args:
        a: First blob
        b: Second blob

    Returns:
        New blob with concatenated data
    """
    return Blob(a.data + b.data)


def blob_to_string(b: Blob) -> str:
    """Decode blob as UTF-8 string.

    Args:
        b: Blob

    Returns:
        Decoded string

    Raises:
        UnicodeDecodeError: If blob is not valid UTF-8
    """
    return b.data.decode("utf-8")


def blob_decode_utf16(b: Blob) -> str:
    """Decode blob as UTF-16 string.

    Args:
        b: Blob

    Returns:
        Decoded string

    Raises:
        UnicodeDecodeError: If blob is not valid UTF-16
    """
    return b.data.decode("utf-16")


def string_to_blob(s: str) -> Blob:
    """Encode string as UTF-8 blob.

    Args:
        s: String

    Returns:
        Encoded blob
    """
    return Blob(s.encode("utf-8"))


def string_encode_utf16(s: str) -> Blob:
    """Encode string as UTF-16 blob.

    Args:
        s: String

    Returns:
        Encoded blob
    """
    return Blob(s.encode("utf-16"))


# Register all blob builtins
register_builtin("BlobSize", blob_length)  # Renamed from BlobLength
register_builtin("BlobGetUint8", blob_get)  # Renamed from BlobGet
register_builtin("BlobSetUint8", blob_set)
register_builtin("BlobCreate", blob_create)
# Note: BlobSlice and BlobConcat not in spec but kept for convenience
register_builtin("BlobSlice", blob_slice)
register_builtin("BlobConcat", blob_concat)
register_builtin("BlobDecodeUtf8", blob_to_string)  # Renamed from BlobToString
register_builtin("BlobDecodeUtf16", blob_decode_utf16)
register_builtin("StringEncodeUtf8", string_to_blob)  # Renamed from StringToBlob
register_builtin("StringEncodeUtf16", string_encode_utf16)


__all__ = [
    "blob_length",
    "blob_get",
    "blob_set",
    "blob_create",
    "blob_slice",
    "blob_concat",
    "blob_to_string",
    "blob_decode_utf16",
    "string_to_blob",
    "string_encode_utf16",
]
