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


def string_to_blob(s: str) -> Blob:
    """Encode string as UTF-8 blob.

    Args:
        s: String

    Returns:
        Encoded blob
    """
    return Blob(s.encode("utf-8"))


# Register all blob builtins
register_builtin("BlobLength", blob_length)
register_builtin("BlobGet", blob_get)
register_builtin("BlobSlice", blob_slice)
register_builtin("BlobConcat", blob_concat)
register_builtin("BlobToString", blob_to_string)
register_builtin("StringToBlob", string_to_blob)


__all__ = [
    "blob_length",
    "blob_get",
    "blob_slice",
    "blob_concat",
    "blob_to_string",
    "string_to_blob",
]
