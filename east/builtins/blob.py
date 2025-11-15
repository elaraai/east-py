"""Blob builtin functions."""

from typing import Any

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


def blob_decode_beast(blob: Blob, T: Any) -> Any:
    """Decode blob from Beast binary format.

    Args:
        blob: Blob to decode
        T: Expected EastType

    Returns:
        Decoded East value

    Raises:
        ValueError: If blob is not valid Beast format
    """
    from east.serialization.beast import decode_beast_for

    decoder = decode_beast_for(T)
    return decoder(blob.data)


def blob_encode_beast(value: Any, T: Any) -> Blob:
    """Encode value to Beast binary format.

    Args:
        value: East value to encode
        T: EastType of the value

    Returns:
        Encoded blob

    Raises:
        TypeError: If value cannot be encoded
    """
    from east.serialization.beast import encode_beast_for

    encoder = encode_beast_for(T)
    data = encoder(value)
    return Blob(data)


def blob_decode_beast2(blob: Blob, T: Any) -> Any:
    """Decode blob from Beast2 binary format.

    Args:
        blob: Blob to decode
        T: Expected EastType

    Returns:
        Decoded East value
    """
    from east.serialization.beast2 import decode_beast2_with_header_for

    decoder = decode_beast2_with_header_for(T)
    return decoder(blob.data)


def blob_encode_beast2(value: Any, T: Any) -> Blob:
    """Encode value to Beast2 binary format.

    Args:
        value: East value to encode
        T: EastType of the value

    Returns:
        Encoded blob
    """
    from east.serialization.beast2 import encode_beast2_with_header_for

    encoder = encode_beast2_with_header_for(T)
    data = encoder(value)
    return Blob(data)


# Register all blob builtins
register_builtin("BlobSize", blob_length)  # Renamed from BlobLength
register_builtin("BlobGetUint8", blob_get)  # Renamed from BlobGet
register_builtin("BlobDecodeUtf8", blob_to_string)  # Renamed from BlobToString
register_builtin("BlobDecodeUtf16", blob_decode_utf16)
register_builtin("BlobDecodeBeast", blob_decode_beast)
register_builtin("BlobEncodeBeast", blob_encode_beast)
register_builtin("BlobDecodeBeast2", blob_decode_beast2)
register_builtin("BlobEncodeBeast2", blob_encode_beast2)
register_builtin("StringEncodeUtf8", string_to_blob)  # Renamed from StringToBlob
register_builtin("StringEncodeUtf16", string_encode_utf16)


__all__ = [
    "blob_length",
    "blob_get",
    "blob_to_string",
    "blob_decode_utf16",
    "blob_decode_beast",
    "blob_encode_beast",
    "blob_decode_beast2",
    "blob_encode_beast2",
    "string_to_blob",
    "string_encode_utf16",
]
