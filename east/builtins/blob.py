"""Blob builtin functions.

These are factory builtins that take type parameters at compile time.
"""

from collections.abc import Callable
from typing import Any

from east.builtins.registry import register_builtin
from east.types.primitives import Blob


def blob_length(b: Blob) -> int:
    """Get length of blob."""
    return len(b.data)


def blob_get(b: Blob, index: int) -> int:
    """Get byte at index."""
    return b.data[index]


def blob_to_string(b: Blob) -> str:
    """Decode blob as UTF-8 string."""
    return b.data.decode("utf-8")


def blob_decode_utf16(b: Blob) -> str:
    """Decode blob as UTF-16 string."""
    return b.data.decode("utf-16")


def string_to_blob(s: str) -> Blob:
    """Encode string as UTF-8 blob."""
    return Blob(s.encode("utf-8"))


def string_encode_utf16(s: str) -> Blob:
    """Encode string as UTF-16 blob."""
    return Blob(s.encode("utf-16"))


def blob_decode_beast_for(T: Any) -> Callable[[Blob], Any]:
    """Factory for decoding blob from Beast binary format.

    Args:
        T: Expected EastType

    Returns:
        Function that decodes blobs of this type
    """
    from east.serialization.beast import decode_beast_for

    decoder = decode_beast_for(T)

    def blob_decode_beast(blob: Blob) -> Any:
        return decoder(blob.data)

    return blob_decode_beast


def blob_encode_beast_for(T: Any) -> Callable[[Any], Blob]:
    """Factory for encoding value to Beast binary format.

    Args:
        T: EastType of the value

    Returns:
        Function that encodes values of this type
    """
    from east.serialization.beast import encode_beast_for

    encoder = encode_beast_for(T)

    def blob_encode_beast(value: Any) -> Blob:
        return Blob(encoder(value))

    return blob_encode_beast


def blob_decode_beast2_for(T: Any) -> Callable[[Blob], Any]:
    """Factory for decoding blob from Beast2 binary format.

    Args:
        T: Expected EastType

    Returns:
        Function that decodes blobs of this type
    """
    from east.serialization.beast2 import decode_beast2_with_header_for

    decoder = decode_beast2_with_header_for(T)

    def blob_decode_beast2(blob: Blob) -> Any:
        return decoder(blob.data)

    return blob_decode_beast2


def blob_encode_beast2_for(T: Any) -> Callable[[Any], Blob]:
    """Factory for encoding value to Beast2 binary format.

    Args:
        T: EastType of the value

    Returns:
        Function that encodes values of this type
    """
    from east.serialization.beast2 import encode_beast2_with_header_for

    encoder = encode_beast2_with_header_for(T)

    def blob_encode_beast2(value: Any) -> Blob:
        return Blob(encoder(value))

    return blob_encode_beast2


# Register all blob builtins as factories
register_builtin("BlobSize", lambda: blob_length)
register_builtin("BlobGetUint8", lambda: blob_get)
register_builtin("BlobDecodeUtf8", lambda: blob_to_string)
register_builtin("BlobDecodeUtf16", lambda: blob_decode_utf16)
register_builtin("BlobDecodeBeast", blob_decode_beast_for)
register_builtin("BlobEncodeBeast", blob_encode_beast_for)
register_builtin("BlobDecodeBeast2", blob_decode_beast2_for)
register_builtin("BlobEncodeBeast2", blob_encode_beast2_for)
register_builtin("StringEncodeUtf8", lambda: string_to_blob)
register_builtin("StringEncodeUtf16", lambda: string_encode_utf16)


__all__ = [
    "blob_length",
    "blob_get",
    "blob_to_string",
    "blob_decode_utf16",
    "blob_decode_beast_for",
    "blob_encode_beast_for",
    "blob_decode_beast2_for",
    "blob_encode_beast2_for",
    "string_to_blob",
    "string_encode_utf16",
]
