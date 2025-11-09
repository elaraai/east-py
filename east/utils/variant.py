"""Variant helper functions for creating Option-type returns."""

from typing import Any, TypedDict


class Variant(TypedDict):
    """Variant object with type tag and value."""

    type: str
    value: Any


def variant(tag: str, value: Any) -> Variant:
    """Create a variant object.

    Args:
        tag: The variant tag (e.g., "some", "none")
        value: The variant value

    Returns:
        Variant dict with type and value
    """
    return {"type": tag, "value": value}


def some(value: Any) -> Variant:
    """Create a 'some' variant.

    Args:
        value: The value to wrap

    Returns:
        Variant with type="some"
    """
    return variant("some", value)


def none() -> Variant:
    """Create a 'none' variant.

    Returns:
        Variant with type="none" and value=null
    """
    from east.types.primitives import null

    return variant("none", null)


__all__ = ["variant", "some", "none", "Variant"]
