"""Comparison builtin functions."""

from typing import Any

from east.builtins.registry import register_builtin
from east.utils.ordering import east_compare


def is_identical(a: Any, b: Any) -> bool:
    """Identity comparison (same object).

    Args:
        a: First value
        b: Second value

    Returns:
        True if a and b are the same object
    """
    return a is b


def equals(a: Any, b: Any) -> bool:
    """Structural equality comparison.

    Args:
        a: First value
        b: Second value

    Returns:
        True if values are structurally equal
    """
    return a == b


def not_equals(a: Any, b: Any) -> bool:
    """Structural inequality comparison.

    Args:
        a: First value
        b: Second value

    Returns:
        True if values are not structurally equal
    """
    return a != b


def less_than(a: Any, b: Any) -> bool:
    """Less than comparison using East total ordering.

    Args:
        a: First value
        b: Second value

    Returns:
        True if a < b in East ordering
    """
    return east_compare(a, b) < 0


def less_than_or_equal(a: Any, b: Any) -> bool:
    """Less than or equal comparison using East total ordering.

    Args:
        a: First value
        b: Second value

    Returns:
        True if a <= b in East ordering
    """
    return east_compare(a, b) <= 0


def greater_than(a: Any, b: Any) -> bool:
    """Greater than comparison using East total ordering.

    Args:
        a: First value
        b: Second value

    Returns:
        True if a > b in East ordering
    """
    return east_compare(a, b) > 0


def greater_than_or_equal(a: Any, b: Any) -> bool:
    """Greater than or equal comparison using East total ordering.

    Args:
        a: First value
        b: Second value

    Returns:
        True if a >= b in East ordering
    """
    return east_compare(a, b) >= 0


# Register all comparison builtins with spec-compliant names
register_builtin("Is", is_identical)
register_builtin("Equal", equals)
register_builtin("NotEqual", not_equals)
register_builtin("Less", less_than)
register_builtin("LessEqual", less_than_or_equal)
register_builtin("Greater", greater_than)
register_builtin("GreaterEqual", greater_than_or_equal)


__all__ = [
    "is_identical",
    "equals",
    "not_equals",
    "less_than",
    "less_than_or_equal",
    "greater_than",
    "greater_than_or_equal",
]
