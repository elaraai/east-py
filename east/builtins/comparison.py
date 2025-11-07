"""Comparison builtin functions."""

from typing import Any

from east.builtins.registry import register_builtin
from east.utils.ordering import east_compare


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


# Register all comparison builtins
register_builtin("Equals", equals)
register_builtin("NotEquals", not_equals)
register_builtin("LessThan", less_than)
register_builtin("LessThanOrEqual", less_than_or_equal)
register_builtin("GreaterThan", greater_than)
register_builtin("GreaterThanOrEqual", greater_than_or_equal)


__all__ = [
    "equals",
    "not_equals",
    "less_than",
    "less_than_or_equal",
    "greater_than",
    "greater_than_or_equal",
]
