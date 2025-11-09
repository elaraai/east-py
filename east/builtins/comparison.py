"""Comparison builtin functions."""

from typing import Any

from east.builtins.registry import register_builtin
from east.types.type_system import EastType
from east.utils.ordering import (
    equal_for,
    greater_equal_for,
    greater_for,
    is_for,
    less_equal_for,
    less_for,
)


def is_identical(a: Any, b: Any, T: EastType) -> bool:
    """Identity comparison (same object for mutables, value for immutables).

    Args:
        a: First value
        b: Second value
        T: East type of the values

    Returns:
        True if a and b are identical (uses `is` for mutables, value equality for immutables)
    """
    is_comparer = is_for(T)
    return is_comparer(a, b)


def equals(a: Any, b: Any, T: EastType) -> bool:
    """Structural equality comparison.

    Args:
        a: First value
        b: Second value
        T: East type of the values

    Returns:
        True if values are structurally equal (NaN == NaN, -0.0 != 0.0, cycle detection)
    """
    equal_comparer = equal_for(T)
    return equal_comparer(a, b)


def not_equals(a: Any, b: Any, T: EastType) -> bool:
    """Structural inequality comparison.

    Args:
        a: First value
        b: Second value
        T: East type of the values

    Returns:
        True if values are not structurally equal
    """
    equal_comparer = equal_for(T)
    return not equal_comparer(a, b)


def less_than(a: Any, b: Any, T: EastType) -> bool:
    """Less than comparison using East total ordering.

    Args:
        a: First value
        b: Second value
        T: East type of the values

    Returns:
        True if a < b in East ordering
    """
    less_comparer = less_for(T)
    return less_comparer(a, b)


def less_than_or_equal(a: Any, b: Any, T: EastType) -> bool:
    """Less than or equal comparison using East total ordering.

    Args:
        a: First value
        b: Second value
        T: East type of the values

    Returns:
        True if a <= b in East ordering
    """
    less_equal_comparer = less_equal_for(T)
    return less_equal_comparer(a, b)


def greater_than(a: Any, b: Any, T: EastType) -> bool:
    """Greater than comparison using East total ordering.

    Args:
        a: First value
        b: Second value
        T: East type of the values

    Returns:
        True if a > b in East ordering
    """
    greater_comparer = greater_for(T)
    return greater_comparer(a, b)


def greater_than_or_equal(a: Any, b: Any, T: EastType) -> bool:
    """Greater than or equal comparison using East total ordering.

    Args:
        a: First value
        b: Second value
        T: East type of the values

    Returns:
        True if a >= b in East ordering
    """
    greater_equal_comparer = greater_equal_for(T)
    return greater_equal_comparer(a, b)


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
