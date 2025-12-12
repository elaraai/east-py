#
# Copyright (c) 2025 Elara AI Pty Ltd
# Licensed under the Business Source License 1.1. See LICENSE.md for details.
#
"""Comparison builtin functions.

These are factory builtins - they take type parameters at compile time and
return specialized comparison functions that are called at runtime.
"""

from collections.abc import Callable

from east.builtins.registry import register_builtin
from east.types.types import EastType
from east.types.values import EastValue
from east.utils.ordering import (
    equal_for,
    greater_equal_for,
    greater_for,
    is_for,
    less_equal_for,
    less_for,
)


def not_equal_for(T: EastType) -> Callable[[EastValue, EastValue], bool]:
    """Factory for structural inequality comparison.

    Args:
        T: East type of the values to compare

    Returns:
        A function that returns True if values are not structurally equal
    """
    equal_comparer = equal_for(T)

    def not_equal(a: EastValue, b: EastValue) -> bool:
        return not equal_comparer(a, b)

    return not_equal


# Register all comparison builtins as factory functions
# These are called at compile time with type parameters to produce specialized comparers
register_builtin("Is", is_for)
register_builtin("Equal", equal_for)
register_builtin("NotEqual", not_equal_for)
register_builtin("Less", less_for)
register_builtin("LessEqual", less_equal_for)
register_builtin("Greater", greater_for)
register_builtin("GreaterEqual", greater_equal_for)


__all__ = [
    "not_equal_for",
]
