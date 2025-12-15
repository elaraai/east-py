#
# Copyright (c) 2025 Elara AI Pty Ltd
# Licensed under the Business Source License 1.1. See LICENSE.md for details.
#
"""Comparison builtin functions.

These are factory builtins - they take type parameters at compile time and
return specialized comparison functions that are called at runtime.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from east.runtime.platform import PlatformFunction

from east.builtins.registry import register_builtin
from east.types.types import EastType
from east.types.values import EastValue
from east.utils.ordering import (
    equal_for as _equal_for,
)
from east.utils.ordering import (
    greater_equal_for as _greater_equal_for,
)
from east.utils.ordering import (
    greater_for as _greater_for,
)
from east.utils.ordering import (
    is_for as _is_for,
)
from east.utils.ordering import (
    less_equal_for as _less_equal_for,
)
from east.utils.ordering import (
    less_for as _less_for,
)


# Wrap ordering functions to accept _platform parameter (for consistency with TypeScript)
def is_for(
    _platform: "list[PlatformFunction]", T: EastType
) -> Callable[[EastValue, EastValue], bool]:
    """Factory for identity comparison."""
    return _is_for(T)


def equal_for(
    _platform: "list[PlatformFunction]", T: EastType
) -> Callable[[EastValue, EastValue], bool]:
    """Factory for structural equality comparison."""
    return _equal_for(T)


def less_for(
    _platform: "list[PlatformFunction]", T: EastType
) -> Callable[[EastValue, EastValue], bool]:
    """Factory for less-than comparison."""
    return _less_for(T)


def less_equal_for(
    _platform: "list[PlatformFunction]", T: EastType
) -> Callable[[EastValue, EastValue], bool]:
    """Factory for less-than-or-equal comparison."""
    return _less_equal_for(T)


def greater_for(
    _platform: "list[PlatformFunction]", T: EastType
) -> Callable[[EastValue, EastValue], bool]:
    """Factory for greater-than comparison."""
    return _greater_for(T)


def greater_equal_for(
    _platform: "list[PlatformFunction]", T: EastType
) -> Callable[[EastValue, EastValue], bool]:
    """Factory for greater-than-or-equal comparison."""
    return _greater_equal_for(T)


def not_equal_for(
    _platform: "list[PlatformFunction]", T: EastType
) -> Callable[[EastValue, EastValue], bool]:
    """Factory for structural inequality comparison.

    Args:
        T: East type of the values to compare

    Returns:
        A function that returns True if values are not structurally equal
    """
    equal_comparer = _equal_for(T)

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
