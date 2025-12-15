#
# Copyright (c) 2025 Elara AI Pty Ltd
# Licensed under the Business Source License 1.1. See LICENSE.md for details.
#
"""Boolean builtin functions."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from east.builtins.registry import register_builtin


def boolean_and(a: bool, b: bool) -> bool:
    """Logical AND.

    Args:
        a: First boolean
        b: Second boolean

    Returns:
        a AND b
    """
    return a and b


def boolean_or(a: bool, b: bool) -> bool:
    """Logical OR.

    Args:
        a: First boolean
        b: Second boolean

    Returns:
        a OR b
    """
    return a or b


def boolean_not(a: bool) -> bool:
    """Logical NOT.

    Args:
        a: Boolean to negate

    Returns:
        NOT a
    """
    return not a


def boolean_xor(a: bool, b: bool) -> bool:
    """Logical XOR (exclusive or).

    Args:
        a: First boolean
        b: Second boolean

    Returns:
        a XOR b (true if exactly one is true)
    """
    return a != b


# Register all boolean builtins as factories (no type params, so return impl directly)
register_builtin("BooleanAnd", lambda _platform: boolean_and)
register_builtin("BooleanOr", lambda _platform: boolean_or)
register_builtin("BooleanNot", lambda _platform: boolean_not)
register_builtin("BooleanXor", lambda _platform: boolean_xor)


__all__ = ["boolean_and", "boolean_or", "boolean_not", "boolean_xor"]
