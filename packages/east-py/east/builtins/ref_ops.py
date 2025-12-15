#
# Copyright (c) 2025 Elara AI Pty Ltd
# Licensed under the Business Source License 1.1. See LICENSE.md for details.
#
"""EastRef builtin functions.

These are factory builtins that take type parameters at compile time.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from east.runtime.platform import PlatformFunction

from east.builtins.registry import register_builtin
from east.types.types import EastType
from east.types.values import EastRef, EastValue, deref, set_ref


def ref_get_for(_platform: "list[PlatformFunction]", T: EastType) -> Callable[[EastRef], EastValue]:
    """Factory for getting value from a reference cell.

    Args:
        T: Type parameter (element type)

    Returns:
        Function that takes a east_ref and returns its value
    """

    def ref_get(ref_cell: EastRef) -> EastValue:
        return deref(ref_cell)

    return ref_get


def ref_update_for(
    _platform: "list[PlatformFunction]", T: EastType
) -> Callable[[EastRef, EastValue], None]:
    """Factory for updating a reference cell.

    Args:
        T: Type parameter (element type)

    Returns:
        Function that takes a east_ref and value, and updates the east_ref
    """

    def ref_update(ref_cell: EastRef, value: EastValue) -> None:
        set_ref(ref_cell, value)

    return ref_update


def ref_merge_for(
    _platform: "list[PlatformFunction]", T: EastType, T2: EastType
) -> Callable[[EastRef, EastValue, Any], None]:
    """Factory for merging a value into a reference cell.

    Args:
        T: Type parameter (current value type)
        T2: Type parameter (new value type)

    Returns:
        Function that takes a east_ref, new value, and merge function
    """

    def ref_merge(ref_cell: EastRef, new_value: EastValue, update_fn: Any) -> None:
        current = deref(ref_cell)
        merged = update_fn(current, new_value)
        set_ref(ref_cell, merged)

    return ref_merge


# Register builtins as factories
register_builtin("RefGet", ref_get_for)
register_builtin("RefUpdate", ref_update_for)
register_builtin("RefMerge", ref_merge_for)
