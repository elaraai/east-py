"""Ref builtin functions.

These are factory builtins that take type parameters at compile time.
"""

from collections.abc import Callable
from typing import Any

from east.builtins.registry import register_builtin
from east.types.ref import Ref, deref, set_ref


def ref_get_for(T: Any) -> Callable[[Ref], Any]:
    """Factory for getting value from a reference cell.

    Args:
        T: Type parameter (element type)

    Returns:
        Function that takes a ref and returns its value
    """

    def ref_get(ref_cell: Ref) -> Any:
        return deref(ref_cell)

    return ref_get


def ref_update_for(T: Any) -> Callable[[Ref, Any], None]:
    """Factory for updating a reference cell.

    Args:
        T: Type parameter (element type)

    Returns:
        Function that takes a ref and value, and updates the ref
    """

    def ref_update(ref_cell: Ref, value: Any) -> None:
        set_ref(ref_cell, value)

    return ref_update


def ref_merge_for(T: Any, T2: Any) -> Callable[[Ref, Any, Callable[[Any, Any], Any]], None]:
    """Factory for merging a value into a reference cell.

    Args:
        T: Type parameter (current value type)
        T2: Type parameter (new value type)

    Returns:
        Function that takes a ref, new value, and merge function
    """

    def ref_merge(ref_cell: Ref, new_value: Any, update_fn: Callable[[Any, Any], Any]) -> None:
        current = deref(ref_cell)
        merged = update_fn(current, new_value)
        set_ref(ref_cell, merged)

    return ref_merge


# Register builtins as factories
register_builtin("RefGet", ref_get_for)
register_builtin("RefUpdate", ref_update_for)
register_builtin("RefMerge", ref_merge_for)
