"""Ref builtin functions."""

from collections.abc import Callable
from typing import Any

from east.builtins.registry import register_builtin
from east.types.ref import Ref, deref, set_ref


def ref_get(ref_cell: Ref, T: Any) -> Any:
    """Get the current value from a reference cell.

    Args:
        ref_cell: The reference cell
        T: Type parameter (element type)

    Returns:
        The current value stored in the ref

    Builtin name: Ref.Get
    Type signature: (Ref<T>) -> T
    """
    return deref(ref_cell)


def ref_update(ref_cell: Ref, value: Any, T: Any) -> None:
    """Replace the value in a reference cell.

    Args:
        ref_cell: The reference cell to update
        value: The new value
        T: Type parameter (element type)

    Returns:
        None (side effect only)

    Builtin name: Ref.Update
    Type signature: (Ref<T>, T) -> Null
    """
    set_ref(ref_cell, value)


def ref_merge(
    ref_cell: Ref, new_value: Any, update_fn: Callable[[Any, Any], Any], T: Any, T2: Any
) -> None:
    """Modify reference value by merging with a new value using a function.

    This is useful for patterns where you want to update a reference based on its current value,
    e.g. incrementing a number, appending to a string, updating fields in a struct.

    Args:
        ref_cell: The reference cell to update
        new_value: The new value to merge with
        update_fn: Function (current, new) -> merged
        T: Type parameter (current value type)
        T2: Type parameter (new value type)

    Returns:
        None (side effect only)

    Builtin name: Ref.Merge
    Type signature: (Ref<T>, T2, (T, T2) -> T) -> Null

    Example:
        # Increment counter
        ref_merge(counter, 5, lambda cur, delta: cur + delta, IntegerType, IntegerType)
    """
    current = deref(ref_cell)
    merged = update_fn(current, new_value)
    set_ref(ref_cell, merged)


# Register builtins
register_builtin("Ref.Get", ref_get)
register_builtin("Ref.Update", ref_update)
register_builtin("Ref.Merge", ref_merge)
