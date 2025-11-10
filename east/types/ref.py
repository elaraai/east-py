"""East ref type - mutable reference cells with identity semantics.

Ref-cells provide mutable reference containers with identity semantics.
Similar to OCaml's ref type or Scheme boxes.
"""

from __future__ import annotations

from typing import Any, Generic, TypeVar

T = TypeVar("T")

# Symbol for nominal typing (brand)
REF_SYMBOL = object()


class Ref(Generic[T]):
    """Mutable reference cell containing a value.

    Ref-cells are mutable containers with identity semantics:
    - Two refs are equal only if they're the same object (Object.is)
    - Or if their contents are deeply equal and not circular
    - Refs support aliasing in serialization
    - Refs are invariant in the type system

    Examples:
        >>> counter = ref(0)
        >>> set_ref(counter, deref(counter) + 1)
        >>> deref(counter)
        1

        >>> # Aliasing - both variables point to same ref-cell
        >>> r1 = ref([1, 2, 3])
        >>> r2 = r1  # Same ref-cell
        >>> set_ref(r2, [4, 5, 6])
        >>> deref(r1)
        [4, 5, 6]

    Attributes:
        value: The mutable value contained in this ref-cell
        _brand: Brand symbol for nominal typing (private)
    """

    __slots__ = ("value", "_brand")

    def __init__(self, value: T):
        """Create a new ref-cell (use ref() function instead)."""
        self.value: T = value
        self._brand = REF_SYMBOL

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"ref({self.value!r})"

    def __str__(self) -> str:
        """Human-readable string."""
        return f"&{self.value}"


def ref(value: T) -> Ref[T]:
    """Create a new mutable reference cell containing the specified value.

    Args:
        value: The initial value to store in the ref-cell

    Returns:
        A branded ref-cell object

    Examples:
        >>> counter = ref(0)
        >>> set_ref(counter, deref(counter) + 1)
        >>> deref(counter)
        1

        >>> # Refs have identity semantics
        >>> original = ref([1, 2, 3])
        >>> alias = original  # Same ref-cell
        >>> set_ref(alias, [4, 5, 6])
        >>> deref(original)
        [4, 5, 6]
    """
    return Ref(value)


def is_ref(v: Any) -> bool:
    """Check if a value is a ref-cell.

    Args:
        v: The value to check

    Returns:
        True if the value is a ref-cell, False otherwise

    Examples:
        >>> r = ref(42)
        >>> is_ref(r)
        True
        >>> is_ref(42)
        False
        >>> is_ref({'value': 42})
        False
    """
    return isinstance(v, Ref) and hasattr(v, "_brand") and v._brand is REF_SYMBOL


def deref(r: Ref[T]) -> T:
    """Retrieve the current value from a ref-cell.

    Args:
        r: The ref-cell to dereference

    Returns:
        The current value stored in the ref-cell

    Examples:
        >>> counter = ref(10)
        >>> deref(counter)
        10
    """
    return r.value


def set_ref(r: Ref[T], value: T) -> None:
    """Update the value stored in a ref-cell.

    This mutates the ref-cell in place. All aliases to the same
    ref-cell will see the updated value.

    Args:
        r: The ref-cell to update
        value: The new value to store

    Examples:
        >>> counter = ref(0)
        >>> set_ref(counter, 1)
        >>> set_ref(counter, deref(counter) + 1)
        >>> deref(counter)
        2

        >>> # Aliasing - both variables point to same ref-cell
        >>> r1 = ref("hello")
        >>> r2 = r1
        >>> set_ref(r2, "world")
        >>> deref(r1)
        'world'
    """
    r.value = value


__all__ = ["Ref", "ref", "is_ref", "deref", "set_ref", "REF_SYMBOL"]
