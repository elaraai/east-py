"""Total ordering for East values.

East defines a total ordering across all types, which is used for:
- Sorted sets
- Sorted dicts (by key)
- Comparison operations

The ordering is defined lexicographically by:
1. Type order (Null < Boolean < Integer < Float < String < Blob < DateTime < Array < Set < Dict < Struct < Variant)
2. Within each type, use the natural ordering

Special cases:
- NaN floats are ordered (NaN < -Infinity < ... < Infinity)
- Containers are compared lexicographically
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

from east.types.primitives import Blob, Null

# Type ordering: lower number means comes first
TYPE_ORDER = {
    "Null": 0,
    "Boolean": 1,
    "Integer": 2,
    "Float": 3,
    "String": 4,
    "Blob": 5,
    "DateTime": 6,
    "Array": 7,
    "Set": 8,
    "Dict": 9,
    "Struct": 10,
    "Variant": 11,
}


def get_type_name(value: Any) -> str:
    """Get the East type name for a value.

    Args:
        value: The value to get the type name for

    Returns:
        The East type name (e.g., "Integer", "String", "Array")
    """
    if isinstance(value, Null):
        return "Null"
    if isinstance(value, bool):
        return "Boolean"
    if isinstance(value, int):
        return "Integer"
    if isinstance(value, float):
        return "Float"
    if isinstance(value, str):
        return "String"
    if isinstance(value, Blob):
        return "Blob"
    if isinstance(value, datetime):
        return "DateTime"
    if isinstance(value, list):
        return "Array"
    if isinstance(value, set):
        return "Set"
    if isinstance(value, dict):
        return "Dict"
    if hasattr(value, "_east_type") and hasattr(value._east_type, "fields"):
        return "Struct"
    if hasattr(value, "_east_type") and hasattr(value._east_type, "cases"):
        return "Variant"
    raise TypeError(f"Unknown East type for value: {type(value)}")


def compare_floats(a: float, b: float) -> int:
    """Compare two floats with East semantics.

    In East, NaN is ordered: NaN < -Infinity < ... < Infinity

    Args:
        a: First float
        b: Second float

    Returns:
        -1 if a < b, 0 if a == b, 1 if a > b
    """
    # Handle NaN specially
    a_is_nan = math.isnan(a)
    b_is_nan = math.isnan(b)

    if a_is_nan and b_is_nan:
        return 0
    if a_is_nan:
        return -1  # NaN < everything
    if b_is_nan:
        return 1  # everything > NaN

    # Normal float comparison
    if a < b:
        return -1
    if a > b:
        return 1
    return 0


def east_compare(a: Any, b: Any) -> int:
    """Compare two East values with total ordering.

    Args:
        a: First value
        b: Second value

    Returns:
        -1 if a < b, 0 if a == b, 1 if a > b

    Raises:
        TypeError: If values are not comparable East types
    """
    # First compare by type
    type_a = get_type_name(a)
    type_b = get_type_name(b)

    order_a = TYPE_ORDER[type_a]
    order_b = TYPE_ORDER[type_b]

    if order_a < order_b:
        return -1
    if order_a > order_b:
        return 1

    # Same type - compare within type
    if type_a == "Null":
        return 0  # All nulls are equal

    if type_a == "Boolean":
        # False < True
        return int(a) - int(b)

    if type_a == "Integer":
        if a < b:
            return -1
        if a > b:
            return 1
        return 0

    if type_a == "Float":
        return compare_floats(a, b)

    if type_a == "String" or type_a == "Blob" or type_a == "DateTime":
        if a < b:
            return -1
        if a > b:
            return 1
        return 0

    if type_a == "Array":
        # Lexicographic comparison
        for item_a, item_b in zip(a, b, strict=False):
            cmp = east_compare(item_a, item_b)
            if cmp != 0:
                return cmp
        # All equal so far, compare lengths
        if len(a) < len(b):
            return -1
        if len(a) > len(b):
            return 1
        return 0

    if type_a == "Set":
        # Compare as sorted arrays
        list_a = sorted(a, key=EastKey)
        list_b = sorted(b, key=EastKey)
        return east_compare(list_a, list_b)

    if type_a == "Dict":
        # Compare as sorted arrays of (key, value) pairs
        items_a = sorted(a.items(), key=lambda kv: EastKey(kv[0]))
        items_b = sorted(b.items(), key=lambda kv: EastKey(kv[0]))
        return east_compare(items_a, items_b)

    if type_a == "Struct":
        # Compare fields in order
        for val_a, val_b in zip(a._values, b._values, strict=False):
            cmp = east_compare(val_a, val_b)
            if cmp != 0:
                return cmp
        return 0

    if type_a == "Variant":
        # First compare by tag
        if a.tag < b.tag:
            return -1
        if a.tag > b.tag:
            return 1
        # Same tag, compare values
        return east_compare(a.value, b.value)

    raise TypeError(f"Cannot compare East type: {type_a}")


class EastKey:
    """Wrapper for East values to use in Python sorted() and SortedContainers.

    This provides a key function that implements East's total ordering.

    Example:
        >>> sorted(values, key=EastKey)
        >>> SortedSet(key=EastKey)
    """

    __slots__ = ("value",)

    def __init__(self, value: Any):
        """Create a key wrapper for a value."""
        self.value = value

    def __lt__(self, other: EastKey) -> bool:
        """Less than comparison using East semantics."""
        return east_compare(self.value, other.value) < 0

    def __le__(self, other: EastKey) -> bool:
        """Less than or equal comparison using East semantics."""
        return east_compare(self.value, other.value) <= 0

    def __gt__(self, other: EastKey) -> bool:
        """Greater than comparison using East semantics."""
        return east_compare(self.value, other.value) > 0

    def __ge__(self, other: EastKey) -> bool:
        """Greater than or equal comparison using East semantics."""
        return east_compare(self.value, other.value) >= 0

    def __eq__(self, other: object) -> bool:
        """Equality comparison using East semantics."""
        if not isinstance(other, EastKey):
            return NotImplemented
        return east_compare(self.value, other.value) == 0

    def __hash__(self) -> int:
        """Hash based on the value."""
        return hash(self.value)


__all__ = [
    "TYPE_ORDER",
    "get_type_name",
    "compare_floats",
    "east_compare",
    "EastKey",
]
