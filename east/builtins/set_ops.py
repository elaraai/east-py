"""Set builtin functions."""

from typing import Any

from east.builtins.registry import register_builtin
from east.types.containers import EastArray, EastSet


def set_size(s: EastSet) -> int:
    """Get size of set.

    Args:
        s: Set

    Returns:
        Number of elements in set
    """
    return len(s)


def set_has(s: EastSet, value: Any) -> bool:
    """Check if set contains value.

    Args:
        s: Set
        value: Value to check

    Returns:
        True if value is in set
    """
    return value in s


def set_add(s: EastSet, value: Any) -> None:
    """Add value to set (mutation).

    Args:
        s: Set
        value: Value to add
    """
    s.add(value)


def set_remove(s: EastSet, value: Any) -> None:
    """Remove value from set (mutation).

    Args:
        s: Set
        value: Value to remove

    Raises:
        KeyError: If value not in set
    """
    s.discard(value)


def set_clear(s: EastSet) -> None:
    """Remove all elements from set (mutation).

    Args:
        s: Set
    """
    s.clear()


def set_union(a: EastSet, b: EastSet) -> EastSet:
    """Union of two sets.

    Args:
        a: First set
        b: Second set

    Returns:
        New set with union
    """
    return EastSet(a.element_type, list(a) + [x for x in b if x not in a])


def set_intersection(a: EastSet, b: EastSet) -> EastSet:
    """Intersection of two sets.

    Args:
        a: First set
        b: Second set

    Returns:
        New set with intersection
    """
    return EastSet(a.element_type, [x for x in a if x in b])


def set_difference(a: EastSet, b: EastSet) -> EastSet:
    """Difference of two sets (a - b).

    Args:
        a: First set
        b: Second set

    Returns:
        New set with elements in a but not in b
    """
    return EastSet(a.element_type, [x for x in a if x not in b])


def set_symmetric_difference(a: EastSet, b: EastSet) -> EastSet:
    """Symmetric difference of two sets.

    Args:
        a: First set
        b: Second set

    Returns:
        New set with elements in either a or b but not both
    """
    in_a_not_b = [x for x in a if x not in b]
    in_b_not_a = [x for x in b if x not in a]
    return EastSet(a.element_type, in_a_not_b + in_b_not_a)


def set_is_subset(a: EastSet, b: EastSet) -> bool:
    """Check if a is subset of b.

    Args:
        a: First set
        b: Second set

    Returns:
        True if a is subset of b
    """
    return all(x in b for x in a)


def set_is_superset(a: EastSet, b: EastSet) -> bool:
    """Check if a is superset of b.

    Args:
        a: First set
        b: Second set

    Returns:
        True if a is superset of b
    """
    return all(x in a for x in b)


def set_to_array(s: EastSet) -> EastArray:
    """Convert set to array.

    Args:
        s: Set

    Returns:
        Array with set elements (sorted)
    """
    return EastArray(s.element_type, list(s))


# Register all set builtins
register_builtin("SetSize", set_size)
register_builtin("SetHas", set_has)
register_builtin("SetAdd", set_add)
register_builtin("SetRemove", set_remove)
register_builtin("SetClear", set_clear)
register_builtin("SetUnion", set_union)
register_builtin("SetIntersection", set_intersection)
register_builtin("SetDifference", set_difference)
register_builtin("SetSymmetricDifference", set_symmetric_difference)
register_builtin("SetIsSubset", set_is_subset)
register_builtin("SetIsSuperset", set_is_superset)
register_builtin("SetToArray", set_to_array)


__all__ = [
    "set_size",
    "set_has",
    "set_add",
    "set_remove",
    "set_clear",
    "set_union",
    "set_intersection",
    "set_difference",
    "set_symmetric_difference",
    "set_is_subset",
    "set_is_superset",
    "set_to_array",
]
