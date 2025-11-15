"""Set builtin functions."""

from typing import Any

from east.builtins.registry import register_builtin
from east.types.containers import EastArray, EastSet


def set_size(s: EastSet, K: Any) -> int:
    """Get size of set.

    Args:
        s: Set

    Returns:
        Number of elements in set
    """
    return len(s)


def set_has(s: EastSet, value: Any, K: Any) -> bool:
    """Check if set contains value.

    Args:
        s: Set
        value: Value to check

    Returns:
        True if value is in set
    """
    return value in s


def set_add(s: EastSet, value: Any, K: Any) -> None:
    """Add value to set (mutation).

    Args:
        s: Set
        value: Value to add
    """
    s.add(value)


def set_remove(s: EastSet, value: Any, K: Any) -> None:
    """Remove value from set (mutation).

    Args:
        s: Set
        value: Value to remove

    Raises:
        KeyError: If value not in set
    """
    s.discard(value)


def set_clear(s: EastSet, K: Any) -> None:
    """Remove all elements from set (mutation).

    Args:
        s: Set
    """
    s.clear()


def set_union(a: EastSet, b: EastSet, K: Any) -> EastSet:
    """Union of two sets.

    Args:
        a: First set
        b: Second set

    Returns:
        New set with union
    """
    return EastSet(K, list(a) + [x for x in b if x not in a])


def set_intersection(a: EastSet, b: EastSet, K: Any) -> EastSet:
    """Intersection of two sets.

    Args:
        a: First set
        b: Second set

    Returns:
        New set with intersection
    """
    return EastSet(K, [x for x in a if x in b])


def set_difference(a: EastSet, b: EastSet, K: Any) -> EastSet:
    """Difference of two sets (a - b).

    Args:
        a: First set
        b: Second set

    Returns:
        New set with elements in a but not in b
    """
    return EastSet(K, [x for x in a if x not in b])


def set_symmetric_difference(a: EastSet, b: EastSet, K: Any) -> EastSet:
    """Symmetric difference of two sets.

    Args:
        a: First set
        b: Second set

    Returns:
        New set with elements in either a or b but not both
    """
    in_a_not_b = [x for x in a if x not in b]
    in_b_not_a = [x for x in b if x not in a]
    return EastSet(K, in_a_not_b + in_b_not_a)


def set_is_subset(a: EastSet, b: EastSet, K: Any) -> bool:
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


def set_is_disjoint(a: EastSet, b: EastSet, K: Any) -> bool:
    """Check if two sets have no elements in common.

    Args:
        a: First set
        b: Second set

    Returns:
        True if a and b have no elements in common
    """
    return all(x not in b for x in a)


def set_copy(s: EastSet, K: Any) -> EastSet:
    """Create shallow copy of set.

    Args:
        s: Set

    Returns:
        New set with same elements
    """
    return EastSet(K, list(s))


def set_union_in_place(a: EastSet, b: EastSet, K: Any) -> None:
    """Union in place (mutation).

    Args:
        a: Set to modify
        b: Set to union with
    """
    for x in b:
        a.add(x)


def set_to_array(s: EastSet, func: Any, K: Any, T2: Any) -> EastArray:
    """Convert set to array using map function.

    Args:
        s: Set
        func: Callable to apply to each element

    Returns:
        Array with mapped values
    """
    s._lock_for_iteration()
    try:
        mapped = [func(item) for item in s]
        # Note: We return an array with the same element type as the set
        # The function might transform the type, but we don't have that info here
        return EastArray(K, mapped)
    finally:
        s._unlock_for_iteration()


# Additional set operations


def set_generate(n: int, gen_fn: Any, validate_fn: Any, K: Any) -> EastSet:
    """Generate set by calling functions.

    Args:
        n: Number of elements to generate
        gen_fn: Callable taking (Integer) -> K
        validate_fn: Callable taking (K) -> Null for validation (called only on duplicates)
        K: Key type

    Returns:
        Set of generated elements
    """
    elements = set()
    for i in range(n):
        element = gen_fn(i)
        # Only call validate_fn if element already exists (conflict)
        if element in elements:
            validate_fn(element)
        elements.add(element)
    return EastSet(K, elements)


def set_try_insert(s: EastSet, element: Any, K: Any) -> bool:
    """Try to insert element, return success.

    Args:
        s: Set
        element: Element to insert

    Returns:
        True if element was new (inserted), False if already present
    """
    was_new = element not in s
    s.add(element)
    return was_new


def set_try_delete(s: EastSet, element: Any, K: Any) -> bool:
    """Try to remove element, return success.

    Args:
        s: Set
        element: Element to remove

    Returns:
        True if element was present (removed), False if not found
    """
    if element in s:
        s.remove(element)
        return True
    return False


def set_for_each(s: EastSet, func: Any, K: Any, T2: Any) -> None:
    """Iterate over set (for side effects).

    Args:
        s: Set
        func: Callable taking (element) -> Any
    """
    s._lock_for_iteration()
    try:
        for element in s:
            func(element)
    finally:
        s._unlock_for_iteration()


def set_map(s: EastSet, func: Any, K: Any, T2: Any) -> Any:
    """Map set to dict.

    Args:
        s: Set
        func: Callable taking element and returning value
        K: Key type
        T2: Output value type

    Returns:
        Dict mapping each element to its mapped value
    """
    from east.types.containers import EastDict

    s._lock_for_iteration()
    try:
        result = EastDict(K, T2, {})
        for element in s:
            result[element] = func(element)
        return result
    finally:
        s._unlock_for_iteration()


def set_filter(s: EastSet, func: Any, K: Any) -> EastSet:
    """Filter set by predicate.

    Args:
        s: Set
        func: Callable taking element and returning boolean

    Returns:
        New set with filtered elements
    """
    s._lock_for_iteration()
    try:
        filtered = {element for element in s if func(element)}
        return EastSet(K, filtered)
    finally:
        s._unlock_for_iteration()


def set_filter_map(s: EastSet, func: Any, K: Any, V2: Any) -> Any:
    """Filter and map to dict.

    Args:
        s: Set
        func: Callable taking element -> Variant<none: Null, some: V2>

    Returns:
        Dict of unwrapped "some" values
    """
    from east.types.containers import EastDict

    s._lock_for_iteration()
    try:
        result = EastDict(K, V2, {})
        for element in s:
            variant = func(element)
            if variant.get("type") == "some":
                result[element] = variant["value"]
        return result
    finally:
        s._unlock_for_iteration()


def set_first_map(s: EastSet, func: Any, K: Any, T2: Any) -> Any:
    """Find first element that maps to "some".

    Args:
        s: Set
        func: Callable taking element -> Variant<none: Null, some: T2>

    Returns:
        First "some" value or "none"
    """
    from east.utils.variant import none

    s._lock_for_iteration()
    try:
        for element in s:
            variant = func(element)
            if variant.get("type") == "some":
                return variant
        return none()
    finally:
        s._unlock_for_iteration()


def set_map_reduce(s: EastSet, map_fn: Any, reduce_fn: Any, K: Any, T2: Any) -> Any:
    """Map then reduce.

    Args:
        s: Set
        map_fn: Callable taking element -> T2
        reduce_fn: Callable taking (T2, T2) -> T2 (associative)

    Returns:
        Reduced value
    """
    if len(s) == 0:
        raise ValueError("Cannot reduce empty set")

    s._lock_for_iteration()
    try:
        mapped = [map_fn(element) for element in s]
        result = mapped[0]
        for item in mapped[1:]:
            result = reduce_fn(result, item)
        return result
    finally:
        s._unlock_for_iteration()


def set_reduce(s: EastSet, func: Any, initial: Any, K: Any, T2: Any) -> Any:
    """Fold over set.

    Args:
        s: Set
        func: Callable taking (accumulator, element) -> accumulator
        initial: Initial accumulator value

    Returns:
        Final accumulator value
    """
    s._lock_for_iteration()
    try:
        accumulator = initial
        for element in s:
            accumulator = func(accumulator, element)
        return accumulator
    finally:
        s._unlock_for_iteration()


def set_to_set(s: EastSet, func: Any, K: Any, K2: Any) -> EastSet:
    """Map set to new set.

    Args:
        s: Set
        func: Callable taking element -> K2

    Returns:
        New set with mapped elements
    """
    s._lock_for_iteration()
    try:
        mapped = {func(element) for element in s}
        return EastSet(K2, mapped)
    finally:
        s._unlock_for_iteration()


def set_to_dict(
    s: EastSet, key_fn: Any, value_fn: Any, merge_fn: Any, K: Any, K2: Any, T2: Any
) -> Any:
    """Convert set to dict using key and value functions.

    Args:
        s: Set
        key_fn: Callable taking element -> K2
        value_fn: Callable taking element -> V2
        merge_fn: Callable taking (V2, V2, K2) -> V2 for duplicate keys
        K: Set element type
        K2: Dict key type
        T2: Dict value type

    Returns:
        Dict
    """
    from east.types.containers import EastDict

    s._lock_for_iteration()
    try:
        result = EastDict(K2, T2, {})
        for element in s:
            key = key_fn(element)
            value = value_fn(element)
            if key in result:
                result[key] = merge_fn(result[key], value, key)
            else:
                result[key] = value
        return result
    finally:
        s._unlock_for_iteration()


def set_flatten_to_array(s: EastSet, func: Any, K: Any, T2: Any) -> EastArray:
    """Flat map to array.

    Args:
        s: Set
        func: Callable taking element -> Array<T2>
        K: Input key type
        T2: Output element type

    Returns:
        Flattened array
    """
    s._lock_for_iteration()
    try:
        results = []
        for element in s:
            mapped = func(element)
            results.extend(mapped)
        return EastArray(T2, results)
    finally:
        s._unlock_for_iteration()


def set_flatten_to_set(s: EastSet, func: Any, K: Any, K2: Any) -> EastSet:
    """Flat map to set.

    Args:
        s: Set
        func: Callable taking element -> Set<K2>
        K: Input key type
        K2: Output key type

    Returns:
        Union of all mapped sets
    """
    s._lock_for_iteration()
    try:
        result = set()
        for element in s:
            mapped = func(element)
            result.update(mapped)
        return EastSet(K2, result)
    finally:
        s._unlock_for_iteration()


def set_flatten_to_dict(s: EastSet, func: Any, merge_fn: Any, K: Any, K2: Any, T2: Any) -> Any:
    """Flat map to dict.

    Args:
        s: Set
        func: Callable taking element -> Dict<K2, V2>
        merge_fn: Callable taking (V2, V2, K2) -> V2
        K: Input key type
        K2: Output key type
        T2: Output value type

    Returns:
        Merged dict
    """
    from east.types.containers import EastDict

    s._lock_for_iteration()
    try:
        result = EastDict(K2, T2, {})
        for element in s:
            mapped = func(element)
            for key, value in mapped.items():
                if key in result:
                    result[key] = merge_fn(result[key], value, key)
                else:
                    result[key] = value
        return result
    finally:
        s._unlock_for_iteration()


def set_group_fold(
    s: EastSet, key_fn: Any, init_fn: Any, fold_fn: Any, K: Any, K2: Any, T2: Any
) -> Any:
    """Group by key and fold each group.

    Args:
        s: Set
        key_fn: Callable taking element -> K2
        init_fn: Callable taking (K2) -> T2 for initial value
        fold_fn: Callable taking (T2, element) -> T2

    Returns:
        Dict mapping keys to folded values
    """
    from east.types.containers import EastDict

    s._lock_for_iteration()
    try:
        result = EastDict(K2, T2, {})
        for element in s:
            key = key_fn(element)
            if key not in result:
                result[key] = init_fn(key)
            result[key] = fold_fn(result[key], element)
        return result
    finally:
        s._unlock_for_iteration()


# Register all set builtins
register_builtin("SetGenerate", set_generate)
register_builtin("SetSize", set_size)
register_builtin("SetHas", set_has)
register_builtin("SetInsert", set_add)  # Renamed from SetAdd
register_builtin("SetTryInsert", set_try_insert)
register_builtin("SetDelete", set_remove)  # Renamed from SetRemove
register_builtin("SetTryDelete", set_try_delete)
register_builtin("SetClear", set_clear)
register_builtin("SetUnionInPlace", set_union_in_place)
register_builtin("SetUnion", set_union)
register_builtin("SetIntersect", set_intersection)  # Renamed from SetIntersection
register_builtin("SetDiff", set_difference)  # Renamed from SetDifference
register_builtin("SetSymDiff", set_symmetric_difference)  # Renamed from SetSymmetricDifference
register_builtin("SetIsSubset", set_is_subset)
register_builtin("SetIsDisjoint", set_is_disjoint)
register_builtin("SetCopy", set_copy)
register_builtin("SetForEach", set_for_each)
register_builtin("SetMap", set_map)
register_builtin("SetFilter", set_filter)
register_builtin("SetFilterMap", set_filter_map)
register_builtin("SetFirstMap", set_first_map)
register_builtin("SetMapReduce", set_map_reduce)
register_builtin("SetReduce", set_reduce)
register_builtin("SetToArray", set_to_array)
register_builtin("SetToSet", set_to_set)
register_builtin("SetToDict", set_to_dict)
register_builtin("SetFlattenToArray", set_flatten_to_array)
register_builtin("SetFlattenToSet", set_flatten_to_set)
register_builtin("SetFlattenToDict", set_flatten_to_dict)
register_builtin("SetGroupFold", set_group_fold)


__all__ = [
    "set_generate",
    "set_size",
    "set_has",
    "set_add",
    "set_try_insert",
    "set_remove",
    "set_try_delete",
    "set_clear",
    "set_union",
    "set_union_in_place",
    "set_intersection",
    "set_difference",
    "set_symmetric_difference",
    "set_is_subset",
    "set_is_disjoint",
    "set_copy",
    "set_for_each",
    "set_map",
    "set_filter",
    "set_filter_map",
    "set_first_map",
    "set_map_reduce",
    "set_reduce",
    "set_to_array",
    "set_to_set",
    "set_to_dict",
    "set_flatten_to_array",
    "set_flatten_to_set",
    "set_flatten_to_dict",
    "set_group_fold",
]
