"""Dict builtin functions."""

from typing import Any

from east.builtins.registry import register_builtin
from east.types.containers import EastArray, EastDict, EastSet


def dict_size(d: EastDict, K: Any, V: Any) -> int:
    """Get size of dict.

    Args:
        d: Dict

    Returns:
        Number of entries in dict
    """
    return len(d)


def dict_has(d: EastDict, key: Any, K: Any, V: Any) -> bool:
    """Check if dict has key.

    Args:
        d: Dict
        key: Key to check

    Returns:
        True if key is in dict
    """
    return key in d


def dict_get(d: EastDict, key: Any, K: Any, V: Any) -> Any:
    """Get value for key.

    Args:
        d: Dict
        key: Key to look up

    Returns:
        Value for key

    Raises:
        KeyError: If key not in dict
    """
    return d[key]


def dict_set(d: EastDict, key: Any, value: Any, K: Any, V: Any) -> None:
    """Set value for key (mutation).

    Args:
        d: Dict
        key: Key to set
        value: Value to set
    """
    d[key] = value


def dict_remove(d: EastDict, key: Any, K: Any, V: Any) -> None:
    """Remove key from dict (mutation).

    Args:
        d: Dict
        key: Key to remove

    Raises:
        KeyError: If key not in dict
    """
    del d[key]


def dict_clear(d: EastDict, K: Any, V: Any) -> None:
    """Remove all entries from dict (mutation).

    Args:
        d: Dict
    """
    d.clear()


def dict_keys(d: EastDict, K: Any, V: Any) -> EastSet:
    """Get set of dict keys.

    Args:
        d: Dict

    Returns:
        Set of keys
    """
    return EastSet(K, set(d.keys()))


def dict_merge(
    d: EastDict, key: Any, value: Any, merge_fn: Any, initial_fn: Any, K: Any, V: Any, V2: Any
) -> None:
    """Merge single key-value pair into dict (mutation).

    Args:
        d: Dict to modify
        key: Key to merge
        value: Value to merge (type V2)
        merge_fn: Callable taking (V, V2, K) -> V for merging
        initial_fn: Callable taking K -> V for missing keys

    Returns:
        None (mutates dict)
    """
    if key in d:
        existing = d[key]
    else:
        existing = initial_fn(key)
    d[key] = merge_fn(existing, value, key)


def dict_get_or_default(d: EastDict, key: Any, default_fn: Any, K: Any, V: Any) -> Any:
    """Get value for key or call default function if key not in dict.

    Args:
        d: Dict
        key: Key to look up
        default_fn: Function taking key and returning default value

    Returns:
        Value for key, or result of default_fn(key) if key not in dict
    """
    if key in d:
        return d[key]
    return default_fn(key)


def dict_copy(d: EastDict, K: Any, V: Any) -> EastDict:
    """Create shallow copy of dict.

    Args:
        d: Dict

    Returns:
        New dict with same entries
    """
    return EastDict(K, V, dict(d.items()))


def dict_update(d: EastDict, key: Any, value: Any, K: Any, V: Any) -> None:
    """Update value for existing key (mutation).

    Args:
        d: Dict
        key: Key to update
        value: New value

    Raises:
        KeyError: If key not in dict
    """
    if key not in d:
        raise KeyError(f"Key not in dict: {key}")
    d[key] = value


# Additional dict operations


def dict_generate(n: int, key_fn: Any, value_fn: Any, merge_fn: Any, K: Any, V: Any) -> EastDict:
    """Generate dict by calling functions.

    Args:
        n: Number of entries to generate
        key_fn: Callable taking (Integer) -> K
        value_fn: Callable taking (Integer) -> V
        merge_fn: Callable taking (V, V, K) -> V for duplicate keys
        K: Key type
        V: Value type

    Returns:
        Dict with generated entries
    """
    result = EastDict(K, V, {})
    for i in range(n):
        key = key_fn(i)
        value = value_fn(i)
        if key in result:
            result[key] = merge_fn(result[key], value, key)
        else:
            result[key] = value
    return result


def dict_try_get(d: EastDict, key: Any, K: Any, V: Any) -> Any:
    """Get value as Option variant.

    Args:
        d: Dict
        key: Key to lookup

    Returns:
        {type: "some", value: value} or {type: "none", value: null}
    """
    from east.utils.variant import none, some

    if key in d:
        return some(d[key])
    return none()


def dict_get_or_insert(d: EastDict, key: Any, default_fn: Any, K: Any, V: Any) -> Any:
    """Get existing value or insert default.

    Args:
        d: Dict
        key: Key to lookup/insert
        default_fn: Callable taking (K) -> V for default value

    Returns:
        Existing value or newly inserted default
    """
    if key in d:
        return d[key]
    value = default_fn(key)
    d[key] = value
    return value


def dict_insert_or_update(d: EastDict, key: Any, value: Any, merge_fn: Any, K: Any, V: Any) -> None:
    """Insert or merge with existing value.

    Args:
        d: Dict
        key: Key to insert/update
        value: Value to insert
        merge_fn: Callable taking (V, V, K) -> V for merging
    """
    if key in d:
        d[key] = merge_fn(d[key], value, key)
    else:
        d[key] = value


def dict_swap(d: EastDict, key: Any, value: Any, K: Any, V: Any) -> Any:
    """Replace value and return old value.

    Args:
        d: Dict
        key: Key to replace
        value: New value

    Returns:
        Old value

    Raises:
        KeyError: If key not found
    """
    if key not in d:
        raise KeyError(f"Key not in dict: {key}")
    old_value = d[key]
    d[key] = value
    return old_value


def dict_try_delete(d: EastDict, key: Any, K: Any, V: Any) -> bool:
    """Try to remove key, return success.

    Args:
        d: Dict
        key: Key to remove

    Returns:
        True if key was present (removed), False if not found
    """
    if key in d:
        del d[key]
        return True
    return False


def dict_pop(d: EastDict, key: Any, K: Any, V: Any) -> Any:
    """Remove key and return value.

    Args:
        d: Dict
        key: Key to remove

    Returns:
        Value for removed key

    Raises:
        KeyError: If key not found
    """
    return d.pop(key)


def dict_union_in_place(d: EastDict, other: EastDict, merge_fn: Any, K: Any, V: Any) -> None:
    """Merge another dict into this one (mutates).

    Args:
        d: Dict to modify
        other: Dict to merge from
        merge_fn: Callable taking (V, V, K) -> V for duplicate keys
    """
    for key, value in other.items():
        if key in d:
            d[key] = merge_fn(d[key], value, key)
        else:
            d[key] = value


def dict_merge_all(
    d: EastDict, other: EastDict, merge_fn: Any, default_fn: Any, K: Any, V: Any, V2: Any
) -> None:
    """Merge another dict with different value type.

    Args:
        d: Dict to modify
        other: Dict to merge from
        merge_fn: Callable taking (V, V2, K) -> V for merging
        default_fn: Callable taking (K) -> V for missing keys
    """
    for key, value in other.items():
        if key in d:
            d[key] = merge_fn(d[key], value, key)
        else:
            d[key] = default_fn(key)


def dict_get_keys(d: EastDict, keys: EastSet, default_fn: Any, K: Any, V: Any) -> EastDict:
    """Get multiple keys, using default for missing.

    Args:
        d: Dict
        keys: Set of keys to get
        default_fn: Callable taking (K) -> V for missing keys

    Returns:
        Dict with requested keys
    """
    result = EastDict(K, V, {})
    for key in keys:
        if key in d:
            result[key] = d[key]
        else:
            result[key] = default_fn(key)
    return result


def dict_for_each(d: EastDict, func: Any, K: Any, V: Any, T2: Any) -> None:
    """Iterate over dict (for side effects).

    Args:
        d: Dict
        func: Callable taking (value, key) -> Any
    """
    for key, value in d.items():
        func(value, key)


def dict_map(d: EastDict, func: Any, K: Any, V: Any, V2: Any) -> EastDict:
    """Map values to new type.

    Args:
        d: Dict
        func: Callable taking (value, key) -> V2
        K: Key type
        V: Input value type
        V2: Output value type

    Returns:
        Dict with mapped values
    """
    result = EastDict(K, V2, {})
    for key, value in d.items():
        result[key] = func(value, key)
    return result


def dict_filter(d: EastDict, func: Any, K: Any, V: Any) -> EastDict:
    """Filter dict by predicate.

    Args:
        d: Dict
        func: Callable taking (value, key) -> Boolean

    Returns:
        Dict with filtered entries
    """
    result = EastDict(K, V, {})
    for key, value in d.items():
        if func(value, key):
            result[key] = value
    return result


def dict_filter_map(d: EastDict, func: Any, K: Any, V: Any, V2: Any) -> EastDict:
    """Filter and map values.

    Args:
        d: Dict
        func: Callable taking (value, key) -> Variant<none: Null, some: V2>

    Returns:
        Dict of unwrapped "some" values
    """
    result = EastDict(K, V2, {})
    for key, value in d.items():
        variant = func(value, key)
        if variant.get("type") == "some":
            result[key] = variant["value"]
    return result


def dict_first_map(d: EastDict, func: Any, K: Any, V: Any, T2: Any) -> Any:
    """Find first entry that maps to "some".

    Args:
        d: Dict
        func: Callable taking (value, key) -> Variant<none: Null, some: T2>

    Returns:
        First "some" value or "none"
    """
    from east.utils.variant import none

    for key, value in d.items():
        variant = func(value, key)
        if variant.get("type") == "some":
            return variant
    return none()


def dict_map_reduce(d: EastDict, map_fn: Any, reduce_fn: Any, K: Any, V: Any, T2: Any) -> Any:
    """Map then reduce.

    Args:
        d: Dict
        map_fn: Callable taking (value, key) -> T2
        reduce_fn: Callable taking (T2, T2) -> T2 (associative)

    Returns:
        Reduced value
    """
    if len(d) == 0:
        raise ValueError("Cannot reduce empty dict")

    mapped = [map_fn(value, key) for key, value in d.items()]
    result = mapped[0]
    for item in mapped[1:]:
        result = reduce_fn(result, item)
    return result


def dict_reduce(d: EastDict, initial: Any, func: Any, K: Any, V: Any, T2: Any) -> Any:
    """Fold over dict.

    Args:
        d: Dict
        func: Callable taking (accumulator, value, key) -> accumulator
        initial: Initial accumulator value

    Returns:
        Final accumulator value
    """
    accumulator = initial
    for key, value in d.items():
        accumulator = func(accumulator, value, key)
    return accumulator


def dict_to_array(d: EastDict, func: Any, K: Any, V: Any, T2: Any) -> EastArray:
    """Convert dict to array using map function.

    Args:
        d: Dict
        func: Callable taking (value, key) -> T2

    Returns:
        Array of mapped values
    """
    # Sort by keys for deterministic ordering
    sorted_items = sorted(d.items(), key=lambda x: (type(x[0]).__name__, x[0]))
    mapped = [func(value, key) for key, value in sorted_items]
    return EastArray(T2, mapped)


def dict_to_set(d: EastDict, func: Any, K: Any, V: Any, K2: Any) -> EastSet:
    """Convert dict to set using map function.

    Args:
        d: Dict
        func: Callable taking (value, key) -> K2

    Returns:
        Set of mapped keys
    """
    mapped = {func(value, key) for key, value in d.items()}
    return EastSet(K2, mapped)


def dict_to_dict(
    d: EastDict, key_fn: Any, value_fn: Any, merge_fn: Any, K: Any, V: Any, K2: Any, V2: Any
) -> EastDict:
    """Map dict to new dict with different key/value types.

    Args:
        d: Dict
        key_fn: Callable taking (value, key) -> K2
        value_fn: Callable taking (value, key) -> V2
        merge_fn: Callable taking (V2, V2, K2) -> V2 for duplicate keys

    Returns:
        New dict with mapped keys and values
    """
    result = EastDict(K2, V2, {})
    for key, value in d.items():
        new_key = key_fn(value, key)
        new_value = value_fn(value, key)
        if new_key in result:
            result[new_key] = merge_fn(result[new_key], new_value, new_key)
        else:
            result[new_key] = new_value
    return result


def dict_flatten_to_array(d: EastDict, func: Any, K: Any, V: Any, T2: Any) -> EastArray:
    """Flat map to array.

    Args:
        d: Dict
        func: Callable taking (value, key) -> Array<T2>
        K: Input key type
        V: Input value type
        T2: Output element type

    Returns:
        Flattened array
    """
    results = []
    for key, value in d.items():
        mapped = func(value, key)
        results.extend(mapped)
    return EastArray(T2, results)


def dict_flatten_to_set(d: EastDict, func: Any, K: Any, V: Any, K2: Any) -> EastSet:
    """Flat map to set.

    Args:
        d: Dict
        func: Callable taking (value, key) -> Set<K2>
        K: Input key type
        V: Input value type
        K2: Output key type

    Returns:
        Union of all mapped sets
    """
    result = set()
    for key, value in d.items():
        mapped = func(value, key)
        result.update(mapped)
    return EastSet(K2, result)


def dict_flatten_to_dict(
    d: EastDict, func: Any, merge_fn: Any, K: Any, V: Any, K2: Any, V2: Any
) -> EastDict:
    """Flat map to dict.

    Args:
        d: Dict
        func: Callable taking (value, key) -> Dict<K2, V2>
        merge_fn: Callable taking (V2, V2, K2) -> V2
        K: Input key type
        V: Input value type
        K2: Output key type
        V2: Output value type

    Returns:
        Merged dict
    """
    result = EastDict(K2, V2, {})
    for key, value in d.items():
        mapped = func(value, key)
        for k, v in mapped.items():
            if k in result:
                result[k] = merge_fn(result[k], v, k)
            else:
                result[k] = v
    return result


def dict_group_fold(
    d: EastDict, key_fn: Any, init_fn: Any, fold_fn: Any, K: Any, V: Any, K2: Any, T2: Any
) -> EastDict:
    """Group by key and fold each group.

    Args:
        d: Dict
        key_fn: Callable taking (value, key) -> K2
        init_fn: Callable taking (K2) -> T2 for initial value
        fold_fn: Callable taking (T2, value, key) -> T2

    Returns:
        Dict mapping keys to folded values
    """
    result = EastDict(K2, T2, {})
    for key, value in d.items():
        group_key = key_fn(value, key)
        if group_key not in result:
            result[group_key] = init_fn(group_key)
        result[group_key] = fold_fn(result[group_key], value, key)
    return result


# Register all dict builtins
register_builtin("DictGenerate", dict_generate)
register_builtin("DictSize", dict_size)
register_builtin("DictHas", dict_has)
register_builtin("DictGet", dict_get)
register_builtin("DictGetOrDefault", dict_get_or_default)
register_builtin("DictTryGet", dict_try_get)
register_builtin("DictInsert", dict_set)  # Renamed from DictSet
register_builtin("DictGetOrInsert", dict_get_or_insert)
register_builtin("DictInsertOrUpdate", dict_insert_or_update)
register_builtin("DictUpdate", dict_update)
register_builtin("DictSwap", dict_swap)
register_builtin("DictMerge", dict_merge)
register_builtin("DictDelete", dict_remove)  # Renamed from DictRemove
register_builtin("DictTryDelete", dict_try_delete)
register_builtin("DictPop", dict_pop)
register_builtin("DictClear", dict_clear)
register_builtin("DictUnionInPlace", dict_union_in_place)
register_builtin("DictMergeAll", dict_merge_all)
register_builtin("DictKeys", dict_keys)
register_builtin("DictGetKeys", dict_get_keys)
register_builtin("DictForEach", dict_for_each)
register_builtin("DictCopy", dict_copy)
register_builtin("DictMap", dict_map)
register_builtin("DictFilter", dict_filter)
register_builtin("DictFilterMap", dict_filter_map)
register_builtin("DictFirstMap", dict_first_map)
register_builtin("DictMapReduce", dict_map_reduce)
register_builtin("DictReduce", dict_reduce)
register_builtin("DictToArray", dict_to_array)
register_builtin("DictToSet", dict_to_set)
register_builtin("DictToDict", dict_to_dict)
register_builtin("DictFlattenToArray", dict_flatten_to_array)
register_builtin("DictFlattenToSet", dict_flatten_to_set)
register_builtin("DictFlattenToDict", dict_flatten_to_dict)
register_builtin("DictGroupFold", dict_group_fold)


__all__ = [
    "dict_generate",
    "dict_size",
    "dict_has",
    "dict_get",
    "dict_get_or_default",
    "dict_try_get",
    "dict_set",
    "dict_get_or_insert",
    "dict_insert_or_update",
    "dict_update",
    "dict_swap",
    "dict_merge",
    "dict_remove",
    "dict_try_delete",
    "dict_pop",
    "dict_clear",
    "dict_union_in_place",
    "dict_merge_all",
    "dict_keys",
    "dict_get_keys",
    "dict_for_each",
    "dict_copy",
    "dict_map",
    "dict_filter",
    "dict_filter_map",
    "dict_first_map",
    "dict_map_reduce",
    "dict_reduce",
    "dict_to_array",
    "dict_to_set",
    "dict_to_dict",
    "dict_flatten_to_array",
    "dict_flatten_to_set",
    "dict_flatten_to_dict",
    "dict_group_fold",
]
