"""Array builtin functions."""

from typing import Any

from east.builtins.registry import register_builtin
from east.types.containers import EastArray


def array_length(arr: EastArray, T: Any) -> int:
    """Get length of array.

    Args:
        arr: Array

    Returns:
        Number of elements in array
    """
    return len(arr)


def array_get(arr: EastArray, index: int, T: Any) -> Any:
    """Get element at index.

    Args:
        arr: Array
        index: Element index (0-based)

    Returns:
        Element at index

    Raises:
        IndexError: If index out of bounds
    """
    return arr[index]


def array_set(arr: EastArray, index: int, value: Any, T: Any) -> None:
    """Set element at index (mutation).

    Args:
        arr: Array
        index: Element index (0-based)
        value: New value

    Raises:
        IndexError: If index out of bounds
    """
    arr[index] = value


def array_push_first(arr: EastArray, value: Any, T: Any) -> None:
    """Prepend element to array (mutation).

    Args:
        arr: Array
        value: Value to prepend
    """
    arr.insert(0, value)


def array_push_last(arr: EastArray, value: Any, T: Any) -> None:
    """Append element to array (mutation).

    Args:
        arr: Array
        value: Value to append
    """
    arr.append(value)


def array_pop_first(arr: EastArray, T: Any) -> Any:
    """Remove and return first element (mutation).

    Args:
        arr: Array

    Returns:
        First element

    Raises:
        IndexError: If array is empty
    """
    return arr.pop(0)


def array_pop_last(arr: EastArray, T: Any) -> Any:
    """Remove and return last element (mutation).

    Args:
        arr: Array

    Returns:
        Last element

    Raises:
        IndexError: If array is empty
    """
    return arr.pop()


def array_slice(arr: EastArray, start: int, end: int, T: Any) -> EastArray:
    """Get array slice.

    Args:
        arr: Array
        start: Start index (inclusive)
        end: End index (exclusive)

    Returns:
        New array with slice
    """
    return EastArray(T, arr[start:end])


def array_concat(a: EastArray, b: EastArray, T: Any) -> EastArray:
    """Concatenate two arrays.

    Args:
        a: First array
        b: Second array

    Returns:
        New array with concatenated elements
    """
    return EastArray(T, list(a) + list(b))


def array_reverse(arr: EastArray, T: Any) -> EastArray:
    """Reverse array.

    Args:
        arr: Array

    Returns:
        New array with reversed elements
    """
    return EastArray(T, list(reversed(arr)))


def array_sort(arr: EastArray, key_fn: Any, T: Any, T2: Any) -> EastArray:
    """Sort array by key function.

    Args:
        arr: Array
        key_fn: Callable taking element and returning sort key

    Returns:
        New sorted array
    """
    from functools import cmp_to_key

    from east.utils.ordering import compare_for

    # Compute keys for each element
    keys = [key_fn(item) for item in arr]

    # Sort by keys using East ordering
    compare = compare_for(T2)
    sorted_indices = sorted(range(len(arr)), key=lambda i: cmp_to_key(compare)(keys[i]))
    sorted_items = [arr[i] for i in sorted_indices]
    return EastArray(T, sorted_items)


def array_get_or_default(arr: EastArray, index: int, default_fn: Any, T: Any) -> Any:
    """Get element at index or call default function if out of bounds.

    Args:
        arr: Array
        index: Element index (0-based)
        default_fn: Callable taking (Integer) -> T

    Returns:
        Element at index, or default_fn(index) if index out of bounds
    """
    if 0 <= index < len(arr):
        return arr[index]
    return default_fn(index)


def array_clear(arr: EastArray, T: Any) -> None:
    """Remove all elements from array (mutation).

    Args:
        arr: Array
    """
    arr.clear()


def array_copy(arr: EastArray, T: Any) -> EastArray:
    """Create shallow copy of array.

    Args:
        arr: Array

    Returns:
        New array with same elements
    """
    return EastArray(T, list(arr))


def array_reverse_in_place(arr: EastArray, T: Any) -> None:
    """Reverse array in place (mutation).

    Args:
        arr: Array
    """
    arr.reverse()


def array_sort_in_place(arr: EastArray, key_fn: Any, T: Any, T2: Any) -> None:
    """Sort array in place by key function (mutation).

    Args:
        arr: Array
        key_fn: Callable taking element and returning sort key
    """
    from functools import cmp_to_key

    from east.utils.ordering import compare_for

    # Compute keys for each element
    keys = [key_fn(item) for item in arr]

    # Sort by keys using East ordering
    compare = compare_for(T2)
    sorted_indices = sorted(range(len(arr)), key=lambda i: cmp_to_key(compare)(keys[i]))
    sorted_items = [arr[i] for i in sorted_indices]
    arr.clear()
    arr.extend(sorted_items)


def array_range(start: int, end: int, step: int) -> EastArray:
    """Create array from range.

    Args:
        start: Start value (inclusive)
        end: End value (exclusive)
        step: Step size

    Returns:
        Array of integers from start to end by step
    """
    from east.types.types import IntegerType

    return EastArray(IntegerType, list(range(start, end, step)))


# Higher-order functions


def array_map(arr: EastArray, func: Any, T: Any, T2: Any) -> EastArray:
    """Map function over array elements.

    Args:
        arr: Array
        func: Callable taking (element, index) and returning new value

    Returns:
        New array with mapped values
    """
    arr._lock_for_iteration()
    try:
        mapped = [func(item, index) for index, item in enumerate(arr)]
        return EastArray(T2, mapped)
    finally:
        arr._unlock_for_iteration()


def array_filter(arr: EastArray, func: Any, T: Any) -> EastArray:
    """Filter array elements by predicate.

    Args:
        arr: Array
        func: Callable taking (element, index) and returning boolean

    Returns:
        New array with filtered elements
    """
    arr._lock_for_iteration()
    try:
        filtered = [item for index, item in enumerate(arr) if func(item, index)]
        return EastArray(T, filtered)
    finally:
        arr._unlock_for_iteration()


def array_reduce(arr: EastArray, initial: Any, func: Any, T: Any, T2: Any) -> Any:
    """Reduce array to single value.

    Args:
        arr: Array
        initial: Initial accumulator value
        func: Callable taking (accumulator, element, index) and returning new accumulator

    Returns:
        Final accumulator value
    """
    arr._lock_for_iteration()
    try:
        accumulator = initial
        for index, item in enumerate(arr):
            accumulator = func(accumulator, item, index)
        return accumulator
    finally:
        arr._unlock_for_iteration()


# Additional array operations


def array_generate(n: int, func: Any, T: Any) -> EastArray:
    """Generate array by calling function for each index.

    Args:
        n: Number of elements to generate
        func: Callable taking (Integer) -> T
        T: Element type

    Returns:
        Array of n elements
    """
    elements = [func(i) for i in range(n)]
    return EastArray(T, elements)


def array_linspace(start: float, end: float, n: int) -> EastArray:
    """Generate linearly spaced floats.

    Args:
        start: Start value (inclusive)
        end: End value (inclusive)
        n: Number of elements

    Returns:
        Array of n evenly spaced values from start to end
    """
    from east.types.types import FloatType

    if n == 1:
        return EastArray(FloatType, [start])
    step = (end - start) / (n - 1)
    elements = [start + i * step for i in range(n)]
    return EastArray(FloatType, elements)


def array_has(arr: EastArray, index: int, T: Any) -> bool:
    """Check if index exists in array.

    Args:
        arr: Array
        index: Index to check

    Returns:
        True if 0 <= index < len(array)
    """
    return 0 <= index < len(arr)


def array_try_get(arr: EastArray, index: int, T: Any) -> Any:
    """Get element as Option variant.

    Args:
        arr: Array
        index: Element index

    Returns:
        {type: "some", value: element} or {type: "none", value: null}
    """
    from east.utils.variant import none, some

    if 0 <= index < len(arr):
        return some(arr[index])
    return none()


def array_merge(arr: EastArray, index: int, value: Any, func: Any, T: Any, T2: Any) -> None:
    """Merge value at index using function.

    This is a mutation operation that returns Null (None).

    Args:
        arr: Array
        index: Index to merge at
        value: New value
        func: Callable taking (old_value, new_value, index) -> merged_value

    Returns:
        None (side effect only)
    """
    old_value = arr[index]
    arr[index] = func(old_value, value, index)


def array_append(arr: EastArray, other: EastArray, T: Any) -> None:
    """Append another array to end (mutation).

    Args:
        arr: Array to extend
        other: Array to append
    """
    arr.extend(other)


def array_prepend(arr: EastArray, other: EastArray, T: Any) -> None:
    """Prepend another array to start (mutation).

    Args:
        arr: Array to extend
        other: Array to prepend
    """
    for i, item in enumerate(other):
        arr.insert(i, item)


def array_merge_all(arr: EastArray, other: EastArray, func: Any, T: Any, T2: Any) -> None:
    """Merge another array element-wise (mutation).

    Args:
        arr: Array to modify
        other: Array to merge from
        func: Callable taking (T, T2, Integer) -> T
    """
    for i, item in enumerate(other):
        if i < len(arr):
            arr[i] = func(arr[i], item, i)


def array_is_sorted(arr: EastArray, key_fn: Any, T: Any, T2: Any) -> bool:
    """Check if array is sorted by key function.

    Args:
        arr: Array
        key_fn: Callable taking element and returning sort key

    Returns:
        True if all adjacent pairs are ordered
    """
    from east.utils.ordering import compare_for

    if len(arr) <= 1:
        return True

    compare = compare_for(T2)
    keys = [key_fn(item) for item in arr]
    for i in range(len(keys) - 1):
        if compare(keys[i], keys[i + 1]) > 0:
            return False
    return True


def array_find_sorted_first(arr: EastArray, target: Any, key_fn: Any, T: Any, T2: Any) -> int:
    """Binary search for first occurrence.

    Args:
        arr: Sorted array
        target: Target key value
        key_fn: Callable taking element and returning sort key

    Returns:
        Index of first element with key >= target
    """
    from east.utils.ordering import compare_for

    compare = compare_for(T2)
    left, right = 0, len(arr)
    while left < right:
        mid = (left + right) // 2
        key = key_fn(arr[mid])
        if compare(key, target) < 0:
            left = mid + 1
        else:
            right = mid
    return left


def array_find_sorted_last(arr: EastArray, target: Any, key_fn: Any, T: Any, T2: Any) -> int:
    """Binary search for last occurrence.

    Args:
        arr: Sorted array
        target: Target key value
        key_fn: Callable taking element and returning sort key

    Returns:
        Index of first element with key > target (exclusive end of range)
    """
    from east.utils.ordering import compare_for

    compare = compare_for(T2)
    left, right = 0, len(arr)
    while left < right:
        mid = (left + right) // 2
        key = key_fn(arr[mid])
        if compare(key, target) <= 0:
            left = mid + 1
        else:
            right = mid
    return left


def array_find_sorted_range(arr: EastArray, target: Any, key_fn: Any, T: Any, T2: Any) -> Any:
    """Binary search for range of occurrences.

    Args:
        arr: Sorted array
        target: Target key value
        key_fn: Callable taking element and returning sort key

    Returns:
        {start: first_index, end: last_index}
    """
    start = array_find_sorted_first(arr, target, key_fn, T, T2)
    end = array_find_sorted_last(arr, target, key_fn, T, T2)
    return {"start": start, "end": end}


def array_find_first(arr: EastArray, target: Any, key_fn: Any, T: Any, T2: Any) -> Any:
    """Linear search for first occurrence.

    Args:
        arr: Array
        target: Target key value
        key_fn: Callable taking element and returning sort key

    Returns:
        {type: "some", value: index} or {type: "none", value: null}
    """
    from east.utils.ordering import compare_for
    from east.utils.variant import none, some

    arr._lock_for_iteration()
    try:
        compare = compare_for(T2)
        for index, item in enumerate(arr):
            key = key_fn(item)
            if compare(key, target) == 0:
                return some(index)
        return none()
    finally:
        arr._unlock_for_iteration()


def array_get_keys(arr: EastArray, indices: EastArray, default_fn: Any, T: Any) -> EastArray:
    """Get multiple elements by index array.

    Args:
        arr: Array
        indices: Array of indices
        default_fn: Callable taking (Integer) -> T for invalid indices

    Returns:
        Array of elements
    """
    elements = []
    for index in indices:
        if 0 <= index < len(arr):
            elements.append(arr[index])
        else:
            elements.append(default_fn(index))
    return EastArray(T, elements)


def array_for_each(arr: EastArray, func: Any, T: Any, T2: Any) -> None:
    """Iterate over array (for side effects).

    Args:
        arr: Array
        func: Callable taking (element, index) -> Any
    """
    arr._lock_for_iteration()
    try:
        for index, item in enumerate(arr):
            func(item, index)
    finally:
        arr._unlock_for_iteration()


def array_filter_map(arr: EastArray, func: Any, T: Any, T2: Any) -> EastArray:
    """Filter and map in one pass.

    Args:
        arr: Array
        func: Callable taking (element, index) -> Variant<none: Null, some: T2>
        T: Input element type
        T2: Output element type

    Returns:
        Array of unwrapped "some" values
    """
    arr._lock_for_iteration()
    try:
        results = []
        for index, item in enumerate(arr):
            result = func(item, index)
            if result.get("type") == "some":
                results.append(result["value"])
        return EastArray(T2, results)
    finally:
        arr._unlock_for_iteration()


def array_first_map(arr: EastArray, func: Any, T: Any, T2: Any) -> Any:
    """Find first element that maps to "some".

    Args:
        arr: Array
        func: Callable taking (element, index) -> Variant<none: Null, some: T2>

    Returns:
        First "some" value or "none"
    """
    from east.utils.variant import none

    arr._lock_for_iteration()
    try:
        for index, item in enumerate(arr):
            result = func(item, index)
            if result.get("type") == "some":
                return result
        return none()
    finally:
        arr._unlock_for_iteration()


def array_map_reduce(arr: EastArray, map_fn: Any, reduce_fn: Any, T: Any, T2: Any) -> Any:
    """Map then reduce.

    Args:
        arr: Array
        map_fn: Callable taking (element, index) -> T2
        reduce_fn: Callable taking (T2, T2) -> T2 (associative operator)

    Returns:
        Reduced value
    """
    if len(arr) == 0:
        raise ValueError("Cannot reduce empty array")

    arr._lock_for_iteration()
    try:
        mapped = [map_fn(item, index) for index, item in enumerate(arr)]
        result = mapped[0]
        for item in mapped[1:]:
            result = reduce_fn(result, item)
        return result
    finally:
        arr._unlock_for_iteration()


def array_string_join(arr: EastArray, delimiter: str) -> str:
    """Join string array with delimiter.

    Args:
        arr: Array of strings
        delimiter: Delimiter string

    Returns:
        Joined string
    """
    return delimiter.join(arr)


def array_to_set(arr: EastArray, key_fn: Any, T: Any, K2: Any) -> Any:
    """Convert array to set using key function.

    Args:
        arr: Array
        key_fn: Callable taking (element, index) -> K2
        T: Array element type
        K2: Set key type

    Returns:
        EastSet
    """
    from east.types.containers import EastSet

    arr._lock_for_iteration()
    try:
        keys = {key_fn(item, index) for index, item in enumerate(arr)}
        return EastSet(K2, keys)
    finally:
        arr._unlock_for_iteration()


def array_to_dict(
    arr: EastArray, key_fn: Any, value_fn: Any, merge_fn: Any, T: Any, K2: Any, T2: Any
) -> Any:
    """Convert array to dict using key and value functions.

    Args:
        arr: Array
        key_fn: Callable taking (element, index) -> K2
        value_fn: Callable taking (element, index) -> V2
        merge_fn: Callable taking (V2, V2, K2) -> V2 for duplicate keys
        T: Array element type
        K2: Dict key type
        T2: Dict value type

    Returns:
        EastDict
    """
    from east.types.containers import EastDict

    arr._lock_for_iteration()
    try:
        result = EastDict(K2, T2, {})
        for index, item in enumerate(arr):
            key = key_fn(item, index)
            value = value_fn(item, index)
            if key in result:
                result[key] = merge_fn(result[key], value, key)
            else:
                result[key] = value
        return result
    finally:
        arr._unlock_for_iteration()


def array_flatten_to_array(arr: EastArray, func: Any, T: Any, T2: Any) -> EastArray:
    """Flat map to array.

    Args:
        arr: Array
        func: Callable taking (element, index) -> Array<T2>
        T: Input element type
        T2: Output element type

    Returns:
        Flattened array
    """
    arr._lock_for_iteration()
    try:
        results = []
        for index, item in enumerate(arr):
            mapped = func(item, index)
            results.extend(mapped)
        return EastArray(T2, results)
    finally:
        arr._unlock_for_iteration()


def array_flatten_to_set(arr: EastArray, func: Any, T: Any, K2: Any) -> Any:
    """Flat map to set.

    Args:
        arr: Array
        func: Callable taking (element, index) -> Set<K2>
        T: Array element type
        K2: Set key type

    Returns:
        Union of all mapped sets
    """
    from east.types.containers import EastSet

    arr._lock_for_iteration()
    try:
        result = set()
        for index, item in enumerate(arr):
            mapped = func(item, index)
            result.update(mapped)
        return EastSet(K2, result)
    finally:
        arr._unlock_for_iteration()


def array_flatten_to_dict(
    arr: EastArray, func: Any, merge_fn: Any, T: Any, K2: Any, T2: Any
) -> Any:
    """Flat map to dict.

    Args:
        arr: Array
        func: Callable taking (element, index) -> Dict<K2, V2>
        merge_fn: Callable taking (V2, V2, K2) -> V2
        T: Array element type
        K2: Dict key type
        T2: Dict value type

    Returns:
        Merged dict
    """
    from east.types.containers import EastDict

    arr._lock_for_iteration()
    try:
        result = EastDict(K2, T2, {})
        for index, item in enumerate(arr):
            mapped = func(item, index)
            for key, value in mapped.items():
                if key in result:
                    result[key] = merge_fn(result[key], value, key)
                else:
                    result[key] = value
        return result
    finally:
        arr._unlock_for_iteration()


def array_group_fold(
    arr: EastArray, key_fn: Any, init_fn: Any, fold_fn: Any, T: Any, K2: Any, T2: Any
) -> Any:
    """Group by key and fold each group.

    Args:
        arr: Array
        key_fn: Callable taking (element, index) -> K2
        init_fn: Callable taking (K2) -> V2 for initial value of each group
        fold_fn: Callable taking (V2, element, index) -> V2
        T: Array element type
        K2: Key type
        T2: Value type

    Returns:
        Dict mapping keys to folded values
    """
    from east.types.containers import EastDict

    arr._lock_for_iteration()
    try:
        result = EastDict(K2, T2, {})
        for index, item in enumerate(arr):
            key = key_fn(item, index)
            if key not in result:
                result[key] = init_fn(key)
            result[key] = fold_fn(result[key], item, index)
        return result
    finally:
        arr._unlock_for_iteration()


# Register all array builtins
register_builtin("ArrayGenerate", array_generate)
register_builtin("ArrayRange", array_range)
register_builtin("ArrayLinspace", array_linspace)
register_builtin("ArraySize", array_length)  # Renamed from ArrayLength
register_builtin("ArrayHas", array_has)
register_builtin("ArrayGet", array_get)
register_builtin("ArrayGetOrDefault", array_get_or_default)
register_builtin("ArrayTryGet", array_try_get)
register_builtin("ArrayUpdate", array_set)  # Renamed from ArraySet
register_builtin("ArrayMerge", array_merge)
register_builtin("ArrayPushLast", array_push_last)
register_builtin("ArrayPopLast", array_pop_last)
register_builtin("ArrayPushFirst", array_push_first)
register_builtin("ArrayPopFirst", array_pop_first)
register_builtin("ArrayAppend", array_append)
register_builtin("ArrayPrepend", array_prepend)
register_builtin("ArrayMergeAll", array_merge_all)
register_builtin("ArrayClear", array_clear)
register_builtin("ArraySortInPlace", array_sort_in_place)
register_builtin("ArrayReverseInPlace", array_reverse_in_place)
register_builtin("ArraySort", array_sort)
register_builtin("ArrayReverse", array_reverse)
register_builtin("ArrayIsSorted", array_is_sorted)
register_builtin("ArrayFindSortedFirst", array_find_sorted_first)
register_builtin("ArrayFindSortedLast", array_find_sorted_last)
register_builtin("ArrayFindSortedRange", array_find_sorted_range)
register_builtin("ArrayFindFirst", array_find_first)
register_builtin("ArrayConcat", array_concat)
register_builtin("ArraySlice", array_slice)
register_builtin("ArrayGetKeys", array_get_keys)
register_builtin("ArrayForEach", array_for_each)
register_builtin("ArrayCopy", array_copy)
register_builtin("ArrayMap", array_map)
register_builtin("ArrayFilter", array_filter)
register_builtin("ArrayFilterMap", array_filter_map)
register_builtin("ArrayFirstMap", array_first_map)
register_builtin("ArrayMapReduce", array_map_reduce)
register_builtin("ArrayFold", array_reduce)  # Renamed from ArrayReduce
register_builtin("ArrayStringJoin", array_string_join)
register_builtin("ArrayToSet", array_to_set)
register_builtin("ArrayToDict", array_to_dict)
register_builtin("ArrayFlattenToArray", array_flatten_to_array)
register_builtin("ArrayFlattenToSet", array_flatten_to_set)
register_builtin("ArrayFlattenToDict", array_flatten_to_dict)
register_builtin("ArrayGroupFold", array_group_fold)


__all__ = [
    "array_generate",
    "array_linspace",
    "array_range",
    "array_length",
    "array_has",
    "array_get",
    "array_get_or_default",
    "array_try_get",
    "array_set",
    "array_merge",
    "array_push_first",
    "array_push_last",
    "array_pop_first",
    "array_pop_last",
    "array_append",
    "array_prepend",
    "array_merge_all",
    "array_clear",
    "array_slice",
    "array_concat",
    "array_reverse",
    "array_reverse_in_place",
    "array_sort",
    "array_sort_in_place",
    "array_is_sorted",
    "array_find_sorted_first",
    "array_find_sorted_last",
    "array_find_sorted_range",
    "array_find_first",
    "array_get_keys",
    "array_copy",
    "array_for_each",
    "array_map",
    "array_filter",
    "array_filter_map",
    "array_first_map",
    "array_map_reduce",
    "array_reduce",
    "array_string_join",
    "array_to_set",
    "array_to_dict",
    "array_flatten_to_array",
    "array_flatten_to_set",
    "array_flatten_to_dict",
    "array_group_fold",
]
