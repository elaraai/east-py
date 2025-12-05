"""Array builtin functions.

These are factory builtins that take type parameters at compile time.
"""

from collections.abc import Callable
from typing import Any

from east.builtins.registry import register_builtin
from east.types.types import EastType
from east.types.values import EastArray, EastBlob, EastStruct, EastValue

# Factory functions for array operations


def array_length_for(T: EastType) -> Callable[[EastArray], int]:
    """Factory for getting array length."""

    def array_length(arr: EastArray) -> int:
        return len(arr)

    return array_length


def array_get_for(T: EastType) -> Callable[[EastArray, int], EastValue]:
    """Factory for getting element at index."""

    def array_get(arr: EastArray, index: int) -> EastValue:
        return arr[index]

    return array_get


def array_set_for(T: EastType) -> Callable[[EastArray, int, EastValue], None]:
    """Factory for setting element at index."""

    def array_set(arr: EastArray, index: int, value: EastValue) -> None:
        arr[index] = value

    return array_set


def array_push_first_for(T: EastType) -> Callable[[EastArray, EastValue], None]:
    """Factory for prepending element."""

    def array_push_first(arr: EastArray, value: EastValue) -> None:
        arr.insert(0, value)

    return array_push_first


def array_push_last_for(T: EastType) -> Callable[[EastArray, EastValue], None]:
    """Factory for appending element."""

    def array_push_last(arr: EastArray, value: EastValue) -> None:
        arr.append(value)

    return array_push_last


def array_pop_first_for(T: EastType) -> Callable[[EastArray], EastValue]:
    """Factory for removing first element."""

    def array_pop_first(arr: EastArray) -> EastValue:
        return arr.pop(0)

    return array_pop_first


def array_pop_last_for(T: EastType) -> Callable[[EastArray], EastValue]:
    """Factory for removing last element."""

    def array_pop_last(arr: EastArray) -> EastValue:
        return arr.pop()

    return array_pop_last


def array_slice_for(T: EastType) -> Callable[[EastArray, int, int], EastArray]:
    """Factory for getting array slice."""

    def array_slice(arr: EastArray, start: int, end: int) -> EastArray:
        return EastArray(T, arr[start:end])

    return array_slice


def array_concat_for(T: EastType) -> Callable[[EastArray, EastArray], EastArray]:
    """Factory for concatenating arrays."""

    def array_concat(a: EastArray, b: EastArray) -> EastArray:
        return EastArray(T, list(a) + list(b))

    return array_concat


def array_reverse_for(T: EastType) -> Callable[[EastArray], EastArray]:
    """Factory for reversing array."""

    def array_reverse(arr: EastArray) -> EastArray:
        return EastArray(T, list(reversed(arr)))

    return array_reverse


def array_sort_for(T: EastType, T2: EastType) -> Callable[[EastArray, Any], EastArray]:
    """Factory for sorting array by key function."""
    from functools import cmp_to_key

    from east.utils.ordering import compare_for

    compare = compare_for(T2)  # Computed once at compile time

    def array_sort(arr: EastArray, key_fn: Any) -> EastArray:
        keys = [key_fn(item) for item in arr]
        sorted_indices = sorted(range(len(arr)), key=lambda i: cmp_to_key(compare)(keys[i]))
        sorted_items = [arr[i] for i in sorted_indices]
        return EastArray(T, sorted_items)

    return array_sort


def array_get_or_default_for(T: EastType) -> Callable[[EastArray, int, Any], EastValue]:
    """Factory for getting element or default."""

    def array_get_or_default(arr: EastArray, index: int, default_fn: Any) -> EastValue:
        if 0 <= index < len(arr):
            return arr[index]
        return default_fn(index)

    return array_get_or_default


def array_clear_for(T: EastType) -> Callable[[EastArray], None]:
    """Factory for clearing array."""

    def array_clear(arr: EastArray) -> None:
        arr.clear()

    return array_clear


def array_copy_for(T: EastType) -> Callable[[EastArray], EastArray]:
    """Factory for copying array."""

    def array_copy(arr: EastArray) -> EastArray:
        return EastArray(T, list(arr))

    return array_copy


def array_reverse_in_place_for(T: EastType) -> Callable[[EastArray], None]:
    """Factory for reversing array in place."""

    def array_reverse_in_place(arr: EastArray) -> None:
        arr.reverse()

    return array_reverse_in_place


def array_sort_in_place_for(T: EastType, T2: EastType) -> Callable[[EastArray, EastValue], None]:
    """Factory for sorting array in place."""
    from functools import cmp_to_key

    from east.utils.ordering import compare_for

    compare = compare_for(T2)  # Computed once at compile time

    def array_sort_in_place(arr: EastArray, key_fn: Any) -> None:
        keys = [key_fn(item) for item in arr]
        sorted_indices = sorted(range(len(arr)), key=lambda i: cmp_to_key(compare)(keys[i]))
        sorted_items = [arr[i] for i in sorted_indices]
        arr.clear()
        arr.extend(sorted_items)

    return array_sort_in_place


def array_range(start: int, end: int, step: int) -> EastArray:
    """Create array from range (no type params)."""
    from east.types.types import IntegerType

    return EastArray(IntegerType, list(range(start, end, step)))


def array_map_for(T: EastType, T2: EastType) -> Callable[[EastArray, Any], EastArray]:
    """Factory for mapping over array."""

    def array_map(arr: EastArray, func: Any) -> EastArray:
        arr._lock_for_iteration()
        try:
            mapped = [func(item, index) for index, item in enumerate(arr)]
            return EastArray(T2, mapped)
        finally:
            arr._unlock_for_iteration()

    return array_map


def array_filter_for(T: EastType) -> Callable[[EastArray, Any], EastArray]:
    """Factory for filtering array."""

    def array_filter(arr: EastArray, func: Any) -> EastArray:
        arr._lock_for_iteration()
        try:
            filtered = [item for index, item in enumerate(arr) if func(item, index)]
            return EastArray(T, filtered)
        finally:
            arr._unlock_for_iteration()

    return array_filter


def array_reduce_for(T: EastType, T2: EastType) -> Callable[[EastArray, EastValue, Any], EastValue]:
    """Factory for reducing array."""

    def array_reduce(arr: EastArray, initial: EastValue, func: Any) -> EastValue:
        arr._lock_for_iteration()
        try:
            accumulator = initial
            for index, item in enumerate(arr):
                accumulator = func(accumulator, item, index)
            return accumulator
        finally:
            arr._unlock_for_iteration()

    return array_reduce


def array_generate_for(T: EastType) -> Callable[[int, Any], EastArray]:
    """Factory for generating array."""

    def array_generate(n: int, func: Any) -> EastArray:
        elements = [func(i) for i in range(n)]
        return EastArray(T, elements)

    return array_generate


def array_linspace(start: float, end: float, n: int) -> EastArray:
    """Generate linearly spaced floats (no type params)."""
    from east.types.types import FloatType

    if n == 1:
        return EastArray(FloatType, [start])
    step = (end - start) / (n - 1)
    elements = [start + i * step for i in range(n)]
    return EastArray(FloatType, elements)


def array_has_for(T: EastType) -> Callable[[EastArray, int], bool]:
    """Factory for checking if index exists."""

    def array_has(arr: EastArray, index: int) -> bool:
        return 0 <= index < len(arr)

    return array_has


def array_try_get_for(T: EastType) -> Callable[[EastArray, int], EastValue]:
    """Factory for getting element as Option."""

    def array_try_get(arr: EastArray, index: int) -> EastValue:
        from east.types.values import EastNone, EastSome

        if 0 <= index < len(arr):
            return EastSome(arr[index])
        return EastNone()

    return array_try_get


def array_merge_for(T: EastType, T2: EastType) -> Callable[[EastArray, int, EastValue, Any], None]:
    """Factory for merging value at index."""

    def array_merge(arr: EastArray, index: int, value: EastValue, func: Any) -> None:
        old_value = arr[index]
        arr[index] = func(old_value, value, index)

    return array_merge


def array_append_for(T: EastType) -> Callable[[EastArray, EastArray], None]:
    """Factory for appending another array."""

    def array_append(arr: EastArray, other: EastArray) -> None:
        arr.extend(other)

    return array_append


def array_prepend_for(T: EastType) -> Callable[[EastArray, EastArray], None]:
    """Factory for prepending another array."""

    def array_prepend(arr: EastArray, other: EastArray) -> None:
        for i, item in enumerate(other):
            arr.insert(i, item)

    return array_prepend


def array_merge_all_for(T: EastType, T2: EastType) -> Callable[[EastArray, EastArray, Any], None]:
    """Factory for merging arrays element-wise."""

    def array_merge_all(arr: EastArray, other: EastArray, func: Any) -> None:
        for i, item in enumerate(other):
            if i < len(arr):
                arr[i] = func(arr[i], item, i)

    return array_merge_all


def array_is_sorted_for(T: EastType, T2: EastType) -> Callable[[EastArray, Any], bool]:
    """Factory for checking if array is sorted."""
    from east.utils.ordering import compare_for

    compare = compare_for(T2)  # Computed once at compile time

    def array_is_sorted(arr: EastArray, key_fn: Any) -> bool:
        if len(arr) <= 1:
            return True
        keys = [key_fn(item) for item in arr]
        for i in range(len(keys) - 1):
            if compare(keys[i], keys[i + 1]) > 0:
                return False
        return True

    return array_is_sorted


def array_find_sorted_first_for(
    T: EastType, T2: EastType
) -> Callable[[EastArray, EastValue, Any], int]:
    """Factory for binary search first occurrence."""
    from east.utils.ordering import compare_for

    compare = compare_for(T2)  # Computed once at compile time

    def array_find_sorted_first(arr: EastArray, target: EastValue, key_fn: Any) -> int:
        left, right = 0, len(arr)
        while left < right:
            mid = (left + right) // 2
            key = key_fn(arr[mid])
            if compare(key, target) < 0:
                left = mid + 1
            else:
                right = mid
        return left

    return array_find_sorted_first


def array_find_sorted_last_for(
    T: EastType, T2: EastType
) -> Callable[[EastArray, EastValue, Any], int]:
    """Factory for binary search last occurrence."""
    from east.utils.ordering import compare_for

    compare = compare_for(T2)  # Computed once at compile time

    def array_find_sorted_last(arr: EastArray, target: EastValue, key_fn: Any) -> int:
        left, right = 0, len(arr)
        while left < right:
            mid = (left + right) // 2
            key = key_fn(arr[mid])
            if compare(key, target) <= 0:
                left = mid + 1
            else:
                right = mid
        return left

    return array_find_sorted_last


def array_find_sorted_range_for(
    T: EastType, T2: EastType
) -> Callable[[EastArray, EastValue, Any], EastStruct]:
    """Factory for binary search range."""
    first_fn = array_find_sorted_first_for(T, T2)
    last_fn = array_find_sorted_last_for(T, T2)

    def array_find_sorted_range(arr: EastArray, target: EastValue, key_fn: Any) -> EastStruct:
        start = first_fn(arr, target, key_fn)
        end = last_fn(arr, target, key_fn)
        return EastStruct({"start": start, "end": end})

    return array_find_sorted_range


def array_find_first_for(
    T: EastType, T2: EastType
) -> Callable[[EastArray, EastValue, Any], EastValue]:
    """Factory for linear search first occurrence."""
    from east.utils.ordering import compare_for

    compare = compare_for(T2)  # Computed once at compile time

    def array_find_first(arr: EastArray, target: EastValue, key_fn: Any) -> EastValue:
        from east.types.values import EastNone, EastSome

        arr._lock_for_iteration()
        try:
            for index, item in enumerate(arr):
                key = key_fn(item)
                if compare(key, target) == 0:
                    return EastSome(index)
            return EastNone()
        finally:
            arr._unlock_for_iteration()

    return array_find_first


def array_get_keys_for(T: EastType) -> Callable[[EastArray, EastArray, Any], EastArray]:
    """Factory for getting multiple elements by indices."""

    def array_get_keys(arr: EastArray, indices: EastArray, default_fn: Any) -> EastArray:
        elements = []
        for index in indices:
            if 0 <= index < len(arr):
                elements.append(arr[index])
            else:
                elements.append(default_fn(index))
        return EastArray(T, elements)

    return array_get_keys


def array_for_each_for(T: EastType, T2: EastType) -> Callable[[EastArray, EastValue], None]:
    """Factory for iterating over array."""

    def array_for_each(arr: EastArray, func: Any) -> None:
        arr._lock_for_iteration()
        try:
            for index, item in enumerate(arr):
                func(item, index)
        finally:
            arr._unlock_for_iteration()

    return array_for_each


def array_filter_map_for(T: EastType, T2: EastType) -> Callable[[EastArray, Any], EastArray]:
    """Factory for filter and map in one pass."""

    def array_filter_map(arr: EastArray, func: Any) -> EastArray:
        arr._lock_for_iteration()
        try:
            results = []
            for index, item in enumerate(arr):
                result = func(item, index)
                if result.type == "some":
                    results.append(result.value)
            return EastArray(T2, results)
        finally:
            arr._unlock_for_iteration()

    return array_filter_map


def array_first_map_for(T: EastType, T2: EastType) -> Callable[[EastArray, Any], EastValue]:
    """Factory for finding first element that maps to some."""

    def array_first_map(arr: EastArray, func: Any) -> EastValue:
        from east.types.values import EastNone

        arr._lock_for_iteration()
        try:
            for index, item in enumerate(arr):
                result = func(item, index)
                if result.type == "some":
                    return result
            return EastNone()
        finally:
            arr._unlock_for_iteration()

    return array_first_map


def array_map_reduce_for(T: EastType, T2: EastType) -> Callable[[EastArray, Any, Any], EastValue]:
    """Factory for map then reduce."""

    def array_map_reduce(arr: EastArray, map_fn: Any, reduce_fn: Any) -> EastValue:
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

    return array_map_reduce


def array_string_join(arr: EastArray, delimiter: str) -> str:
    """Join string array (no type params)."""
    return delimiter.join(arr)


def array_to_set_for(T: EastType, K2: EastType) -> Callable[[EastArray, Any], EastValue]:
    """Factory for converting array to set."""

    def array_to_set(arr: EastArray, key_fn: Any) -> EastValue:
        from east.types.values import EastSet

        arr._lock_for_iteration()
        try:
            keys = {key_fn(item, index) for index, item in enumerate(arr)}
            return EastSet(K2, keys)
        finally:
            arr._unlock_for_iteration()

    return array_to_set


def array_to_dict_for(
    T: EastType, K2: EastType, T2: EastType
) -> Callable[[EastArray, Any, Any, Any], EastValue]:
    """Factory for converting array to dict."""

    def array_to_dict(arr: EastArray, key_fn: Any, value_fn: Any, merge_fn: Any) -> EastValue:
        from east.types.values import EastDict

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

    return array_to_dict


def array_flatten_to_array_for(T: EastType, T2: EastType) -> Callable[[EastArray, Any], EastArray]:
    """Factory for flat map to array."""

    def array_flatten_to_array(arr: EastArray, func: Any) -> EastArray:
        arr._lock_for_iteration()
        try:
            results = []
            for index, item in enumerate(arr):
                mapped = func(item, index)
                results.extend(mapped)
            return EastArray(T2, results)
        finally:
            arr._unlock_for_iteration()

    return array_flatten_to_array


def array_flatten_to_set_for(T: EastType, K2: EastType) -> Callable[[EastArray, Any], EastValue]:
    """Factory for flat map to set."""

    def array_flatten_to_set(arr: EastArray, func: Any) -> EastValue:
        from east.types.values import EastSet

        arr._lock_for_iteration()
        try:
            result = set()
            for index, item in enumerate(arr):
                mapped = func(item, index)
                result.update(mapped)
            return EastSet(K2, result)
        finally:
            arr._unlock_for_iteration()

    return array_flatten_to_set


def array_flatten_to_dict_for(
    T: EastType, K2: EastType, T2: EastType
) -> Callable[[EastArray, Any, Any], EastValue]:
    """Factory for flat map to dict."""

    def array_flatten_to_dict(arr: EastArray, func: Any, merge_fn: Any) -> EastValue:
        from east.types.values import EastDict

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

    return array_flatten_to_dict


def array_group_fold_for(
    T: EastType, K2: EastType, T2: EastType
) -> Callable[[EastArray, Any, Any, Any], EastValue]:
    """Factory for grouping and folding."""

    def array_group_fold(arr: EastArray, key_fn: Any, init_fn: Any, fold_fn: Any) -> EastValue:
        from east.types.values import EastDict

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

    return array_group_fold


def array_encode_csv_for(
    T: EastType, Config: EastType
) -> Callable[[list[EastValue], EastValue], EastBlob]:
    """Factory for encoding array to CSV.

    Args:
        T: Struct type for each row
        Config: CsvSerializeConfigType

    Returns:
        Function that encodes arrays to CSV blobs with config
    """
    from east.serialization.csv import encode_csv_for

    def array_encode_csv(array: list[EastValue], config: EastValue) -> EastBlob:
        encoder = encode_csv_for(T, config)
        return EastBlob(encoder(array))

    return array_encode_csv


# Register all array builtins as factories
register_builtin("ArrayGenerate", array_generate_for)
register_builtin("ArrayRange", lambda: array_range)
register_builtin("ArrayLinspace", lambda: array_linspace)
register_builtin("ArraySize", array_length_for)
register_builtin("ArrayHas", array_has_for)
register_builtin("ArrayGet", array_get_for)
register_builtin("ArrayGetOrDefault", array_get_or_default_for)
register_builtin("ArrayTryGet", array_try_get_for)
register_builtin("ArrayUpdate", array_set_for)
register_builtin("ArrayMerge", array_merge_for)
register_builtin("ArrayPushLast", array_push_last_for)
register_builtin("ArrayPopLast", array_pop_last_for)
register_builtin("ArrayPushFirst", array_push_first_for)
register_builtin("ArrayPopFirst", array_pop_first_for)
register_builtin("ArrayAppend", array_append_for)
register_builtin("ArrayPrepend", array_prepend_for)
register_builtin("ArrayMergeAll", array_merge_all_for)
register_builtin("ArrayClear", array_clear_for)
register_builtin("ArraySortInPlace", array_sort_in_place_for)
register_builtin("ArrayReverseInPlace", array_reverse_in_place_for)
register_builtin("ArraySort", array_sort_for)
register_builtin("ArrayReverse", array_reverse_for)
register_builtin("ArrayIsSorted", array_is_sorted_for)
register_builtin("ArrayFindSortedFirst", array_find_sorted_first_for)
register_builtin("ArrayFindSortedLast", array_find_sorted_last_for)
register_builtin("ArrayFindSortedRange", array_find_sorted_range_for)
register_builtin("ArrayFindFirst", array_find_first_for)
register_builtin("ArrayConcat", array_concat_for)
register_builtin("ArraySlice", array_slice_for)
register_builtin("ArrayGetKeys", array_get_keys_for)
register_builtin("ArrayForEach", array_for_each_for)
register_builtin("ArrayCopy", array_copy_for)
register_builtin("ArrayMap", array_map_for)
register_builtin("ArrayFilter", array_filter_for)
register_builtin("ArrayFilterMap", array_filter_map_for)
register_builtin("ArrayFirstMap", array_first_map_for)
register_builtin("ArrayMapReduce", array_map_reduce_for)
register_builtin("ArrayFold", array_reduce_for)
register_builtin("ArrayStringJoin", lambda: array_string_join)
register_builtin("ArrayToSet", array_to_set_for)
register_builtin("ArrayToDict", array_to_dict_for)
register_builtin("ArrayFlattenToArray", array_flatten_to_array_for)
register_builtin("ArrayFlattenToSet", array_flatten_to_set_for)
register_builtin("ArrayFlattenToDict", array_flatten_to_dict_for)
register_builtin("ArrayGroupFold", array_group_fold_for)
register_builtin("ArrayEncodeCsv", array_encode_csv_for)


__all__ = [
    "array_generate_for",
    "array_linspace",
    "array_range",
    "array_length_for",
    "array_has_for",
    "array_get_for",
    "array_get_or_default_for",
    "array_try_get_for",
    "array_set_for",
    "array_merge_for",
    "array_push_first_for",
    "array_push_last_for",
    "array_pop_first_for",
    "array_pop_last_for",
    "array_append_for",
    "array_prepend_for",
    "array_merge_all_for",
    "array_clear_for",
    "array_slice_for",
    "array_concat_for",
    "array_reverse_for",
    "array_reverse_in_place_for",
    "array_sort_for",
    "array_sort_in_place_for",
    "array_is_sorted_for",
    "array_find_sorted_first_for",
    "array_find_sorted_last_for",
    "array_find_sorted_range_for",
    "array_find_first_for",
    "array_get_keys_for",
    "array_copy_for",
    "array_for_each_for",
    "array_map_for",
    "array_filter_for",
    "array_filter_map_for",
    "array_first_map_for",
    "array_map_reduce_for",
    "array_reduce_for",
    "array_string_join",
    "array_to_set_for",
    "array_to_dict_for",
    "array_flatten_to_array_for",
    "array_flatten_to_set_for",
    "array_flatten_to_dict_for",
    "array_group_fold_for",
    "array_encode_csv_for",
]
