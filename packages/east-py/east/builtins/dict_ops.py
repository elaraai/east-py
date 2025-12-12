#
# Copyright (c) 2025 Elara AI Pty Ltd
# Licensed under the Business Source License 1.1. See LICENSE.md for details.
#
"""Dict builtin functions.

These are factory builtins that take type parameters at compile time.
"""

from collections.abc import Callable
from typing import Any

from east.builtins.registry import register_builtin
from east.types.types import EastType
from east.types.values import EastArray, EastDict, EastSet, EastValue


def dict_size_for(K: EastType, V: EastType) -> Callable[[EastDict], int]:
    """Factory for getting dict size."""

    def dict_size(d: EastDict) -> int:
        return len(d)

    return dict_size


def dict_has_for(K: EastType, V: EastType) -> Callable[[EastDict, EastValue], bool]:
    """Factory for checking if dict has key."""

    def dict_has(d: EastDict, key: EastValue) -> bool:
        return key in d

    return dict_has


def dict_get_for(K: EastType, V: EastType) -> Callable[[EastDict, EastValue], EastValue]:
    """Factory for getting value for key."""

    def dict_get(d: EastDict, key: EastValue) -> EastValue:
        return d[key]

    return dict_get


def dict_set_for(K: EastType, V: EastType) -> Callable[[EastDict, EastValue, EastValue], None]:
    """Factory for setting value for key."""

    def dict_set(d: EastDict, key: EastValue, value: EastValue) -> None:
        d[key] = value

    return dict_set


def dict_remove_for(K: EastType, V: EastType) -> Callable[[EastDict, EastValue], None]:
    """Factory for removing key from dict."""

    def dict_remove(d: EastDict, key: EastValue) -> None:
        del d[key]

    return dict_remove


def dict_clear_for(K: EastType, V: EastType) -> Callable[[EastDict], None]:
    """Factory for clearing dict."""

    def dict_clear(d: EastDict) -> None:
        d.clear()

    return dict_clear


def dict_keys_for(K: EastType, V: EastType) -> Callable[[EastDict], EastSet]:
    """Factory for getting keys as set."""

    def dict_keys(d: EastDict) -> EastSet:
        return EastSet(K, set(d.keys()))

    return dict_keys


def dict_merge_for(
    K: EastType, V: EastType, V2: EastType
) -> Callable[[EastDict, EastValue, EastValue, Any, Any], None]:
    """Factory for merging single key-value pair."""

    def dict_merge(
        d: EastDict, key: EastValue, value: EastValue, merge_fn: Any, initial_fn: Any
    ) -> None:
        if key in d:
            existing = d[key]
        else:
            existing = initial_fn(key)
        d[key] = merge_fn(existing, value, key)

    return dict_merge


def dict_get_or_default_for(
    K: EastType, V: EastType
) -> Callable[[EastDict, EastValue, Any], EastValue]:
    """Factory for getting value or default."""

    def dict_get_or_default(d: EastDict, key: EastValue, default_fn: Any) -> EastValue:
        if key in d:
            return d[key]
        return default_fn(key)

    return dict_get_or_default


def dict_copy_for(K: EastType, V: EastType) -> Callable[[EastDict], EastDict]:
    """Factory for copying dict."""

    def dict_copy(d: EastDict) -> EastDict:
        return EastDict(K, V, dict(d.items()))

    return dict_copy


def dict_update_for(K: EastType, V: EastType) -> Callable[[EastDict, EastValue, EastValue], None]:
    """Factory for updating existing key."""

    def dict_update(d: EastDict, key: EastValue, value: EastValue) -> None:
        if key not in d:
            raise KeyError(f"Key not in dict: {key}")
        d[key] = value

    return dict_update


def dict_generate_for(K: EastType, V: EastType) -> Callable[[int, Any, Any, Any], EastDict]:
    """Factory for generating dict."""

    def dict_generate(n: int, key_fn: Any, value_fn: Any, merge_fn: Any) -> EastDict:
        result: EastDict = EastDict(K, V, {})
        for i in range(n):
            key = key_fn(i)
            value = value_fn(i)
            if key in result:
                result[key] = merge_fn(result[key], value, key)
            else:
                result[key] = value
        return result

    return dict_generate


def dict_try_get_for(K: EastType, V: EastType) -> Callable[[EastDict, EastValue], EastValue]:
    """Factory for getting value as Option."""

    def dict_try_get(d: EastDict, key: EastValue) -> EastValue:
        from east.types.values import EastNone, EastSome

        if key in d:
            return EastSome(d[key])
        return EastNone()

    return dict_try_get


def dict_get_or_insert_for(
    K: EastType, V: EastType
) -> Callable[[EastDict, EastValue, Any], EastValue]:
    """Factory for getting or inserting value."""

    def dict_get_or_insert(d: EastDict, key: EastValue, default_fn: Any) -> EastValue:
        if key in d:
            return d[key]
        value = default_fn(key)
        d[key] = value
        return value

    return dict_get_or_insert


def dict_insert_or_update_for(
    K: EastType, V: EastType
) -> Callable[[EastDict, EastValue, EastValue, Any], None]:
    """Factory for inserting or merging."""

    def dict_insert_or_update(d: EastDict, key: EastValue, value: EastValue, merge_fn: Any) -> None:
        if key in d:
            d[key] = merge_fn(d[key], value, key)
        else:
            d[key] = value

    return dict_insert_or_update


def dict_swap_for(
    K: EastType, V: EastType
) -> Callable[[EastDict, EastValue, EastValue], EastValue]:
    """Factory for swapping value."""

    def dict_swap(d: EastDict, key: EastValue, value: EastValue) -> EastValue:
        if key not in d:
            raise KeyError(f"Key not in dict: {key}")
        old_value = d[key]
        d[key] = value
        return old_value

    return dict_swap


def dict_try_delete_for(K: EastType, V: EastType) -> Callable[[EastDict, EastValue], bool]:
    """Factory for trying to delete key."""

    def dict_try_delete(d: EastDict, key: EastValue) -> bool:
        if key in d:
            del d[key]
            return True
        return False

    return dict_try_delete


def dict_pop_for(K: EastType, V: EastType) -> Callable[[EastDict, EastValue], EastValue]:
    """Factory for popping key."""

    def dict_pop(d: EastDict, key: EastValue) -> EastValue:
        return d.pop(key)

    return dict_pop


def dict_union_in_place_for(K: EastType, V: EastType) -> Callable[[EastDict, EastDict, Any], None]:
    """Factory for union in place."""

    def dict_union_in_place(d: EastDict, other: EastDict, merge_fn: Any) -> None:
        for key, value in other.items():
            if key in d:
                d[key] = merge_fn(d[key], value, key)
            else:
                d[key] = value

    return dict_union_in_place


def dict_merge_all_for(
    K: EastType, V: EastType, V2: EastType
) -> Callable[[EastDict, EastDict, Any, Any], None]:
    """Factory for merging all entries."""

    def dict_merge_all(d: EastDict, other: EastDict, merge_fn: Any, default_fn: Any) -> None:
        for key, value in other.items():
            if key in d:
                d[key] = merge_fn(d[key], value, key)
            else:
                d[key] = merge_fn(default_fn(key), value, key)

    return dict_merge_all


def dict_get_keys_for(K: EastType, V: EastType) -> Callable[[EastDict, EastSet, Any], EastDict]:
    """Factory for getting multiple keys."""

    def dict_get_keys(d: EastDict, keys: EastSet, default_fn: Any) -> EastDict:
        result: EastDict = EastDict(K, V, {})
        for key in keys:
            if key in d:
                result[key] = d[key]
            else:
                result[key] = default_fn(key)
        return result

    return dict_get_keys


def dict_for_each_for(K: EastType, V: EastType, T2: EastType) -> Callable[[EastDict, Any], None]:
    """Factory for iterating over dict."""

    def dict_for_each(d: EastDict, func: Any) -> None:
        d._lock_for_iteration()
        try:
            for key, value in d.items():
                func(value, key)
        finally:
            d._unlock_for_iteration()

    return dict_for_each


def dict_map_for(K: EastType, V: EastType, V2: EastType) -> Callable[[EastDict, Any], EastDict]:
    """Factory for mapping dict values."""

    def dict_map(d: EastDict, func: Any) -> EastDict:
        d._lock_for_iteration()
        try:
            result: EastDict = EastDict(K, V2, {})
            for key, value in d.items():
                result[key] = func(value, key)
            return result
        finally:
            d._unlock_for_iteration()

    return dict_map


def dict_filter_for(K: EastType, V: EastType) -> Callable[[EastDict, Any], EastDict]:
    """Factory for filtering dict."""

    def dict_filter(d: EastDict, func: Any) -> EastDict:
        d._lock_for_iteration()
        try:
            result: EastDict = EastDict(K, V, {})
            for key, value in d.items():
                if func(value, key):
                    result[key] = value
            return result
        finally:
            d._unlock_for_iteration()

    return dict_filter


def dict_filter_map_for(
    K: EastType, V: EastType, V2: EastType
) -> Callable[[EastDict, Any], EastDict]:
    """Factory for filter and map."""

    def dict_filter_map(d: EastDict, func: Any) -> EastDict:
        d._lock_for_iteration()
        try:
            result: EastDict = EastDict(K, V2, {})
            for key, value in d.items():
                variant = func(value, key)
                if variant.type == "some":
                    result[key] = variant.value
            return result
        finally:
            d._unlock_for_iteration()

    return dict_filter_map


def dict_first_map_for(
    K: EastType, V: EastType, T2: EastType
) -> Callable[[EastDict, Any], EastValue]:
    """Factory for finding first mapping to some."""

    def dict_first_map(d: EastDict, func: Any) -> EastValue:
        from east.types.values import EastNone

        d._lock_for_iteration()
        try:
            for key, value in d.items():
                variant = func(value, key)
                if variant.type == "some":
                    return variant
            return EastNone()
        finally:
            d._unlock_for_iteration()

    return dict_first_map


def dict_map_reduce_for(
    K: EastType, V: EastType, T2: EastType
) -> Callable[[EastDict, Any, Any], EastValue]:
    """Factory for map then reduce."""

    def dict_map_reduce(d: EastDict, map_fn: Any, reduce_fn: Any) -> EastValue:
        if len(d) == 0:
            raise ValueError("Cannot reduce empty dict")
        d._lock_for_iteration()
        try:
            mapped = [map_fn(value, key) for key, value in d.items()]
            result = mapped[0]
            for item in mapped[1:]:
                result = reduce_fn(result, item)
            return result
        finally:
            d._unlock_for_iteration()

    return dict_map_reduce


def dict_reduce_for(
    K: EastType, V: EastType, T2: EastType
) -> Callable[[EastDict, Any, EastValue], EastValue]:
    """Factory for folding over dict."""

    def dict_reduce(d: EastDict, func: Any, initial: EastValue) -> EastValue:
        d._lock_for_iteration()
        try:
            accumulator = initial
            for key, value in d.items():
                accumulator = func(accumulator, value, key)
            return accumulator
        finally:
            d._unlock_for_iteration()

    return dict_reduce


def dict_to_array_for(
    K: EastType, V: EastType, T2: EastType
) -> Callable[[EastDict, Any], EastArray]:
    """Factory for converting dict to array."""

    def dict_to_array(d: EastDict, func: Any) -> EastArray:
        d._lock_for_iteration()
        try:
            sorted_items = sorted(d.items(), key=lambda x: (type(x[0]).__name__, x[0]))
            mapped = [func(value, key) for key, value in sorted_items]
            return EastArray(T2, mapped)
        finally:
            d._unlock_for_iteration()

    return dict_to_array


def dict_to_set_for(K: EastType, V: EastType, K2: EastType) -> Callable[[EastDict, Any], EastSet]:
    """Factory for converting dict to set."""

    def dict_to_set(d: EastDict, func: Any) -> EastSet:
        d._lock_for_iteration()
        try:
            mapped = {func(value, key) for key, value in d.items()}
            return EastSet(K2, mapped)
        finally:
            d._unlock_for_iteration()

    return dict_to_set


def dict_to_dict_for(
    K: EastType, V: EastType, K2: EastType, V2: EastType
) -> Callable[[EastDict, Any, Any, Any], EastDict]:
    """Factory for mapping dict to new dict."""

    def dict_to_dict(d: EastDict, key_fn: Any, value_fn: Any, merge_fn: Any) -> EastDict:
        d._lock_for_iteration()
        try:
            result: EastDict = EastDict(K2, V2, {})
            for key, value in d.items():
                new_key = key_fn(value, key)
                new_value = value_fn(value, key)
                if new_key in result:
                    result[new_key] = merge_fn(result[new_key], new_value, new_key)
                else:
                    result[new_key] = new_value
            return result
        finally:
            d._unlock_for_iteration()

    return dict_to_dict


def dict_flatten_to_array_for(
    K: EastType, V: EastType, T2: EastType
) -> Callable[[EastDict, Any], EastArray]:
    """Factory for flat map to array."""

    def dict_flatten_to_array(d: EastDict, func: Any) -> EastArray:
        d._lock_for_iteration()
        try:
            results = []
            for key, value in d.items():
                mapped = func(value, key)
                results.extend(mapped)
            return EastArray(T2, results)
        finally:
            d._unlock_for_iteration()

    return dict_flatten_to_array


def dict_flatten_to_set_for(
    K: EastType, V: EastType, K2: EastType
) -> Callable[[EastDict, Any], EastSet]:
    """Factory for flat map to set."""

    def dict_flatten_to_set(d: EastDict, func: Any) -> EastSet:
        d._lock_for_iteration()
        try:
            result = set()
            for key, value in d.items():
                mapped = func(value, key)
                result.update(mapped)
            return EastSet(K2, result)
        finally:
            d._unlock_for_iteration()

    return dict_flatten_to_set


def dict_flatten_to_dict_for(
    K: EastType, V: EastType, K2: EastType, V2: EastType
) -> Callable[[EastDict, Any, Any], EastDict]:
    """Factory for flat map to dict."""

    def dict_flatten_to_dict(d: EastDict, func: Any, merge_fn: Any) -> EastDict:
        d._lock_for_iteration()
        try:
            result: EastDict = EastDict(K2, V2, {})
            for key, value in d.items():
                mapped = func(value, key)
                for k, v in mapped.items():
                    if k in result:
                        result[k] = merge_fn(result[k], v, k)
                    else:
                        result[k] = v
            return result
        finally:
            d._unlock_for_iteration()

    return dict_flatten_to_dict


def dict_group_fold_for(
    K: EastType, V: EastType, K2: EastType, T2: EastType
) -> Callable[[EastDict, Any, Any, Any], EastDict]:
    """Factory for grouping and folding."""

    def dict_group_fold(d: EastDict, key_fn: Any, init_fn: Any, fold_fn: Any) -> EastDict:
        d._lock_for_iteration()
        try:
            result: EastDict = EastDict(K2, T2, {})
            for key, value in d.items():
                group_key = key_fn(value, key)
                if group_key not in result:
                    result[group_key] = init_fn(group_key)
                result[group_key] = fold_fn(result[group_key], value, key)
            return result
        finally:
            d._unlock_for_iteration()

    return dict_group_fold


# Register all dict builtins as factories
register_builtin("DictGenerate", dict_generate_for)
register_builtin("DictSize", dict_size_for)
register_builtin("DictHas", dict_has_for)
register_builtin("DictGet", dict_get_for)
register_builtin("DictGetOrDefault", dict_get_or_default_for)
register_builtin("DictTryGet", dict_try_get_for)
register_builtin("DictInsert", dict_set_for)
register_builtin("DictGetOrInsert", dict_get_or_insert_for)
register_builtin("DictInsertOrUpdate", dict_insert_or_update_for)
register_builtin("DictUpdate", dict_update_for)
register_builtin("DictSwap", dict_swap_for)
register_builtin("DictMerge", dict_merge_for)
register_builtin("DictDelete", dict_remove_for)
register_builtin("DictTryDelete", dict_try_delete_for)
register_builtin("DictPop", dict_pop_for)
register_builtin("DictClear", dict_clear_for)
register_builtin("DictUnionInPlace", dict_union_in_place_for)
register_builtin("DictMergeAll", dict_merge_all_for)
register_builtin("DictKeys", dict_keys_for)
register_builtin("DictGetKeys", dict_get_keys_for)
register_builtin("DictForEach", dict_for_each_for)
register_builtin("DictCopy", dict_copy_for)
register_builtin("DictMap", dict_map_for)
register_builtin("DictFilter", dict_filter_for)
register_builtin("DictFilterMap", dict_filter_map_for)
register_builtin("DictFirstMap", dict_first_map_for)
register_builtin("DictMapReduce", dict_map_reduce_for)
register_builtin("DictReduce", dict_reduce_for)
register_builtin("DictToArray", dict_to_array_for)
register_builtin("DictToSet", dict_to_set_for)
register_builtin("DictToDict", dict_to_dict_for)
register_builtin("DictFlattenToArray", dict_flatten_to_array_for)
register_builtin("DictFlattenToSet", dict_flatten_to_set_for)
register_builtin("DictFlattenToDict", dict_flatten_to_dict_for)
register_builtin("DictGroupFold", dict_group_fold_for)


__all__ = [
    "dict_generate_for",
    "dict_size_for",
    "dict_has_for",
    "dict_get_for",
    "dict_get_or_default_for",
    "dict_try_get_for",
    "dict_set_for",
    "dict_get_or_insert_for",
    "dict_insert_or_update_for",
    "dict_update_for",
    "dict_swap_for",
    "dict_merge_for",
    "dict_remove_for",
    "dict_try_delete_for",
    "dict_pop_for",
    "dict_clear_for",
    "dict_union_in_place_for",
    "dict_merge_all_for",
    "dict_keys_for",
    "dict_get_keys_for",
    "dict_for_each_for",
    "dict_copy_for",
    "dict_map_for",
    "dict_filter_for",
    "dict_filter_map_for",
    "dict_first_map_for",
    "dict_map_reduce_for",
    "dict_reduce_for",
    "dict_to_array_for",
    "dict_to_set_for",
    "dict_to_dict_for",
    "dict_flatten_to_array_for",
    "dict_flatten_to_set_for",
    "dict_flatten_to_dict_for",
    "dict_group_fold_for",
]
