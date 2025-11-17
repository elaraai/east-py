"""Set builtin functions.

These are factory builtins that take type parameters at compile time.
"""

from collections.abc import Callable
from typing import Any

from east.builtins.registry import register_builtin
from east.types.containers import EastArray, EastSet


def set_size_for(K: Any) -> Callable[[EastSet], int]:
    """Factory for getting set size."""

    def set_size(s: EastSet) -> int:
        return len(s)

    return set_size


def set_has_for(K: Any) -> Callable[[EastSet, Any], bool]:
    """Factory for checking if set contains value."""

    def set_has(s: EastSet, value: Any) -> bool:
        return value in s

    return set_has


def set_add_for(K: Any) -> Callable[[EastSet, Any], None]:
    """Factory for adding value to set."""

    def set_add(s: EastSet, value: Any) -> None:
        s.add(value)

    return set_add


def set_remove_for(K: Any) -> Callable[[EastSet, Any], None]:
    """Factory for removing value from set."""

    def set_remove(s: EastSet, value: Any) -> None:
        s.discard(value)

    return set_remove


def set_clear_for(K: Any) -> Callable[[EastSet], None]:
    """Factory for clearing set."""

    def set_clear(s: EastSet) -> None:
        s.clear()

    return set_clear


def set_union_for(K: Any) -> Callable[[EastSet, EastSet], EastSet]:
    """Factory for union of sets."""

    def set_union(a: EastSet, b: EastSet) -> EastSet:
        return EastSet(K, list(a) + [x for x in b if x not in a])

    return set_union


def set_intersection_for(K: Any) -> Callable[[EastSet, EastSet], EastSet]:
    """Factory for intersection of sets."""

    def set_intersection(a: EastSet, b: EastSet) -> EastSet:
        return EastSet(K, [x for x in a if x in b])

    return set_intersection


def set_difference_for(K: Any) -> Callable[[EastSet, EastSet], EastSet]:
    """Factory for difference of sets."""

    def set_difference(a: EastSet, b: EastSet) -> EastSet:
        return EastSet(K, [x for x in a if x not in b])

    return set_difference


def set_symmetric_difference_for(K: Any) -> Callable[[EastSet, EastSet], EastSet]:
    """Factory for symmetric difference of sets."""

    def set_symmetric_difference(a: EastSet, b: EastSet) -> EastSet:
        in_a_not_b = [x for x in a if x not in b]
        in_b_not_a = [x for x in b if x not in a]
        return EastSet(K, in_a_not_b + in_b_not_a)

    return set_symmetric_difference


def set_is_subset_for(K: Any) -> Callable[[EastSet, EastSet], bool]:
    """Factory for checking subset."""

    def set_is_subset(a: EastSet, b: EastSet) -> bool:
        return all(x in b for x in a)

    return set_is_subset


def set_is_disjoint_for(K: Any) -> Callable[[EastSet, EastSet], bool]:
    """Factory for checking disjoint sets."""

    def set_is_disjoint(a: EastSet, b: EastSet) -> bool:
        return all(x not in b for x in a)

    return set_is_disjoint


def set_copy_for(K: Any) -> Callable[[EastSet], EastSet]:
    """Factory for copying set."""

    def set_copy(s: EastSet) -> EastSet:
        return EastSet(K, list(s))

    return set_copy


def set_union_in_place_for(K: Any) -> Callable[[EastSet, EastSet], None]:
    """Factory for union in place."""

    def set_union_in_place(a: EastSet, b: EastSet) -> None:
        for x in b:
            a.add(x)

    return set_union_in_place


def set_to_array_for(K: Any, T2: Any) -> Callable[[EastSet, Any], EastArray]:
    """Factory for converting set to array."""

    def set_to_array(s: EastSet, func: Any) -> EastArray:
        s._lock_for_iteration()
        try:
            mapped = [func(item) for item in s]
            return EastArray(K, mapped)
        finally:
            s._unlock_for_iteration()

    return set_to_array


def set_generate_for(K: Any) -> Callable[[int, Any, Any], EastSet]:
    """Factory for generating set."""

    def set_generate(n: int, gen_fn: Any, validate_fn: Any) -> EastSet:
        elements = set()
        for i in range(n):
            element = gen_fn(i)
            if element in elements:
                validate_fn(element)
            elements.add(element)
        return EastSet(K, elements)

    return set_generate


def set_try_insert_for(K: Any) -> Callable[[EastSet, Any], bool]:
    """Factory for trying to insert element."""

    def set_try_insert(s: EastSet, element: Any) -> bool:
        was_new = element not in s
        s.add(element)
        return was_new

    return set_try_insert


def set_try_delete_for(K: Any) -> Callable[[EastSet, Any], bool]:
    """Factory for trying to delete element."""

    def set_try_delete(s: EastSet, element: Any) -> bool:
        if element in s:
            s.remove(element)
            return True
        return False

    return set_try_delete


def set_for_each_for(K: Any, T2: Any) -> Callable[[EastSet, Any], None]:
    """Factory for iterating over set."""

    def set_for_each(s: EastSet, func: Any) -> None:
        s._lock_for_iteration()
        try:
            for element in s:
                func(element)
        finally:
            s._unlock_for_iteration()

    return set_for_each


def set_map_for(K: Any, T2: Any) -> Callable[[EastSet, Any], Any]:
    """Factory for mapping set to dict."""

    def set_map(s: EastSet, func: Any) -> Any:
        from east.types.containers import EastDict

        s._lock_for_iteration()
        try:
            result = EastDict(K, T2, {})
            for element in s:
                result[element] = func(element)
            return result
        finally:
            s._unlock_for_iteration()

    return set_map


def set_filter_for(K: Any) -> Callable[[EastSet, Any], EastSet]:
    """Factory for filtering set."""

    def set_filter(s: EastSet, func: Any) -> EastSet:
        s._lock_for_iteration()
        try:
            filtered = {element for element in s if func(element)}
            return EastSet(K, filtered)
        finally:
            s._unlock_for_iteration()

    return set_filter


def set_filter_map_for(K: Any, V2: Any) -> Callable[[EastSet, Any], Any]:
    """Factory for filter and map to dict."""

    def set_filter_map(s: EastSet, func: Any) -> Any:
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

    return set_filter_map


def set_first_map_for(K: Any, T2: Any) -> Callable[[EastSet, Any], Any]:
    """Factory for finding first element that maps to some."""

    def set_first_map(s: EastSet, func: Any) -> Any:
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

    return set_first_map


def set_map_reduce_for(K: Any, T2: Any) -> Callable[[EastSet, Any, Any], Any]:
    """Factory for map then reduce."""

    def set_map_reduce(s: EastSet, map_fn: Any, reduce_fn: Any) -> Any:
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

    return set_map_reduce


def set_reduce_for(K: Any, T2: Any) -> Callable[[EastSet, Any, Any], Any]:
    """Factory for folding over set."""

    def set_reduce(s: EastSet, func: Any, initial: Any) -> Any:
        s._lock_for_iteration()
        try:
            accumulator = initial
            for element in s:
                accumulator = func(accumulator, element)
            return accumulator
        finally:
            s._unlock_for_iteration()

    return set_reduce


def set_to_set_for(K: Any, K2: Any) -> Callable[[EastSet, Any], EastSet]:
    """Factory for mapping set to new set."""

    def set_to_set(s: EastSet, func: Any) -> EastSet:
        s._lock_for_iteration()
        try:
            mapped = {func(element) for element in s}
            return EastSet(K2, mapped)
        finally:
            s._unlock_for_iteration()

    return set_to_set


def set_to_dict_for(K: Any, K2: Any, T2: Any) -> Callable[[EastSet, Any, Any, Any], Any]:
    """Factory for converting set to dict."""

    def set_to_dict(s: EastSet, key_fn: Any, value_fn: Any, merge_fn: Any) -> Any:
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

    return set_to_dict


def set_flatten_to_array_for(K: Any, T2: Any) -> Callable[[EastSet, Any], EastArray]:
    """Factory for flat map to array."""

    def set_flatten_to_array(s: EastSet, func: Any) -> EastArray:
        s._lock_for_iteration()
        try:
            results = []
            for element in s:
                mapped = func(element)
                results.extend(mapped)
            return EastArray(T2, results)
        finally:
            s._unlock_for_iteration()

    return set_flatten_to_array


def set_flatten_to_set_for(K: Any, K2: Any) -> Callable[[EastSet, Any], EastSet]:
    """Factory for flat map to set."""

    def set_flatten_to_set(s: EastSet, func: Any) -> EastSet:
        s._lock_for_iteration()
        try:
            result = set()
            for element in s:
                mapped = func(element)
                result.update(mapped)
            return EastSet(K2, result)
        finally:
            s._unlock_for_iteration()

    return set_flatten_to_set


def set_flatten_to_dict_for(K: Any, K2: Any, T2: Any) -> Callable[[EastSet, Any, Any], Any]:
    """Factory for flat map to dict."""

    def set_flatten_to_dict(s: EastSet, func: Any, merge_fn: Any) -> Any:
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

    return set_flatten_to_dict


def set_group_fold_for(K: Any, K2: Any, T2: Any) -> Callable[[EastSet, Any, Any, Any], Any]:
    """Factory for grouping and folding."""

    def set_group_fold(s: EastSet, key_fn: Any, init_fn: Any, fold_fn: Any) -> Any:
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

    return set_group_fold


# Register all set builtins as factories
register_builtin("SetGenerate", set_generate_for)
register_builtin("SetSize", set_size_for)
register_builtin("SetHas", set_has_for)
register_builtin("SetInsert", set_add_for)
register_builtin("SetTryInsert", set_try_insert_for)
register_builtin("SetDelete", set_remove_for)
register_builtin("SetTryDelete", set_try_delete_for)
register_builtin("SetClear", set_clear_for)
register_builtin("SetUnionInPlace", set_union_in_place_for)
register_builtin("SetUnion", set_union_for)
register_builtin("SetIntersect", set_intersection_for)
register_builtin("SetDiff", set_difference_for)
register_builtin("SetSymDiff", set_symmetric_difference_for)
register_builtin("SetIsSubset", set_is_subset_for)
register_builtin("SetIsDisjoint", set_is_disjoint_for)
register_builtin("SetCopy", set_copy_for)
register_builtin("SetForEach", set_for_each_for)
register_builtin("SetMap", set_map_for)
register_builtin("SetFilter", set_filter_for)
register_builtin("SetFilterMap", set_filter_map_for)
register_builtin("SetFirstMap", set_first_map_for)
register_builtin("SetMapReduce", set_map_reduce_for)
register_builtin("SetReduce", set_reduce_for)
register_builtin("SetToArray", set_to_array_for)
register_builtin("SetToSet", set_to_set_for)
register_builtin("SetToDict", set_to_dict_for)
register_builtin("SetFlattenToArray", set_flatten_to_array_for)
register_builtin("SetFlattenToSet", set_flatten_to_set_for)
register_builtin("SetFlattenToDict", set_flatten_to_dict_for)
register_builtin("SetGroupFold", set_group_fold_for)


__all__ = [
    "set_generate_for",
    "set_size_for",
    "set_has_for",
    "set_add_for",
    "set_try_insert_for",
    "set_remove_for",
    "set_try_delete_for",
    "set_clear_for",
    "set_union_for",
    "set_union_in_place_for",
    "set_intersection_for",
    "set_difference_for",
    "set_symmetric_difference_for",
    "set_is_subset_for",
    "set_is_disjoint_for",
    "set_copy_for",
    "set_for_each_for",
    "set_map_for",
    "set_filter_for",
    "set_filter_map_for",
    "set_first_map_for",
    "set_map_reduce_for",
    "set_reduce_for",
    "set_to_array_for",
    "set_to_set_for",
    "set_to_dict_for",
    "set_flatten_to_array_for",
    "set_flatten_to_set_for",
    "set_flatten_to_dict_for",
    "set_group_fold_for",
]
