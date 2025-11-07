"""Array builtin functions."""

from typing import Any

from east.builtins.registry import register_builtin
from east.types.containers import EastArray


def array_length(arr: EastArray) -> int:
    """Get length of array.

    Args:
        arr: Array

    Returns:
        Number of elements in array
    """
    return len(arr)


def array_get(arr: EastArray, index: int) -> Any:
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


def array_set(arr: EastArray, index: int, value: Any) -> None:
    """Set element at index (mutation).

    Args:
        arr: Array
        index: Element index (0-based)
        value: New value

    Raises:
        IndexError: If index out of bounds
    """
    arr[index] = value


def array_push_first(arr: EastArray, value: Any) -> None:
    """Prepend element to array (mutation).

    Args:
        arr: Array
        value: Value to prepend
    """
    arr.insert(0, value)


def array_push_last(arr: EastArray, value: Any) -> None:
    """Append element to array (mutation).

    Args:
        arr: Array
        value: Value to append
    """
    arr.append(value)


def array_pop_first(arr: EastArray) -> Any:
    """Remove and return first element (mutation).

    Args:
        arr: Array

    Returns:
        First element

    Raises:
        IndexError: If array is empty
    """
    return arr.pop(0)


def array_pop_last(arr: EastArray) -> Any:
    """Remove and return last element (mutation).

    Args:
        arr: Array

    Returns:
        Last element

    Raises:
        IndexError: If array is empty
    """
    return arr.pop()


def array_insert(arr: EastArray, index: int, value: Any) -> None:
    """Insert element at index (mutation).

    Args:
        arr: Array
        index: Index to insert at
        value: Value to insert
    """
    arr.insert(index, value)


def array_remove(arr: EastArray, index: int) -> Any:
    """Remove and return element at index (mutation).

    Args:
        arr: Array
        index: Index to remove at

    Returns:
        Removed element

    Raises:
        IndexError: If index out of bounds
    """
    return arr.pop(index)


def array_slice(arr: EastArray, start: int, end: int) -> EastArray:
    """Get array slice.

    Args:
        arr: Array
        start: Start index (inclusive)
        end: End index (exclusive)

    Returns:
        New array with slice
    """
    return EastArray(arr.element_type, arr[start:end])


def array_concat(a: EastArray, b: EastArray) -> EastArray:
    """Concatenate two arrays.

    Args:
        a: First array
        b: Second array

    Returns:
        New array with concatenated elements
    """
    return EastArray(a.element_type, list(a) + list(b))


def array_reverse(arr: EastArray) -> EastArray:
    """Reverse array.

    Args:
        arr: Array

    Returns:
        New array with reversed elements
    """
    return EastArray(arr.element_type, list(reversed(arr)))


def array_sort(arr: EastArray) -> EastArray:
    """Sort array using East ordering.

    Args:
        arr: Array

    Returns:
        New sorted array
    """
    from functools import cmp_to_key

    from east.utils.ordering import east_compare

    sorted_items = sorted(arr, key=cmp_to_key(east_compare))
    return EastArray(arr.element_type, sorted_items)


def array_contains(arr: EastArray, value: Any) -> bool:
    """Check if array contains value.

    Args:
        arr: Array
        value: Value to search for

    Returns:
        True if value is in array
    """
    return value in arr


def array_index_of(arr: EastArray, value: Any) -> int:
    """Find first index of value.

    Args:
        arr: Array
        value: Value to search for

    Returns:
        Index of first occurrence, or -1 if not found
    """
    try:
        return arr.index(value)
    except ValueError:
        return -1


# Higher-order functions


def array_map(arr: EastArray, func: dict[str, Any]) -> EastArray:
    """Map function over array elements.

    Args:
        arr: Array
        func: Closure to apply to each element

    Returns:
        New array with mapped values
    """
    from east.runtime.interpreter import Interpreter, ReturnException

    interp = Interpreter()
    func_node = func["node"]
    func_env = func["env"]

    mapped = []
    for item in arr:
        # Create environment and bind parameter
        call_env = func_env.extend()
        call_env.bind(func_node.param_names[0], item, False)

        # Execute function body
        try:
            result = interp.eval(func_node.body, call_env)
        except ReturnException as e:
            result = e.value

        mapped.append(result)

    return EastArray(func_node.return_type, mapped)


def array_filter(arr: EastArray, func: dict[str, Any]) -> EastArray:
    """Filter array elements by predicate.

    Args:
        arr: Array
        func: Closure returning boolean for each element

    Returns:
        New array with filtered elements
    """
    from east.runtime.interpreter import Interpreter, ReturnException

    interp = Interpreter()
    func_node = func["node"]
    func_env = func["env"]

    filtered = []
    for item in arr:
        # Create environment and bind parameter
        call_env = func_env.extend()
        call_env.bind(func_node.param_names[0], item, False)

        # Execute function body
        try:
            result = interp.eval(func_node.body, call_env)
        except ReturnException as e:
            result = e.value

        if result:
            filtered.append(item)

    return EastArray(arr.element_type, filtered)


def array_reduce(arr: EastArray, func: dict[str, Any], initial: Any) -> Any:
    """Reduce array to single value.

    Args:
        arr: Array
        func: Closure taking (accumulator, element) and returning new accumulator
        initial: Initial accumulator value

    Returns:
        Final accumulator value
    """
    from east.runtime.interpreter import Interpreter, ReturnException

    interp = Interpreter()
    func_node = func["node"]
    func_env = func["env"]

    accumulator = initial
    for item in arr:
        # Create environment and bind parameters
        call_env = func_env.extend()
        call_env.bind(func_node.param_names[0], accumulator, False)
        call_env.bind(func_node.param_names[1], item, False)

        # Execute function body
        try:
            accumulator = interp.eval(func_node.body, call_env)
        except ReturnException as e:
            accumulator = e.value

    return accumulator


def array_find(arr: EastArray, func: dict[str, Any]) -> Any:
    """Find first element matching predicate.

    Args:
        arr: Array
        func: Closure returning boolean for each element

    Returns:
        First matching element, or null if none found
    """
    from east.runtime.interpreter import Interpreter, ReturnException
    from east.types.primitives import null

    interp = Interpreter()
    func_node = func["node"]
    func_env = func["env"]

    for item in arr:
        # Create environment and bind parameter
        call_env = func_env.extend()
        call_env.bind(func_node.param_names[0], item, False)

        # Execute function body
        try:
            result = interp.eval(func_node.body, call_env)
        except ReturnException as e:
            result = e.value

        if result:
            return item

    return null


def array_find_index(arr: EastArray, func: dict[str, Any]) -> int:
    """Find index of first element matching predicate.

    Args:
        arr: Array
        func: Closure returning boolean for each element

    Returns:
        Index of first matching element, or -1 if none found
    """
    from east.runtime.interpreter import Interpreter, ReturnException

    interp = Interpreter()
    func_node = func["node"]
    func_env = func["env"]

    for index, item in enumerate(arr):
        # Create environment and bind parameter
        call_env = func_env.extend()
        call_env.bind(func_node.param_names[0], item, False)

        # Execute function body
        try:
            result = interp.eval(func_node.body, call_env)
        except ReturnException as e:
            result = e.value

        if result:
            return index

    return -1


# Register all array builtins
register_builtin("ArrayLength", array_length)
register_builtin("ArrayGet", array_get)
register_builtin("ArraySet", array_set)
register_builtin("ArrayPushFirst", array_push_first)
register_builtin("ArrayPushLast", array_push_last)
register_builtin("ArrayPopFirst", array_pop_first)
register_builtin("ArrayPopLast", array_pop_last)
register_builtin("ArrayInsert", array_insert)
register_builtin("ArrayRemove", array_remove)
register_builtin("ArraySlice", array_slice)
register_builtin("ArrayConcat", array_concat)
register_builtin("ArrayReverse", array_reverse)
register_builtin("ArraySort", array_sort)
register_builtin("ArrayContains", array_contains)
register_builtin("ArrayIndexOf", array_index_of)
register_builtin("ArrayMap", array_map)
register_builtin("ArrayFilter", array_filter)
register_builtin("ArrayReduce", array_reduce)
register_builtin("ArrayFind", array_find)
register_builtin("ArrayFindIndex", array_find_index)


__all__ = [
    "array_length",
    "array_get",
    "array_set",
    "array_push_first",
    "array_push_last",
    "array_pop_first",
    "array_pop_last",
    "array_insert",
    "array_remove",
    "array_slice",
    "array_concat",
    "array_reverse",
    "array_sort",
    "array_contains",
    "array_index_of",
    "array_map",
    "array_filter",
    "array_reduce",
    "array_find",
    "array_find_index",
]
