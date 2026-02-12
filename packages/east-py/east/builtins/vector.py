#
# Copyright (c) 2025 Elara AI Pty Ltd
# Licensed under the Business Source License 1.1. See LICENSE.md for details.
#
"""Vector builtin functions.

These are factory builtins that take type parameters at compile time.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from east.runtime.platform import PlatformFunction

from east.builtins.registry import register_builtin
from east.runtime.errors import EastError
from east.types.types import EastType
from east.types.values import EAST_ELEMENT_TO_DTYPE, EastArray, EastMatrix, EastVector, east_null


def vector_length_for(
    _platform: "list[PlatformFunction]", T: EastType
) -> Callable[[EastVector], int]:
    """Factory for getting vector length."""

    def vector_length(vec: EastVector) -> int:
        return len(vec)

    return vector_length


def vector_get_for(
    _platform: "list[PlatformFunction]", T: EastType
) -> Callable[[EastVector, int], Any]:
    """Factory for getting element at index."""
    is_boolean = T.type == "Boolean"

    def vector_get(vec: EastVector, index: int) -> Any:
        if index < 0 or index >= len(vec.data):
            raise EastError(
                f"Vector index {index} out of bounds for length {len(vec.data)}",
                {"filename": "", "line": 0, "column": 0},
            )
        val = vec.data[index]
        if is_boolean:
            return bool(val)
        return val.item()

    return vector_get


def vector_set_for(
    _platform: "list[PlatformFunction]", T: EastType
) -> Callable[[EastVector, int, Any], Any]:
    """Factory for setting element at index."""

    def vector_set(vec: EastVector, index: int, value: Any) -> Any:
        if index < 0 or index >= len(vec.data):
            raise EastError(
                f"Vector index {index} out of bounds for length {len(vec.data)}",
                {"filename": "", "line": 0, "column": 0},
            )
        vec.data[index] = value
        return east_null

    return vector_set


def vector_slice_for(
    _platform: "list[PlatformFunction]", T: EastType
) -> Callable[[EastVector, int, int], EastVector]:
    """Factory for slicing a vector."""

    def vector_slice(vec: EastVector, start: int, end: int) -> EastVector:
        return EastVector(vec.element_type, vec.data[start:end].copy())

    return vector_slice


def vector_concat_for(
    _platform: "list[PlatformFunction]", T: EastType
) -> Callable[[EastVector, EastVector], EastVector]:
    """Factory for concatenating two vectors."""

    def vector_concat(a: EastVector, b: EastVector) -> EastVector:
        return EastVector(a.element_type, np.concatenate([a.data, b.data]))

    return vector_concat


def vector_from_array_for(
    _platform: "list[PlatformFunction]", T: EastType
) -> Callable[[EastArray], EastVector]:
    """Factory for converting array to vector."""
    dtype = EAST_ELEMENT_TO_DTYPE[T.type]

    def vector_from_array(arr: EastArray) -> EastVector:
        return EastVector(T, np.array(list(arr), dtype=dtype))

    return vector_from_array


def vector_to_array_for(
    _platform: "list[PlatformFunction]", T: EastType
) -> Callable[[EastVector], EastArray]:
    """Factory for converting vector to array."""
    is_boolean = T.type == "Boolean"

    def vector_to_array(vec: EastVector) -> EastArray:
        if is_boolean:
            return EastArray(T, [bool(x) for x in vec.data])
        return EastArray(T, [x.item() for x in vec.data])

    return vector_to_array


def vector_to_matrix_for(
    _platform: "list[PlatformFunction]", T: EastType
) -> Callable[[EastVector, int, int], EastMatrix]:
    """Factory for reshaping vector to matrix."""

    def vector_to_matrix(vec: EastVector, rows: int, cols: int) -> EastMatrix:
        if rows * cols != len(vec.data):
            raise EastError(
                f"Cannot reshape vector of length {len(vec.data)} to {rows}x{cols} matrix",
                {"filename": "", "line": 0, "column": 0},
            )
        return EastMatrix(vec.element_type, vec.data.copy().reshape(rows, cols), rows, cols)

    return vector_to_matrix


def vector_zeros_for(
    _platform: "list[PlatformFunction]", T: EastType
) -> Callable[[int], EastVector]:
    """Factory for creating zero-filled vector."""
    dtype = EAST_ELEMENT_TO_DTYPE[T.type]

    def vector_zeros(length: int) -> EastVector:
        return EastVector(T, np.zeros(length, dtype=dtype))

    return vector_zeros


def vector_ones_for(
    _platform: "list[PlatformFunction]", T: EastType
) -> Callable[[int], EastVector]:
    """Factory for creating one-filled vector."""
    dtype = EAST_ELEMENT_TO_DTYPE[T.type]

    def vector_ones(length: int) -> EastVector:
        return EastVector(T, np.ones(length, dtype=dtype))

    return vector_ones


def vector_fill_for(
    _platform: "list[PlatformFunction]", T: EastType
) -> Callable[[int, Any], EastVector]:
    """Factory for creating a filled vector."""
    dtype = EAST_ELEMENT_TO_DTYPE[T.type]

    def vector_fill(length: int, value: Any) -> EastVector:
        return EastVector(T, np.full(length, value, dtype=dtype))

    return vector_fill


def vector_map_for(
    _platform: "list[PlatformFunction]", T: EastType, T2: EastType
) -> Callable[[EastVector, Callable], EastVector]:
    """Factory for mapping over vector elements."""
    dtype = EAST_ELEMENT_TO_DTYPE[T2.type]
    is_boolean_in = T.type == "Boolean"

    def vector_map(vec: EastVector, fn: Callable) -> EastVector:
        results = []
        for i in range(len(vec.data)):
            elem = bool(vec.data[i]) if is_boolean_in else vec.data[i].item()
            results.append(fn(elem, i))
        return EastVector(T2, np.array(results, dtype=dtype))

    return vector_map


def vector_fold_for(
    _platform: "list[PlatformFunction]", T: EastType, T2: EastType
) -> Callable[[EastVector, Any, Callable], Any]:
    """Factory for folding/reducing a vector."""
    is_boolean = T.type == "Boolean"

    def vector_fold(vec: EastVector, init: Any, fn: Callable) -> Any:
        acc = init
        for i in range(len(vec.data)):
            elem = bool(vec.data[i]) if is_boolean else vec.data[i].item()
            acc = fn(acc, elem, i)
        return acc

    return vector_fold


# Register all vector builtins
register_builtin("VectorLength", vector_length_for)
register_builtin("VectorGet", vector_get_for)
register_builtin("VectorSet", vector_set_for)
register_builtin("VectorSlice", vector_slice_for)
register_builtin("VectorConcat", vector_concat_for)
register_builtin("VectorFromArray", vector_from_array_for)
register_builtin("VectorToArray", vector_to_array_for)
register_builtin("VectorToMatrix", vector_to_matrix_for)
register_builtin("VectorZeros", vector_zeros_for)
register_builtin("VectorOnes", vector_ones_for)
register_builtin("VectorFill", vector_fill_for)
register_builtin("VectorMap", vector_map_for)
register_builtin("VectorFold", vector_fold_for)
