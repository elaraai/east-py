#
# Copyright (c) 2025 Elara AI Pty Ltd
# Licensed under the Business Source License 1.1. See LICENSE.md for details.
#
"""Matrix builtin functions.

These are factory builtins that take type parameters at compile time.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from east.runtime.platform import PlatformFunction

from east.builtins.registry import register_builtin
from east.runtime.errors import EastError
from east.types.types import ArrayType, EastType
from east.types.values import EAST_ELEMENT_TO_DTYPE, EastArray, EastMatrix, EastVector, east_null


def matrix_rows_for(
    _platform: "list[PlatformFunction]", T: EastType
) -> Callable[[EastMatrix], int]:
    """Factory for getting matrix row count."""

    def matrix_rows(mat: EastMatrix) -> int:
        return mat.rows

    return matrix_rows


def matrix_cols_for(
    _platform: "list[PlatformFunction]", T: EastType
) -> Callable[[EastMatrix], int]:
    """Factory for getting matrix column count."""

    def matrix_cols(mat: EastMatrix) -> int:
        return mat.cols

    return matrix_cols


def matrix_get_for(
    _platform: "list[PlatformFunction]", T: EastType
) -> Callable[[EastMatrix, int, int], Any]:
    """Factory for getting element at (row, col)."""
    is_boolean = T.type == "Boolean"

    def matrix_get(mat: EastMatrix, row: int, col: int) -> Any:
        if row < 0 or row >= mat.rows or col < 0 or col >= mat.cols:
            raise EastError(
                f"Matrix index ({row}, {col}) out of bounds for {mat.rows}x{mat.cols} matrix",
                {"filename": "", "line": 0, "column": 0},
            )
        val = mat.data[row, col]
        if is_boolean:
            return bool(val)
        return val.item()

    return matrix_get


def matrix_set_for(
    _platform: "list[PlatformFunction]", T: EastType
) -> Callable[[EastMatrix, int, int, Any], Any]:
    """Factory for setting element at (row, col)."""

    def matrix_set(mat: EastMatrix, row: int, col: int, value: Any) -> Any:
        if row < 0 or row >= mat.rows or col < 0 or col >= mat.cols:
            raise EastError(
                f"Matrix index ({row}, {col}) out of bounds for {mat.rows}x{mat.cols} matrix",
                {"filename": "", "line": 0, "column": 0},
            )
        mat.data[row, col] = value
        return east_null

    return matrix_set


def matrix_get_row_for(
    _platform: "list[PlatformFunction]", T: EastType
) -> Callable[[EastMatrix, int], EastVector]:
    """Factory for getting a row as a vector (copy)."""

    def matrix_get_row(mat: EastMatrix, row: int) -> EastVector:
        if row < 0 or row >= mat.rows:
            raise EastError(
                f"Matrix row {row} out of bounds for {mat.rows}x{mat.cols} matrix",
                {"filename": "", "line": 0, "column": 0},
            )
        return EastVector(mat.element_type, mat.data[row].copy())

    return matrix_get_row


def matrix_get_col_for(
    _platform: "list[PlatformFunction]", T: EastType
) -> Callable[[EastMatrix, int], EastVector]:
    """Factory for getting a column as a vector (copy)."""

    def matrix_get_col(mat: EastMatrix, col: int) -> EastVector:
        if col < 0 or col >= mat.cols:
            raise EastError(
                f"Matrix column {col} out of bounds for {mat.rows}x{mat.cols} matrix",
                {"filename": "", "line": 0, "column": 0},
            )
        return EastVector(mat.element_type, mat.data[:, col].copy())

    return matrix_get_col


def matrix_to_vector_for(
    _platform: "list[PlatformFunction]", T: EastType
) -> Callable[[EastMatrix], EastVector]:
    """Factory for flattening matrix to vector (row-major)."""

    def matrix_to_vector(mat: EastMatrix) -> EastVector:
        return EastVector(mat.element_type, mat.data.ravel(order="C").copy())

    return matrix_to_vector


def matrix_from_array_for(
    _platform: "list[PlatformFunction]", T: EastType
) -> Callable[[EastArray], EastMatrix]:
    """Factory for converting nested array to matrix."""
    dtype = EAST_ELEMENT_TO_DTYPE[T.type]

    def matrix_from_array(arr: EastArray) -> EastMatrix:
        if len(arr) == 0:
            return EastMatrix(T, np.empty((0, 0), dtype=dtype))
        cols = len(arr[0])
        for i, row in enumerate(arr):
            if len(row) != cols:
                raise EastError(
                    f"Jagged array: row 0 has {cols} columns but row {i} has {len(row)}",
                    {"filename": "", "line": 0, "column": 0},
                )
        rows = len(arr)
        data = np.array([list(row) for row in arr], dtype=dtype)
        return EastMatrix(T, data, rows, cols)

    return matrix_from_array


def matrix_to_array_for(
    _platform: "list[PlatformFunction]", T: EastType
) -> Callable[[EastMatrix], EastArray]:
    """Factory for converting matrix to nested array."""
    is_boolean = T.type == "Boolean"

    def matrix_to_array(mat: EastMatrix) -> EastArray:
        rows = []
        for r in range(mat.rows):
            if is_boolean:
                row: EastArray = EastArray(T, [bool(mat.data[r, c]) for c in range(mat.cols)])
            else:
                row = EastArray(T, [mat.data[r, c].item() for c in range(mat.cols)])
            rows.append(row)
        return EastArray(ArrayType(T), rows)

    return matrix_to_array


def matrix_transpose_for(
    _platform: "list[PlatformFunction]", T: EastType
) -> Callable[[EastMatrix], EastMatrix]:
    """Factory for transposing a matrix."""

    def matrix_transpose(mat: EastMatrix) -> EastMatrix:
        transposed = np.ascontiguousarray(mat.data.T)
        return EastMatrix(mat.element_type, transposed)

    return matrix_transpose


def matrix_zeros_for(
    _platform: "list[PlatformFunction]", T: EastType
) -> Callable[[int, int], EastMatrix]:
    """Factory for creating zero-filled matrix."""
    dtype = EAST_ELEMENT_TO_DTYPE[T.type]

    def matrix_zeros(rows: int, cols: int) -> EastMatrix:
        return EastMatrix(T, np.zeros((rows, cols), dtype=dtype))

    return matrix_zeros


def matrix_ones_for(
    _platform: "list[PlatformFunction]", T: EastType
) -> Callable[[int, int], EastMatrix]:
    """Factory for creating one-filled matrix."""
    dtype = EAST_ELEMENT_TO_DTYPE[T.type]

    def matrix_ones(rows: int, cols: int) -> EastMatrix:
        return EastMatrix(T, np.ones((rows, cols), dtype=dtype))

    return matrix_ones


def matrix_fill_for(
    _platform: "list[PlatformFunction]", T: EastType
) -> Callable[[int, int, Any], EastMatrix]:
    """Factory for creating a filled matrix."""
    dtype = EAST_ELEMENT_TO_DTYPE[T.type]

    def matrix_fill(rows: int, cols: int, value: Any) -> EastMatrix:
        return EastMatrix(T, np.full((rows, cols), value, dtype=dtype))

    return matrix_fill


def matrix_map_elements_for(
    _platform: "list[PlatformFunction]", T: EastType, T2: EastType
) -> Callable[[EastMatrix, Callable], EastMatrix]:
    """Factory for mapping over all matrix elements."""
    dtype = EAST_ELEMENT_TO_DTYPE[T2.type]
    is_boolean_in = T.type == "Boolean"

    def matrix_map_elements(mat: EastMatrix, fn: Callable) -> EastMatrix:
        results = []
        for r in range(mat.rows):
            for c in range(mat.cols):
                elem = bool(mat.data[r, c]) if is_boolean_in else mat.data[r, c].item()
                results.append(fn(elem, r, c))
        data = np.array(results, dtype=dtype).reshape(mat.rows, mat.cols)
        return EastMatrix(T2, data)

    return matrix_map_elements


def matrix_map_rows_for(
    _platform: "list[PlatformFunction]", T: EastType, T2: EastType
) -> Callable[[EastMatrix, Callable], EastMatrix]:
    """Factory for mapping over matrix rows."""

    def matrix_map_rows(mat: EastMatrix, fn: Callable) -> EastMatrix:
        row_vecs = []
        for r in range(mat.rows):
            row_vec = EastVector(mat.element_type, mat.data[r].copy())
            result_vec = fn(row_vec, r)
            row_vecs.append(result_vec.data)
        if not row_vecs:
            dtype = EAST_ELEMENT_TO_DTYPE[T2.type]
            return EastMatrix(T2, np.empty((0, 0), dtype=dtype))
        data = np.stack(row_vecs)
        return EastMatrix(T2, data)

    return matrix_map_rows


def matrix_to_rows_for(
    _platform: "list[PlatformFunction]", T: EastType
) -> Callable[[EastMatrix], EastArray]:
    """Factory for decomposing a matrix into an array of row vectors."""
    from east.types.types import VectorType

    vec_type = VectorType(T)

    def matrix_to_rows(mat: EastMatrix) -> EastArray:
        rows = []
        for r in range(mat.rows):
            rows.append(EastVector(mat.element_type, mat.data[r].copy()))
        return EastArray(vec_type, rows)

    return matrix_to_rows


def matrix_from_rows_for(
    _platform: "list[PlatformFunction]", T: EastType
) -> Callable[[EastArray], EastMatrix]:
    """Factory for constructing a matrix from an array of row vectors."""
    dtype = EAST_ELEMENT_TO_DTYPE[T.type]

    def matrix_from_rows(arr: EastArray) -> EastMatrix:
        if len(arr) == 0:
            return EastMatrix(T, np.empty((0, 0), dtype=dtype))
        cols = len(arr[0].data)
        for i in range(1, len(arr)):
            if len(arr[i].data) != cols:
                raise EastError(
                    f"Jagged rows: row 0 has {cols} columns but row {i} has {len(arr[i].data)}",
                    {"filename": "", "line": 0, "column": 0},
                )
        data = np.empty((len(arr), cols), dtype=dtype)
        for i in range(len(arr)):
            data[i] = arr[i].data
        return EastMatrix(T, data)

    return matrix_from_rows


# Register all matrix builtins
register_builtin("MatrixRows", matrix_rows_for)
register_builtin("MatrixCols", matrix_cols_for)
register_builtin("MatrixGet", matrix_get_for)
register_builtin("MatrixSet", matrix_set_for)
register_builtin("MatrixGetRow", matrix_get_row_for)
register_builtin("MatrixGetCol", matrix_get_col_for)
register_builtin("MatrixToVector", matrix_to_vector_for)
register_builtin("MatrixFromArray", matrix_from_array_for)
register_builtin("MatrixToArray", matrix_to_array_for)
register_builtin("MatrixTranspose", matrix_transpose_for)
register_builtin("MatrixZeros", matrix_zeros_for)
register_builtin("MatrixOnes", matrix_ones_for)
register_builtin("MatrixFill", matrix_fill_for)
register_builtin("MatrixMapElements", matrix_map_elements_for)
register_builtin("MatrixMapRows", matrix_map_rows_for)
register_builtin("MatrixToRows", matrix_to_rows_for)
register_builtin("MatrixFromRows", matrix_from_rows_for)
