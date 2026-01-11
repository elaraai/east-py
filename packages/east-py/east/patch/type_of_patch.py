#
# Copyright (c) 2025 Elara AI Pty Ltd
# Licensed under the Business Source License 1.1. See LICENSE.md for details.
#
"""PatchType - Compute the patch type for a given East type.

This module provides the PatchType function that constructs the
patch type structure for any East type.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from east.types.types import (
    ArrayType,
    DictType,
    IntegerType,
    NullType,
    StructType,
    VariantType,
    is_array_type,
    is_async_function_type,
    is_blob_type,
    is_boolean_type,
    is_datetime_type,
    is_dict_type,
    is_float_type,
    is_function_type,
    is_integer_type,
    is_never_type,
    is_null_type,
    is_recursive_type,
    is_ref_type,
    is_set_type,
    is_string_type,
    is_struct_type,
    is_variant_type,
)

if TYPE_CHECKING:
    from east.types.types import EastType


def PatchType(type_val: EastType, ctx: dict[int, EastType] | None = None) -> EastType:
    """Construct the patch type for a given East type.

    Args:
        type_val: The East type to compute patch type for
        ctx: Context for caching recursive types (internal)

    Returns:
        The patch type (a VariantType with unchanged/replace/patch cases)
    """
    if ctx is None:
        ctx = {}

    # Check cache for recursive types (by object id)
    type_id = id(type_val)
    if type_id in ctx:
        return ctx[type_id]

    # Primitives: unchanged | replace (no patch case)
    if (
        is_never_type(type_val)
        or is_null_type(type_val)
        or is_boolean_type(type_val)
        or is_integer_type(type_val)
        or is_float_type(type_val)
        or is_string_type(type_val)
        or is_datetime_type(type_val)
        or is_blob_type(type_val)
    ):
        return VariantType(
            [
                ("unchanged", NullType),
                ("replace", StructType([("before", type_val), ("after", type_val)])),
            ]
        )

    if is_array_type(type_val):
        element_type = type_val.value
        element_patch = PatchType(element_type, ctx)
        operation_type = VariantType(
            [
                ("delete", element_type),
                ("insert", element_type),
                ("update", element_patch),
            ]
        )
        entry_type = StructType(
            [
                ("key", IntegerType),
                ("offset", IntegerType),
                ("operation", operation_type),
            ]
        )
        return VariantType(
            [
                ("unchanged", NullType),
                ("replace", StructType([("before", type_val), ("after", type_val)])),
                ("patch", ArrayType(entry_type)),
            ]
        )

    if is_set_type(type_val):
        key_type = type_val.value
        operation_type = VariantType(
            [
                ("delete", NullType),
                ("insert", NullType),
            ]
        )
        return VariantType(
            [
                ("unchanged", NullType),
                ("replace", StructType([("before", type_val), ("after", type_val)])),
                ("patch", DictType(key_type, operation_type)),
            ]
        )

    if is_dict_type(type_val):
        key_type = type_val.value["key"]
        value_type = type_val.value["value"]
        value_patch = PatchType(value_type, ctx)
        operation_type = VariantType(
            [
                ("delete", value_type),
                ("insert", value_type),
                ("update", value_patch),
            ]
        )
        return VariantType(
            [
                ("unchanged", NullType),
                ("replace", StructType([("before", type_val), ("after", type_val)])),
                ("patch", DictType(key_type, operation_type)),
            ]
        )

    if is_struct_type(type_val):
        patch_fields: list[tuple[str, EastType]] = []
        for field_spec in type_val.value:
            name = field_spec["name"]
            field_type = field_spec["type"]
            patch_fields.append((name, PatchType(field_type, ctx)))
        return VariantType(
            [
                ("unchanged", NullType),
                ("replace", StructType([("before", type_val), ("after", type_val)])),
                ("patch", StructType(patch_fields)),
            ]
        )

    if is_variant_type(type_val):
        patch_cases: list[tuple[str, EastType]] = []
        for case_spec in type_val.value:
            name = case_spec["name"]
            case_type = case_spec["type"]
            patch_cases.append((name, PatchType(case_type, ctx)))
        return VariantType(
            [
                ("unchanged", NullType),
                ("replace", StructType([("before", type_val), ("after", type_val)])),
                ("patch", VariantType(patch_cases)),
            ]
        )

    if is_ref_type(type_val):
        inner_type = type_val.value
        inner_patch = PatchType(inner_type, ctx)
        return VariantType(
            [
                ("unchanged", NullType),
                ("replace", StructType([("before", type_val), ("after", type_val)])),
                ("patch", inner_patch),
            ]
        )

    if is_recursive_type(type_val):
        # Check if we've already seen this type (handles circular back-references)
        if type_id in ctx:
            return ctx[type_id]

        # For back-references within the recursive structure, use replace-only semantics
        # Register this BEFORE recursing so circular refs are caught
        replace_only_type = VariantType(
            [
                ("unchanged", NullType),
                ("replace", StructType([("before", type_val), ("after", type_val)])),
            ]
        )
        ctx[type_id] = replace_only_type

        # For RecursiveType, use replace-only semantics
        return replace_only_type

    if is_function_type(type_val) or is_async_function_type(type_val):
        return VariantType(
            [
                ("unchanged", NullType),
                ("replace", StructType([("before", type_val), ("after", type_val)])),
            ]
        )

    raise RuntimeError(f"Unhandled type in PatchType: {type_val.type}")


__all__ = ["PatchType"]
