#
# Copyright (c) 2025 Elara AI Pty Ltd
# Licensed under the Business Source License 1.1. See LICENSE.md for details.
#
"""invert_for - Invert a patch.

This module provides the invertFor function that creates type-specific
invert functions for reversing patches.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from east.patch.types import InvertContext
from east.types.types import (
    NullType,
    is_array_type,
    is_async_function_type,
    is_blob_type,
    is_boolean_type,
    is_datetime_type,
    is_dict_type,
    is_float_type,
    is_function_type,
    is_integer_type,
    is_matrix_type,
    is_never_type,
    is_null_type,
    is_recursive_type,
    is_ref_type,
    is_set_type,
    is_string_type,
    is_struct_type,
    is_variant_type,
    is_vector_type,
)
from east.types.values import EastDict, EastVariant
from east.utils.ordering import equal_for

if TYPE_CHECKING:
    from east.types.types import EastType


def invert_for(
    type_val: EastType,
    ctx: InvertContext | None = None,
) -> Callable[[Any], Any]:
    """Create an invert function for a given type.

    Args:
        type_val: The East type
        ctx: Context for recursive type handling (internal)

    Returns:
        A function (patch) -> inverted_patch
    """
    if ctx is None:
        ctx = InvertContext()

    if is_never_type(type_val):

        def invert_never(_patch: Any) -> Any:
            raise RuntimeError("Cannot invert patches for type Never")

        return invert_never

    # Primitives: swap before/after in replace
    if (
        is_null_type(type_val)
        or is_boolean_type(type_val)
        or is_integer_type(type_val)
        or is_float_type(type_val)
        or is_string_type(type_val)
        or is_datetime_type(type_val)
        or is_blob_type(type_val)
        or is_vector_type(type_val)
        or is_matrix_type(type_val)
    ):

        def invert_primitive(patch: EastVariant) -> EastVariant:
            if patch.type == "unchanged":
                return patch
            if patch.type == "replace":
                return EastVariant(
                    "replace",
                    {"before": patch.value["after"], "after": patch.value["before"]},
                )
            raise RuntimeError(f"Invalid patch type for primitive inversion: {patch.type}")

        return invert_primitive

    if is_array_type(type_val):
        element_invert: Callable[[Any], Any]

        def invert_array(patch: EastVariant) -> EastVariant:
            if patch.type == "unchanged":
                return patch
            if patch.type == "replace":
                return EastVariant(
                    "replace",
                    {"before": patch.value["after"], "after": patch.value["before"]},
                )
            if patch.type == "patch":
                operations = patch.value
                inverted: list[dict[str, Any]] = []

                # Reverse order of operations
                for i in range(len(operations) - 1, -1, -1):
                    op = operations[i]
                    if op["operation"].type == "delete":
                        # Delete becomes insert at same position
                        inverted.append(
                            {
                                "key": op["key"],
                                "offset": 0,
                                "operation": EastVariant("insert", op["operation"].value),
                            }
                        )
                    elif op["operation"].type == "insert":
                        # Insert becomes delete at same position
                        inverted.append(
                            {
                                "key": op["key"],
                                "offset": 0,
                                "operation": EastVariant("delete", op["operation"].value),
                            }
                        )
                    elif op["operation"].type == "update":
                        inverted.append(
                            {
                                "key": op["key"],
                                "offset": 0,
                                "operation": EastVariant(
                                    "update", element_invert(op["operation"].value)
                                ),
                            }
                        )

                return EastVariant("patch", inverted)
            raise RuntimeError(f"Invalid patch type for array inversion: {patch.type}")

        # Build array equality using equal_for with current context
        array_equal = equal_for(type_val, ctx.equal)

        ctx.invert.append(invert_array)
        ctx.types.append(type_val)
        ctx.equal.append(array_equal)
        element_invert = invert_for(type_val.value, ctx)
        ctx.invert.pop()
        ctx.types.pop()
        ctx.equal.pop()

        return invert_array

    if is_set_type(type_val):

        def invert_set(patch: EastVariant) -> EastVariant:
            if patch.type == "unchanged":
                return patch
            if patch.type == "replace":
                return EastVariant(
                    "replace",
                    {"before": patch.value["after"], "after": patch.value["before"]},
                )
            if patch.type == "patch":
                operations = patch.value
                inverted: EastDict[Any, Any] = EastDict(type_val.value, NullType)

                for key, op in operations.items():
                    if op.type == "delete":
                        inverted[key] = EastVariant("insert", None)
                    elif op.type == "insert":
                        inverted[key] = EastVariant("delete", None)

                return EastVariant("patch", inverted)
            raise RuntimeError(f"Invalid patch type for set inversion: {patch.type}")

        return invert_set

    if is_dict_type(type_val):
        value_invert: Callable[[Any], Any]

        def invert_dict(patch: EastVariant) -> EastVariant:
            if patch.type == "unchanged":
                return patch
            if patch.type == "replace":
                return EastVariant(
                    "replace",
                    {"before": patch.value["after"], "after": patch.value["before"]},
                )
            if patch.type == "patch":
                operations = patch.value
                inverted: EastDict[Any, Any] = EastDict(type_val.value["key"], NullType)

                for key, op in operations.items():
                    if op.type == "delete":
                        inverted[key] = EastVariant("insert", op.value)
                    elif op.type == "insert":
                        inverted[key] = EastVariant("delete", op.value)
                    elif op.type == "update":
                        inverted[key] = EastVariant("update", value_invert(op.value))

                return EastVariant("patch", inverted)
            raise RuntimeError(f"Invalid patch type for dict inversion: {patch.type}")

        # Build dict equality using equal_for with current context
        dict_equal = equal_for(type_val, ctx.equal)

        ctx.invert.append(invert_dict)
        ctx.types.append(type_val)
        ctx.equal.append(dict_equal)
        value_invert = invert_for(type_val.value["value"], ctx)
        ctx.invert.pop()
        ctx.types.pop()
        ctx.equal.pop()

        return invert_dict

    if is_struct_type(type_val):
        field_inverts: dict[str, Callable[[Any], Any]] = {}

        def invert_struct(patch: EastVariant) -> EastVariant:
            if patch.type == "unchanged":
                return patch
            if patch.type == "replace":
                return EastVariant(
                    "replace",
                    {"before": patch.value["after"], "after": patch.value["before"]},
                )
            if patch.type == "patch":
                result: dict[str, Any] = {}
                all_unchanged = True

                for field_spec in type_val.value:
                    name = field_spec["name"]
                    inverted = field_inverts[name](patch.value[name])
                    result[name] = inverted
                    if inverted.type != "unchanged":
                        all_unchanged = False

                if all_unchanged:
                    return EastVariant("unchanged", None)

                return EastVariant("patch", result)
            raise RuntimeError(f"Invalid patch type for struct inversion: {patch.type}")

        # Build struct equality using equal_for with current context
        struct_equal = equal_for(type_val, ctx.equal)

        ctx.invert.append(invert_struct)
        ctx.types.append(type_val)
        ctx.equal.append(struct_equal)
        for field_spec in type_val.value:
            name = field_spec["name"]
            field_type = field_spec["type"]
            field_inverts[name] = invert_for(field_type, ctx)
        ctx.invert.pop()
        ctx.types.pop()
        ctx.equal.pop()

        return invert_struct

    if is_variant_type(type_val):
        case_inverts: dict[str, Callable[[Any], Any]] = {}

        def invert_variant(patch: EastVariant) -> EastVariant:
            if patch.type == "unchanged":
                return patch
            if patch.type == "replace":
                return EastVariant(
                    "replace",
                    {"before": patch.value["after"], "after": patch.value["before"]},
                )
            if patch.type == "patch":
                case_name = patch.value.type
                inverted = case_inverts[case_name](patch.value.value)

                if inverted.type == "unchanged":
                    return EastVariant("unchanged", None)

                return EastVariant("patch", EastVariant(case_name, inverted))
            raise RuntimeError(f"Invalid patch type for variant inversion: {patch.type}")

        # Build variant equality using equal_for with current context
        variant_equal = equal_for(type_val, ctx.equal)

        ctx.invert.append(invert_variant)
        ctx.types.append(type_val)
        ctx.equal.append(variant_equal)
        for case_spec in type_val.value:
            name = case_spec["name"]
            case_type = case_spec["type"]
            case_inverts[name] = invert_for(case_type, ctx)
        ctx.invert.pop()
        ctx.types.pop()
        ctx.equal.pop()

        return invert_variant

    if is_ref_type(type_val):
        inner_invert: Callable[[Any], Any]

        def invert_ref(patch: EastVariant) -> EastVariant:
            if patch.type == "unchanged":
                return patch
            if patch.type == "replace":
                return EastVariant(
                    "replace",
                    {"before": patch.value["after"], "after": patch.value["before"]},
                )
            if patch.type == "patch":
                inverted = inner_invert(patch.value)
                if inverted.type == "unchanged":
                    return EastVariant("unchanged", None)
                return EastVariant("patch", inverted)
            raise RuntimeError(f"Invalid patch type for ref inversion: {patch.type}")

        # Build ref equality using equal_for with current context
        ref_equal = equal_for(type_val, ctx.equal)

        ctx.invert.append(invert_ref)
        ctx.types.append(type_val)
        ctx.equal.append(ref_equal)
        inner_invert = invert_for(type_val.value, ctx)
        ctx.invert.pop()
        ctx.types.pop()
        ctx.equal.pop()

        return invert_ref

    if is_recursive_type(type_val):
        # Recursive types use replace-only semantics
        def invert_recursive(patch: EastVariant) -> EastVariant:
            if patch.type == "unchanged":
                return patch
            if patch.type == "replace":
                return EastVariant(
                    "replace",
                    {"before": patch.value["after"], "after": patch.value["before"]},
                )
            raise RuntimeError(
                f"Invalid patch type for recursive type inversion: {patch.type}"
            )

        return invert_recursive

    if is_function_type(type_val) or is_async_function_type(type_val):

        def invert_function(patch: EastVariant) -> EastVariant:
            if patch.type == "unchanged":
                return patch
            if patch.type == "replace":
                return EastVariant(
                    "replace",
                    {"before": patch.value["after"], "after": patch.value["before"]},
                )
            raise RuntimeError(f"Invalid patch type for function inversion: {patch.type}")

        return invert_function

    raise RuntimeError(f"Unhandled type in invert_for: {type_val.type}")


__all__ = ["invert_for"]
