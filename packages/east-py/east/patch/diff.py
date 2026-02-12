#
# Copyright (c) 2025 Elara AI Pty Ltd
# Licensed under the Business Source License 1.1. See LICENSE.md for details.
#
"""diff_for - Compute difference between two East values.

This module provides the diffFor function that creates type-specific
diff functions for computing patches between East values.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from east.patch.types import DiffContext, compute_lcs
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
from east.utils.ordering import equal_for, is_for

if TYPE_CHECKING:
    from east.types.types import EastType


def diff_for(
    type_val: EastType,
    ctx: DiffContext | None = None,
) -> Callable[[Any, Any], Any]:
    """Create a diff function for a given type.

    Args:
        type_val: The East type to create a diff function for
        ctx: Context for recursive type handling (internal)

    Returns:
        A function (before, after) -> patch
    """
    if ctx is None:
        ctx = DiffContext()

    if is_never_type(type_val):

        def diff_never(_before: Any, _after: Any) -> Any:
            raise RuntimeError("Cannot diff values of type Never")

        return diff_never

    # Primitives: unchanged or replace
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
        equal = equal_for(type_val)

        def diff_primitive(before: Any, after: Any) -> EastVariant:
            if equal(before, after):
                return EastVariant("unchanged", None)
            return EastVariant("replace", {"before": before, "after": after})

        return diff_primitive

    if is_array_type(type_val):
        element_equal: Callable[[Any, Any], bool]
        is_same: Callable[[Any, Any], bool]

        def diff_array(before: Any, after: Any) -> EastVariant:
            if is_same(before, after):
                return EastVariant("unchanged", None)

            before_list = list(before)
            after_list = list(after)

            before_indices, after_indices = compute_lcs(before_list, after_list, element_equal)

            operations: list[dict[str, Any]] = []
            before_ptr = 0
            after_ptr = 0
            lcs_ptr = 0
            delete_count = 0
            insert_count = 0

            while before_ptr < len(before_list) or after_ptr < len(after_list):
                next_before_lcs = (
                    before_indices[lcs_ptr] if lcs_ptr < len(before_indices) else len(before_list)
                )
                next_after_lcs = (
                    after_indices[lcs_ptr] if lcs_ptr < len(after_indices) else len(after_list)
                )

                while before_ptr < next_before_lcs:
                    # Delete: key is the position in the mutating array
                    actual_position = before_ptr - delete_count + insert_count
                    operations.append(
                        {
                            "key": actual_position,
                            "offset": 0,
                            "operation": EastVariant("delete", before_list[before_ptr]),
                        }
                    )
                    delete_count += 1
                    before_ptr += 1

                while after_ptr < next_after_lcs:
                    # Insert: key is the position in the target array
                    operations.append(
                        {
                            "key": after_ptr,
                            "offset": 0,
                            "operation": EastVariant("insert", after_list[after_ptr]),
                        }
                    )
                    insert_count += 1
                    after_ptr += 1

                if lcs_ptr < len(before_indices):
                    before_ptr += 1
                    after_ptr += 1
                    lcs_ptr += 1

            if len(operations) == 0:
                return EastVariant("unchanged", None)

            return EastVariant("patch", operations)

        # Build array equality using equal_for with current context
        array_equal = equal_for(type_val, ctx.equal)

        ctx.diff.append(diff_array)
        ctx.types.append(type_val)
        ctx.equal.append(array_equal)
        is_same = is_for(type_val, ctx.equal)
        element_equal = equal_for(type_val.value, ctx.equal)
        ctx.diff.pop()
        ctx.types.pop()
        ctx.equal.pop()

        return diff_array

    if is_set_type(type_val):
        # Set keys cannot contain recursive types, so no context needed
        is_same = is_for(type_val)

        def diff_set(before: Any, after: Any) -> EastVariant:
            if is_same(before, after):
                return EastVariant("unchanged", None)

            operations: EastDict[Any, Any] = EastDict(type_val.value, NullType)

            for key in before:
                if key not in after:
                    operations[key] = EastVariant("delete", None)

            for key in after:
                if key not in before:
                    operations[key] = EastVariant("insert", None)

            if len(operations) == 0:
                return EastVariant("unchanged", None)

            # Check if this is a complete replacement
            delete_count = sum(1 for op in operations.values() if op.type == "delete")
            insert_count = sum(1 for op in operations.values() if op.type == "insert")
            if delete_count == len(before) and insert_count == len(after) and len(before) > 0:
                return EastVariant("replace", {"before": before, "after": after})

            return EastVariant("patch", operations)

        return diff_set

    if is_dict_type(type_val):
        value_diff: Callable[[Any, Any], Any]
        value_equal: Callable[[Any, Any], bool]
        is_same: Callable[[Any, Any], bool]

        def diff_dict(before: Any, after: Any) -> EastVariant:
            if is_same(before, after):
                return EastVariant("unchanged", None)

            operations: EastDict[Any, Any] = EastDict(type_val.value["key"], NullType)

            for key, before_value in before.items():
                if key not in after:
                    operations[key] = EastVariant("delete", before_value)
                else:
                    after_value = after[key]
                    if not value_equal(before_value, after_value):
                        patch = value_diff(before_value, after_value)
                        operations[key] = EastVariant("update", patch)

            for key, after_value in after.items():
                if key not in before:
                    operations[key] = EastVariant("insert", after_value)

            if len(operations) == 0:
                return EastVariant("unchanged", None)

            # Check if this is a complete replacement
            insert_count = sum(1 for op in operations.values() if op.type == "insert")
            delete_count = sum(1 for op in operations.values() if op.type == "delete")
            if insert_count == len(after) and delete_count == len(before) and len(before) > 0:
                return EastVariant("replace", {"before": before, "after": after})

            return EastVariant("patch", operations)

        # Build dict equality using equal_for with current context
        dict_equal = equal_for(type_val, ctx.equal)

        ctx.diff.append(diff_dict)
        ctx.types.append(type_val)
        ctx.equal.append(dict_equal)
        is_same = is_for(type_val, ctx.equal)
        value_diff = diff_for(type_val.value["value"], ctx)
        value_equal = equal_for(type_val.value["value"], ctx.equal)
        ctx.diff.pop()
        ctx.types.pop()
        ctx.equal.pop()

        return diff_dict

    if is_struct_type(type_val):
        field_diffs: dict[str, Callable[[Any, Any], Any]] = {}
        field_equals: dict[str, Callable[[Any, Any], bool]] = {}

        def diff_struct(before: Any, after: Any) -> EastVariant:
            if before is after:
                return EastVariant("unchanged", None)

            patch_fields: dict[str, Any] = {}
            all_unchanged = True

            for field_spec in type_val.value:
                name = field_spec["name"]
                before_value = before[name]
                after_value = after[name]

                if field_equals[name](before_value, after_value):
                    patch_fields[name] = EastVariant("unchanged", None)
                else:
                    patch_fields[name] = field_diffs[name](before_value, after_value)
                    all_unchanged = False

            if all_unchanged:
                return EastVariant("unchanged", None)

            return EastVariant("patch", patch_fields)

        # Build struct equality using equal_for with current context
        struct_equal = equal_for(type_val, ctx.equal)

        ctx.diff.append(diff_struct)
        ctx.types.append(type_val)
        ctx.equal.append(struct_equal)
        for field_spec in type_val.value:
            name = field_spec["name"]
            field_type = field_spec["type"]
            field_diffs[name] = diff_for(field_type, ctx)
            field_equals[name] = equal_for(field_type, ctx.equal)
        ctx.diff.pop()
        ctx.types.pop()
        ctx.equal.pop()

        return diff_struct

    if is_variant_type(type_val):
        case_diffs: dict[str, Callable[[Any, Any], Any]] = {}
        case_equals: dict[str, Callable[[Any, Any], bool]] = {}

        def diff_variant(before: Any, after: Any) -> EastVariant:
            if before is after:
                return EastVariant("unchanged", None)

            if before.type != after.type:
                return EastVariant("replace", {"before": before, "after": after})

            case_name = before.type
            if case_equals[case_name](before.value, after.value):
                return EastVariant("unchanged", None)

            case_patch = case_diffs[case_name](before.value, after.value)

            if case_patch.type == "unchanged":
                return EastVariant("unchanged", None)

            return EastVariant("patch", EastVariant(case_name, case_patch))

        # Build variant equality using equal_for with current context
        variant_equal = equal_for(type_val, ctx.equal)

        ctx.diff.append(diff_variant)
        ctx.types.append(type_val)
        ctx.equal.append(variant_equal)
        for case_spec in type_val.value:
            name = case_spec["name"]
            case_type = case_spec["type"]
            case_diffs[name] = diff_for(case_type, ctx)
            case_equals[name] = equal_for(case_type, ctx.equal)
        ctx.diff.pop()
        ctx.types.pop()
        ctx.equal.pop()

        return diff_variant

    if is_ref_type(type_val):
        inner_diff: Callable[[Any, Any], Any]
        inner_equal: Callable[[Any, Any], bool]
        is_same: Callable[[Any, Any], bool]

        def diff_ref(before: Any, after: Any) -> EastVariant:
            if is_same(before, after):
                return EastVariant("unchanged", None)

            if inner_equal(before.value, after.value):
                return EastVariant("unchanged", None)

            inner_patch = inner_diff(before.value, after.value)

            if inner_patch.type == "unchanged":
                return EastVariant("unchanged", None)

            return EastVariant("patch", inner_patch)

        # Build ref equality using equal_for with current context
        ref_equal = equal_for(type_val, ctx.equal)

        ctx.diff.append(diff_ref)
        ctx.types.append(type_val)
        ctx.equal.append(ref_equal)
        is_same = is_for(type_val, ctx.equal)
        inner_diff = diff_for(type_val.value, ctx)
        inner_equal = equal_for(type_val.value, ctx.equal)
        ctx.diff.pop()
        ctx.types.pop()
        ctx.equal.pop()

        return diff_ref

    if is_recursive_type(type_val):
        # Recursive types use replace-only semantics
        scope_id = type_val.value
        if isinstance(scope_id, int):
            resolved_idx = len(ctx.types) - scope_id
            if resolved_idx < 0 or resolved_idx >= len(ctx.equal):
                raise RuntimeError("Internal error: Recursive type context not found in diff_for")
            equal = ctx.equal[resolved_idx]
        else:
            raise RuntimeError(f"Unexpected recursive type marker: {scope_id}")

        def diff_recursive(before: Any, after: Any) -> EastVariant:
            if equal(before, after):
                return EastVariant("unchanged", None)
            return EastVariant("replace", {"before": before, "after": after})

        return diff_recursive

    if is_function_type(type_val) or is_async_function_type(type_val):

        def diff_function(_before: Any, _after: Any) -> EastVariant:
            return EastVariant("unchanged", None)

        return diff_function

    raise RuntimeError(f"Unhandled type in diff_for: {type_val.type}")


__all__ = ["diff_for"]
