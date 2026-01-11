#
# Copyright (c) 2025 Elara AI Pty Ltd
# Licensed under the Business Source License 1.1. See LICENSE.md for details.
#
"""apply_for - Apply a patch to an East value.

This module provides the applyFor function that creates type-specific
apply functions for applying patches to East values.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from east.patch.types import ApplyContext, ConflictError
from east.serialization.east_printer import print_east
from east.types.types import (
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
from east.types.values import EastDict, EastRef, EastSet, EastVariant
from east.utils.ordering import equal_for

if TYPE_CHECKING:
    from east.types.types import EastType


def _print_for(type_val: EastType) -> Callable[[Any], str]:
    """Create a print function for error messages."""
    return lambda value: print_east(value, type_val)


def apply_for(
    type_val: EastType,
    ctx: ApplyContext | None = None,
) -> Callable[[Any, Any], Any]:
    """Create an apply function for a given type.

    Args:
        type_val: The East type
        ctx: Context for recursive type handling (internal)

    Returns:
        A function (base, patch) -> patched_value

    Raises:
        ConflictError: If the patch conflicts with the base value
    """
    if ctx is None:
        ctx = ApplyContext()

    if is_never_type(type_val):

        def apply_never(_base: Any, _patch: Any) -> Any:
            raise RuntimeError("Cannot apply patch to values of type Never")

        return apply_never

    # Primitives: unchanged or replace only
    if (
        is_null_type(type_val)
        or is_boolean_type(type_val)
        or is_integer_type(type_val)
        or is_float_type(type_val)
        or is_string_type(type_val)
        or is_datetime_type(type_val)
        or is_blob_type(type_val)
    ):
        equal = equal_for(type_val)
        print_val = _print_for(type_val)

        def apply_primitive(base: Any, patch: EastVariant) -> Any:
            if patch.type == "unchanged":
                return base
            if patch.type == "replace":
                if not equal(base, patch.value["before"]):
                    raise ConflictError(
                        f"Cannot apply replace - expected {print_val(patch.value['before'])}, "
                        f"found {print_val(base)}"
                    )
                return patch.value["after"]
            raise RuntimeError(f"Invalid patch type for primitive: {patch.type}")

        return apply_primitive

    if is_array_type(type_val):
        element_apply: Callable[[Any, Any], Any]
        element_equal: Callable[[Any, Any], bool]
        element_print: Callable[[Any], str]
        array_equal: Callable[[Any, Any], bool]

        def apply_array(base: Any, patch: EastVariant) -> Any:
            if patch.type == "unchanged":
                return base
            if patch.type == "replace":
                if not array_equal(base, patch.value["before"]):
                    raise ConflictError(
                        "Cannot apply replace - base array does not match expected"
                    )
                return list(patch.value["after"])
            if patch.type == "patch":
                result = list(base)
                operations = patch.value

                for op in operations:
                    key = int(op["key"])
                    offset = int(op["offset"])
                    old_key = key + offset

                    if op["operation"].type == "delete":
                        if old_key < 0 or old_key >= len(result):
                            raise ConflictError(
                                f"Cannot delete at index {old_key} - array length is {len(result)}"
                            )
                        if not element_equal(result[old_key], op["operation"].value):
                            raise ConflictError(
                                f"Cannot delete at index {old_key} - "
                                f"expected {element_print(op['operation'].value)}, "
                                f"found {element_print(result[old_key])}"
                            )
                        result.pop(old_key)
                    elif op["operation"].type == "insert":
                        if key < 0 or key > len(result):
                            raise ConflictError(
                                f"Cannot insert at index {key} - array length is {len(result)}"
                            )
                        result.insert(key, op["operation"].value)
                    elif op["operation"].type == "update":
                        if old_key < 0 or old_key >= len(result):
                            raise ConflictError(
                                f"Cannot update at index {old_key} - array length is {len(result)}"
                            )
                        result[old_key] = element_apply(result[old_key], op["operation"].value)

                return result
            raise RuntimeError(f"Invalid patch type for array: {patch.type}")

        # Build print handler for this array type
        array_print = _print_for(type_val)

        # Build array equality using equal_for with current context
        array_equal = equal_for(type_val, ctx.equal)

        ctx.apply.append(apply_array)
        ctx.types.append(type_val)
        ctx.equal.append(array_equal)
        ctx.print.append(array_print)
        element_apply = apply_for(type_val.value, ctx)
        element_equal = equal_for(type_val.value, ctx.equal)
        element_print = _print_for(type_val.value)
        ctx.apply.pop()
        ctx.types.pop()
        ctx.equal.pop()
        ctx.print.pop()

        return apply_array

    if is_set_type(type_val):
        key_print = _print_for(type_val.value)
        set_equal = equal_for(type_val, ctx.equal)

        def apply_set(base: Any, patch: EastVariant) -> Any:
            if patch.type == "unchanged":
                return base
            if patch.type == "replace":
                if not set_equal(base, patch.value["before"]):
                    raise ConflictError(
                        "Cannot apply replace - base set does not match expected"
                    )
                return EastSet(type_val.value, patch.value["after"])
            if patch.type == "patch":
                result: EastSet[Any] = EastSet(type_val.value, base)
                operations = patch.value

                for key, op in operations.items():
                    if op.type == "delete":
                        if key not in result:
                            raise ConflictError(
                                f"Cannot delete key {key_print(key)} - key does not exist"
                            )
                        result.discard(key)
                    elif op.type == "insert":
                        if key in result:
                            raise ConflictError(
                                f"Cannot insert key {key_print(key)} - key already exists"
                            )
                        result.add(key)

                return result
            raise RuntimeError(f"Invalid patch type for set: {patch.type}")

        return apply_set

    if is_dict_type(type_val):
        value_apply: Callable[[Any, Any], Any]
        value_equal: Callable[[Any, Any], bool]
        dict_equal: Callable[[Any, Any], bool]
        key_print: Callable[[Any], str]
        value_print: Callable[[Any], str]

        def apply_dict(base: Any, patch: EastVariant) -> Any:
            if patch.type == "unchanged":
                return base
            if patch.type == "replace":
                if not dict_equal(base, patch.value["before"]):
                    raise ConflictError(
                        "Cannot apply replace - base dict does not match expected"
                    )
                return EastDict(
                    type_val.value["key"],
                    type_val.value["value"],
                    patch.value["after"],
                )
            if patch.type == "patch":
                result: EastDict[Any, Any] = EastDict(
                    type_val.value["key"],
                    type_val.value["value"],
                    base,
                )
                operations = patch.value

                for key, op in operations.items():
                    if op.type == "delete":
                        if key not in result:
                            raise ConflictError(
                                f"Cannot delete key {key_print(key)} - key does not exist"
                            )
                        if not value_equal(result[key], op.value):
                            raise ConflictError(
                                f"Cannot delete key {key_print(key)} - "
                                f"expected value {value_print(op.value)}, "
                                f"found {value_print(result[key])}"
                            )
                        del result[key]
                    elif op.type == "insert":
                        if key in result:
                            raise ConflictError(
                                f"Cannot insert key {key_print(key)} - "
                                f"key already exists with value {value_print(result[key])}"
                            )
                        result[key] = op.value
                    elif op.type == "update":
                        if key not in result:
                            raise ConflictError(
                                f"Cannot update key {key_print(key)} - key does not exist"
                            )
                        result[key] = value_apply(result[key], op.value)

                return result
            raise RuntimeError(f"Invalid patch type for dict: {patch.type}")

        # Build print handler for this dict type
        dict_print = _print_for(type_val)

        # Build dict equality using equal_for with current context
        dict_equal = equal_for(type_val, ctx.equal)

        ctx.apply.append(apply_dict)
        ctx.types.append(type_val)
        ctx.equal.append(dict_equal)
        ctx.print.append(dict_print)
        value_apply = apply_for(type_val.value["value"], ctx)
        value_equal = equal_for(type_val.value["value"], ctx.equal)
        key_print = _print_for(type_val.value["key"])
        value_print = _print_for(type_val.value["value"])
        ctx.apply.pop()
        ctx.types.pop()
        ctx.equal.pop()
        ctx.print.pop()

        return apply_dict

    if is_struct_type(type_val):
        field_applies: dict[str, Callable[[Any, Any], Any]] = {}
        equal: Callable[[Any, Any], bool]

        def apply_struct(base: Any, patch: EastVariant) -> Any:
            if patch.type == "unchanged":
                return base
            if patch.type == "replace":
                if not equal(base, patch.value["before"]):
                    raise ConflictError(
                        "Cannot apply replace - base struct does not match expected"
                    )
                return dict(patch.value["after"])
            if patch.type == "patch":
                result: dict[str, Any] = {}

                for field_spec in type_val.value:
                    name = field_spec["name"]
                    field_patch = patch.value[name]
                    result[name] = field_applies[name](base[name], field_patch)

                return result
            raise RuntimeError(f"Invalid patch type for struct: {patch.type}")

        # Build print handler for this struct type
        struct_print = _print_for(type_val)

        # Build struct equality using equal_for with current context
        equal = equal_for(type_val, ctx.equal)

        ctx.apply.append(apply_struct)
        ctx.types.append(type_val)
        ctx.equal.append(equal)
        ctx.print.append(struct_print)
        for field_spec in type_val.value:
            name = field_spec["name"]
            field_type = field_spec["type"]
            field_applies[name] = apply_for(field_type, ctx)
        ctx.apply.pop()
        ctx.types.pop()
        ctx.equal.pop()
        ctx.print.pop()

        return apply_struct

    if is_variant_type(type_val):
        case_applies: dict[str, Callable[[Any, Any], Any]] = {}
        equal: Callable[[Any, Any], bool]

        def apply_variant(base: Any, patch: EastVariant) -> Any:
            if patch.type == "unchanged":
                return base
            if patch.type == "replace":
                if not equal(base, patch.value["before"]):
                    raise ConflictError(
                        "Cannot apply replace - base variant does not match expected"
                    )
                return patch.value["after"]
            if patch.type == "patch":
                case_name = patch.value.type
                if base.type != case_name:
                    raise ConflictError(
                        f"Cannot apply patch for case {case_name} to variant with case {base.type}"
                    )
                case_patch = patch.value.value
                new_value = case_applies[case_name](base.value, case_patch)
                return EastVariant(case_name, new_value)
            raise RuntimeError(f"Invalid patch type for variant: {patch.type}")

        # Build print handler for this variant type
        variant_print = _print_for(type_val)

        # Build variant equality using equal_for with current context
        equal = equal_for(type_val, ctx.equal)

        ctx.apply.append(apply_variant)
        ctx.types.append(type_val)
        ctx.equal.append(equal)
        ctx.print.append(variant_print)
        for case_spec in type_val.value:
            name = case_spec["name"]
            case_type = case_spec["type"]
            case_applies[name] = apply_for(case_type, ctx)
        ctx.apply.pop()
        ctx.types.pop()
        ctx.equal.pop()
        ctx.print.pop()

        return apply_variant

    if is_ref_type(type_val):
        inner_apply: Callable[[Any, Any], Any]
        equal: Callable[[Any, Any], bool]

        def apply_ref(base: Any, patch: EastVariant) -> Any:
            if patch.type == "unchanged":
                return base
            if patch.type == "replace":
                if not equal(base, patch.value["before"]):
                    raise ConflictError(
                        "Cannot apply replace - base ref does not match expected"
                    )
                return EastRef(patch.value["after"].value)
            if patch.type == "patch":
                new_value = inner_apply(base.value, patch.value)
                return EastRef(new_value)
            raise RuntimeError(f"Invalid patch type for ref: {patch.type}")

        # Build print handler for this ref type
        ref_print = _print_for(type_val)

        # Build ref equality using equal_for with current context
        equal = equal_for(type_val, ctx.equal)

        ctx.apply.append(apply_ref)
        ctx.types.append(type_val)
        ctx.equal.append(equal)
        ctx.print.append(ref_print)
        inner_apply = apply_for(type_val.value, ctx)
        ctx.apply.pop()
        ctx.types.pop()
        ctx.equal.pop()
        ctx.print.pop()

        return apply_ref

    if is_recursive_type(type_val):
        # Recursive types use replace-only semantics
        def apply_recursive(base: Any, patch: EastVariant) -> Any:
            if patch.type == "unchanged":
                return base
            if patch.type == "replace":
                return patch.value["after"]
            raise RuntimeError(f"Invalid patch type for recursive type: {patch.type}")

        return apply_recursive

    if is_function_type(type_val) or is_async_function_type(type_val):

        def apply_function(base: Any, patch: EastVariant) -> Any:
            if patch.type == "unchanged":
                return base
            if patch.type == "replace":
                return patch.value["after"]
            raise RuntimeError(f"Invalid patch type for function: {patch.type}")

        return apply_function

    raise RuntimeError(f"Unhandled type in apply_for: {type_val.type}")


__all__ = ["apply_for"]
