#
# Copyright (c) 2025 Elara AI Pty Ltd
# Licensed under the Business Source License 1.1. See LICENSE.md for details.
#
"""compose_for - Combine two sequential patches.

This module provides the composeFor function that creates type-specific
compose functions for combining patches.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from east.patch.apply import apply_for
from east.patch.invert import invert_for
from east.patch.types import ApplyContext, ComposeContext, ConflictError, InvertContext
from east.serialization.east_printer import print_east
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
    is_never_type,
    is_null_type,
    is_recursive_type,
    is_ref_type,
    is_set_type,
    is_string_type,
    is_struct_type,
    is_variant_type,
)
from east.types.values import EastDict, EastVariant
from east.utils.ordering import equal_for

if TYPE_CHECKING:
    from east.types.types import EastType


def _print_for(type_val: EastType) -> Callable[[Any], str]:
    """Create a print function for error messages."""
    return lambda value: print_east(value, type_val)


def compose_for(
    type_val: EastType,
    ctx: ComposeContext | None = None,
) -> Callable[[Any, Any], Any]:
    """Create a compose function for a given type.

    Args:
        type_val: The East type
        ctx: Context for recursive type handling (internal)

    Returns:
        A function (first, second) -> combined_patch
    """
    if ctx is None:
        ctx = ComposeContext()

    if is_never_type(type_val):

        def compose_never(_first: Any, _second: Any) -> Any:
            raise RuntimeError("Cannot compose patches for type Never")

        return compose_never

    # Primitives: straightforward composition
    if (
        is_null_type(type_val)
        or is_boolean_type(type_val)
        or is_integer_type(type_val)
        or is_float_type(type_val)
        or is_string_type(type_val)
        or is_datetime_type(type_val)
        or is_blob_type(type_val)
    ):

        def compose_primitive(first: EastVariant, second: EastVariant) -> EastVariant:
            if first.type == "unchanged":
                return second
            if second.type == "unchanged":
                return first
            if first.type == "replace" and second.type == "replace":
                return EastVariant(
                    "replace",
                    {"before": first.value["before"], "after": second.value["after"]},
                )
            raise RuntimeError("Invalid patch composition for primitive")

        return compose_primitive

    if is_array_type(type_val):
        apply_ret: Callable[[Any, Any], Any]
        invert_ret: Callable[[Any], Any]

        def compose_array(first: EastVariant, second: EastVariant) -> EastVariant:
            if first.type == "unchanged":
                return second
            if second.type == "unchanged":
                return first
            if first.type == "replace" and second.type == "replace":
                return EastVariant(
                    "replace",
                    {"before": first.value["before"], "after": second.value["after"]},
                )
            if first.type == "replace" and second.type == "patch":
                after_second = apply_ret(first.value["after"], second)
                return EastVariant(
                    "replace",
                    {"before": first.value["before"], "after": after_second},
                )
            if first.type == "patch" and second.type == "replace":
                inverted_first = invert_ret(first)
                original_before = apply_ret(second.value["before"], inverted_first)
                return EastVariant(
                    "replace",
                    {"before": original_before, "after": second.value["after"]},
                )
            # patch + patch: concatenate operations
            p1_ops = first.value
            p2_ops = second.value
            result = list(p1_ops) + list(p2_ops)
            if len(result) == 0:
                return EastVariant("unchanged", None)
            return EastVariant("patch", result)

        array_equal = equal_for(type_val, ctx.equal)
        array_print = _print_for(type_val)

        # Push compose context first
        ctx.compose.append(compose_array)
        ctx.types.append(type_val)
        ctx.equal.append(array_equal)
        ctx.print.append(array_print)

        # Build apply/invert handlers
        apply_ctx = ApplyContext(
            apply=ctx.apply, types=ctx.types, equal=ctx.equal, print=ctx.print
        )
        invert_ctx = InvertContext(invert=ctx.invert, types=ctx.types, equal=ctx.equal)
        apply_ret = apply_for(type_val, apply_ctx)
        invert_ret = invert_for(type_val, invert_ctx)

        # Push them so .Recursive lookups work during element compose recursion
        ctx.apply.append(apply_ret)
        ctx.invert.append(invert_ret)

        # Recurse into element type for compose
        compose_for(type_val.value, ctx)

        ctx.compose.pop()
        ctx.apply.pop()
        ctx.invert.pop()
        ctx.types.pop()
        ctx.equal.pop()
        ctx.print.pop()

        return compose_array

    if is_set_type(type_val):
        key_print = _print_for(type_val.value)

        # Pass full context so recursive type references can be resolved
        equal_for(type_val, ctx.equal)
        apply_ctx = ApplyContext(
            apply=ctx.apply, types=ctx.types, equal=ctx.equal, print=ctx.print
        )
        invert_ctx = InvertContext(invert=ctx.invert, types=ctx.types, equal=ctx.equal)
        apply_fn = apply_for(type_val, apply_ctx)
        invert_fn = invert_for(type_val, invert_ctx)

        def compose_set(first: EastVariant, second: EastVariant) -> EastVariant:
            if first.type == "unchanged":
                return second
            if second.type == "unchanged":
                return first
            if first.type == "replace" and second.type == "replace":
                return EastVariant(
                    "replace",
                    {"before": first.value["before"], "after": second.value["after"]},
                )
            if first.type == "patch" and second.type == "patch":
                result: EastDict[Any, Any] = EastDict(type_val.value, NullType)

                for key, op in first.value.items():
                    result[key] = op

                for key, op in second.value.items():
                    if key in result:
                        first_op = result[key]
                        if first_op.type == "insert" and op.type == "delete" or first_op.type == "delete" and op.type == "insert":
                            del result[key]
                        else:
                            raise ConflictError(
                                f"Cannot compose patches - conflicting operations on key "
                                f"{key_print(key)}"
                            )
                    else:
                        result[key] = op

                if len(result) == 0:
                    return EastVariant("unchanged", None)

                return EastVariant("patch", result)
            if first.type == "replace":
                after_second = apply_fn(first.value["after"], second)
                return EastVariant(
                    "replace",
                    {"before": first.value["before"], "after": after_second},
                )
            inverted_first = invert_fn(first)
            original_before = apply_fn(second.value["before"], inverted_first)
            return EastVariant(
                "replace",
                {"before": original_before, "after": second.value["after"]},
            )

        return compose_set

    if is_dict_type(type_val):
        value_compose: Callable[[Any, Any], Any]
        value_apply: Callable[[Any, Any], Any]
        apply_ret: Callable[[Any, Any], Any]
        invert_ret: Callable[[Any], Any]
        key_print = _print_for(type_val.value["key"])

        def compose_dict(first: EastVariant, second: EastVariant) -> EastVariant:
            if first.type == "unchanged":
                return second
            if second.type == "unchanged":
                return first
            if first.type == "replace" and second.type == "replace":
                return EastVariant(
                    "replace",
                    {"before": first.value["before"], "after": second.value["after"]},
                )
            if first.type == "patch" and second.type == "patch":
                result: EastDict[Any, Any] = EastDict(type_val.value["key"], NullType)

                for key, op in first.value.items():
                    result[key] = op

                for key, op in second.value.items():
                    if key in result:
                        first_op = result[key]

                        if first_op.type == "insert" and op.type == "delete":
                            del result[key]
                        elif first_op.type == "insert" and op.type == "update":
                            new_value = value_apply(first_op.value, op.value)
                            result[key] = EastVariant("insert", new_value)
                        elif first_op.type == "delete" and op.type == "insert":
                            result[key] = EastVariant(
                                "update",
                                EastVariant(
                                    "replace",
                                    {"before": first_op.value, "after": op.value},
                                ),
                            )
                        elif first_op.type == "update" and op.type == "delete":
                            raise ConflictError(
                                f"Cannot compose patches - update then delete on key "
                                f"{key_print(key)}"
                            )
                        elif first_op.type == "update" and op.type == "update":
                            composed = value_compose(first_op.value, op.value)
                            result[key] = EastVariant("update", composed)
                        else:
                            raise ConflictError(
                                f"Cannot compose patches - conflicting operations on key "
                                f"{key_print(key)}"
                            )
                    else:
                        result[key] = op

                if len(result) == 0:
                    return EastVariant("unchanged", None)

                return EastVariant("patch", result)
            if first.type == "replace":
                after_second = apply_ret(first.value["after"], second)
                return EastVariant(
                    "replace",
                    {"before": first.value["before"], "after": after_second},
                )
            inverted_first = invert_ret(first)
            original_before = apply_ret(second.value["before"], inverted_first)
            return EastVariant(
                "replace",
                {"before": original_before, "after": second.value["after"]},
            )

        # Build print handler for this dict type
        dict_print = _print_for(type_val)

        # Build dict equality using equal_for with current context
        dict_equal = equal_for(type_val, ctx.equal)

        # Push compose context first
        ctx.compose.append(compose_dict)
        ctx.types.append(type_val)
        ctx.equal.append(dict_equal)
        ctx.print.append(dict_print)

        # Build Dict apply/invert handlers
        apply_ctx = ApplyContext(
            apply=ctx.apply, types=ctx.types, equal=ctx.equal, print=ctx.print
        )
        invert_ctx = InvertContext(invert=ctx.invert, types=ctx.types, equal=ctx.equal)
        apply_ret = apply_for(type_val, apply_ctx)
        invert_ret = invert_for(type_val, invert_ctx)

        # Push them so .Recursive lookups work during value compose recursion
        ctx.apply.append(apply_ret)
        ctx.invert.append(invert_ret)

        # Recurse into value type for compose
        value_compose = compose_for(type_val.value["value"], ctx)
        # Build value apply with proper context
        value_apply_ctx = ApplyContext(
            apply=ctx.apply, types=ctx.types, equal=ctx.equal, print=ctx.print
        )
        value_apply = apply_for(type_val.value["value"], value_apply_ctx)

        ctx.compose.pop()
        ctx.apply.pop()
        ctx.invert.pop()
        ctx.types.pop()
        ctx.equal.pop()
        ctx.print.pop()

        return compose_dict

    if is_struct_type(type_val):
        field_composes: dict[str, Callable[[Any, Any], Any]] = {}
        apply_ret: Callable[[Any, Any], Any]
        invert_ret: Callable[[Any], Any]

        def compose_struct(first: EastVariant, second: EastVariant) -> EastVariant:
            if first.type == "unchanged":
                return second
            if second.type == "unchanged":
                return first
            if first.type == "replace" and second.type == "replace":
                return EastVariant(
                    "replace",
                    {"before": first.value["before"], "after": second.value["after"]},
                )
            if first.type == "patch" and second.type == "patch":
                result: dict[str, Any] = {}
                all_unchanged = True

                for field_spec in type_val.value:
                    name = field_spec["name"]
                    composed = field_composes[name](first.value[name], second.value[name])
                    result[name] = composed
                    if composed.type != "unchanged":
                        all_unchanged = False

                if all_unchanged:
                    return EastVariant("unchanged", None)

                return EastVariant("patch", result)
            if first.type == "replace":
                after_second = apply_ret(first.value["after"], second)
                return EastVariant(
                    "replace",
                    {"before": first.value["before"], "after": after_second},
                )
            inverted_first = invert_ret(first)
            original_before = apply_ret(second.value["before"], inverted_first)
            return EastVariant(
                "replace",
                {"before": original_before, "after": second.value["after"]},
            )

        struct_equal = equal_for(type_val, ctx.equal)
        struct_print = _print_for(type_val)

        # Push compose context first
        ctx.compose.append(compose_struct)
        ctx.types.append(type_val)
        ctx.equal.append(struct_equal)
        ctx.print.append(struct_print)

        # Build apply/invert handlers using applyFor/invertFor
        apply_ctx = ApplyContext(
            apply=ctx.apply, types=ctx.types, equal=ctx.equal, print=ctx.print
        )
        invert_ctx = InvertContext(invert=ctx.invert, types=ctx.types, equal=ctx.equal)
        apply_ret = apply_for(type_val, apply_ctx)
        invert_ret = invert_for(type_val, invert_ctx)

        # Push them so .Recursive lookups work during field compose recursion
        ctx.apply.append(apply_ret)
        ctx.invert.append(invert_ret)

        # Recurse into field types for compose
        for field_spec in type_val.value:
            name = field_spec["name"]
            field_type = field_spec["type"]
            field_composes[name] = compose_for(field_type, ctx)

        ctx.compose.pop()
        ctx.apply.pop()
        ctx.invert.pop()
        ctx.types.pop()
        ctx.equal.pop()
        ctx.print.pop()

        return compose_struct

    if is_variant_type(type_val):
        case_composes: dict[str, Callable[[Any, Any], Any]] = {}
        apply_ret: Callable[[Any, Any], Any]
        invert_ret: Callable[[Any], Any]

        def compose_variant(first: EastVariant, second: EastVariant) -> EastVariant:
            if first.type == "unchanged":
                return second
            if second.type == "unchanged":
                return first
            if first.type == "replace" and second.type == "replace":
                return EastVariant(
                    "replace",
                    {"before": first.value["before"], "after": second.value["after"]},
                )
            if first.type == "patch" and second.type == "patch":
                if first.value.type != second.value.type:
                    raise ConflictError(
                        f"Cannot compose variant patches for different cases: "
                        f"{first.value.type} and {second.value.type}"
                    )
                case_name = first.value.type
                composed = case_composes[case_name](first.value.value, second.value.value)

                if composed.type == "unchanged":
                    return EastVariant("unchanged", None)

                return EastVariant("patch", EastVariant(case_name, composed))
            if first.type == "replace":
                after_second = apply_ret(first.value["after"], second)
                return EastVariant(
                    "replace",
                    {"before": first.value["before"], "after": after_second},
                )
            # first is "patch", second is "replace"
            inverted_first = invert_ret(first)
            original_before = apply_ret(second.value["before"], inverted_first)
            return EastVariant(
                "replace",
                {"before": original_before, "after": second.value["after"]},
            )

        variant_equal = equal_for(type_val, ctx.equal)
        variant_print = _print_for(type_val)

        # Push compose context first
        ctx.compose.append(compose_variant)
        ctx.types.append(type_val)
        ctx.equal.append(variant_equal)
        ctx.print.append(variant_print)

        # Build apply/invert handlers using applyFor/invertFor
        apply_ctx = ApplyContext(
            apply=ctx.apply, types=ctx.types, equal=ctx.equal, print=ctx.print
        )
        invert_ctx = InvertContext(invert=ctx.invert, types=ctx.types, equal=ctx.equal)
        apply_ret = apply_for(type_val, apply_ctx)
        invert_ret = invert_for(type_val, invert_ctx)

        # Push them so .Recursive lookups work during case compose recursion
        ctx.apply.append(apply_ret)
        ctx.invert.append(invert_ret)

        # Recurse into case types for compose
        for case_spec in type_val.value:
            name = case_spec["name"]
            case_type = case_spec["type"]
            case_composes[name] = compose_for(case_type, ctx)

        ctx.compose.pop()
        ctx.apply.pop()
        ctx.invert.pop()
        ctx.types.pop()
        ctx.equal.pop()
        ctx.print.pop()

        return compose_variant

    if is_ref_type(type_val):
        inner_compose: Callable[[Any, Any], Any]
        apply_ret: Callable[[Any, Any], Any]
        invert_ret: Callable[[Any], Any]

        def compose_ref(first: EastVariant, second: EastVariant) -> EastVariant:
            if first.type == "unchanged":
                return second
            if second.type == "unchanged":
                return first
            if first.type == "replace" and second.type == "replace":
                return EastVariant(
                    "replace",
                    {"before": first.value["before"], "after": second.value["after"]},
                )
            if first.type == "patch" and second.type == "patch":
                composed = inner_compose(first.value, second.value)
                if composed.type == "unchanged":
                    return EastVariant("unchanged", None)
                return EastVariant("patch", composed)
            if first.type == "replace":
                after_second = apply_ret(first.value["after"], second)
                return EastVariant(
                    "replace",
                    {"before": first.value["before"], "after": after_second},
                )
            inverted_first = invert_ret(first)
            original_before = apply_ret(second.value["before"], inverted_first)
            return EastVariant(
                "replace",
                {"before": original_before, "after": second.value["after"]},
            )

        # Build print handler for this ref type
        ref_print = _print_for(type_val)

        # Build ref equality using equal_for with current context
        ref_equal = equal_for(type_val, ctx.equal)

        # Push compose context first
        ctx.compose.append(compose_ref)
        ctx.types.append(type_val)
        ctx.equal.append(ref_equal)
        ctx.print.append(ref_print)

        # Build Ref apply/invert handlers
        apply_ctx = ApplyContext(
            apply=ctx.apply, types=ctx.types, equal=ctx.equal, print=ctx.print
        )
        invert_ctx = InvertContext(invert=ctx.invert, types=ctx.types, equal=ctx.equal)
        apply_ret = apply_for(type_val, apply_ctx)
        invert_ret = invert_for(type_val, invert_ctx)

        # Push them so .Recursive lookups work during inner compose recursion
        ctx.apply.append(apply_ret)
        ctx.invert.append(invert_ret)

        # Recurse into inner type for compose
        inner_compose = compose_for(type_val.value, ctx)

        ctx.compose.pop()
        ctx.apply.pop()
        ctx.invert.pop()
        ctx.types.pop()
        ctx.equal.pop()
        ctx.print.pop()

        return compose_ref

    if is_recursive_type(type_val):
        # Recursive types use replace-only semantics
        def compose_recursive(first: EastVariant, second: EastVariant) -> EastVariant:
            if first.type == "unchanged":
                return second
            if second.type == "unchanged":
                return first
            if first.type == "replace" and second.type == "replace":
                return EastVariant(
                    "replace",
                    {"before": first.value["before"], "after": second.value["after"]},
                )
            raise RuntimeError(
                f"Invalid patch types for recursive type composition: "
                f"{first.type}, {second.type}"
            )

        return compose_recursive

    if is_function_type(type_val) or is_async_function_type(type_val):

        def compose_function(first: EastVariant, second: EastVariant) -> EastVariant:
            if first.type == "unchanged":
                return second
            if second.type == "unchanged":
                return first
            return EastVariant(
                "replace",
                {"before": first.value["before"], "after": second.value["after"]},
            )

        return compose_function

    raise RuntimeError(f"Unhandled type in compose_for: {type_val.type}")


__all__ = ["compose_for"]
