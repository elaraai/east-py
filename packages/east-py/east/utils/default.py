"""Default value generation for East types.

Provides functions to create default and minimal values for any East type.
"""

from __future__ import annotations

from datetime import UTC
from datetime import datetime as DateTime

from east.types.types import (
    EastType,
    is_array_type,
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
from east.types.values import EastBlob, EastValue, east_null


def default_value(type_val: EastType) -> EastValue:
    """Create a default value for a given East type.

    Args:
        type_val: The East type to create a default value for

    Returns:
        A default value of the given type

    Examples:
        - NullType → None
        - BooleanType → False
        - IntegerType → 0
        - FloatType → 0.0
        - StringType → ""
        - ArrayType → []
        - StructType → struct with default field values

    Raises:
        RuntimeError: For Never, Recursive, or Function types
    """
    from east.types.values import EastArray, EastDict, EastSet, EastStruct, EastVariant

    if is_never_type(type_val):
        raise RuntimeError("Cannot create a default value of type .Never")

    if is_null_type(type_val):
        return east_null

    if is_boolean_type(type_val):
        return False

    if is_integer_type(type_val):
        return 0

    if is_float_type(type_val):
        return 0.0

    if is_string_type(type_val):
        return ""

    if is_datetime_type(type_val):
        return DateTime.fromtimestamp(0, tz=UTC)

    if is_blob_type(type_val):
        return EastBlob(b"")

    if is_array_type(type_val):
        return EastArray(type_val.value, [])

    if is_set_type(type_val):
        return EastSet(type_val.value, [])

    if is_dict_type(type_val):
        dict_struct = type_val.value
        return EastDict(dict_struct.key, dict_struct.value, {})

    if is_ref_type(type_val):
        from east.types.values import east_ref

        return east_ref(default_value(type_val.value))

    if is_struct_type(type_val):
        field_values = {}
        for field in type_val.value:
            field_values[field.name] = default_value(field.type)
        return EastStruct(field_values)

    if is_variant_type(type_val):
        cases = type_val.value
        if len(cases) == 0:
            raise RuntimeError("Cannot create a value of an empty variant")

        # Return first case (cases are sorted alphabetically)
        first_case = cases[0]
        return EastVariant(first_case.name, default_value(first_case.type))

    if is_recursive_type(type_val):
        raise RuntimeError("Cannot create a default value of type .Recursive")

    if is_function_type(type_val):
        raise RuntimeError("Cannot create a default value of type .Function")

    raise RuntimeError(f"Unknown type encountered: {type_val.type}")


def minimal_value(type_val: EastType) -> EastValue:
    """Create the minimal possible value for a given East type.

    Currently identical to default_value. This function exists for potential
    future differentiation (e.g., Float could return -Infinity).

    Args:
        type_val: The East type to create a minimal value for

    Returns:
        A minimal value of the given type

    Examples:
        - NullType → None
        - BooleanType → False
        - IntegerType → 0
        - FloatType → 0.0
        - StringType → ""

    Raises:
        RuntimeError: For Never, Recursive, or Function types
    """
    return default_value(type_val)


__all__ = ["default_value", "minimal_value"]
