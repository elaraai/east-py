"""Default value generation for East types.

Provides functions to create default and minimal values for any East type.
"""

from __future__ import annotations

from datetime import UTC
from datetime import datetime as DateTime
from typing import Any

from east.types.primitives import Blob


def default_value(type_val: Any) -> Any:
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
    from east.types.containers import EastArray, EastDict, EastSet

    tag = type_val["type"]

    if tag == "Never":
        raise RuntimeError("Cannot create a default value of type .Never")

    if tag == "Null":
        return None

    if tag == "Boolean":
        return False

    if tag == "Integer":
        return 0

    if tag == "Float":
        return 0.0

    if tag == "String":
        return ""

    if tag == "DateTime":
        return DateTime.fromtimestamp(0, tz=UTC)

    if tag == "Blob":
        return Blob(b"")

    if tag == "Array":
        element_type = type_val["value"]  # type: ignore[attr-defined]
        return EastArray(element_type, [])

    if tag == "Set":
        element_type = type_val["value"]  # type: ignore[attr-defined]
        return EastSet(element_type, [])

    if tag == "Dict":
        dict_struct = type_val["value"]  # type: ignore[attr-defined]
        key_type = dict_struct["key"]  # type: ignore[attr-defined]
        value_type = dict_struct["value"]  # type: ignore[attr-defined]
        return EastDict(key_type, value_type, {})

    if tag == "Ref":
        from east.types.ref import ref

        inner_default = default_value(type_val["value"])  # type: ignore[attr-defined]
        return ref(inner_default)

    if tag == "Struct":
        fields = type_val["value"]  # type: ignore[attr-defined]
        result = {}
        for field in fields:
            field_name = field["name"]  # type: ignore[attr-defined]
            field_type = field["type"]  # type: ignore[attr-defined]
            result[field_name] = default_value(field_type)
        return result

    if tag == "Variant":
        cases = type_val["value"]  # type: ignore[attr-defined]
        if len(cases) == 0:
            raise RuntimeError("Cannot create a value of an empty variant")

        # Return first case (cases are sorted alphabetically)
        first_case = cases[0]
        case_name = first_case["name"]  # type: ignore[attr-defined]
        case_type = first_case["type"]  # type: ignore[attr-defined]
        return {"type": case_name, "value": default_value(case_type)}

    if tag == "Recursive":
        raise RuntimeError("Cannot create a default value of type .Recursive")

    if tag == "Function":
        raise RuntimeError("Cannot create a default value of type .Function")

    raise RuntimeError(f"Unknown type encountered: {tag}")


def minimal_value(type_val: Any) -> Any:
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
