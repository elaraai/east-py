"""Printer for East text format (not JSON or BEAST).

The printer converts East values to text format.

This module handles the East text format specifically. Other printers:
- JSON format: east/serialization/json_printer.py (TODO)
- BEAST binary format: east/serialization/beast_printer.py (TODO)
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import TYPE_CHECKING, Any

from east.types.primitives import Blob, Null

if TYPE_CHECKING:
    from east.types.type_system import EastType


def print_east(value: Any, value_type: EastType) -> str:
    """Print East value to text format.

    Args:
        value: The value to print
        value_type: The type of the value

    Returns:
        East text representation
    """
    tag = value_type.tag

    if tag == "Null":
        return print_null(value)
    if tag == "Boolean":
        return print_boolean(value)
    if tag == "Integer":
        return print_integer(value)
    if tag == "Float":
        return print_float(value)
    if tag == "String":
        return print_string(value)
    if tag == "Blob":
        return print_blob(value)
    if tag == "DateTime":
        return print_datetime(value)
    if tag == "Array":
        return print_array(value, value_type)
    if tag == "Set":
        return print_set(value, value_type)
    if tag == "Dict":
        return print_dict(value, value_type)
    if tag == "Struct":
        return print_struct(value, value_type)
    if tag == "Variant":
        return print_variant(value, value_type)

    raise ValueError(f"Cannot print type {tag}")


def print_null(_value: Any) -> str:
    """Print null value.

    Args:
        _value: null value (unused)

    Returns:
        "null"
    """
    return "null"


def print_boolean(value: bool) -> str:
    """Print boolean value.

    Args:
        value: Boolean value

    Returns:
        "true" or "false"
    """
    return "true" if value else "false"


def print_integer(value: int) -> str:
    """Print integer value.

    Args:
        value: Integer value

    Returns:
        Integer as string
    """
    return str(value)


def print_float(value: float) -> str:
    """Print float value.

    Args:
        value: Float value

    Returns:
        Float as string (including NaN, Infinity)
    """
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "Infinity" if value > 0 else "-Infinity"
    return str(value)


def print_string(value: str) -> str:
    """Print string value.

    Args:
        value: String value

    Returns:
        Quoted and escaped string
    """
    # Escape special characters
    escaped = value.replace("\\", "\\\\")
    escaped = escaped.replace('"', '\\"')
    escaped = escaped.replace("\n", "\\n")
    escaped = escaped.replace("\t", "\\t")
    escaped = escaped.replace("\r", "\\r")

    return f'"{escaped}"'


def print_blob(value: Blob) -> str:
    """Print blob value.

    Args:
        value: Blob value

    Returns:
        Hex string (0x...)
    """
    hex_str = value._data.hex()
    return f"0x{hex_str}"


def print_datetime(value: datetime) -> str:
    """Print datetime value.

    Args:
        value: DateTime value

    Returns:
        ISO 8601 format
    """
    return value.isoformat()


def print_array(value: Any, array_type: EastType) -> str:
    """Print array value.

    Args:
        value: EastArray instance
        array_type: Array type

    Returns:
        Array as text
    """
    element_type = array_type.value

    if len(value) == 0:
        return "[]"

    items = [print_east(item, element_type) for item in value]
    return "[" + ", ".join(items) + "]"


def print_set(value: Any, set_type: EastType) -> str:
    """Print set value.

    Args:
        value: EastSet instance
        set_type: Set type

    Returns:
        Set as text
    """
    element_type = set_type.value

    if len(value) == 0:
        return "{}"

    items = [print_east(item, element_type) for item in value]
    return "{" + ", ".join(items) + "}"


def print_dict(value: Any, dict_type: EastType) -> str:
    """Print dict value.

    Args:
        value: EastDict instance
        dict_type: Dict type

    Returns:
        Dict as text
    """
    dict_struct = dict_type.value
    key_type = dict_struct.key
    value_type = dict_struct.value

    if len(value) == 0:
        return "{:}"

    items = []
    for k, v in value.items():
        key_str = print_east(k, key_type)
        val_str = print_east(v, value_type)
        items.append(f"{key_str}: {val_str}")

    return "{" + ", ".join(items) + "}"


def print_struct(value: Any, struct_type: EastType) -> str:
    """Print struct value.

    Args:
        value: EastStruct instance
        struct_type: Struct type

    Returns:
        Struct as text
    """
    field_specs = struct_type.value

    if len(field_specs) == 0:
        return "()"

    fields = []
    for field in field_specs:
        field_name = field.name
        field_type = field.type
        field_value = getattr(value, field_name)

        # Check if field name needs escaping
        field_name_str = f"`{field_name}`" if needs_escaping(field_name) else field_name

        field_value_str = print_east(field_value, field_type)
        fields.append(f"{field_name_str}={field_value_str}")

    return "(" + ", ".join(fields) + ")"


def print_variant(value: Any, variant_type: EastType) -> str:
    """Print variant value.

    Args:
        value: EastVariant instance
        variant_type: Variant type

    Returns:
        Variant as text
    """
    tag = value.tag
    val = value.value

    # Find the type for this case
    case_specs = variant_type.value
    case_type = None
    for case in case_specs:
        if case.name == tag:
            case_type = case.type
            break

    if case_type is None:
        raise ValueError(f"Unknown variant case: {tag}")

    # Print tag
    result = f".{tag}"

    # Print value (if not null)
    if not isinstance(val, Null):
        val_str = print_east(val, case_type)
        result += f" {val_str}"

    return result


def needs_escaping(identifier: str) -> bool:
    """Check if identifier needs backtick escaping.

    Args:
        identifier: Identifier to check

    Returns:
        True if needs escaping
    """
    if not identifier:
        return True

    # Check first character
    if not (identifier[0].isalpha() or identifier[0] == "_"):
        return True

    # Check remaining characters
    return any(not (char.isalnum() or char == "_") for char in identifier[1:])


__all__: list[str] = ["print_east"]
