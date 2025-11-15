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
    from east.types.types import EastType


# =============================================================================
# Aliasing support for circular references
# =============================================================================


def _common_prefix_length(a: list[str], b: list[str]) -> int:
    """Find the length of the common prefix between two path arrays.

    Args:
        a: First path array
        b: Second path array

    Returns:
        Length of common prefix
    """
    i = 0
    while i < len(a) and i < len(b) and a[i] == b[i]:
        i += 1
    return i


def _encode_relative_ref(current_path: list[str], target_path: list[str]) -> str:
    """Compute a relative reference string from currentPath to targetPath.

    Returns a string like "2#.foo[0]" or "1#"

    The format is: "upLevels#remaining_path_components"

    Args:
        current_path: Current location in the value tree
        target_path: Target location we're referencing

    Returns:
        Relative reference string
    """
    common_len = _common_prefix_length(current_path, target_path)
    up_levels = len(current_path) - common_len
    remaining = target_path[common_len:]

    if not remaining:
        return f"{up_levels}#"

    remaining_str = "".join(remaining)
    return f"{up_levels}#{remaining_str}"


def _decode_relative_ref(ref_str: str, current_path: list[str]) -> list[str]:
    """Decode a relative reference string and return the target path array.

    Input like "2#.foo[0]" returns the target path array.
    Input like "1#" returns the target path array.

    Args:
        ref_str: Relative reference string (e.g., "2#.foo[0]")
        current_path: Current location in the value tree

    Returns:
        Target path array

    Raises:
        ValueError: If reference is invalid
    """
    hash_idx = ref_str.find("#")
    if hash_idx == -1:
        raise ValueError(f"Invalid relative reference: {ref_str}")

    up_level_str = ref_str[:hash_idx]
    remaining_str = ref_str[hash_idx + 1 :]

    try:
        up_levels = int(up_level_str)
    except ValueError as e:
        raise ValueError(f"Invalid relative reference: {ref_str}") from e

    if up_levels < 0 or up_levels > len(current_path):
        raise ValueError(
            f"Invalid relative reference: going up {up_levels} levels "
            f"from depth {len(current_path)}"
        )

    # Build target path
    target_path = current_path[: len(current_path) - up_levels]

    # Add remaining components if any
    if remaining_str:
        # Parse the remaining punctuated path
        # Format: .field[0][key] etc.
        pos = 0
        while pos < len(remaining_str):
            if remaining_str[pos] == ".":
                # Identifier follows
                pos += 1
                end = pos
                while end < len(remaining_str) and (
                    remaining_str[end].isalnum() or remaining_str[end] == "_"
                ):
                    end += 1
                target_path.append(f".{remaining_str[pos:end]}")
                pos = end
            elif remaining_str[pos] == "[":
                # Bracket expression
                end = pos + 1
                depth = 1
                while end < len(remaining_str) and depth > 0:
                    if remaining_str[end] == "[":
                        depth += 1
                    elif remaining_str[end] == "]":
                        depth -= 1
                    end += 1
                target_path.append(remaining_str[pos:end])
                pos = end
            else:
                pos += 1

    return target_path


# =============================================================================
# Main printing functions
# =============================================================================


def _find_recursive_marker(typ: EastType) -> Any | None:
    """Find the RecursiveTypeMarker that this type owns (if any).

    For a Struct/Variant type created with recursive_type(), this returns the marker
    by checking if any Recursive refs in the type point back to this type as their node.

    Args:
        typ: The type to search

    Returns:
        The RecursiveTypeMarker if found, None otherwise
    """
    from east.types.types import RecursiveTypeMarker

    # Helper to find all markers in a type
    def find_all_markers(t: EastType, markers: set[Any]) -> None:
        if not hasattr(t, "tag"):
            return

        tag = t["type"]

        if tag == "Recursive":
            marker = t.value
            if isinstance(marker, RecursiveTypeMarker):
                markers.add(marker)
            return

        if tag in ("Array", "Set"):
            find_all_markers(t.value, markers)
            return

        if tag == "Dict":
            find_all_markers(t["value"]["key"], markers)
            find_all_markers(t["value"]["value"], markers)
            return

        if tag == "Struct":
            for field in t.value:
                find_all_markers(field["type"], markers)
            return

        if tag == "Variant":
            for case in t.value:
                find_all_markers(case["type"], markers)
            return

    # Find all markers referenced in this type
    markers: set[Any] = set()
    find_all_markers(typ, markers)

    # Check if any marker's node points to this type (object identity)
    for marker in markers:
        if hasattr(marker, "node") and marker.node is typ:
            return marker

    return None


def print_east(value: Any, value_type: EastType) -> str:
    """Print East value to text format.

    Args:
        value: The value to print
        value_type: The type of the value

    Returns:
        East text representation
    """
    # Initialize alias tracking and recursive type context for top-level call
    seen_values: dict[int, list[str]] = {}
    current_path: list[str] = []
    type_ctx: list[EastType] = []
    marker_map: dict[Any, int] = {}

    # Find the recursive marker for this type (if any) and register it
    marker = _find_recursive_marker(value_type)
    if marker is not None:
        type_ctx.append(value_type)
        marker_map[id(marker)] = 0

    return _print_east_internal(value, value_type, seen_values, current_path, type_ctx, marker_map)


def _print_east_internal(
    value: Any,
    value_type: EastType,
    seen_values: dict[int, list[str]],
    current_path: list[str],
    type_ctx: list[EastType],
    marker_map: dict[Any, int],
) -> str:
    """Internal print function with alias tracking and recursive type context.

    Args:
        value: The value to print
        value_type: The type of the value
        seen_values: Map from id(value) to path for alias detection
        current_path: Current location in value tree
        type_ctx: Stack of types for recursive type resolution
        marker_map: Map from marker id() to type_ctx index

    Returns:
        East text representation
    """
    tag = value_type["type"]

    # Check for aliases on mutable collections
    if tag in ("Array", "Set", "Dict", "Ref", "Struct"):
        value_id = id(value)
        if value_id in seen_values:
            # Emit reference to previously seen value
            target_path = seen_values[value_id]
            return _encode_relative_ref(current_path, target_path)
        # Mark this value as seen
        seen_values[value_id] = list(current_path)

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
        # Push array type onto context stack
        type_ctx.append(value_type)
        marker = _find_recursive_marker(value_type)
        if marker is not None and id(marker) not in marker_map:
            marker_map[id(marker)] = len(type_ctx) - 1
        try:
            return print_array_internal(
                value, value_type, seen_values, current_path, type_ctx, marker_map
            )
        finally:
            type_ctx.pop()
    if tag == "Set":
        # Push set type onto context stack
        type_ctx.append(value_type)
        marker = _find_recursive_marker(value_type)
        if marker is not None and id(marker) not in marker_map:
            marker_map[id(marker)] = len(type_ctx) - 1
        try:
            return print_set_internal(
                value, value_type, seen_values, current_path, type_ctx, marker_map
            )
        finally:
            type_ctx.pop()
    if tag == "Dict":
        # Push dict type onto context stack
        type_ctx.append(value_type)
        marker = _find_recursive_marker(value_type)
        if marker is not None and id(marker) not in marker_map:
            marker_map[id(marker)] = len(type_ctx) - 1
        try:
            return print_dict_internal(
                value, value_type, seen_values, current_path, type_ctx, marker_map
            )
        finally:
            type_ctx.pop()
    if tag == "Ref":
        # Push ref type onto context stack
        type_ctx.append(value_type)
        marker = _find_recursive_marker(value_type)
        if marker is not None and id(marker) not in marker_map:
            marker_map[id(marker)] = len(type_ctx) - 1
        try:
            return print_ref_internal(
                value, value_type, seen_values, current_path, type_ctx, marker_map
            )
        finally:
            type_ctx.pop()
    if tag == "Struct":
        # Push struct type onto context stack for recursive type resolution
        type_ctx.append(value_type)
        marker = _find_recursive_marker(value_type)
        if marker is not None and id(marker) not in marker_map:
            marker_map[id(marker)] = len(type_ctx) - 1
        try:
            return print_struct_internal(
                value, value_type, seen_values, current_path, type_ctx, marker_map
            )
        finally:
            type_ctx.pop()
    if tag == "Variant":
        # Push variant type onto context stack for recursive type resolution
        type_ctx.append(value_type)
        marker = _find_recursive_marker(value_type)
        if marker is not None and id(marker) not in marker_map:
            marker_map[id(marker)] = len(type_ctx) - 1
        try:
            return print_variant_internal(
                value, value_type, seen_values, current_path, type_ctx, marker_map
            )
        finally:
            type_ctx.pop()
    if tag == "Function":
        return print_function(value)
    if tag == "Recursive":
        # Resolve recursive reference to actual type
        from east.types.types import RecursiveTypeMarker

        marker = value_type["value"]
        if isinstance(marker, RecursiveTypeMarker):
            marker_id = id(marker)
            if marker_id not in marker_map:
                raise ValueError(f"Unresolved recursive type marker: marker_id={marker_id}")
            ctx_index = marker_map[marker_id]
            resolved_type = type_ctx[ctx_index]
        elif isinstance(marker, int):
            # Integer scope_id from TypeScript exports
            ctx_index = len(type_ctx) - marker
            if ctx_index < 0 or ctx_index >= len(type_ctx):
                raise ValueError(
                    f"Invalid recursive scope_id {marker} (ctx len={len(type_ctx)}, calculated index={ctx_index})"
                )
            resolved_type = type_ctx[ctx_index]
        else:
            raise ValueError(f"Expected RecursiveTypeMarker or int, got {type(marker)}")
        return _print_east_internal(
            value, resolved_type, seen_values, current_path, type_ctx, marker_map
        )

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

    # Use Python's default str() which gives shortest accurate representation
    # This matches JavaScript's String() behavior better than using .17g
    result = str(value)

    # Ensure we have a decimal point for float distinction
    if "." not in result and "e" not in result and "E" not in result:
        result += ".0"

    return result


def print_string(value: str) -> str:
    """Print string value.

    Args:
        value: String value

    Returns:
        Quoted and escaped string
    """
    # Escape special characters
    # Note: East text format only supports \\ and \" escapes
    # Newlines, tabs, etc. must be included literally in the string
    escaped = value.replace("\\", "\\\\")
    escaped = escaped.replace('"', '\\"')

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
        ISO 8601 format with milliseconds, no timezone (matches JavaScript toISOString().substring(0,23))
    """
    # Format: "YYYY-MM-DDTHH:MM:SS.mmm" (23 characters, like JavaScript's toISOString().substring(0, 23))
    # Python's isoformat() includes timezone, so we format manually
    return value.strftime("%Y-%m-%dT%H:%M:%S.%f")[
        :-3
    ]  # Remove last 3 digits of microseconds to get milliseconds


def print_array(value: Any, array_type: EastType) -> str:
    """Print array value (no alias tracking).

    Args:
        value: EastArray instance
        array_type: Array type

    Returns:
        Array as text
    """
    element_type = array_type["value"]

    if len(value) == 0:
        return "[]"

    items = [print_east(item, element_type) for item in value]
    return "[" + ", ".join(items) + "]"


def print_array_internal(
    value: Any,
    array_type: EastType,
    seen_values: dict[int, list[str]],
    current_path: list[str],
    type_ctx: list[EastType],
    marker_map: dict[Any, int],
) -> str:
    """Print array value with alias tracking.

    Args:
        value: EastArray instance
        array_type: Array type
        seen_values: Alias tracking dict
        current_path: Current path
        type_ctx: Stack of types for recursive type resolution
        marker_map: Map from marker id() to type_ctx index

    Returns:
        Array as text
    """
    element_type = array_type["value"]

    # Register marker for element type if it's recursive
    marker = _find_recursive_marker(element_type)
    if marker is not None and id(marker) not in marker_map:
        type_ctx.append(element_type)
        marker_map[id(marker)] = len(type_ctx) - 1

    if len(value) == 0:
        return "[]"

    items = []
    for i, item in enumerate(value):
        # Update path for this element
        item_path = current_path + [f"[{i}]"]
        items.append(
            _print_east_internal(item, element_type, seen_values, item_path, type_ctx, marker_map)
        )

    return "[" + ", ".join(items) + "]"


def print_set(value: Any, set_type: EastType) -> str:
    """Print set value (no alias tracking).

    Args:
        value: EastSet instance
        set_type: Set type

    Returns:
        Set as text
    """
    element_type = set_type["value"]

    if len(value) == 0:
        return "{}"

    items = [print_east(item, element_type) for item in value]
    return "{" + ",".join(items) + "}"


def print_set_internal(
    value: Any,
    set_type: EastType,
    seen_values: dict[int, list[str]],
    current_path: list[str],
    type_ctx: list[EastType],
    marker_map: dict[Any, int],
) -> str:
    """Print set value with alias tracking.

    Args:
        value: EastSet instance
        set_type: Set type
        seen_values: Alias tracking dict
        current_path: Current path
        type_ctx: Stack of types for recursive type resolution
        marker_map: Map from marker id() to type_ctx index

    Returns:
        Set as text
    """
    element_type = set_type["value"]

    if len(value) == 0:
        return "{}"

    items = []
    for i, item in enumerate(value):
        # Sets don't have meaningful paths, just use index
        item_path = current_path + [f"[{i}]"]
        items.append(
            _print_east_internal(item, element_type, seen_values, item_path, type_ctx, marker_map)
        )

    return "{" + ",".join(items) + "}"


def print_dict(value: Any, dict_type: EastType) -> str:
    """Print dict value (no alias tracking).

    Args:
        value: EastDict instance
        dict_type: Dict type

    Returns:
        Dict as text
    """
    dict_struct = dict_type["value"]
    key_type = dict_struct["key"]
    value_type = dict_struct["value"]

    if len(value) == 0:
        return "{:}"

    items = []
    for k, v in value.items():
        key_str = print_east(k, key_type)
        val_str = print_east(v, value_type)
        items.append(f"{key_str}:{val_str}")

    return "{" + ",".join(items) + "}"


def print_dict_internal(
    value: Any,
    dict_type: EastType,
    seen_values: dict[int, list[str]],
    current_path: list[str],
    type_ctx: list[EastType],
    marker_map: dict[Any, int],
) -> str:
    """Print dict value with alias tracking.

    Args:
        value: EastDict instance
        dict_type: Dict type
        seen_values: Alias tracking dict
        current_path: Current path
        type_ctx: Stack of types for recursive type resolution
        marker_map: Map from marker id() to type_ctx index

    Returns:
        Dict as text
    """
    dict_struct = dict_type["value"]
    key_type = dict_struct["key"]
    value_type = dict_struct["value"]

    if len(value) == 0:
        return "{:}"

    items = []
    for k, v in value.items():
        # Keys use print_east directly (no alias tracking for primitives)
        key_str = print_east(k, key_type)
        # Values track path with key
        val_path = current_path + [f"[{key_str}]"]
        val_str = _print_east_internal(v, value_type, seen_values, val_path, type_ctx, marker_map)
        items.append(f"{key_str}:{val_str}")

    return "{" + ",".join(items) + "}"


def print_ref(value: Any, ref_type: EastType) -> str:
    """Print ref value (no alias tracking).

    Args:
        value: Ref instance
        ref_type: Ref type

    Returns:
        Ref as text
    """
    from east.types.ref import deref

    inner_type = ref_type["value"]
    inner_value = deref(value)
    inner_str = print_east(inner_value, inner_type)
    return f"&{inner_str}"


def print_ref_internal(
    value: Any,
    ref_type: EastType,
    seen_values: dict[int, list[str]],
    current_path: list[str],
    type_ctx: list[EastType],
    marker_map: dict[Any, int],
) -> str:
    """Print ref value with alias tracking.

    Args:
        value: Ref instance
        ref_type: Ref type
        seen_values: Alias tracking dict
        current_path: Current path
        type_ctx: Stack of types for recursive type resolution
        marker_map: Map from marker id() to type_ctx index

    Returns:
        Ref as text
    """
    from east.types.ref import deref

    inner_type = ref_type["value"]

    # Register marker for inner type if it's recursive
    marker = _find_recursive_marker(inner_type)
    if marker is not None and id(marker) not in marker_map:
        type_ctx.append(inner_type)
        marker_map[id(marker)] = len(type_ctx) - 1

    inner_value = deref(value)
    # Path for inner value
    inner_path = current_path + ["&"]
    inner_str = _print_east_internal(
        inner_value, inner_type, seen_values, inner_path, type_ctx, marker_map
    )
    return f"&{inner_str}"


def print_struct(value: Any, struct_type: EastType) -> str:
    """Print struct value (no alias tracking).

    Args:
        value: EastStruct instance
        struct_type: Struct type

    Returns:
        Struct as text
    """
    field_specs = struct_type["value"]

    if len(field_specs) == 0:
        return "()"

    fields = []
    for field in field_specs:
        field_name = field["name"]
        field_type = field["type"]
        # Handle both dict and EastStruct objects
        if isinstance(value, dict):
            field_value = value[field_name]
        else:
            field_value = getattr(value, field_name)

        # Check if field name needs escaping
        field_name_str = f"`{field_name}`" if needs_escaping(field_name) else field_name

        field_value_str = print_east(field_value, field_type)
        fields.append(f"{field_name_str}={field_value_str}")

    return "(" + ", ".join(fields) + ")"


def print_struct_internal(
    value: Any,
    struct_type: EastType,
    seen_values: dict[int, list[str]],
    current_path: list[str],
    type_ctx: list[EastType],
    marker_map: dict[Any, int],
) -> str:
    """Print struct value with alias tracking.

    Args:
        value: EastStruct instance
        struct_type: Struct type
        seen_values: Alias tracking dict
        current_path: Current path
        type_ctx: Stack of types for recursive type resolution
        marker_map: Map from marker id() to type_ctx index

    Returns:
        Struct as text
    """
    field_specs = struct_type["value"]

    if len(field_specs) == 0:
        return "()"

    fields = []
    for field in field_specs:
        field_name = field["name"]
        field_type = field["type"]

        # Register marker for field type if it's recursive
        marker = _find_recursive_marker(field_type)
        if marker is not None and id(marker) not in marker_map:
            type_ctx.append(field_type)
            marker_map[id(marker)] = len(type_ctx) - 1

        # Handle both dict and EastStruct objects
        if isinstance(value, dict):
            field_value = value[field_name]
        else:
            field_value = getattr(value, field_name)

        # Check if field name needs escaping
        field_name_str = f"`{field_name}`" if needs_escaping(field_name) else field_name

        # Track path for field
        field_path = current_path + [f".{field_name}"]
        field_value_str = _print_east_internal(
            field_value, field_type, seen_values, field_path, type_ctx, marker_map
        )
        fields.append(f"{field_name_str}={field_value_str}")

    return "(" + ", ".join(fields) + ")"


def print_variant(value: Any, variant_type: EastType) -> str:
    """Print variant value (no alias tracking).

    Args:
        value: EastVariant instance
        variant_type: Variant type

    Returns:
        Variant as text
    """
    tag = value["type"]
    val = value["value"]

    # Find the type for this case
    case_specs = variant_type["value"]
    case_type = None
    for case in case_specs:
        if case["name"] == tag:
            case_type = case["type"]
            # DEBUG
            print(
                f"DEBUG: Found case {tag}, case_type type={type(case_type)}, case_type={case_type}"
            )
            case_type_kind = case_type["type"]
            print(f"DEBUG: case_type type={case_type_kind}")
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


def print_variant_internal(
    value: Any,
    variant_type: EastType,
    seen_values: dict[int, list[str]],
    current_path: list[str],
    type_ctx: list[EastType],
    marker_map: dict[Any, int],
) -> str:
    """Print variant value with alias tracking.

    Args:
        value: EastVariant instance
        variant_type: Variant type
        seen_values: Alias tracking dict
        current_path: Current path
        type_ctx: Stack of types for recursive type resolution
        marker_map: Map from marker id() to type_ctx index

    Returns:
        Variant as text
    """
    print(f"DEBUG print_variant_internal: value type={type(value)}, value={value}")
    variant_type_kind = variant_type["type"]
    variant_type_value = variant_type["value"]
    print(
        f"DEBUG print_variant_internal: variant_type type={variant_type_kind}, variant_type value={variant_type_value}"
    )
    tag = value["type"]
    val = value["value"]

    # Find the type for this case
    case_specs = variant_type["value"]
    case_type = None
    for case in case_specs:
        if case["name"] == tag:
            case_type = case["type"]
            # DEBUG
            print(
                f"DEBUG: Found case {tag}, case_type type={type(case_type)}, case_type={case_type}"
            )
            case_type_kind = case_type["type"]
            print(f"DEBUG: case_type type={case_type_kind}")
            break

    if case_type is None:
        raise ValueError(f"Unknown variant case: {tag}")

    # Register marker for case type if it's recursive
    marker = _find_recursive_marker(case_type)
    if marker is not None and id(marker) not in marker_map:
        type_ctx.append(case_type)
        marker_map[id(marker)] = len(type_ctx) - 1

    # Print tag
    result = f".{tag}"

    # Print value (if not null)
    if not isinstance(val, Null):
        # Variants don't add to path since they're transparent
        val_str = _print_east_internal(
            val, case_type, seen_values, current_path, type_ctx, marker_map
        )
        result += f" {val_str}"

    return result


def print_function(_value: Any) -> str:
    """Print function value.

    Args:
        _value: Function value (unused)

    Returns:
        "λ" (lambda symbol)
    """
    return "λ"


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


def print_identifier(identifier: str) -> str:
    """Print an identifier, escaping with backticks if necessary.

    Args:
        identifier: The identifier to print

    Returns:
        The identifier as-is if valid, or escaped with backticks if invalid
    """
    if not needs_escaping(identifier):
        return identifier
    # Escape backslashes and backticks inside the identifier
    escaped = identifier.replace("\\", "\\\\").replace("`", "\\`")
    return f"`{escaped}`"


def print_for(type_val: EastType):
    """Create a printer function for values of a given type.

    Args:
        type_val: The East type to create a printer for

    Returns:
        A function that prints values of the given type
    """
    return lambda value: print_east(value, type_val)


def print_type(type_val: EastType, stack: list[EastType] | None = None) -> str:
    """Print an East type.

    This is a bootstrap function that prints the same output as printing an `EastType`
    as an `EastTypeType`, but is available before value printing is fully defined.

    Args:
        type_val: The East type (must be EastType variant)
        stack: Stack for tracking recursive types (internal use)

    Returns:
        String representation of the type
    """
    import json

    # _StructTypeClass and _VariantTypeClass removed

    if stack is None:
        stack = []

    # Reject raw _StructTypeClass/_VariantTypeClass - these are internal helpers, not valid types
    if False:  # _StructTypeClass and _VariantTypeClass removed
        raise TypeError(
            f"Cannot print raw {type(type_val).__name__} - use StructType() or VariantType() instead"
        )

    type_kind = type_val["type"]

    if type_kind == "Never":
        return ".Never"
    if type_kind == "Null":
        return ".Null"
    if type_kind == "Boolean":
        return ".Boolean"
    if type_kind == "Integer":
        return ".Integer"
    if type_kind == "Float":
        return ".Float"
    if type_kind == "String":
        return ".String"
    if type_kind == "DateTime":
        return ".DateTime"
    if type_kind == "Blob":
        return ".Blob"

    if type_kind == "Array":
        stack.append(type_val)
        elem_type = type_val["value"]
        ret = f".Array {print_type(elem_type, stack)}"  # type: ignore[arg-type]
        stack.pop()
        return ret

    if type_kind == "Set":
        stack.append(type_val)
        elem_type = type_val["value"]
        ret = f".Set {print_type(elem_type, stack)}"  # type: ignore[arg-type]
        stack.pop()
        return ret

    if type_kind == "Dict":
        stack.append(type_val)
        dict_type_struct = type_val["value"]
        key_str = print_type(dict_type_struct["key"], stack)
        value_str = print_type(dict_type_struct["value"], stack)
        ret = f".Dict (key={key_str}, value={value_str})"
        stack.pop()
        return ret

    if type_kind == "Ref":
        stack.append(type_val)
        elem_type = type_val["value"]
        ret = f".Ref {print_type(elem_type, stack)}"  # type: ignore[arg-type]
        stack.pop()
        return ret

    if type_kind == "Struct":
        stack.append(type_val)
        fields = []
        # EastType with Struct tag: value contains field structs
        for field_struct in type_val["value"]:  # type: ignore[attr-defined]
            field_name = field_struct["name"]  # type: ignore[attr-defined]
            field_type = field_struct["type"]  # type: ignore[attr-defined]
            name_json = json.dumps(field_name)
            type_str = print_type(field_type, stack)
            fields.append(f"(name={name_json}, type={type_str})")
        ret = f".Struct [{', '.join(fields)}]"
        stack.pop()
        return ret

    if type_kind == "Variant":
        stack.append(type_val)
        cases = []
        # EastType with Variant tag: value contains case structs
        for case_struct in type_val["value"]:  # type: ignore[attr-defined]
            case_name = case_struct["name"]  # type: ignore[attr-defined]
            case_type = case_struct["type"]  # type: ignore[attr-defined]
            name_json = json.dumps(case_name)
            type_str = print_type(case_type, stack)
            cases.append(f"(name={name_json}, type={type_str})")
        ret = f".Variant [{', '.join(cases)}]"
        stack.pop()
        return ret

    if type_kind == "Recursive":
        # Find index in stack to determine recursion depth
        depth = type_val["value"]  # type: ignore[attr-defined]
        return f".Recursive {depth}"

    if type_kind == "Function":
        stack.append(type_val)
        func_struct = type_val["value"]  # type: ignore[attr-defined]
        inputs = func_struct["inputs"]  # type: ignore[attr-defined]
        output = func_struct["output"]  # type: ignore[attr-defined]
        platforms = func_struct["platforms"]  # type: ignore[attr-defined]

        input_strs = [print_type(inp, stack) for inp in inputs]
        output_str = print_type(output, stack)
        platform_strs = [json.dumps(p) for p in platforms]

        ret = f".Function (inputs=[{', '.join(input_strs)}], output={output_str}, platforms=[{', '.join(platform_strs)}])"
        stack.pop()
        return ret

    raise ValueError(f"Unknown type encountered during type printing: {type_kind}")


__all__: list[str] = ["print_east", "print_for", "print_type"]
