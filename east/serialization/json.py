"""JSON serialization for East types.

This module provides JSON encoding and decoding for East values, following
the TypeScript reference implementation at /home/crambelsoupy/src/East/src/serialization/json.ts

Key features:
- Type-driven encoding/decoding
- Special handling for Integer (as string), Float (special values), DateTime (RFC 3339), Blob (hex)
- Circular reference tracking via relative JSON Pointers
- Recursive type support
"""

import json
from datetime import UTC
from datetime import datetime as DateTime
from typing import Any

from east.serialization.east_printer import print_type
from east.types.containers import EastArray, EastDict, EastSet
from east.types.primitives import Blob, null
from east.types.type_system import EastType


class JSONDecodeError(Exception):
    """Error during JSON decoding."""

    def __init__(self, message: str, path: str = "", type_str: str = ""):
        """Initialize decode error.

        Args:
            message: Error message
            path: JSON path where error occurred
            type_str: String representation of the type being parsed
        """
        super().__init__(message)
        self.path = path
        self.message = message
        self.type_str = type_str

    def __str__(self) -> str:
        """Return error message in TypeScript-compatible format."""
        # Format: "Error occurred because <message> at <path> (line 1, col 1) while parsing value of type "<type>""
        result = f"Error occurred because {self.message}"
        if self.path:
            result += f" at {self.path}"
        result += " (line 1, col 1)"
        if self.type_str:
            result += f' while parsing value of type "{self.type_str}"'
        return result


def encode_json_pointer_component(component: str) -> str:
    """Encode a JSON Pointer component according to RFC 6901.

    Per RFC 6901, '~' is encoded as '~0' and '/' is encoded as '~1'.
    The order matters: we must escape '~' first to avoid double-escaping.

    Args:
        component: The path component to encode

    Returns:
        The encoded component with ~ and / escaped
    """
    return component.replace("~", "~0").replace("/", "~1")


def decode_json_pointer_component(component: str) -> str:
    """Decode a RFC 6901 JSON Pointer component.

    Args:
        component: Encoded component

    Returns:
        Decoded component
    """
    return component.replace("~1", "/").replace("~0", "~")


def common_prefix_length(a: list[str], b: list[str]) -> int:
    """Find the length of the common prefix between two path arrays.

    Args:
        a: First path
        b: Second path

    Returns:
        Length of common prefix
    """
    i = 0
    while i < len(a) and i < len(b) and a[i] == b[i]:
        i += 1
    return i


def encode_relative_ref(current_path: list[str], target_path: list[str]) -> str:
    """Compute a relative JSON Pointer reference from currentPath to targetPath.

    Returns a string like "2#foo/bar" (up 2 levels, then follow foo/bar)
    or "1#" (up 1 level, no remaining path).

    Args:
        current_path: Current location in JSON structure
        target_path: Target location to reference

    Returns:
        Relative JSON pointer reference string
    """
    common_len = common_prefix_length(current_path, target_path)
    up_levels = len(current_path) - common_len
    remaining = target_path[common_len:]

    if not remaining:
        return f"{up_levels}#"

    # Escape each component according to RFC 6901
    escaped_remaining = "/".join(encode_json_pointer_component(c) for c in remaining)
    return f"{up_levels}#{escaped_remaining}"


def decode_relative_ref(ref_str: str, current_path: list[str]) -> list[str]:
    """Decode a relative JSON Pointer reference and return the target path array.

    Input like "2#foo/bar" returns the target path array.
    Input like "1#" returns the target path array.

    Args:
        ref_str: Relative reference string
        current_path: Current location

    Returns:
        Target path array

    Raises:
        ValueError: If reference is invalid
    """
    hash_idx = ref_str.find("#")
    if hash_idx == -1:
        raise ValueError(f"Invalid relative JSON Pointer reference: {ref_str}")

    up_level_str = ref_str[:hash_idx]
    remaining_str = ref_str[hash_idx + 1 :]

    try:
        up_levels = int(up_level_str)
    except ValueError:
        raise ValueError(f"Invalid relative JSON Pointer reference: {ref_str}") from None

    if up_levels < 0 or up_levels > len(current_path):
        raise ValueError(
            f"Invalid relative JSON Pointer reference: going up {up_levels} levels from depth {len(current_path)}"
        )

    # Build target path
    target_path = current_path[: len(current_path) - up_levels]

    # Add remaining components if any
    if remaining_str:
        components = remaining_str.split("/")
        for component in components:
            # Decode RFC 6901 escaping
            unescaped = decode_json_pointer_component(component)
            target_path.append(unescaped)

    return target_path


# Type contexts for recursive types
JSONEncodeTypeContext = list[Any]  # Stack of encoders
JSONDecodeTypeContext = list[Any]  # Stack of decoders


class JSONEncodeValueContext:
    """Value-level context for tracking seen mutable containers during JSON encoding.

    Tracks mutable containers (Array, Set, Dict) and their path arrays for generating
    relative references.
    """

    def __init__(self):
        """Initialize encoding context."""
        self.refs: dict[int, list[str]] = {}  # id(container) -> path array
        self.current_path: list[str] = []  # Current position in JSON structure


class JSONDecodeValueContext:
    """Value-level context for tracking decoded mutable containers during JSON decoding.

    Maps path arrays to decoded mutable containers (Array, Set, Dict) for resolving references.
    Containers are added to the map BEFORE their contents are populated to handle circular references.
    """

    def __init__(self):
        """Initialize decoding context."""
        self.refs: dict[str, Any] = {}  # Stringified path -> decoded mutable container
        self.current_path: list[str] = []  # Current position during traversal


def _find_recursive_scope_ids(typ: Any, found: set[int]) -> None:
    """Find all recursive scope IDs in a type."""
    # Handle raw _StructTypeClass and _VariantTypeClass
    from east.types.type_system import _StructTypeClass as StructTypeClass
    from east.types.type_system import _VariantTypeClass as VariantTypeClass

    if isinstance(typ, StructTypeClass):
        for _name, field_type in typ.fields:
            _find_recursive_scope_ids(field_type, found)
        return
    if isinstance(typ, VariantTypeClass):
        for _name, case_type in typ.cases:
            _find_recursive_scope_ids(case_type, found)
        return

    tag = typ.tag
    if tag == "Recursive":
        found.add(typ.value)
    elif tag in ("Array", "Set"):
        _find_recursive_scope_ids(typ.value, found)
    elif tag == "Dict":
        _find_recursive_scope_ids(typ.value.key, found)
        _find_recursive_scope_ids(typ.value.value, found)
    elif tag == "Struct":
        for field in typ.value:
            _find_recursive_scope_ids(field.type, found)
    elif tag == "Variant":
        for case in typ.value:
            _find_recursive_scope_ids(case.type, found)


def _find_recursive_marker(typ: Any) -> Any | None:
    """Find the RecursiveTypeMarker that this type owns (if any).

    For a Struct/Variant type created with recursive_type(), this returns the marker
    by checking if any Recursive refs in the type point back to this type as their node.

    Args:
        typ: The type to check

    Returns:
        The RecursiveTypeMarker if this is a recursive type, None otherwise
    """
    from east.types.type_system import RecursiveTypeMarker

    # Helper to find all markers in a type
    def find_all_markers(t: Any, markers: set[Any]) -> None:
        if hasattr(t, "tag"):
            if t.tag == "Recursive":
                marker = t.value
                if isinstance(marker, RecursiveTypeMarker):
                    markers.add(marker)
            elif t.tag in ("Array", "Set"):
                find_all_markers(t.value, markers)
            elif t.tag == "Dict":
                find_all_markers(t.value.key, markers)
                find_all_markers(t.value.value, markers)
            elif t.tag == "Struct":
                for field in t.value:
                    find_all_markers(field.type, markers)
            elif t.tag == "Variant":
                for case in t.value:
                    find_all_markers(case.type, markers)

    # Find all markers referenced in this type
    markers: set[Any] = set()
    find_all_markers(typ, markers)

    # Check if any marker's node points to this type (object identity)
    for marker in markers:
        if marker.node is typ:
            return marker

    return None


def to_json_for(
    type_val: Any, type_ctx: list[Any] | None = None, marker_map: dict[Any, int] | None = None
) -> Any:
    """Create a JSON encoder function for a given East type.

    The returned function converts East values to JSON-serializable values.

    Args:
        type_val: The East type to create an encoder for
        type_ctx: Stack of encoders for recursive types (internal use)
        marker_map: Mapping from RecursiveTypeMarker to stack index (internal use)

    Returns:
        A function that converts East values to JSON-serializable values
    """
    if type_ctx is None:
        # Top-level call: create empty stack and marker map
        type_ctx = []
        marker_map = {}
        # Build the encoder with the context
        encoder = _build_json_encoder(type_val, type_ctx, marker_map)
        return encoder
    # Recursive call: use existing context
    return _build_json_encoder(type_val, type_ctx, marker_map)


def _build_json_encoder(type_val: Any, type_ctx: list[Any], marker_map: dict[Any, int]) -> Any:
    """Build the JSON encoder for a type.

    Args:
        type_val: The East type (must be EastType variant)
        type_ctx: Stack of encoders for recursive types
        marker_map: Mapping from RecursiveTypeMarker to stack index
    """
    from east.types.type_system import _StructTypeClass, _VariantTypeClass

    # Reject raw _StructTypeClass/_VariantTypeClass - these are internal helpers, not valid types
    if isinstance(type_val, (_StructTypeClass, _VariantTypeClass)):
        raise TypeError(
            f"Cannot encode raw {type(type_val).__name__} - use StructType() or VariantType() instead"
        )

    type_kind = type_val.tag

    # Simple types
    if type_kind == "Never":
        raise ValueError("Cannot encode Never type to JSON")
    if type_kind == "Null":
        return lambda _value, _ctx=None: None
    if type_kind == "Boolean":
        return lambda value, _ctx=None: value
    if type_kind == "Integer":
        return lambda value, _ctx=None: str(value)
    if type_kind == "Float":

        def encode_float(value: float, _ctx=None):
            if value == 0 and str(value).startswith("-"):
                return "-0.0"
            if value != value:  # NaN
                return "NaN"
            if value == float("inf"):
                return "Infinity"
            if value == float("-inf"):
                return "-Infinity"
            return value

        return encode_float
    if type_kind == "String":
        return lambda value, _ctx=None: value
    if type_kind == "DateTime":

        def encode_datetime(dt: DateTime, _ctx=None):
            year = dt.year
            month = str(dt.month).zfill(2)
            day = str(dt.day).zfill(2)
            hour = str(dt.hour).zfill(2)
            minute = str(dt.minute).zfill(2)
            second = str(dt.second).zfill(2)
            ms = str(dt.microsecond // 1000).zfill(3)
            return f"{year}-{month}-{day}T{hour}:{minute}:{second}.{ms}+00:00"

        return encode_datetime
    if type_kind == "Blob":

        def encode_blob(blob: Blob, _ctx=None):
            hex_str = "".join(f"{b:02x}" for b in blob.data)
            return f"0x{hex_str}"

        return encode_blob

    # Container types
    if type_kind == "Array":
        value_encoder = to_json_for(type_val.value, type_ctx, marker_map)  # type: ignore

        def encode_array(arr: EastArray, ctx: JSONEncodeValueContext | None = None):
            if ctx is None:
                ctx = JSONEncodeValueContext()
            arr_id = id(arr)
            if arr_id in ctx.refs:
                target_path = ctx.refs[arr_id]
                ref_str = encode_relative_ref(ctx.current_path, target_path)
                return {"$ref": ref_str}
            ctx.refs[arr_id] = list(ctx.current_path)
            result = []
            for i, item in enumerate(arr):
                ctx.current_path.append(str(i))
                result.append(value_encoder(item, ctx))
                ctx.current_path.pop()
            return result

        return encode_array

    if type_kind == "Set":
        key_encoder = to_json_for(type_val.value, type_ctx, marker_map)  # type: ignore

        def encode_set(s: EastSet, ctx: JSONEncodeValueContext | None = None):
            if ctx is None:
                ctx = JSONEncodeValueContext()
            if ctx:
                set_id = id(s)
                if set_id in ctx.refs:
                    target_path = ctx.refs[set_id]
                    ref_str = encode_relative_ref(ctx.current_path, target_path)
                    return {"$ref": ref_str}
                ctx.refs[set_id] = list(ctx.current_path)
            result = []
            for i, item in enumerate(s):
                if ctx:
                    ctx.current_path.append(str(i))
                result.append(key_encoder(item, ctx))
                if ctx:
                    ctx.current_path.pop()
            return result

        return encode_set

    if type_kind == "Dict":
        key_encoder = to_json_for(type_val.value.key, type_ctx, marker_map)  # type: ignore
        value_encoder = to_json_for(type_val.value.value, type_ctx, marker_map)  # type: ignore

        def encode_dict(d: EastDict, ctx: JSONEncodeValueContext | None = None):
            if ctx is None:
                ctx = JSONEncodeValueContext()
            dict_id = id(d)
            if dict_id in ctx.refs:
                target_path = ctx.refs[dict_id]
                ref_str = encode_relative_ref(ctx.current_path, target_path)
                return {"$ref": ref_str}
            ctx.refs[dict_id] = list(ctx.current_path)
            result = []
            for i, (k, v) in enumerate(d.items()):
                entry: dict[str, Any] = {}
                ctx.current_path.extend([str(i), "key"])
                entry["key"] = key_encoder(k, ctx)
                ctx.current_path.pop()
                ctx.current_path.pop()
                ctx.current_path.extend([str(i), "value"])
                entry["value"] = value_encoder(v, ctx)
                ctx.current_path.pop()
                ctx.current_path.pop()
                result.append(entry)
            return result

        return encode_dict

    if type_kind == "Ref":
        from east.types.ref import Ref

        inner_encoder = to_json_for(type_val.value, type_ctx, marker_map)  # type: ignore

        def encode_ref(r: Ref, ctx: JSONEncodeValueContext | None = None):
            if ctx is None:
                ctx = JSONEncodeValueContext()
            ref_id = id(r)
            if ref_id in ctx.refs:
                target_path = ctx.refs[ref_id]
                ref_str = encode_relative_ref(ctx.current_path, target_path)
                return {"$ref": ref_str}
            ctx.refs[ref_id] = list(ctx.current_path)
            # Encode inner value
            ctx.current_path.append("value")
            encoded_value = inner_encoder(r.value, ctx)
            ctx.current_path.pop()
            return {"value": encoded_value}

        return encode_ref

    # Structural types - need to pre-register for recursion
    if type_kind == "Struct":
        field_encoders: dict[str, Any] = {}

        def encode_struct(obj, ctx: JSONEncodeValueContext | None = None):
            if ctx is None:
                ctx = JSONEncodeValueContext()
            result: dict[str, Any] = {}
            for field_name, encoder in field_encoders.items():
                ctx.current_path.append(field_name)
                # Handle both dict and EastStruct objects
                field_value = obj[field_name] if isinstance(obj, dict) else getattr(obj, field_name)
                result[field_name] = encoder(field_value, ctx)
                ctx.current_path.pop()
            return result

        # Push this encoder onto the stack BEFORE building field encoders
        # This allows fields to reference this type recursively
        stack_index = len(type_ctx)
        type_ctx.append(encode_struct)

        # If this type owns a RecursiveTypeMarker (from recursive_type()), register it
        marker = _find_recursive_marker(type_val)
        if marker is not None:
            marker_map[id(marker)] = stack_index

        # Build field encoders
        # EastType with Struct tag: value contains field structs
        for field_struct in type_val.value:  # type: ignore
            field_encoders[field_struct.name] = to_json_for(field_struct.type, type_ctx, marker_map)

        # Pop from stack after building
        type_ctx.pop()

        return encode_struct

    if type_kind == "Variant":
        case_encoders: dict[str, Any] = {}

        def encode_variant(variant, ctx: JSONEncodeValueContext | None = None):
            if ctx is None:
                ctx = JSONEncodeValueContext()
            # Handle both dict and EastVariant objects
            variant_type = variant["type"] if isinstance(variant, dict) else variant.tag
            variant_value = variant["value"] if isinstance(variant, dict) else variant.value
            ctx.current_path.append(variant_type)
            encoded_value = case_encoders[variant_type](variant_value, ctx)
            ctx.current_path.pop()
            return {"type": variant_type, "value": encoded_value}

        # Push this encoder onto the stack BEFORE building case encoders
        # This allows cases to reference this type recursively
        stack_index = len(type_ctx)
        type_ctx.append(encode_variant)

        # If this type owns a RecursiveTypeMarker (from recursive_type()), register it
        marker = _find_recursive_marker(type_val)
        if marker is not None:
            marker_map[id(marker)] = stack_index

        # Build case encoders
        # EastType with Variant tag: value contains case structs
        for case_struct in type_val.value:  # type: ignore
            case_encoders[case_struct.name] = to_json_for(case_struct.type, type_ctx, marker_map)

        # Pop from stack after building
        type_ctx.pop()

        return encode_variant

    if type_kind == "Recursive":
        # Look up encoder by marker
        from east.types.type_system import RecursiveTypeMarker

        marker = type_val.value
        if isinstance(marker, RecursiveTypeMarker):
            # Look up marker in map to get stack index
            marker_id = id(marker)
            if marker_id not in marker_map:
                raise ValueError("Internal error: Recursive type marker not yet registered")
            stack_index = marker_map[marker_id]
            return type_ctx[stack_index]
        # Old-style integer scope_id (shouldn't happen with new code)
        raise ValueError(f"Internal error: Expected RecursiveTypeMarker, got {type(marker)}")

    if type_kind == "Function":
        raise ValueError("Cannot encode function type to JSON")

    raise ValueError(f"Unhandled type {type_kind} for JSON encoding")


def from_json_for(
    type_val: Any,
    frozen: bool = False,
    type_ctx: list[Any] | None = None,
    marker_map: dict[int, int] | None = None,
    type_str: str | None = None,
) -> Any:
    """Create a JSON decoder function for a given East type.

    The returned function converts JSON values to East values.

    Args:
        type_val: The East type to create a decoder for
        frozen: Whether to freeze decoded objects
        type_ctx: Stack of decoders for recursive types (internal use)
        marker_map: Mapping from RecursiveTypeMarker id to stack index (internal use)
        type_str: String representation of the type (internal use)

    Returns:
        A function that converts JSON values to East values
    """
    if type_ctx is None:
        # Top-level call: create empty stack and marker map
        type_ctx = []
        marker_map = {}

        # Generate type string for error messages
        if type_str is None:
            type_str = print_type(type_val)

        # Build the decoder with the context
        decoder = _build_json_decoder(type_val, frozen, type_ctx, marker_map, type_str)
        return decoder
    # Recursive call: use existing context
    # Generate type string if not provided
    if type_str is None:
        type_str = print_type(type_val)
    return _build_json_decoder(type_val, frozen, type_ctx, marker_map, type_str)


def _build_json_decoder(
    type_val: Any,
    frozen: bool,
    type_ctx: list[Any],
    marker_map: dict[int, int],
    type_str: str,
) -> Any:
    """Build the JSON decoder for a type.

    Args:
        type_val: The East type (must be EastType variant)
        frozen: Whether to freeze mutable containers
        type_ctx: Stack of decoders for recursive types
        marker_map: Mapping from RecursiveTypeMarker id to stack index
        type_str: String representation of the type for error messages
    """
    from datetime import datetime

    from east.types.type_system import _StructTypeClass, _VariantTypeClass

    # Reject raw _StructTypeClass/_VariantTypeClass - these are internal helpers, not valid types
    if isinstance(type_val, (_StructTypeClass, _VariantTypeClass)):
        raise TypeError(
            f"Cannot decode raw {type(type_val).__name__} - use StructType() or VariantType() instead"
        )

    type_kind = type_val.tag

    if type_kind == "Never":
        raise ValueError("Cannot decode Never type from JSON")

    if type_kind == "Null":

        def decode_null(value, _ctx=None):
            if value is not None:
                raise JSONDecodeError(f"expected null, got {json.dumps(value)}", type_str=type_str)
            return null

        return decode_null

    if type_kind == "Boolean":

        def decode_boolean(value, _ctx=None):
            if not isinstance(value, bool):
                raise JSONDecodeError(
                    f"expected boolean, got {json.dumps(value)}", type_str=type_str
                )
            return value

        return decode_boolean

    if type_kind == "Integer":

        def decode_integer(value, _ctx=None):
            if not isinstance(value, str) or not value:
                raise JSONDecodeError(
                    f"expected string representing integer, got {json.dumps(value)}",
                    type_str=type_str,
                )
            try:
                result = int(value)
            except ValueError:
                raise JSONDecodeError(
                    f"expected string representing integer, got {json.dumps(value)}",
                    type_str=type_str,
                ) from None
            # Check for 64-bit signed integer range
            if result < -(2**63) or result > 2**63 - 1:
                raise JSONDecodeError(
                    f"integer out of range (must be 64-bit signed), got {json.dumps(value)}",
                    type_str=type_str,
                )
            return result

        return decode_integer

    if type_kind == "Float":

        def decode_float(value, _ctx=None):
            if isinstance(value, int | float):
                return float(value)
            if value == "-0.0":
                return -0.0
            if value == "NaN":
                return float("nan")
            if value == "Infinity":
                return float("inf")
            if value == "-Infinity":
                return float("-inf")
            raise JSONDecodeError(
                f"expected number or string representing special float value, got {json.dumps(value)}",
                type_str=type_str,
            )

        return decode_float

    if type_kind == "String":

        def decode_string(value, _ctx=None):
            if not isinstance(value, str):
                raise JSONDecodeError(
                    f"expected string, got {json.dumps(value)}", type_str=type_str
                )
            return value

        return decode_string

    if type_kind == "DateTime":

        def decode_datetime(value, _ctx=None):
            if not isinstance(value, str):
                raise JSONDecodeError(
                    f"expected string for DateTime, got {json.dumps(value)}",
                    type_str=type_str,
                )
            # Require RFC 3339 date-time format with timezone
            # Format: YYYY-MM-DDTHH:mm:ss.sss(Z|±HH:mm)
            import re

            iso8601_with_timezone = re.compile(
                r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}(Z|[+-]\d{2}:\d{2})$"
            )
            if not iso8601_with_timezone.match(value):
                raise JSONDecodeError(
                    f'expected ISO 8601 date string with timezone (e.g. "2022-06-29T13:43:00.123Z" or "2022-06-29T13:43:00.123+05:00"), got {json.dumps(value)}',
                    type_str=type_str,
                )
            try:
                dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
                # Convert to UTC
                return dt.astimezone(UTC)
            except ValueError:
                raise JSONDecodeError(
                    f"invalid date string, got {json.dumps(value)}", type_str=type_str
                ) from None

        return decode_datetime

    if type_kind == "Blob":

        def decode_blob(value, _ctx=None):
            if not isinstance(value, str) or not value.startswith("0x"):
                raise JSONDecodeError(
                    f"expected hex string starting with 0x, got {json.dumps(value)}",
                    type_str=type_str,
                )
            hex_str = value[2:]
            if len(hex_str) % 2 != 0 or not all(c in "0123456789abcdefABCDEF" for c in hex_str):
                raise JSONDecodeError(
                    f"invalid hex string, got {json.dumps(value)}", type_str=type_str
                )
            # Decode hex string
            data = bytes.fromhex(hex_str)
            return Blob(data)

        return decode_blob

    if type_kind == "Array":
        # Generate type_str for value type
        value_type_str = print_type(type_val.value)  # type: ignore
        value_decoder = from_json_for(
            type_val.value,
            frozen,
            type_ctx,
            marker_map,
            value_type_str,  # type: ignore
        )

        def decode_array(json_val, ctx: JSONDecodeValueContext | None = None):
            if ctx is None:
                ctx = JSONDecodeValueContext()

            # Check for reference first
            if isinstance(json_val, dict) and "$ref" in json_val and len(json_val) == 1:
                ref_str = json_val["$ref"]
                if isinstance(ref_str, str):
                    try:
                        target_path = decode_relative_ref(ref_str, ctx.current_path)
                        path_key = "/" + "/".join(
                            encode_json_pointer_component(c) for c in target_path
                        )
                        if path_key not in ctx.refs:
                            raise JSONDecodeError(
                                f"undefined reference {ref_str}", type_str=type_str
                            )
                        return ctx.refs[path_key]
                    except ValueError:
                        raise JSONDecodeError(
                            f"invalid reference {ref_str}", type_str=type_str
                        ) from None

            if not isinstance(json_val, list):
                raise JSONDecodeError(
                    f"expected array, got {json.dumps(json_val)}", type_str=type_str
                )

            # Create array and pre-register
            array = EastArray(type_val.value, [])  # type: ignore
            path_key = "/" + "/".join(encode_json_pointer_component(c) for c in ctx.current_path)
            ctx.refs[path_key] = array

            # Populate array
            for i, item in enumerate(json_val):
                ctx.current_path.append(str(i))
                try:
                    array.append(value_decoder(item, ctx))
                except JSONDecodeError as e:
                    new_path = f"[{i}]" + (e.path if e.path else "")
                    raise JSONDecodeError(e.message, new_path, type_str) from None
                finally:
                    ctx.current_path.pop()

            return array

        result = decode_array
        return result

    if type_kind == "Set":
        # Generate type_str for element type
        element_type_str = print_type(type_val.value)  # type: ignore
        key_decoder = from_json_for(
            type_val.value,
            frozen,
            type_ctx,
            marker_map,
            element_type_str,  # type: ignore
        )

        def decode_set(json_val, ctx: JSONDecodeValueContext | None = None):
            if ctx is None:
                ctx = JSONDecodeValueContext()

            # Check for reference first
            if isinstance(json_val, dict) and "$ref" in json_val and len(json_val) == 1:
                ref_str = json_val["$ref"]
                if isinstance(ref_str, str):
                    try:
                        target_path = decode_relative_ref(ref_str, ctx.current_path)
                        path_key = "/" + "/".join(
                            encode_json_pointer_component(c) for c in target_path
                        )
                        if path_key not in ctx.refs:
                            raise JSONDecodeError(
                                f"undefined reference {ref_str}", type_str=type_str
                            )
                        return ctx.refs[path_key]
                    except ValueError:
                        raise JSONDecodeError(
                            f"invalid reference {ref_str}", type_str=type_str
                        ) from None

            if not isinstance(json_val, list):
                raise JSONDecodeError(
                    f"expected array for Set, got {json.dumps(json_val)}", type_str=type_str
                )

            # Create set and pre-register
            s = EastSet(type_val.value, [])  # type: ignore
            path_key = "/" + "/".join(encode_json_pointer_component(c) for c in ctx.current_path)
            ctx.refs[path_key] = s

            # Populate set
            for i, item in enumerate(json_val):
                ctx.current_path.append(str(i))
                try:
                    s.add(key_decoder(item, ctx))
                except JSONDecodeError as e:
                    new_path = f"[{i}]" + (e.path if e.path else "")
                    raise JSONDecodeError(e.message, new_path, type_str) from None
                finally:
                    ctx.current_path.pop()

            return s

        return decode_set

    if type_kind == "Dict":
        # Generate type_str for key and value types
        key_type_str = print_type(type_val.value.key)  # type: ignore
        value_type_str = print_type(type_val.value.value)  # type: ignore
        key_decoder = from_json_for(
            type_val.value.key,
            frozen,
            type_ctx,
            marker_map,
            key_type_str,  # type: ignore
        )
        value_decoder = from_json_for(
            type_val.value.value,
            frozen,
            type_ctx,
            marker_map,
            value_type_str,  # type: ignore
        )

        def decode_dict(json_val, ctx: JSONDecodeValueContext | None = None):
            if ctx is None:
                ctx = JSONDecodeValueContext()

            # Check for reference first
            if isinstance(json_val, dict) and "$ref" in json_val and len(json_val) == 1:
                ref_str = json_val["$ref"]
                if isinstance(ref_str, str):
                    try:
                        target_path = decode_relative_ref(ref_str, ctx.current_path)
                        path_key = "/" + "/".join(
                            encode_json_pointer_component(c) for c in target_path
                        )
                        if path_key not in ctx.refs:
                            raise JSONDecodeError(
                                f"undefined reference {ref_str}", type_str=type_str
                            )
                        return ctx.refs[path_key]
                    except ValueError:
                        raise JSONDecodeError(
                            f"invalid reference {ref_str}", type_str=type_str
                        ) from None

            if not isinstance(json_val, list):
                raise JSONDecodeError(
                    f"expected array for Dict, got {json.dumps(json_val)}", type_str=type_str
                )

            # Create dict and pre-register
            d = EastDict(type_val.value.key, type_val.value.value, {})  # type: ignore
            path_key = "/" + "/".join(encode_json_pointer_component(c) for c in ctx.current_path)
            ctx.refs[path_key] = d

            # Populate dict
            for i, entry in enumerate(json_val):
                if not isinstance(entry, dict) or "key" not in entry or "value" not in entry:
                    raise JSONDecodeError(
                        f"expected object with key and value for Dict entry, got {json.dumps(entry)}",
                        f"[{i}]",
                        type_str,
                    )
                # Check for extra fields
                for k in entry:
                    if k not in ("key", "value"):
                        raise JSONDecodeError(
                            f'unexpected field "{k}" in Dict entry, got {json.dumps(entry)}',
                            f"[{i}]",
                            type_str,
                        )

                # Decode key
                ctx.current_path.extend([str(i), "key"])
                try:
                    dict_key = key_decoder(entry["key"], ctx)
                except JSONDecodeError as e:
                    new_path = f"[{i}].key" + (e.path if e.path else "")
                    raise JSONDecodeError(e.message, new_path, type_str) from None
                finally:
                    ctx.current_path.pop()
                    ctx.current_path.pop()

                # Decode value
                ctx.current_path.extend([str(i), "value"])
                try:
                    dict_value = value_decoder(entry["value"], ctx)
                    d[dict_key] = dict_value
                except JSONDecodeError as e:
                    new_path = f"[{i}].value" + (e.path if e.path else "")
                    raise JSONDecodeError(e.message, new_path, type_str) from None
                finally:
                    ctx.current_path.pop()
                    ctx.current_path.pop()

            return d

        result = decode_dict
        return result

    if type_kind == "Ref":
        from east.types.ref import ref

        # Generate type_str for inner type
        inner_type_str = print_type(type_val.value)  # type: ignore
        inner_decoder = from_json_for(
            type_val.value,
            frozen,
            type_ctx,
            marker_map,
            inner_type_str,  # type: ignore
        )

        def decode_ref(json_val, ctx: JSONDecodeValueContext | None = None):
            if ctx is None:
                ctx = JSONDecodeValueContext()

            # Check for reference first
            if isinstance(json_val, dict) and "$ref" in json_val and len(json_val) == 1:
                ref_str = json_val["$ref"]
                if isinstance(ref_str, str):
                    try:
                        target_path = decode_relative_ref(ref_str, ctx.current_path)
                        path_key = "/" + "/".join(
                            encode_json_pointer_component(c) for c in target_path
                        )
                        if path_key not in ctx.refs:
                            raise JSONDecodeError(
                                f"undefined reference {ref_str}", type_str=type_str
                            )
                        return ctx.refs[path_key]
                    except ValueError:
                        raise JSONDecodeError(
                            f"invalid reference {ref_str}", type_str=type_str
                        ) from None

            if not isinstance(json_val, dict) or "value" not in json_val:
                raise JSONDecodeError(
                    f"expected object with value field for Ref, got {json.dumps(json_val)}",
                    type_str=type_str,
                )

            # Check for extra fields
            for k in json_val:
                if k != "value":
                    raise JSONDecodeError(
                        f'unexpected field "{k}" in Ref, got {json.dumps(json_val)}',
                        type_str=type_str,
                    )

            # Create ref and pre-register
            r = ref(None)  # Placeholder
            path_key = "/" + "/".join(encode_json_pointer_component(c) for c in ctx.current_path)
            ctx.refs[path_key] = r

            # Decode inner value
            ctx.current_path.append("value")
            try:
                inner_value = inner_decoder(json_val["value"], ctx)
                from east.types.ref import set_ref

                set_ref(r, inner_value)
            except JSONDecodeError as e:
                new_path = ".value" + (e.path if e.path else "")
                raise JSONDecodeError(e.message, new_path, type_str) from None
            finally:
                ctx.current_path.pop()

            return r

        return decode_ref

    if type_kind == "Struct":
        field_decoders: dict[str, Any] = {}

        def decode_struct(json_val, ctx: JSONDecodeValueContext | None = None):
            if ctx is None:
                ctx = JSONDecodeValueContext()

            if not isinstance(json_val, dict):
                raise JSONDecodeError(
                    f"expected object for Struct, got {json.dumps(json_val)}", type_str=type_str
                )

            # Check for extra fields
            for k in json_val:
                if k not in field_decoders:
                    raise JSONDecodeError(
                        f'unexpected field "{k}" in Struct, got {json.dumps(json_val)}',
                        type_str=type_str,
                    )

            # Create struct
            obj: dict[str, Any] = {}

            # Populate fields
            for field_name, decoder in field_decoders.items():
                if field_name not in json_val:
                    raise JSONDecodeError(
                        f'missing field "{field_name}" in Struct, got {json.dumps(json_val)}',
                        type_str=type_str,
                    )

                ctx.current_path.append(field_name)
                try:
                    obj[field_name] = decoder(json_val[field_name], ctx)
                except JSONDecodeError as e:
                    new_path = f".{field_name}" + (e.path if e.path else "")
                    raise JSONDecodeError(e.message, new_path, type_str) from None
                finally:
                    ctx.current_path.pop()

            # Build runtime _StructTypeClass and create EastStruct instance
            from east.types.type_system import _StructTypeClass

            fields = [(field.name, field.type) for field in type_val.value]  # type: ignore[attr-defined]
            runtime_type = _StructTypeClass(tuple(fields))
            return runtime_type.create(**obj)

        # Push this decoder onto the stack BEFORE building field decoders
        # This allows fields to reference this type recursively
        stack_index = len(type_ctx)
        type_ctx.append(decode_struct)

        # If this type owns a RecursiveTypeMarker (from recursive_type()), register it
        marker = _find_recursive_marker(type_val)
        if marker is not None:
            marker_map[id(marker)] = stack_index

        # Build field decoders
        # EastType with Struct tag: value contains field structs
        for field_struct in type_val.value:  # type: ignore
            field_type_str = print_type(field_struct.type)
            field_decoders[field_struct.name] = from_json_for(
                field_struct.type, frozen, type_ctx, marker_map, field_type_str
            )

        # Pop from stack after building
        type_ctx.pop()

        return decode_struct

    if type_kind == "Variant":
        case_decoders: dict[str, Any] = {}

        def decode_variant(json_val, ctx: JSONDecodeValueContext | None = None):
            if ctx is None:
                ctx = JSONDecodeValueContext()

            if not isinstance(json_val, dict) or "type" not in json_val or "value" not in json_val:
                raise JSONDecodeError(
                    f"expected object with type and value for Variant, got {json.dumps(json_val)}",
                    type_str=type_str,
                )

            variant_type = json_val["type"]
            if variant_type not in case_decoders:
                raise JSONDecodeError(
                    f'unknown variant type "{variant_type}", got {json.dumps(json_val)}',
                    type_str=type_str,
                )

            case_decoder = case_decoders[variant_type]

            # Decode the value
            ctx.current_path.append(variant_type)
            try:
                variant_value = case_decoder(json_val["value"], ctx)

                # Build runtime _VariantTypeClass and create EastVariant instance
                from east.types.type_system import _VariantTypeClass

                cases = [(case.name, case.type) for case in type_val.value]  # type: ignore[attr-defined]
                runtime_type = _VariantTypeClass(tuple(cases))
                return runtime_type.create(variant_type, variant_value)
            except JSONDecodeError as e:
                new_path = f".{variant_type}" + (e.path if e.path else "")
                raise JSONDecodeError(e.message, new_path, type_str) from None
            finally:
                ctx.current_path.pop()

        # Push this decoder onto the stack BEFORE building case decoders
        # This allows cases to reference this type recursively
        stack_index = len(type_ctx)
        type_ctx.append(decode_variant)

        # If this type owns a RecursiveTypeMarker (from recursive_type()), register it
        marker = _find_recursive_marker(type_val)
        if marker is not None:
            marker_map[id(marker)] = stack_index

        # Build case decoders
        # EastType with Variant tag: value contains case structs
        for case_struct in type_val.value:  # type: ignore
            case_type_str = print_type(case_struct.type)
            case_decoders[case_struct.name] = from_json_for(
                case_struct.type, frozen, type_ctx, marker_map, case_type_str
            )

        # Pop from stack after building
        type_ctx.pop()

        return decode_variant

    if type_kind == "Recursive":
        # Look up decoder by marker
        from east.types.type_system import RecursiveTypeMarker

        marker = type_val.value
        if isinstance(marker, RecursiveTypeMarker):
            # Look up marker in map to get stack index
            marker_id = id(marker)
            if marker_id not in marker_map:
                raise ValueError("Internal error: Recursive type marker not yet registered")
            stack_index = marker_map[marker_id]
            return type_ctx[stack_index]
        # Old-style integer scope_id (shouldn't happen with new code)
        raise ValueError(f"Internal error: Expected RecursiveTypeMarker, got {type(marker)}")

    if type_kind == "Function":
        raise ValueError("Cannot decode function type from JSON")

    raise ValueError(f"Unhandled type {type_kind} for JSON decoding")


def encode_json_for(type_val: EastType) -> Any:
    """Create a function that encodes East values to JSON bytes.

    Args:
        type_val: The East type to create an encoder for

    Returns:
        A function that converts East values to JSON bytes (as bytes object)
    """
    to_json = to_json_for(type_val)

    def encode(value):
        json_value = to_json(value)
        json_str = json.dumps(json_value, separators=(",", ":"))
        return json_str.encode("utf-8")

    return encode


def decode_json_for(type_val: EastType, frozen: bool = False) -> Any:
    """Create a function that decodes JSON bytes to East values.

    Args:
        type_val: The East type to create a decoder for
        frozen: Whether to freeze decoded objects

    Returns:
        A function that converts JSON bytes to East values
    """
    from_json = from_json_for(type_val, frozen)
    type_str = print_type(type_val)

    def decode(data: bytes):
        json_str = data.decode("utf-8")
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError as e:
            # Extract line and column from the JSON parse error
            # Format: "Error occurred because <msg> (line X, col Y) while parsing value of type "<type>""
            error_msg = f'Error occurred because invalid JSON: {e.msg} (line {e.lineno}, col {e.colno}) while parsing value of type "{type_str}"'
            raise ValueError(error_msg) from e

        try:
            return from_json(parsed)
        except JSONDecodeError as e:
            # Use the formatted error message from JSONDecodeError.__str__
            raise ValueError(str(e)) from None

    return decode


__all__ = [
    "JSONDecodeError",
    "to_json_for",
    "from_json_for",
    "encode_json_for",
    "decode_json_for",
    "encode_json_pointer_component",
    "decode_json_pointer_component",
    "encode_relative_ref",
    "decode_relative_ref",
]
