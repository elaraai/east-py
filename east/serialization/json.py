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
from typing import Any

from east.types.containers import EastArray, EastDict, EastSet
from east.types.primitives import Blob, null
from datetime import datetime as DateTime
from east.types.type_system import EastType


class JSONDecodeError(Exception):
    """Error during JSON decoding."""

    def __init__(self, message: str, path: str = ""):
        """Initialize decode error.

        Args:
            message: Error message
            path: JSON path where error occurred
        """
        super().__init__(message)
        self.path = path


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
        raise ValueError(f"Invalid relative JSON Pointer reference: {ref_str}")

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


def to_json_for(
    type_val: EastType, type_ctx: JSONEncodeTypeContext | None = None
) -> Any:
    """Create a JSON encoder function for a given East type.

    The returned function converts East values to JSON-serializable values.

    Args:
        type_val: The East type to create an encoder for
        type_ctx: Optional context for handling recursive types (internal use)

    Returns:
        A function that converts East values to JSON-serializable values
    """
    if type_ctx is None:
        type_ctx = []

    # Handle raw StructType and VariantType objects (not wrapped in EastType)
    from east.types.type_system import StructType as StructTypeClass, VariantType as VariantTypeClass

    if isinstance(type_val, StructTypeClass):
        # Handle raw StructType
        type_kind = "Struct"
    elif isinstance(type_val, VariantTypeClass):
        # Handle raw VariantType
        type_kind = "Variant"
    else:
        # EastType instance - use tag
        type_kind = type_val.tag

    if type_kind == "Null":
        return lambda _value, _ctx=None: None

    elif type_kind == "Boolean":
        return lambda value, _ctx=None: value

    elif type_kind == "Integer":
        # Encode as string to preserve full 64-bit range
        return lambda value, _ctx=None: str(value)

    elif type_kind == "Float":

        def encode_float(value: float, _ctx=None):
            # Handle negative zero specially since JSON.parse("-0.0") returns 0
            if value == 0 and str(value).startswith("-"):
                return "-0.0"
            # Handle special values
            if value != value:  # NaN
                return "NaN"
            if value == float("inf"):
                return "Infinity"
            if value == float("-inf"):
                return "-Infinity"
            # Normal finite float
            return value

        return encode_float

    elif type_kind == "String":
        return lambda value, _ctx=None: value

    elif type_kind == "DateTime":

        def encode_datetime(dt: DateTime, _ctx=None):
            # Encode as RFC 3339 date-time string
            # Format: YYYY-MM-DDTHH:mm:ss.sss+00:00
            year = dt.year
            month = str(dt.month).zfill(2)
            day = str(dt.day).zfill(2)
            hour = str(dt.hour).zfill(2)
            minute = str(dt.minute).zfill(2)
            second = str(dt.second).zfill(2)
            ms = str(dt.microsecond // 1000).zfill(3)
            return f"{year}-{month}-{day}T{hour}:{minute}:{second}.{ms}+00:00"

        return encode_datetime

    elif type_kind == "Blob":

        def encode_blob(blob: Blob, _ctx=None):
            # Encode as hex string with 0x prefix
            hex_str = "".join(f"{b:02x}" for b in blob.data)
            return f"0x{hex_str}"

        return encode_blob

    elif type_kind == "Array":
        value_encoder = to_json_for(type_val.value, type_ctx)  # type: ignore

        def encode_array(arr: EastArray, ctx: JSONEncodeValueContext | None = None):
            if ctx is None:
                ctx = JSONEncodeValueContext()

            # Check if this array was already seen
            arr_id = id(arr)
            if arr_id in ctx.refs:
                # Return a relative reference
                target_path = ctx.refs[arr_id]
                ref_str = encode_relative_ref(ctx.current_path, target_path)
                return {"$ref": ref_str}

            # First encounter - register the current path
            ctx.refs[arr_id] = list(ctx.current_path)

            # Serialize array elements
            result = []
            for i, item in enumerate(arr):
                ctx.current_path.append(str(i))
                result.append(value_encoder(item, ctx))
                ctx.current_path.pop()
            return result

        type_ctx.append(encode_array)
        result = encode_array
        type_ctx.pop()
        return result

    elif type_kind == "Set":
        key_encoder = to_json_for(type_val.value, type_ctx)  # type: ignore

        def encode_set(s: EastSet, ctx: JSONEncodeValueContext | None = None):
            if ctx is None:
                ctx = JSONEncodeValueContext()

            # Check if we're tracking references
            if ctx:
                # Check if this set was already seen
                set_id = id(s)
                if set_id in ctx.refs:
                    # Return a relative reference
                    target_path = ctx.refs[set_id]
                    ref_str = encode_relative_ref(ctx.current_path, target_path)
                    return {"$ref": ref_str}

                # First encounter - register the current path
                ctx.refs[set_id] = list(ctx.current_path)

            # Serialize set elements as array
            result = []
            for i, item in enumerate(s):
                if ctx:
                    ctx.current_path.append(str(i))
                result.append(key_encoder(item, ctx))
                if ctx:
                    ctx.current_path.pop()
            return result

        return encode_set

    elif type_kind == "Dict":
        key_encoder = to_json_for(type_val.value.key, type_ctx)  # type: ignore
        value_encoder = to_json_for(type_val.value.value, type_ctx)  # type: ignore

        def encode_dict(d: EastDict, ctx: JSONEncodeValueContext | None = None):
            if ctx is None:
                ctx = JSONEncodeValueContext()

            # Check if this dict was already seen
            dict_id = id(d)
            if dict_id in ctx.refs:
                # Return a relative reference
                target_path = ctx.refs[dict_id]
                ref_str = encode_relative_ref(ctx.current_path, target_path)
                return {"$ref": ref_str}

            # First encounter - register the current path
            ctx.refs[dict_id] = list(ctx.current_path)

            # Serialize dict entries as array of {key, value} objects
            result = []
            for i, (k, v) in enumerate(d.items()):
                entry: dict[str, Any] = {}

                # Encode key
                ctx.current_path.extend([str(i), "key"])
                entry["key"] = key_encoder(k, ctx)
                ctx.current_path.pop()
                ctx.current_path.pop()

                # Encode value
                ctx.current_path.extend([str(i), "value"])
                entry["value"] = value_encoder(v, ctx)
                ctx.current_path.pop()
                ctx.current_path.pop()

                result.append(entry)
            return result

        type_ctx.append(encode_dict)
        result = encode_dict
        type_ctx.pop()
        return result

    elif type_kind == "Struct":
        field_encoders: dict[str, Any] = {}

        def encode_struct(obj: dict[str, Any], ctx: JSONEncodeValueContext | None = None):
            if ctx is None:
                ctx = JSONEncodeValueContext()

            result: dict[str, Any] = {}
            for field_name, encoder in field_encoders.items():
                ctx.current_path.append(field_name)
                result[field_name] = encoder(obj[field_name], ctx)
                ctx.current_path.pop()
            return result

        type_ctx.append(encode_struct)
        # Handle both raw StructType and EastType with Struct tag
        if isinstance(type_val, StructTypeClass):
            # Raw StructType: fields is a tuple of (name, type) tuples
            for name, field_type in type_val.fields:
                field_encoders[name] = to_json_for(field_type, type_ctx)
        else:
            # EastType with Struct tag: value contains field structs
            for field_struct in type_val.value:  # type: ignore
                field_encoders[field_struct.name] = to_json_for(field_struct.type, type_ctx)
        type_ctx.pop()
        return encode_struct

    elif type_kind == "Variant":
        case_encoders: dict[str, Any] = {}

        def encode_variant(variant: dict[str, Any], ctx: JSONEncodeValueContext | None = None):
            if ctx is None:
                ctx = JSONEncodeValueContext()

            variant_type = variant["type"]
            ctx.current_path.append(variant_type)
            encoded_value = case_encoders[variant_type](variant["value"], ctx)
            ctx.current_path.pop()
            return {"type": variant_type, "value": encoded_value}

        type_ctx.append(encode_variant)
        # Handle both raw VariantType and EastType with Variant tag
        if isinstance(type_val, VariantTypeClass):
            # Raw VariantType: cases is a tuple of (name, type) tuples
            for name, case_type in type_val.cases:
                case_encoders[name] = to_json_for(case_type, type_ctx)
        else:
            # EastType with Variant tag: value contains case structs
            for case_struct in type_val.value:  # type: ignore
                case_encoders[case_struct.name] = to_json_for(case_struct.type, type_ctx)
        type_ctx.pop()
        return encode_variant

    elif type_kind == "Recursive":
        # Look up the encoder from the type context
        # depth indicates which level of nesting the type refers to (0 = outermost)
        depth = type_val.value  # type: ignore
        if depth < 0 or depth >= len(type_ctx):
            raise ValueError(f"Invalid recursive type reference: depth={depth}, context size={len(type_ctx)}")
        encoder = type_ctx[depth]
        return encoder

    elif type_kind == "Function":
        raise ValueError("Cannot encode function type to JSON")

    else:
        raise ValueError(f"Unhandled type {type_kind} for JSON encoding")


def from_json_for(
    type_val: EastType, frozen: bool = False, type_ctx: JSONDecodeTypeContext | None = None
) -> Any:
    """Create a JSON decoder function for a given East type.

    The returned function converts JSON values to East values.

    Args:
        type_val: The East type to create a decoder for
        frozen: Whether to freeze decoded objects
        type_ctx: Optional context for handling recursive types (internal use)

    Returns:
        A function that converts JSON values to East values
    """
    if type_ctx is None:
        type_ctx = []

    from datetime import datetime, timezone
    from east.types.type_system import StructType as StructTypeClass, VariantType as VariantTypeClass

    # Handle raw StructType and VariantType objects (not wrapped in EastType)
    if isinstance(type_val, StructTypeClass):
        type_kind = "Struct"
    elif isinstance(type_val, VariantTypeClass):
        type_kind = "Variant"
    else:
        type_kind = type_val.tag

    if type_kind == "Null":

        def decode_null(value, _ctx=None):
            if value is not None:
                raise JSONDecodeError(f"expected null, got {json.dumps(value)}")
            return null

        return decode_null

    elif type_kind == "Boolean":

        def decode_boolean(value, _ctx=None):
            if not isinstance(value, bool):
                raise JSONDecodeError(f"expected boolean, got {json.dumps(value)}")
            return value

        return decode_boolean

    elif type_kind == "Integer":

        def decode_integer(value, _ctx=None):
            if not isinstance(value, str) or not value:
                raise JSONDecodeError(f"expected string representing integer, got {json.dumps(value)}")
            try:
                result = int(value)
            except ValueError:
                raise JSONDecodeError(f"expected string representing integer, got {json.dumps(value)}")
            # Check for 64-bit signed integer range
            if result < -(2**63) or result > 2**63 - 1:
                raise JSONDecodeError(f"integer out of range (must be 64-bit signed), got {json.dumps(value)}")
            return result

        return decode_integer

    elif type_kind == "Float":

        def decode_float(value, _ctx=None):
            if isinstance(value, (int, float)):
                return float(value)
            elif value == "-0.0":
                return -0.0
            elif value == "NaN":
                return float("nan")
            elif value == "Infinity":
                return float("inf")
            elif value == "-Infinity":
                return float("-inf")
            else:
                raise JSONDecodeError(
                    f"expected number or string representing special float value, got {json.dumps(value)}"
                )

        return decode_float

    elif type_kind == "String":

        def decode_string(value, _ctx=None):
            if not isinstance(value, str):
                raise JSONDecodeError(f"expected string, got {json.dumps(value)}")
            return value

        return decode_string

    elif type_kind == "DateTime":

        def decode_datetime(value, _ctx=None):
            if not isinstance(value, str):
                raise JSONDecodeError(f"expected string for DateTime, got {json.dumps(value)}")
            # Require RFC 3339 date-time format with timezone
            # Format: YYYY-MM-DDTHH:mm:ss.sss(Z|±HH:mm)
            import re

            iso8601_with_timezone = re.compile(
                r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}(Z|[+-]\d{2}:\d{2})$"
            )
            if not iso8601_with_timezone.match(value):
                raise JSONDecodeError(
                    f'expected ISO 8601 date string with timezone (e.g. "2022-06-29T13:43:00.123Z" or "2022-06-29T13:43:00.123+05:00"), got {json.dumps(value)}'
                )
            try:
                dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
                # Convert to UTC
                dt = dt.astimezone(timezone.utc)
                return dt
            except ValueError:
                raise JSONDecodeError(f"invalid date string, got {json.dumps(value)}")

        return decode_datetime

    elif type_kind == "Blob":

        def decode_blob(value, _ctx=None):
            if not isinstance(value, str) or not value.startswith("0x"):
                raise JSONDecodeError(f"expected hex string starting with 0x, got {json.dumps(value)}")
            hex_str = value[2:]
            if len(hex_str) % 2 != 0 or not all(c in "0123456789abcdefABCDEF" for c in hex_str):
                raise JSONDecodeError(f"invalid hex string, got {json.dumps(value)}")
            # Decode hex string
            data = bytes.fromhex(hex_str)
            return Blob(data)

        return decode_blob

    elif type_kind == "Array":
        value_decoder = from_json_for(type_val.value, frozen, type_ctx)  # type: ignore

        def decode_array(json_val, ctx: JSONDecodeValueContext | None = None):
            if ctx is None:
                ctx = JSONDecodeValueContext()

            # Check for reference first
            if isinstance(json_val, dict) and "$ref" in json_val and len(json_val) == 1:
                ref_str = json_val["$ref"]
                if isinstance(ref_str, str):
                    try:
                        target_path = decode_relative_ref(ref_str, ctx.current_path)
                        path_key = "/" + "/".join(encode_json_pointer_component(c) for c in target_path)
                        if path_key not in ctx.refs:
                            raise JSONDecodeError(f"undefined reference {ref_str}")
                        return ctx.refs[path_key]
                    except ValueError as e:
                        raise JSONDecodeError(f"invalid reference {ref_str}")

            if not isinstance(json_val, list):
                raise JSONDecodeError(f"expected array, got {json.dumps(json_val)}")

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
                    raise JSONDecodeError(e.args[0], new_path)
                finally:
                    ctx.current_path.pop()

            return array

        type_ctx.append(decode_array)
        result = decode_array
        type_ctx.pop()
        return result

    elif type_kind == "Set":
        key_decoder = from_json_for(type_val.value, frozen, type_ctx)  # type: ignore

        def decode_set(json_val, ctx: JSONDecodeValueContext | None = None):
            if ctx is None:
                ctx = JSONDecodeValueContext()

            # Check for reference first
            if isinstance(json_val, dict) and "$ref" in json_val and len(json_val) == 1:
                ref_str = json_val["$ref"]
                if isinstance(ref_str, str):
                    try:
                        target_path = decode_relative_ref(ref_str, ctx.current_path)
                        path_key = "/" + "/".join(encode_json_pointer_component(c) for c in target_path)
                        if path_key not in ctx.refs:
                            raise JSONDecodeError(f"undefined reference {ref_str}")
                        return ctx.refs[path_key]
                    except ValueError:
                        raise JSONDecodeError(f"invalid reference {ref_str}")

            if not isinstance(json_val, list):
                raise JSONDecodeError(f"expected array for Set, got {json.dumps(json_val)}")

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
                    raise JSONDecodeError(e.args[0], new_path)
                finally:
                    ctx.current_path.pop()

            return s

        return decode_set

    elif type_kind == "Dict":
        key_decoder = from_json_for(type_val.value.key, frozen, type_ctx)  # type: ignore
        value_decoder = from_json_for(type_val.value.value, frozen, type_ctx)  # type: ignore

        def decode_dict(json_val, ctx: JSONDecodeValueContext | None = None):
            if ctx is None:
                ctx = JSONDecodeValueContext()

            # Check for reference first
            if isinstance(json_val, dict) and "$ref" in json_val and len(json_val) == 1:
                ref_str = json_val["$ref"]
                if isinstance(ref_str, str):
                    try:
                        target_path = decode_relative_ref(ref_str, ctx.current_path)
                        path_key = "/" + "/".join(encode_json_pointer_component(c) for c in target_path)
                        if path_key not in ctx.refs:
                            raise JSONDecodeError(f"undefined reference {ref_str}")
                        return ctx.refs[path_key]
                    except ValueError:
                        raise JSONDecodeError(f"invalid reference {ref_str}")

            if not isinstance(json_val, list):
                raise JSONDecodeError(f"expected array for Dict, got {json.dumps(json_val)}")

            # Create dict and pre-register
            d = EastDict(type_val.value.key, type_val.value.value, {})  # type: ignore
            path_key = "/" + "/".join(encode_json_pointer_component(c) for c in ctx.current_path)
            ctx.refs[path_key] = d

            # Populate dict
            for i, entry in enumerate(json_val):
                if not isinstance(entry, dict) or "key" not in entry or "value" not in entry:
                    raise JSONDecodeError(
                        f"expected object with key and value for Dict entry, got {json.dumps(entry)}", f"[{i}]"
                    )
                # Check for extra fields
                for k in entry:
                    if k not in ("key", "value"):
                        raise JSONDecodeError(
                            f'unexpected field "{k}" in Dict entry, got {json.dumps(entry)}', f"[{i}]"
                        )

                # Decode key
                ctx.current_path.extend([str(i), "key"])
                try:
                    dict_key = key_decoder(entry["key"], ctx)
                except JSONDecodeError as e:
                    new_path = f"[{i}].key" + (e.path if e.path else "")
                    raise JSONDecodeError(e.args[0], new_path)
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
                    raise JSONDecodeError(e.args[0], new_path)
                finally:
                    ctx.current_path.pop()
                    ctx.current_path.pop()

            return d

        type_ctx.append(decode_dict)
        result = decode_dict
        type_ctx.pop()
        return result

    elif type_kind == "Struct":
        field_decoders: dict[str, Any] = {}

        def decode_struct(json_val, ctx: JSONDecodeValueContext | None = None):
            if ctx is None:
                ctx = JSONDecodeValueContext()

            if not isinstance(json_val, dict):
                raise JSONDecodeError(f"expected object for Struct, got {json.dumps(json_val)}")

            # Check for extra fields
            for k in json_val:
                if k not in field_decoders:
                    raise JSONDecodeError(f'unexpected field "{k}" in Struct, got {json.dumps(json_val)}')

            # Create struct
            obj: dict[str, Any] = {}

            # Populate fields
            for field_name, decoder in field_decoders.items():
                if field_name not in json_val:
                    raise JSONDecodeError(f'missing field "{field_name}" in Struct, got {json.dumps(json_val)}')

                ctx.current_path.append(field_name)
                try:
                    obj[field_name] = decoder(json_val[field_name], ctx)
                except JSONDecodeError as e:
                    new_path = f".{field_name}" + (e.path if e.path else "")
                    raise JSONDecodeError(e.args[0], new_path)
                finally:
                    ctx.current_path.pop()

            return obj

        type_ctx.append(decode_struct)
        # Handle both raw StructType and EastType with Struct tag
        if isinstance(type_val, StructTypeClass):
            # Raw StructType: fields is a tuple of (name, type) tuples
            for name, field_type in type_val.fields:
                field_decoders[name] = from_json_for(field_type, frozen, type_ctx)
        else:
            # EastType with Struct tag: value contains field structs
            for field_struct in type_val.value:  # type: ignore
                field_decoders[field_struct.name] = from_json_for(field_struct.type, frozen, type_ctx)
        type_ctx.pop()
        return decode_struct

    elif type_kind == "Variant":
        case_decoders: dict[str, Any] = {}

        def decode_variant(json_val, ctx: JSONDecodeValueContext | None = None):
            if ctx is None:
                ctx = JSONDecodeValueContext()

            if not isinstance(json_val, dict) or "type" not in json_val or "value" not in json_val:
                raise JSONDecodeError(
                    f"expected object with type and value for Variant, got {json.dumps(json_val)}"
                )

            variant_type = json_val["type"]
            if variant_type not in case_decoders:
                raise JSONDecodeError(f'unknown variant type "{variant_type}", got {json.dumps(json_val)}')

            case_decoder = case_decoders[variant_type]

            # Decode the value
            ctx.current_path.append(variant_type)
            try:
                variant_value = case_decoder(json_val["value"], ctx)
                # Return as dict with type and value
                return {"type": variant_type, "value": variant_value}
            except JSONDecodeError as e:
                new_path = f".{variant_type}" + (e.path if e.path else "")
                raise JSONDecodeError(e.args[0], new_path)
            finally:
                ctx.current_path.pop()

        type_ctx.append(decode_variant)
        # Handle both raw VariantType and EastType with Variant tag
        if isinstance(type_val, VariantTypeClass):
            # Raw VariantType: cases is a tuple of (name, type) tuples
            for name, case_type in type_val.cases:
                case_decoders[name] = from_json_for(case_type, frozen, type_ctx)
        else:
            # EastType with Variant tag: value contains case structs
            for case_struct in type_val.value:  # type: ignore
                case_decoders[case_struct.name] = from_json_for(case_struct.type, frozen, type_ctx)
        type_ctx.pop()
        return decode_variant

    elif type_kind == "Recursive":
        # Look up the decoder from the type context
        # depth indicates which level of nesting the type refers to (0 = outermost)
        depth = type_val.value  # type: ignore
        if depth < 0 or depth >= len(type_ctx):
            raise ValueError(f"Invalid recursive type reference: depth={depth}, context size={len(type_ctx)}")
        decoder = type_ctx[depth]
        return decoder

    elif type_kind == "Function":
        raise ValueError("Cannot decode function type from JSON")

    else:
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

    def decode(data: bytes):
        json_str = data.decode("utf-8")
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")

        try:
            return from_json(parsed)
        except JSONDecodeError as e:
            path_str = f" at {e.path}" if e.path else ""
            raise ValueError(f"{e.args[0]}{path_str}")

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
