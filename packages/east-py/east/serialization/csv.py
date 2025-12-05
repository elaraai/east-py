"""CSV serialization for East types (RFC 4180 compliant).

This module provides CSV encoding and decoding for East values, following
the TypeScript reference implementation.

Key features:
- Type-driven encoding/decoding
- RFC 4180 compliant parsing with extensions
- Support for all primitive types and Option<T>
- Configurable delimiters, quoting, null handling
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC
from datetime import datetime as DateTime
from typing import Any

from east.types.types import (
    ArrayType,
    BooleanType,
    DictType,
    EastType,
    OptionType,
    StringType,
    StructType,
    is_blob_type,
    is_boolean_type,
    is_datetime_type,
    is_float_type,
    is_integer_type,
    is_null_type,
    is_string_type,
    is_struct_type,
    is_variant_type,
)
from east.types.values import (
    EastBlob,
    EastStruct,
    EastVariant,
    east_null,
)

# =============================================================================
# CSV Configuration East Types
# =============================================================================

CsvParseConfigType: EastType = StructType(
    [
        ("delimiter", OptionType(StringType)),
        ("quoteChar", OptionType(StringType)),
        ("escapeChar", OptionType(StringType)),
        ("newline", OptionType(StringType)),
        ("hasHeader", OptionType(BooleanType)),
        ("nullStrings", OptionType(ArrayType(StringType))),
        ("skipEmptyLines", OptionType(BooleanType)),
        ("trimFields", OptionType(BooleanType)),
        ("columnMapping", OptionType(DictType(StringType, StringType))),
        ("strict", OptionType(BooleanType)),
    ]
)

CsvSerializeConfigType: EastType = StructType(
    [
        ("delimiter", OptionType(StringType)),
        ("quoteChar", OptionType(StringType)),
        ("escapeChar", OptionType(StringType)),
        ("newline", OptionType(StringType)),
        ("includeHeader", OptionType(BooleanType)),
        ("nullString", OptionType(StringType)),
        ("alwaysQuote", OptionType(BooleanType)),
    ]
)

# =============================================================================
# CSV Error Types
# =============================================================================


@dataclass
class CsvLocation:
    """Location information for CSV parsing errors."""

    row: int  # 1-indexed row number (excluding header)
    column: int  # 0-indexed column index
    column_name: str | None = None


class CsvError(Exception):
    """Error thrown during CSV parsing with location information."""

    def __init__(self, message: str, location: CsvLocation | None = None):
        self.location = location
        location_str = ""
        if location:
            location_str = f" at row {location.row}, column {location.column}"
            if location.column_name:
                location_str += f" ({location.column_name})"
        super().__init__(f"CSV error: {message}{location_str}")


# =============================================================================
# Configuration Helpers
# =============================================================================


@dataclass
class ResolvedParseConfig:
    """Resolved parse configuration with defaults applied."""

    delimiter: str = ","
    quote_char: str = '"'
    escape_char: str = '"'
    newline: str = ""  # empty = auto-detect
    has_header: bool = True
    null_strings: list[str] = field(default_factory=lambda: [""])
    skip_empty_lines: bool = True
    trim_fields: bool = False
    column_mapping: dict[str, str] = field(default_factory=dict)
    strict: bool = False


@dataclass
class ResolvedSerializeConfig:
    """Resolved serialize configuration with defaults applied."""

    delimiter: str = ","
    quote_char: str = '"'
    escape_char: str = '"'
    newline: str = "\r\n"
    include_header: bool = True
    null_string: str = ""
    always_quote: bool = False


def _get_option_value(val: Any, default: Any) -> Any:
    """Extract value from Option variant or return default."""
    if val is None:
        return default
    if isinstance(val, EastVariant):
        if val.type == "some":
            return val.value
        return default
    # Handle dict-style variant (from EastStruct)
    if isinstance(val, dict) and "type" in val:
        if val["type"] == "some":
            return val.get("value", default)
        return default
    return default


def resolve_parse_config(config: Any) -> ResolvedParseConfig:
    """Extract resolved options from East config value, applying defaults."""
    if config is None:
        return ResolvedParseConfig()

    # Handle both EastStruct and dict (EastStruct extends dict)
    if isinstance(config, dict):
        data = config
    else:
        data = {}

    null_strings_val = _get_option_value(data.get("nullStrings"), [""])
    column_mapping_val = _get_option_value(data.get("columnMapping"), {})

    return ResolvedParseConfig(
        delimiter=_get_option_value(data.get("delimiter"), ","),
        quote_char=_get_option_value(data.get("quoteChar"), '"'),
        escape_char=_get_option_value(data.get("escapeChar"), '"'),
        newline=_get_option_value(data.get("newline"), ""),
        has_header=_get_option_value(data.get("hasHeader"), True),
        null_strings=list(null_strings_val) if null_strings_val else [""],
        skip_empty_lines=_get_option_value(data.get("skipEmptyLines"), True),
        trim_fields=_get_option_value(data.get("trimFields"), False),
        column_mapping=dict(column_mapping_val) if column_mapping_val else {},
        strict=_get_option_value(data.get("strict"), False),
    )


def resolve_serialize_config(config: Any) -> ResolvedSerializeConfig:
    """Extract resolved options from East config value, applying defaults."""
    if config is None:
        return ResolvedSerializeConfig()

    # Handle both EastStruct and dict (EastStruct extends dict)
    if isinstance(config, dict):
        data = config
    else:
        data = {}

    return ResolvedSerializeConfig(
        delimiter=_get_option_value(data.get("delimiter"), ","),
        quote_char=_get_option_value(data.get("quoteChar"), '"'),
        escape_char=_get_option_value(data.get("escapeChar"), '"'),
        newline=_get_option_value(data.get("newline"), "\r\n"),
        include_header=_get_option_value(data.get("includeHeader"), True),
        null_string=_get_option_value(data.get("nullString"), ""),
        always_quote=_get_option_value(data.get("alwaysQuote"), False),
    )


# =============================================================================
# Type Helpers
# =============================================================================


def is_option_type(type_val: EastType) -> bool:
    """Check if type is Option (Variant with exactly 'none' and 'some' cases)."""
    if not is_variant_type(type_val):
        return False
    cases = type_val.value
    if len(cases) != 2:
        return False
    # Variant cases are sorted alphabetically: none at 0, some at 1
    return cases[0]["name"] == "none" and cases[1]["name"] == "some"


def get_option_inner_type(type_val: EastType) -> EastType:
    """Get the inner type of an OptionType (the 'some' case type)."""
    if not is_variant_type(type_val):
        raise ValueError("Not an OptionType")
    # 'some' is at index 1 (alphabetically sorted)
    return type_val.value[1]["type"]


def is_supported_field_type(type_val: EastType) -> bool:
    """Check if type is a supported primitive type for CSV fields."""
    if (
        is_null_type(type_val)
        or is_boolean_type(type_val)
        or is_integer_type(type_val)
        or is_float_type(type_val)
        or is_string_type(type_val)
        or is_datetime_type(type_val)
        or is_blob_type(type_val)
    ):
        return True
    if is_option_type(type_val):
        inner = get_option_inner_type(type_val)
        return is_supported_field_type(inner)
    return False


# =============================================================================
# Field Decoders
# =============================================================================

FieldDecoder = Callable[[str, CsvLocation], Any]


def create_field_decoder(
    type_val: EastType,
    field_name: str,
    null_strings: list[str],
    trim_fields: bool,
) -> FieldDecoder:
    """Create a decoder for a single field based on its type."""
    is_option = is_option_type(type_val)
    base_type = get_option_inner_type(type_val) if is_option else type_val

    def decoder(value: str, location: CsvLocation) -> Any:
        # Apply trim if configured
        if trim_fields:
            value = value.strip()

        # Check for null
        if value in null_strings:
            if is_option:
                return EastVariant("none", east_null)
            raise CsvError(f"null value for required field '{field_name}'", location)

        # Parse based on type
        try:
            parsed = parse_value(value, base_type, location)
        except CsvError:
            raise
        except Exception as e:
            raise CsvError(f"failed to parse '{value}' as {base_type.type}: {e}", location) from e

        # Wrap in Option if needed
        if is_option:
            return EastVariant("some", parsed)
        return parsed

    return decoder


def parse_value(value: str, type_val: EastType, location: CsvLocation) -> Any:
    """Parse a string value to the given type."""
    type_name = type_val.type

    if type_name == "Null":
        if value != "" and value != "null":
            raise CsvError(f"expected null, got '{value}'", location)
        return east_null

    if type_name == "Boolean":
        if value == "true":
            return True
        if value == "false":
            return False
        raise CsvError(f"expected 'true' or 'false', got '{value}'", location)

    if type_name == "Integer":
        trimmed = value.strip()
        # Check for valid integer format
        test_str = trimmed[1:] if trimmed.startswith("-") else trimmed
        if not test_str.isdigit():
            raise CsvError(f"expected integer, got '{value}'", location)
        return int(trimmed)

    if type_name == "Float":
        if value == "NaN":
            return float("nan")
        if value == "Infinity":
            return float("inf")
        if value == "-Infinity":
            return float("-inf")
        if value == "-0" or value == "-0.0":
            return -0.0
        try:
            return float(value)
        except ValueError:
            raise CsvError(f"expected float, got '{value}'", location) from None

    if type_name == "String":
        return value

    if type_name == "DateTime":
        try:
            # Parse ISO 8601 format
            # Handle with/without Z suffix and milliseconds
            dt_str = value
            if dt_str.endswith("Z"):
                dt_str = dt_str[:-1] + "+00:00"
            return DateTime.fromisoformat(dt_str).replace(tzinfo=UTC)
        except ValueError:
            raise CsvError(f"expected ISO 8601 date, got '{value}'", location) from None

    if type_name == "Blob":
        if not value.startswith("0x"):
            raise CsvError(f"expected hex string starting with '0x', got '{value}'", location)
        hex_str = value[2:]
        if len(hex_str) % 2 != 0:
            raise CsvError(f"invalid hex string '{value}'", location)
        try:
            return EastBlob(bytes.fromhex(hex_str))
        except ValueError:
            raise CsvError(f"invalid hex string '{value}'", location) from None

    raise CsvError(f"unsupported field type {type_name}", location)


# =============================================================================
# Field Encoders
# =============================================================================

FieldEncoder = Callable[[Any], str]


def create_field_encoder(type_val: EastType, null_string: str) -> FieldEncoder:
    """Create an encoder for a single field based on its type."""
    is_option = is_option_type(type_val)
    base_type = get_option_inner_type(type_val) if is_option else type_val

    def encoder(value: Any) -> str:
        # Handle Option type
        if is_option:
            if isinstance(value, EastVariant) and value.type == "none":
                return null_string
            if isinstance(value, EastVariant) and value.type == "some":
                value = value.value

        # Handle null
        if value is None or value is east_null:
            return null_string

        return encode_value(value, base_type)

    return encoder


def encode_value(value: Any, type_val: EastType) -> str:
    """Encode a value to a string."""
    type_name = type_val.type

    if type_name == "Null":
        return ""

    if type_name == "Boolean":
        return "true" if value else "false"

    if type_name == "Integer":
        return str(value)

    if type_name == "Float":
        if math.isnan(value):
            return "NaN"
        if value == float("inf"):
            return "Infinity"
        if value == float("-inf"):
            return "-Infinity"
        if value == 0.0 and math.copysign(1, value) < 0:
            return "-0"
        return str(value)

    if type_name == "String":
        return value

    if type_name == "DateTime":
        # ISO 8601 format without timezone suffix
        dt: DateTime = value
        if dt.tzinfo is not None:
            dt = dt.astimezone(UTC).replace(tzinfo=None)
        return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}"

    if type_name == "Blob":
        data = value.data if isinstance(value, EastBlob) else value
        return "0x" + data.hex()

    raise ValueError(f"Unsupported field type {type_name} for CSV encoding")


# =============================================================================
# CSV Parsing
# =============================================================================


def parse_row(
    data: bytes,
    offset: int,
    delimiter: str,
    quote_char: str,
    escape_char: str,
) -> tuple[list[str], int, bool]:
    """Parse a CSV row into an array of fields, handling quotes and escapes.

    Returns: (fields, new_offset, is_end)
    """
    delim_byte = ord(delimiter)
    quote_byte = ord(quote_char)
    escape_byte = ord(escape_char)

    fields: list[str] = []
    in_quote = False
    field_chars: list[int] = []
    i = offset

    while i < len(data):
        byte = data[i]

        if in_quote:
            if byte == escape_byte and i + 1 < len(data) and data[i + 1] == quote_byte:
                # Escaped quote
                field_chars.append(quote_byte)
                i += 2
            elif byte == quote_byte:
                # End of quoted field
                in_quote = False
                i += 1
            else:
                field_chars.append(byte)
                i += 1
        else:
            if byte == quote_byte and len(field_chars) == 0:
                # Start of quoted field
                in_quote = True
                i += 1
            elif byte == delim_byte:
                # End of field
                fields.append(bytes(field_chars).decode("utf-8"))
                field_chars = []
                i += 1
            elif byte == 0x0D:  # CR
                # Check for CRLF
                if i + 1 < len(data) and data[i + 1] == 0x0A:
                    fields.append(bytes(field_chars).decode("utf-8"))
                    return (fields, i + 2, False)
                # Just CR
                fields.append(bytes(field_chars).decode("utf-8"))
                return (fields, i + 1, False)
            elif byte == 0x0A:  # LF
                fields.append(bytes(field_chars).decode("utf-8"))
                return (fields, i + 1, False)
            else:
                field_chars.append(byte)
                i += 1

    # End of file
    if in_quote:
        raise CsvError("unclosed quote at end of file")

    fields.append(bytes(field_chars).decode("utf-8"))
    return (fields, i, True)


def is_empty_row(fields: list[str]) -> bool:
    """Check if a row is empty (all fields are empty strings)."""
    if len(fields) == 0:
        return True
    return all(f == "" for f in fields)


def decode_csv_for(
    struct_type: EastType,
    config: Any = None,
    _frozen: bool = False,
) -> Callable[[bytes], list[Any]]:
    """Create a type-specialized CSV decoder for Array<Struct>.

    Args:
        struct_type: The struct type for each row
        config: Configuration as East value (CsvParseConfigType)
        frozen: Whether to freeze decoded values (no-op in Python)

    Returns:
        Function that decodes CSV bytes to a list of structs
    """
    if not is_struct_type(struct_type):
        raise ValueError("CSV decode requires a struct type")

    fields = struct_type.value  # list of {"name": str, "type": EastType}

    # Validate that all fields are supported types
    for f in fields:
        if not is_supported_field_type(f["type"]):
            raise ValueError(f"CSV field '{f['name']}' has unsupported type")

    # Resolve config with defaults
    resolved = resolve_parse_config(config)

    # Pre-build field info
    field_infos = [
        {
            "name": f["name"],
            "type": f["type"],
            "is_optional": is_option_type(f["type"]),
            "decoder": create_field_decoder(
                f["type"], f["name"], resolved.null_strings, resolved.trim_fields
            ),
        }
        for f in fields
    ]
    field_names = [f["name"] for f in field_infos]

    def decode(data: bytes) -> list[Any]:
        # Skip UTF-8 BOM if present
        offset = 0
        if len(data) >= 3 and data[0:3] == b"\xef\xbb\xbf":
            offset = 3

        # Parse header row
        if resolved.has_header:
            header_result = parse_row(
                data,
                offset,
                resolved.delimiter,
                resolved.quote_char,
                resolved.escape_char,
            )
            headers = [resolved.column_mapping.get(h, h) for h in header_result[0]]
            offset = header_result[1]
        else:
            headers = field_names

        # Build header index lookup
        header_to_index = {h: i for i, h in enumerate(headers)}

        # Validate: check for missing required fields
        for info in field_infos:
            if info["name"] not in header_to_index and not info["is_optional"]:
                raise CsvError(f"missing required column '{info['name']}'")

        # Strict mode: check for extra columns
        if resolved.strict:
            for header in headers:
                if header not in field_names:
                    raise CsvError(f"unexpected column '{header}' in strict mode")

        # Build per-field decoder info with header indices
        decoders = [
            {
                "name": info["name"],
                "is_optional": info["is_optional"],
                "decoder": info["decoder"],
                "header_index": header_to_index.get(info["name"]),
            }
            for info in field_infos
        ]

        # Parse data rows
        result: list[Any] = []
        row_num = 1

        while offset < len(data):
            row_result = parse_row(
                data,
                offset,
                resolved.delimiter,
                resolved.quote_char,
                resolved.escape_char,
            )
            row_fields = row_result[0]
            offset = row_result[1]
            is_end = row_result[2]

            if resolved.skip_empty_lines and is_empty_row(row_fields):
                if is_end:
                    break
                continue

            # Decode row into struct
            row: dict[str, Any] = {}
            for dec in decoders:
                header_idx = dec["header_index"]
                if header_idx is None:
                    row[dec["name"]] = EastVariant("none", east_null)
                elif header_idx >= len(row_fields):
                    if dec["is_optional"]:
                        row[dec["name"]] = EastVariant("none", east_null)
                    else:
                        raise CsvError(
                            f"row has {len(row_fields)} fields, expected at least {header_idx + 1}",
                            CsvLocation(row_num, header_idx, dec["name"]),
                        )
                else:
                    location = CsvLocation(row_num, header_idx, dec["name"])
                    row[dec["name"]] = dec["decoder"](row_fields[header_idx], location)

            result.append(EastStruct(row))
            row_num += 1
            if is_end:
                break

        return result

    return decode


# =============================================================================
# CSV Serialization
# =============================================================================


def needs_quoting(value: str, delimiter: str, quote_char: str) -> bool:
    """Check if a string needs quoting."""
    return delimiter in value or quote_char in value or "\r" in value or "\n" in value


def quote_field(value: str, quote_char: str, escape_char: str) -> str:
    """Quote a string value, escaping internal quotes."""
    escaped = value.replace(quote_char, escape_char + quote_char)
    return quote_char + escaped + quote_char


def encode_csv_for(
    struct_type: EastType,
    config: Any = None,
) -> Callable[[list[Any]], bytes]:
    """Create a type-specialized CSV encoder for Array<Struct>.

    Args:
        struct_type: The struct type for each row
        config: Configuration as East value (CsvSerializeConfigType)

    Returns:
        Function that encodes list of structs to CSV bytes
    """
    if not is_struct_type(struct_type):
        raise ValueError("CSV encode requires a struct type")

    fields = struct_type.value

    # Validate that all fields are supported types
    for f in fields:
        if not is_supported_field_type(f["type"]):
            raise ValueError(f"CSV field '{f['name']}' has unsupported type")

    # Resolve config with defaults
    resolved = resolve_serialize_config(config)

    field_names = [f["name"] for f in fields]

    # Create field encoders
    encoders = [create_field_encoder(f["type"], resolved.null_string) for f in fields]

    def encode(value: list[Any]) -> bytes:
        lines: list[str] = []

        # Write header
        if resolved.include_header:
            header_fields = []
            for name in field_names:
                if resolved.always_quote or needs_quoting(
                    name, resolved.delimiter, resolved.quote_char
                ):
                    header_fields.append(
                        quote_field(name, resolved.quote_char, resolved.escape_char)
                    )
                else:
                    header_fields.append(name)
            lines.append(resolved.delimiter.join(header_fields))

        # Write data rows
        for row in value:
            row_fields: list[str] = []
            # Handle both dict and EastStruct (EastStruct extends dict)
            if isinstance(row, dict):
                row_data = row
            else:
                row_data = {k: getattr(row, k, None) for k in field_names}

            for i, field_name in enumerate(field_names):
                field_value = row_data.get(field_name)
                encoded = encoders[i](field_value)

                if resolved.always_quote or needs_quoting(
                    encoded, resolved.delimiter, resolved.quote_char
                ):
                    encoded = quote_field(encoded, resolved.quote_char, resolved.escape_char)

                row_fields.append(encoded)

            lines.append(resolved.delimiter.join(row_fields))

        return resolved.newline.join(lines).encode("utf-8")

    return encode


__all__ = [
    # Types
    "CsvParseConfigType",
    "CsvSerializeConfigType",
    "CsvLocation",
    "CsvError",
    # Functions
    "decode_csv_for",
    "encode_csv_for",
    "resolve_parse_config",
    "resolve_serialize_config",
]
