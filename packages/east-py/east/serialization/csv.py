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
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime as DateTime
from typing import TYPE_CHECKING, Any, NamedTuple

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

if TYPE_CHECKING:
    pass

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


@dataclass(slots=True)
class CsvLocation:
    """Location information for CSV parsing errors."""

    row: int  # 1-indexed row number (excluding header)
    column: int  # 0-indexed column index
    column_name: str | None = None


class CsvError(Exception):
    """Error thrown during CSV parsing with location information."""

    __slots__ = ("location",)

    def __init__(self, message: str, location: CsvLocation | None = None):
        self.location = location
        location_str = ""
        if location:
            location_str = f" at row {location.row}, column {location.column}"
            if location.column_name:
                location_str += f" ({location.column_name})"
        super().__init__(f"CSV error: {message}{location_str}")


# =============================================================================
# Configuration - Using NamedTuple for memory efficiency
# =============================================================================


class ResolvedParseConfig(NamedTuple):
    """Resolved parse configuration with defaults applied."""

    delimiter: str = ","
    quote_char: str = '"'
    escape_char: str = '"'
    newline: str = ""  # empty = auto-detect
    has_header: bool = True
    null_strings: frozenset[str] = frozenset(("",))
    skip_empty_lines: bool = True
    trim_fields: bool = False
    column_mapping: dict[str, str] | None = None
    strict: bool = False


class ResolvedSerializeConfig(NamedTuple):
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
        return val.value if val.type == "some" else default
    # Handle dict-style variant (from EastStruct)
    if isinstance(val, dict) and val.get("type") == "some":
        return val.get("value", default)
    return default


def resolve_parse_config(config: Any) -> ResolvedParseConfig:
    """Extract resolved options from East config value, applying defaults."""
    if config is None:
        return ResolvedParseConfig()

    # Handle both EastStruct and dict (EastStruct extends dict)
    data = config if isinstance(config, dict) else {}

    null_strings_val = _get_option_value(data.get("nullStrings"), None)
    column_mapping_val = _get_option_value(data.get("columnMapping"), None)

    return ResolvedParseConfig(
        delimiter=_get_option_value(data.get("delimiter"), ","),
        quote_char=_get_option_value(data.get("quoteChar"), '"'),
        escape_char=_get_option_value(data.get("escapeChar"), '"'),
        newline=_get_option_value(data.get("newline"), ""),
        has_header=_get_option_value(data.get("hasHeader"), True),
        null_strings=frozenset(null_strings_val) if null_strings_val else frozenset(("",)),
        skip_empty_lines=_get_option_value(data.get("skipEmptyLines"), True),
        trim_fields=_get_option_value(data.get("trimFields"), False),
        column_mapping=dict(column_mapping_val) if column_mapping_val else None,
        strict=_get_option_value(data.get("strict"), False),
    )


def resolve_serialize_config(config: Any) -> ResolvedSerializeConfig:
    """Extract resolved options from East config value, applying defaults."""
    if config is None:
        return ResolvedSerializeConfig()

    # Handle both EastStruct and dict (EastStruct extends dict)
    data = config if isinstance(config, dict) else {}

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
# Type-specific Parsers (pre-computed for performance)
# =============================================================================

# Type aliases for clarity
ValueParser = Callable[[str, CsvLocation], Any]
FieldDecoder = Callable[[str, CsvLocation], Any]


def _parse_null(value: str, location: CsvLocation) -> Any:
    if value != "" and value != "null":
        raise CsvError(f"expected null, got '{value}'", location)
    return east_null


def _parse_boolean(value: str, location: CsvLocation) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise CsvError(f"expected 'true' or 'false', got '{value}'", location)


def _parse_integer(value: str, location: CsvLocation) -> int:
    # Fast path: check if all digits (with optional leading minus)
    if value:
        start = 1 if value[0] == "-" else 0
        if start < len(value) and value[start:].isdigit():
            return int(value)
    raise CsvError(f"expected integer, got '{value}'", location)


def _parse_float(value: str, location: CsvLocation) -> float:
    # Fast paths for special values
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


def _parse_string(value: str, _location: CsvLocation) -> str:
    return value


def _parse_datetime(value: str, location: CsvLocation) -> DateTime:
    try:
        # Parse ISO 8601 format
        dt_str = value
        if dt_str.endswith("Z"):
            dt_str = dt_str[:-1] + "+00:00"
        return DateTime.fromisoformat(dt_str).replace(tzinfo=UTC)
    except ValueError:
        raise CsvError(f"expected ISO 8601 date, got '{value}'", location) from None


def _parse_blob(value: str, location: CsvLocation) -> EastBlob:
    if not value.startswith("0x"):
        raise CsvError(f"expected hex string starting with '0x', got '{value}'", location)
    hex_str = value[2:]
    if len(hex_str) % 2 != 0:
        raise CsvError(f"invalid hex string '{value}'", location)
    try:
        return EastBlob(bytes.fromhex(hex_str))
    except ValueError:
        raise CsvError(f"invalid hex string '{value}'", location) from None


# Map type names to parser functions (avoid repeated string comparisons)
_TYPE_PARSERS: dict[str, ValueParser] = {
    "Null": _parse_null,
    "Boolean": _parse_boolean,
    "Integer": _parse_integer,
    "Float": _parse_float,
    "String": _parse_string,
    "DateTime": _parse_datetime,
    "Blob": _parse_blob,
}


def get_value_parser(type_val: EastType) -> ValueParser:
    """Get the parser function for a given type."""
    parser = _TYPE_PARSERS.get(type_val.type)
    if parser is None:
        raise ValueError(f"Unsupported field type {type_val.type}")
    return parser


# =============================================================================
# Field Decoders
# =============================================================================


class FieldInfo(NamedTuple):
    """Pre-computed field information for decoding."""

    name: str
    is_optional: bool
    decoder: FieldDecoder
    header_index: int | None = None


def create_field_decoder(
    type_val: EastType,
    field_name: str,
    null_strings: frozenset[str],
    trim_fields: bool,
) -> FieldDecoder:
    """Create a decoder for a single field based on its type."""
    is_option = is_option_type(type_val)
    base_type = get_option_inner_type(type_val) if is_option else type_val
    parser = get_value_parser(base_type)

    # Pre-compute the none variant for optional fields
    none_variant = EastVariant("none", east_null)

    if trim_fields:
        if is_option:

            def decoder_trim_opt(value: str, location: CsvLocation) -> Any:
                value = value.strip()
                if value in null_strings:
                    return none_variant
                return EastVariant("some", parser(value, location))

            return decoder_trim_opt

        def decoder_trim_req(value: str, location: CsvLocation) -> Any:
            value = value.strip()
            if value in null_strings:
                raise CsvError(f"null value for required field '{field_name}'", location)
            return parser(value, location)

        return decoder_trim_req
    if is_option:

        def decoder_opt(value: str, location: CsvLocation) -> Any:
            if value in null_strings:
                return none_variant
            return EastVariant("some", parser(value, location))

        return decoder_opt

    def decoder_req(value: str, location: CsvLocation) -> Any:
        if value in null_strings:
            raise CsvError(f"null value for required field '{field_name}'", location)
        return parser(value, location)

    return decoder_req


# =============================================================================
# Field Encoders
# =============================================================================

FieldEncoder = Callable[[Any], str]


def _encode_null(_value: Any) -> str:
    return ""


def _encode_boolean(value: Any) -> str:
    return "true" if value else "false"


def _encode_integer(value: Any) -> str:
    return str(value)


def _encode_float(value: Any) -> str:
    if math.isnan(value):
        return "NaN"
    if value == float("inf"):
        return "Infinity"
    if value == float("-inf"):
        return "-Infinity"
    if value == 0.0 and math.copysign(1, value) < 0:
        return "-0"
    return str(value)


def _encode_string(value: Any) -> str:
    return value


def _encode_datetime(value: Any) -> str:
    dt: DateTime = value
    if dt.tzinfo is not None:
        dt = dt.astimezone(UTC).replace(tzinfo=None)
    # Use f-string for faster formatting
    return f"{dt.year:04d}-{dt.month:02d}-{dt.day:02d}T{dt.hour:02d}:{dt.minute:02d}:{dt.second:02d}.{dt.microsecond // 1000:03d}"


def _encode_blob(value: Any) -> str:
    data = value.data if isinstance(value, EastBlob) else value
    return "0x" + data.hex()


# Map type names to encoder functions
_TYPE_ENCODERS: dict[str, FieldEncoder] = {
    "Null": _encode_null,
    "Boolean": _encode_boolean,
    "Integer": _encode_integer,
    "Float": _encode_float,
    "String": _encode_string,
    "DateTime": _encode_datetime,
    "Blob": _encode_blob,
}


def create_field_encoder(type_val: EastType, null_string: str) -> FieldEncoder:
    """Create an encoder for a single field based on its type."""
    is_option = is_option_type(type_val)
    base_type = get_option_inner_type(type_val) if is_option else type_val
    base_encoder = _TYPE_ENCODERS.get(base_type.type)

    if base_encoder is None:
        raise ValueError(f"Unsupported field type {base_type.type} for CSV encoding")

    if is_option:

        def encoder_opt(value: Any) -> str:
            if isinstance(value, EastVariant):
                if value.type == "none":
                    return null_string
                value = value.value
            if value is None or value is east_null:
                return null_string
            return base_encoder(value)

        return encoder_opt

    def encoder_req(value: Any) -> str:
        if value is None or value is east_null:
            return null_string
        return base_encoder(value)

    return encoder_req


# =============================================================================
# CSV Parsing - Optimized with bytearray
# =============================================================================


def parse_row(
    data: bytes,
    offset: int,
    delim_byte: int,
    quote_byte: int,
    escape_byte: int,
) -> tuple[list[str], int, bool]:
    """Parse a CSV row into an array of fields, handling quotes and escapes.

    Returns: (fields, new_offset, is_end)

    Note: Takes byte values directly to avoid repeated ord() calls.
    """
    fields: list[str] = []
    field_buf = bytearray()
    in_quote = False
    i = offset
    data_len = len(data)

    while i < data_len:
        byte = data[i]

        if in_quote:
            if byte == escape_byte and i + 1 < data_len and data[i + 1] == quote_byte:
                # Escaped quote
                field_buf.append(quote_byte)
                i += 2
            elif byte == quote_byte:
                # End of quoted field
                in_quote = False
                i += 1
            else:
                field_buf.append(byte)
                i += 1
        else:
            if byte == quote_byte and len(field_buf) == 0:
                # Start of quoted field
                in_quote = True
                i += 1
            elif byte == delim_byte:
                # End of field
                fields.append(field_buf.decode("utf-8"))
                field_buf.clear()
                i += 1
            elif byte == 0x0D:  # CR
                # Check for CRLF
                fields.append(field_buf.decode("utf-8"))
                if i + 1 < data_len and data[i + 1] == 0x0A:
                    return (fields, i + 2, False)
                return (fields, i + 1, False)
            elif byte == 0x0A:  # LF
                fields.append(field_buf.decode("utf-8"))
                return (fields, i + 1, False)
            else:
                field_buf.append(byte)
                i += 1

    # End of file
    if in_quote:
        raise CsvError("unclosed quote at end of file")

    fields.append(field_buf.decode("utf-8"))
    return (fields, i, True)


def is_empty_row(fields: list[str]) -> bool:
    """Check if a row is empty (all fields are empty strings)."""
    # Fast path: single empty field is the most common empty row
    if len(fields) == 1:
        return fields[0] == ""
    if len(fields) == 0:
        return True
    return all(f == "" for f in fields)


# =============================================================================
# Main Decoder
# =============================================================================


def decode_csv_for(
    struct_type: EastType,
    config: Any = None,
    _frozen: bool = False,
) -> Callable[[bytes], list[Any]]:
    """Create a type-specialized CSV decoder for Array<Struct>.

    Args:
        struct_type: The struct type for each row
        config: Configuration as East value (CsvParseConfigType)
        _frozen: Whether to freeze decoded values (no-op in Python)

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

    # Pre-compute byte values for parsing
    delim_byte = ord(resolved.delimiter)
    quote_byte = ord(resolved.quote_char)
    escape_byte = ord(resolved.escape_char)

    # Pre-build field info using NamedTuple
    field_infos = tuple(
        FieldInfo(
            name=f["name"],
            is_optional=is_option_type(f["type"]),
            decoder=create_field_decoder(
                f["type"], f["name"], resolved.null_strings, resolved.trim_fields
            ),
        )
        for f in fields
    )
    field_names = tuple(f.name for f in field_infos)

    # Pre-compute the none variant for missing optional fields
    none_variant = EastVariant("none", east_null)

    # Extract config values for closure (avoid tuple indexing in hot loop)
    has_header = resolved.has_header
    column_mapping = resolved.column_mapping
    skip_empty_lines = resolved.skip_empty_lines
    strict = resolved.strict

    def decode(data: bytes) -> list[Any]:
        # Skip UTF-8 BOM if present
        offset = 3 if len(data) >= 3 and data[0:3] == b"\xef\xbb\xbf" else 0

        # Parse header row
        if has_header:
            header_fields, offset, _ = parse_row(data, offset, delim_byte, quote_byte, escape_byte)
            if column_mapping:
                headers = tuple(column_mapping.get(h, h) for h in header_fields)
            else:
                headers = tuple(header_fields)
        else:
            headers = field_names

        # Build header index lookup
        header_to_index = {h: i for i, h in enumerate(headers)}

        # Validate: check for missing required fields and build decoder list with indices
        decoders: list[tuple[str, bool, FieldDecoder, int | None]] = []
        for info in field_infos:
            idx = header_to_index.get(info.name)
            if idx is None and not info.is_optional:
                raise CsvError(f"missing required column '{info.name}'")
            decoders.append((info.name, info.is_optional, info.decoder, idx))

        # Strict mode: check for extra columns
        if strict:
            field_name_set = set(field_names)
            for header in headers:
                if header not in field_name_set:
                    raise CsvError(f"unexpected column '{header}' in strict mode")

        # Parse data rows
        result: list[Any] = []
        row_num = 1
        data_len = len(data)

        while offset < data_len:
            row_fields, offset, is_end = parse_row(
                data, offset, delim_byte, quote_byte, escape_byte
            )

            if skip_empty_lines and is_empty_row(row_fields):
                if is_end:
                    break
                continue

            # Decode row into struct
            row: dict[str, Any] = {}
            num_fields = len(row_fields)

            for name, is_optional, decoder, header_idx in decoders:
                if header_idx is None:
                    row[name] = none_variant
                elif header_idx >= num_fields:
                    if is_optional:
                        row[name] = none_variant
                    else:
                        raise CsvError(
                            f"row has {num_fields} fields, expected at least {header_idx + 1}",
                            CsvLocation(row_num, header_idx, name),
                        )
                else:
                    row[name] = decoder(
                        row_fields[header_idx], CsvLocation(row_num, header_idx, name)
                    )

            result.append(EastStruct(row))
            row_num += 1
            if is_end:
                break

        return result

    return decode


# =============================================================================
# CSV Serialization - Optimized with direct byte writing
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

    field_names = tuple(f["name"] for f in fields)
    num_fields = len(field_names)

    # Create field encoders
    encoders = tuple(create_field_encoder(f["type"], resolved.null_string) for f in fields)

    # Pre-encode configuration
    delimiter = resolved.delimiter
    quote_char = resolved.quote_char
    escape_char = resolved.escape_char
    newline_bytes = resolved.newline.encode("utf-8")
    delimiter_bytes = delimiter.encode("utf-8")
    include_header = resolved.include_header
    always_quote = resolved.always_quote

    def encode(value: list[Any]) -> bytes:
        # Use bytearray for efficient byte building
        output = bytearray()

        # Write header
        if include_header:
            for i, name in enumerate(field_names):
                if i > 0:
                    output.extend(delimiter_bytes)
                if always_quote or needs_quoting(name, delimiter, quote_char):
                    output.extend(quote_field(name, quote_char, escape_char).encode("utf-8"))
                else:
                    output.extend(name.encode("utf-8"))
            if value:  # Only add newline if there are data rows
                output.extend(newline_bytes)

        # Write data rows
        num_rows = len(value)
        for row_idx, row in enumerate(value):
            # Handle both dict and EastStruct (EastStruct extends dict)
            row_data = row if isinstance(row, dict) else {}

            for i in range(num_fields):
                if i > 0:
                    output.extend(delimiter_bytes)

                field_value = row_data.get(field_names[i])
                encoded = encoders[i](field_value)

                if always_quote or needs_quoting(encoded, delimiter, quote_char):
                    encoded = quote_field(encoded, quote_char, escape_char)

                output.extend(encoded.encode("utf-8"))

            # Add newline between rows (not after last row)
            if row_idx < num_rows - 1:
                output.extend(newline_bytes)

        return bytes(output)

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
