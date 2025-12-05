# CSV Serialization Python Port Design Document

This document specifies the complete design and implementation plan for porting CSV serialization support from the TypeScript implementation (`east/src/serialization/csv.ts`) to Python (`east-py`).

## Table of Contents

1. [Overview](#overview)
2. [Source Analysis](#source-analysis)
3. [Python Implementation Design](#python-implementation-design)
4. [File-by-File Implementation](#file-by-file-implementation)
5. [Type Mappings](#type-mappings)
6. [Implementation Checklist](#implementation-checklist)
7. [Notes](#notes)

---

## Overview

### What Changed in TypeScript (7b4239e..d4f6ed4)

The following commits were added:
- `d4f6ed4` - Merge PR for CSV serialization
- `753e8ae` - Bump version to 0.0.1-beta.5
- `d380406` - feat: add CSV serialization (RFC 4180 compliant)
- `1e309e9` - docs: add CSV serialization design document

### Files Changed

| File | Change Type | Lines Added |
|------|-------------|-------------|
| `src/serialization/csv.ts` | New | 781 |
| `src/serialization/csv.spec.ts` | New | 754 |
| `src/serialization/index.ts` | Modified | +1 export |
| `src/builtins.ts` | Modified | +12 (2 new builtins) |
| `src/compile.ts` | Modified | +22 (compile support) |
| `src/expr/blob.ts` | Modified | +40 (`decodeCsv` method) |
| `src/expr/array.ts` | Modified | +38 (`encodeCsv` method) |
| `devdocs/SERIALIZATION_CSV.md` | New | 839 |
| `STDLIB.md` | Modified | +85 |

### Key Design Decisions from TypeScript

1. **CSV as a Builtin** - Not a platform function, follows BEAST/JSON pattern
2. **Type-driven decode/encode** - Compile-time type specialization
3. **RFC 4180 compliant** - Standard CSV format with extensions
4. **Array<Struct> only** - Top-level must be array of structs
5. **Primitive fields only** - No nested collections in struct fields
6. **Option<T> for nullable** - Empty cells become `none`, present become `some(T)`

---

## Source Analysis

### TypeScript Implementation Structure

```
src/serialization/csv.ts
├── Configuration Types (lines 17-103)
│   ├── CsvParseOptions (TypeScript interface)
│   ├── CsvSerializeOptions (TypeScript interface)
│   ├── CsvParseConfigType (East StructType)
│   └── CsvSerializeConfigType (East StructType)
├── Configuration Conversion (lines 105-177)
│   ├── csvParseOptionsToValue()
│   ├── csvSerializeOptionsToValue()
│   ├── resolveParseConfig()
│   └── resolveSerializeConfig()
├── Error Types (lines 179-209)
│   ├── CsvLocation
│   └── CsvError
├── Type Helpers (lines 211-254)
│   ├── isOptionTypeValue()
│   ├── getOptionInnerTypeValue()
│   └── isSupportedFieldTypeValue()
├── Field Decoders (lines 256-376)
│   ├── createFieldDecoder()
│   └── parseValue()
├── Field Encoders (lines 378-457)
│   ├── createFieldEncoder()
│   └── encodeValue()
├── CSV Parsing (lines 459-682)
│   ├── parseRow()
│   ├── isEmptyRow()
│   └── decodeCsvFor()
└── CSV Serialization (lines 684-781)
    ├── needsQuoting()
    ├── quoteField()
    └── encodeCsvFor()
```

### Builtins Added

```typescript
// In builtins.ts
BlobDecodeCsv: {
  type_parameters: ["T", "Config"],
  inputs: [BlobType, "Config"] as const,
  output: ArrayType("T"),
},
ArrayEncodeCsv: {
  type_parameters: ["T", "Config"],
  inputs: [ArrayType("T"), "Config"] as const,
  output: BlobType,
},
```

### Compiler Support

```typescript
// In compile.ts
BlobDecodeCsv: (location, structType, _configType) => {
  return (data: Uint8Array, config: any) => {
    const decoder = decodeCsvFor(structType, config);
    return decoder(data);
  }
},
ArrayEncodeCsv: (location, structType, _configType) => {
  return (data: any[], config: any) => {
    const encoder = encodeCsvFor(structType, config);
    return encoder(data);
  }
},
```

---

## Python Implementation Design

### Module Structure

```
packages/east-py/east/serialization/
├── __init__.py          # Add csv exports
├── csv.py               # NEW: Core encode/decode functions
├── beast.py             # Existing
├── beast2.py            # Existing
├── json.py              # Existing (pattern to follow)
└── ...

packages/east-py/east/builtins/
├── blob.py              # Add BlobDecodeCsv builtin
├── array.py             # Add ArrayEncodeCsv builtin
└── ...

packages/east-py/tests/serialization/
└── test_csv.py          # NEW: Unit tests
```

### Python Type Mappings

| TypeScript | Python |
|------------|--------|
| `Uint8Array` | `bytes` |
| `string` | `str` |
| `bigint` | `int` |
| `number` | `float` |
| `boolean` | `bool` |
| `Date` | `datetime` (from datetime module) |
| `Map<K,V>` | `EastDict` / `dict` |
| `variant("some", x)` | `EastVariant("some", x)` |
| `variant("none", null)` | `EastVariant("none", east_null)` |
| `EastTypeValue` | `EastType` |
| `StructTypeValue` | `EastType` (with `type == "Struct"`) |

---

## File-by-File Implementation

### 1. `packages/east-py/east/serialization/csv.py`

**Purpose:** Core CSV encode/decode implementation following `json.py` pattern.

```python
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

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime as DateTime
from typing import Any

from east.types.types import (
    ArrayType,
    BooleanType,
    DictType,
    EastType,
    NullType,
    OptionType,
    StringType,
    StructType,
    is_array_type,
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
```

#### Configuration Types

```python
# =============================================================================
# CSV Configuration East Types
# =============================================================================

CsvParseConfigType: EastType = StructType([
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
])

CsvSerializeConfigType: EastType = StructType([
    ("delimiter", OptionType(StringType)),
    ("quoteChar", OptionType(StringType)),
    ("escapeChar", OptionType(StringType)),
    ("newline", OptionType(StringType)),
    ("includeHeader", OptionType(BooleanType)),
    ("nullString", OptionType(StringType)),
    ("alwaysQuote", OptionType(BooleanType)),
])
```

#### Error Types

```python
@dataclass
class CsvLocation:
    """Location information for CSV parsing errors."""
    row: int        # 1-indexed row number (excluding header)
    column: int     # 0-indexed column index
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
```

#### Configuration Helpers

```python
@dataclass
class ResolvedParseConfig:
    """Resolved parse configuration with defaults applied."""
    delimiter: str = ","
    quote_char: str = '"'
    escape_char: str = '"'
    newline: str = ""  # empty = auto-detect
    has_header: bool = True
    null_strings: list[str] = None  # Default [""]
    skip_empty_lines: bool = True
    trim_fields: bool = False
    column_mapping: dict[str, str] = None  # Default {}
    strict: bool = False

    def __post_init__(self):
        if self.null_strings is None:
            self.null_strings = [""]
        if self.column_mapping is None:
            self.column_mapping = {}


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


def resolve_parse_config(config: EastStruct | None) -> ResolvedParseConfig:
    """Extract resolved options from East config value, applying defaults."""
    if config is None:
        return ResolvedParseConfig()

    def get_option(field: str, default: Any) -> Any:
        val = getattr(config, field, None)
        if val is None:
            return default
        if isinstance(val, EastVariant):
            if val.type == "some":
                return val.value
            return default
        return default

    return ResolvedParseConfig(
        delimiter=get_option("delimiter", ","),
        quote_char=get_option("quoteChar", '"'),
        escape_char=get_option("escapeChar", '"'),
        newline=get_option("newline", ""),
        has_header=get_option("hasHeader", True),
        null_strings=list(get_option("nullStrings", [""])),
        skip_empty_lines=get_option("skipEmptyLines", True),
        trim_fields=get_option("trimFields", False),
        column_mapping=dict(get_option("columnMapping", {})),
        strict=get_option("strict", False),
    )


def resolve_serialize_config(config: EastStruct | None) -> ResolvedSerializeConfig:
    """Extract resolved options from East config value, applying defaults."""
    if config is None:
        return ResolvedSerializeConfig()

    def get_option(field: str, default: Any) -> Any:
        val = getattr(config, field, None)
        if val is None:
            return default
        if isinstance(val, EastVariant):
            if val.type == "some":
                return val.value
            return default
        return default

    return ResolvedSerializeConfig(
        delimiter=get_option("delimiter", ","),
        quote_char=get_option("quoteChar", '"'),
        escape_char=get_option("escapeChar", '"'),
        newline=get_option("newline", "\r\n"),
        include_header=get_option("includeHeader", True),
        null_string=get_option("nullString", ""),
        always_quote=get_option("alwaysQuote", False),
    )
```

#### Type Helpers

```python
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
    if (is_null_type(type_val) or is_boolean_type(type_val) or
        is_integer_type(type_val) or is_float_type(type_val) or
        is_string_type(type_val) or is_datetime_type(type_val) or
        is_blob_type(type_val)):
        return True
    if is_option_type(type_val):
        inner = get_option_inner_type(type_val)
        return is_supported_field_type(inner)
    return False
```

#### Field Decoders

```python
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
            else:
                raise CsvError(f"null value for required field '{field_name}'", location)

        # Parse based on type
        try:
            parsed = parse_value(value, base_type, location)
        except CsvError:
            raise
        except Exception as e:
            raise CsvError(f"failed to parse '{value}' as {base_type.type}: {e}", location)

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
        if not trimmed.lstrip('-').isdigit():
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
            raise CsvError(f"expected float, got '{value}'", location)

    if type_name == "String":
        return value

    if type_name == "DateTime":
        try:
            # Parse ISO 8601 format
            # Handle with/without Z suffix and milliseconds
            dt_str = value
            if dt_str.endswith('Z'):
                dt_str = dt_str[:-1] + '+00:00'
            return DateTime.fromisoformat(dt_str).replace(tzinfo=UTC)
        except ValueError:
            raise CsvError(f"expected ISO 8601 date, got '{value}'", location)

    if type_name == "Blob":
        if not value.startswith("0x"):
            raise CsvError(f"expected hex string starting with '0x', got '{value}'", location)
        hex_str = value[2:]
        if len(hex_str) % 2 != 0:
            raise CsvError(f"invalid hex string '{value}'", location)
        try:
            return EastBlob(bytes.fromhex(hex_str))
        except ValueError:
            raise CsvError(f"invalid hex string '{value}'", location)

    raise CsvError(f"unsupported field type {type_name}", location)
```

#### Field Encoders

```python
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
        import math
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
```

#### CSV Parsing

```python
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
    frozen: bool = False,
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
    for field in fields:
        if not is_supported_field_type(field["type"]):
            raise ValueError(f"CSV field '{field['name']}' has unsupported type")

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
        if len(data) >= 3 and data[0:3] == b'\xef\xbb\xbf':
            offset = 3

        # Parse header row
        if resolved.has_header:
            header_result = parse_row(
                data, offset, resolved.delimiter, resolved.quote_char, resolved.escape_char
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
                data, offset, resolved.delimiter, resolved.quote_char, resolved.escape_char
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
                            CsvLocation(row_num, header_idx, dec["name"])
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
```

#### CSV Serialization

```python
def needs_quoting(value: str, delimiter: str, quote_char: str) -> bool:
    """Check if a string needs quoting."""
    return delimiter in value or quote_char in value or '\r' in value or '\n' in value


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
    for field in fields:
        if not is_supported_field_type(field["type"]):
            raise ValueError(f"CSV field '{field['name']}' has unsupported type")

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
                if resolved.always_quote or needs_quoting(name, resolved.delimiter, resolved.quote_char):
                    header_fields.append(quote_field(name, resolved.quote_char, resolved.escape_char))
                else:
                    header_fields.append(name)
            lines.append(resolved.delimiter.join(header_fields))

        # Write data rows
        for row in value:
            row_fields: list[str] = []
            # Handle both dict and EastStruct
            row_data = row if isinstance(row, dict) else row.__dict__

            for i, field_name in enumerate(field_names):
                field_value = row_data.get(field_name)
                encoded = encoders[i](field_value)

                if resolved.always_quote or needs_quoting(encoded, resolved.delimiter, resolved.quote_char):
                    encoded = quote_field(encoded, resolved.quote_char, resolved.escape_char)

                row_fields.append(encoded)

            lines.append(resolved.delimiter.join(row_fields))

        return resolved.newline.join(lines).encode("utf-8")

    return encode
```

#### Module Exports

```python
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
```

---

### 2. `packages/east-py/east/builtins/blob.py` (Additions)

Add these functions and registrations:

```python
def blob_decode_csv_for(T: EastType, Config: EastType) -> Callable[[EastBlob, Any], list[Any]]:
    """Factory for decoding CSV from blob.

    Args:
        T: Struct type for each row
        Config: CsvParseConfigType

    Returns:
        Function that decodes CSV blobs with config
    """
    from east.serialization.csv import decode_csv_for

    def blob_decode_csv(blob: EastBlob, config: Any) -> list[Any]:
        decoder = decode_csv_for(T, config)
        return decoder(blob.data)

    return blob_decode_csv


# Registration
register_builtin("BlobDecodeCsv", blob_decode_csv_for)
```

---

### 3. `packages/east-py/east/builtins/array.py` (Additions)

Add these functions and registrations:

```python
def array_encode_csv_for(T: EastType, Config: EastType) -> Callable[[list[Any], Any], EastBlob]:
    """Factory for encoding array to CSV.

    Args:
        T: Struct type for each row
        Config: CsvSerializeConfigType

    Returns:
        Function that encodes arrays to CSV blobs with config
    """
    from east.serialization.csv import encode_csv_for

    def array_encode_csv(array: list[Any], config: Any) -> EastBlob:
        encoder = encode_csv_for(T, config)
        return EastBlob(encoder(array))

    return array_encode_csv


# Registration
register_builtin("ArrayEncodeCsv", array_encode_csv_for)
```

---

### 4. `packages/east-py/east/serialization/__init__.py` (Update)

```python
"""East serialization and deserialization."""

from east.serialization.csv import (
    CsvParseConfigType,
    CsvSerializeConfigType,
    CsvError,
    CsvLocation,
    decode_csv_for,
    encode_csv_for,
)

__all__ = [
    # CSV
    "CsvParseConfigType",
    "CsvSerializeConfigType",
    "CsvError",
    "CsvLocation",
    "decode_csv_for",
    "encode_csv_for",
]
```

---

## Type Mappings

### East Type to Python Value Mapping for CSV

| East Type | CSV Text | Python Value |
|-----------|----------|--------------|
| `Null` | `""` or `"null"` | `east_null` |
| `Boolean` | `"true"` / `"false"` | `bool` |
| `Integer` | `"123"`, `"-456"` | `int` |
| `Float` | `"3.14"`, `"NaN"`, `"Infinity"` | `float` |
| `String` | as-is | `str` |
| `DateTime` | `"2025-01-15T10:30:00.000"` | `datetime` (UTC) |
| `Blob` | `"0x48656c6c6f"` | `EastBlob` |
| `Option<T>.none` | `""` (or nullString) | `EastVariant("none", east_null)` |
| `Option<T>.some` | value | `EastVariant("some", value)` |

---

## Implementation Checklist

### Phase 1: Core Implementation

- [ ] Create `packages/east-py/east/serialization/csv.py`
  - [ ] Configuration types (`CsvParseConfigType`, `CsvSerializeConfigType`)
  - [ ] Error types (`CsvError`, `CsvLocation`)
  - [ ] Config resolution (`resolve_parse_config`, `resolve_serialize_config`)
  - [ ] Type helpers (`is_option_type`, `get_option_inner_type`, `is_supported_field_type`)
  - [ ] Field decoders (`create_field_decoder`, `parse_value`)
  - [ ] Field encoders (`create_field_encoder`, `encode_value`)
  - [ ] CSV parsing (`parse_row`, `is_empty_row`, `decode_csv_for`)
  - [ ] CSV serialization (`needs_quoting`, `quote_field`, `encode_csv_for`)

### Phase 2: Builtin Integration

- [ ] Update `packages/east-py/east/builtins/blob.py`
  - [ ] Add `blob_decode_csv_for` function
  - [ ] Register `BlobDecodeCsv` builtin

- [ ] Update `packages/east-py/east/builtins/array.py`
  - [ ] Add `array_encode_csv_for` function
  - [ ] Register `ArrayEncodeCsv` builtin

### Phase 3: Module Exports

- [ ] Update `packages/east-py/east/serialization/__init__.py` with CSV exports

### Phase 4: Remove Deprecated CSV from east-py-io

The old platform function CSV implementation must be removed since CSV is now a builtin.

- [ ] Delete `packages/east-py-io/east_py_io/format/csv_impl.py`

- [ ] Update `packages/east-py-io/east_py_io/format/types.py`
  - [ ] Remove `CsvColumnType`
  - [ ] Remove `CsvRowType`
  - [ ] Remove `CsvDataType`
  - [ ] Remove `CsvParseConfigType`
  - [ ] Remove `CsvSerializeConfigType`
  - [ ] Remove from `__all__`

- [ ] Update `packages/east-py-io/east_py_io/format/__init__.py`
  - [ ] Remove all csv imports (`csv_impl`, `csv_parse_impl`, `csv_serialize_impl`)
  - [ ] Remove CSV type imports (`CsvColumnType`, `CsvDataType`, etc.)
  - [ ] Remove from `__all__`
  - [ ] Update module docstring

- [ ] Update `packages/east-py-io/east_py_io/__init__.py`
  - [ ] Remove all CSV imports
  - [ ] Remove `csv_impl` from platform functions list
  - [ ] Remove from `__all__`

- [ ] Update `packages/east-py-io/pyproject.toml`
  - [ ] Remove "csv" from keywords list

### Phase 5: Verification

- [ ] Run existing compliance tests (`make test`) - CSV tests are automatically included via IR exports from `../east`

---

## Notes

### Differences from TypeScript

1. **No frozen support** - Python doesn't have native object freezing like JavaScript's `Object.freeze()`. The `frozen` parameter is accepted for API compatibility but is a no-op.

2. **DateTime handling** - Python's `datetime` uses `datetime.fromisoformat()` which has slightly different parsing rules than JavaScript's `Date` constructor. Special handling for Z suffix required.

3. **Integer type** - Python uses arbitrary precision `int`, TypeScript uses `bigint`. No conversion needed.

4. **Blob type** - Python uses `bytes`, TypeScript uses `Uint8Array`. `EastBlob` wraps `bytes`.

5. **Struct representation** - Python uses `EastStruct` (frozen dataclass-like), TypeScript uses plain objects.

### Performance Considerations

1. **Single-pass parsing** - Maintain streaming approach, don't load entire file into intermediate structures
2. **Pre-compiled decoders** - Create field decoders once at `decode_csv_for()` call time, not per-row
3. **Bytes processing** - Work with `bytes` directly, avoid repeated encoding/decoding
