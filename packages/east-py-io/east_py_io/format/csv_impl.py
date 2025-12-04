"""CSV platform functions for East.

Provides CSV parsing and serialization for East programs with strict RFC 4180 compliance.
"""

from datetime import datetime

from east.runtime.platform import PlatformFunction
from east.types.types import BlobType, StringType
from east.types.values import EastArray, EastBlob, EastDict, EastStruct, EastVariant, east_null

from .types import (
    CsvDataType,
    CsvParseConfigType,
    CsvRowType,
    CsvSerializeConfigType,
    LiteralValueType,
)


def parse_csv_strict(
    text: str,
    delimiter: str,
    quote_char: str,
    escape_char: str | None,
) -> list[list[str]]:
    """Parse CSV with strict validation.

    Raises detailed errors for malformed CSV:
    - Unclosed quotes
    - Text after closing quote
    - Invalid escape sequences (when escape_char is set)

    Args:
        text: CSV text to parse
        delimiter: Field delimiter
        quote_char: Quote character
        escape_char: Escape character (None for double-quote escaping)

    Returns:
        List of rows, each row is a list of field values
    """
    rows: list[list[str]] = []
    current_row: list[str] = []
    current_field = ""
    in_quotes = False
    row_num = 1
    col_num = 1
    field_start_col = 1
    i = 0

    while i < len(text):
        char = text[i]

        # Handle quote character (must be checked before escape when they're the same)
        if char == quote_char:
            if not in_quotes:
                # Starting a quoted field
                if current_field:
                    # Text before quote - error (text after closing quote from previous parse)
                    raise Exception(f"Unexpected quote in row {row_num}, column {col_num}")
                in_quotes = True
            elif i + 1 < len(text) and text[i + 1] == quote_char:
                # Double-quote escape (always supported, e.g., '' -> ')
                current_field += quote_char
                i += 2
                continue
            else:
                # End of quoted field - check for text after closing quote
                in_quotes = False
                # Look ahead for text after closing quote
                j = i + 1
                while j < len(text) and text[j] not in (delimiter, "\n", "\r"):
                    if text[j] != " " and text[j] != "\t":  # Allow trailing whitespace
                        # There's non-whitespace text after the closing quote
                        raise Exception(
                            f"Expected delimiter or newline after closing quote in row {row_num}, column {col_num}"
                        )
                    j += 1
        # Handle escape character (when explicitly set and different from quote char)
        elif escape_char and escape_char != quote_char and char == escape_char and in_quotes:
            if i + 1 < len(text):
                next_char = text[i + 1]
                # Handle known escape sequences
                if next_char == "n":
                    current_field += "\n"
                    i += 2
                    continue
                elif next_char == "r":
                    current_field += "\r"
                    i += 2
                    continue
                elif next_char == "t":
                    current_field += "\t"
                    i += 2
                    continue
                elif next_char in ("x", "u", "U"):
                    # Invalid hex/unicode escape sequence (like \x, \u, \U)
                    raise Exception(f"Invalid escape sequence in row {row_num}, column {col_num}")
                else:
                    # Any other character: escape it literally (quote, escape char, delimiter, etc.)
                    current_field += next_char
                    i += 2
                    continue
            else:
                # Escape at end of input - treat as literal escape character
                current_field += char
                i += 1
                continue
        elif char == delimiter and not in_quotes:
            # End of field
            current_row.append(current_field)
            current_field = ""
            col_num += 1
            field_start_col = col_num
        elif char == "\r":
            if not in_quotes:
                # End of row (handle CRLF)
                current_row.append(current_field)
                rows.append(current_row)
                current_row = []
                current_field = ""
                row_num += 1
                col_num = 1
                field_start_col = 1
                # Skip LF if CRLF
                if i + 1 < len(text) and text[i + 1] == "\n":
                    i += 1
            else:
                current_field += char
        elif char == "\n":
            if not in_quotes:
                # End of row
                current_row.append(current_field)
                rows.append(current_row)
                current_row = []
                current_field = ""
                row_num += 1
                col_num = 1
                field_start_col = 1
            else:
                current_field += char
        else:
            current_field += char

        i += 1

    # Handle unclosed quote
    if in_quotes:
        raise Exception(f"Unclosed quote in row {row_num}, column {field_start_col}")

    # Handle last row/field
    if current_field or current_row:
        current_row.append(current_field)
        rows.append(current_row)

    return rows


def convert_value_to_east(value: str | None, column_type: str) -> EastVariant:
    """Convert a CSV string value to East LiteralValueType variant.

    Matches TypeScript behavior: respects column type, no auto-detection.
    """
    # Handle null/None values
    if value is None:
        return EastVariant("Null", east_null)

    # Parse according to column type (matching TypeScript convertNativeToCell)
    if column_type == "Null":
        return EastVariant("Null", east_null)
    elif column_type == "String":
        return EastVariant("String", value)
    elif column_type == "Integer":
        trimmed = value.strip()
        if trimmed == "":
            raise ValueError("Cannot parse empty string as Integer")
        try:
            int_val = int(trimmed)
            # Check 64-bit signed integer range
            if int_val < -9223372036854775808 or int_val > 9223372036854775807:
                raise ValueError("Integer out of range (must be 64-bit signed)")
            return EastVariant("Integer", int_val)
        except ValueError as e:
            if "out of range" in str(e):
                raise
            raise ValueError(f'Cannot parse "{value}" as Integer') from None
    elif column_type == "Float":
        trimmed = value.strip()
        if trimmed == "":
            raise ValueError("Cannot parse empty string as Float")
        # Handle special values
        if trimmed == "NaN":
            return EastVariant("Float", float("nan"))
        elif trimmed == "Infinity":
            return EastVariant("Float", float("inf"))
        elif trimmed == "-Infinity":
            return EastVariant("Float", float("-inf"))
        try:
            float_val = float(trimmed)
            return EastVariant("Float", float_val)
        except ValueError:
            raise ValueError(f'Cannot parse "{value}" as Float') from None
    elif column_type == "Boolean":
        trimmed = value.strip()
        if trimmed == "true":
            return EastVariant("Boolean", True)
        elif trimmed == "false":
            return EastVariant("Boolean", False)
        else:
            raise ValueError(f"Cannot parse \"{value}\" as Boolean (expected 'true' or 'false')")
    elif column_type == "DateTime":
        trimmed = value.strip()
        if trimmed == "":
            raise ValueError("Cannot parse empty string as DateTime")
        try:
            dt = datetime.fromisoformat(trimmed.replace("Z", "+00:00"))
            return EastVariant("DateTime", dt)
        except ValueError:
            raise ValueError(
                f'Cannot parse "{value}" as DateTime (expected ISO 8601 format)'
            ) from None
    elif column_type == "Blob":
        trimmed = value.strip()
        if trimmed == "":
            raise ValueError("Cannot parse empty string as Blob")
        if not trimmed.startswith("0x"):
            raise ValueError(f'Cannot parse "{value}" as Blob (expected 0x-prefixed hex string)')
        hex_str = trimmed[2:]
        if len(hex_str) % 2 != 0:
            raise ValueError(f'Cannot parse "{value}" as Blob (odd length hex string)')
        try:
            blob_bytes = bytes.fromhex(hex_str)
            return EastVariant("Blob", EastBlob(blob_bytes))
        except ValueError:
            raise ValueError(f'Cannot parse "{value}" as Blob (invalid hex character)') from None
    else:
        # Default to String
        return EastVariant("String", value)


def convert_east_to_value(value: EastVariant) -> str:
    """Convert East LiteralValueType variant to string for CSV output."""
    tag = value.type
    val = value.value

    if tag == "Null" or val is None:
        return ""
    elif tag == "Boolean":
        return "true" if val else "false"
    elif tag == "DateTime":
        return val.isoformat() if hasattr(val, "isoformat") else str(val)
    elif tag == "Blob":
        return val.hex() if hasattr(val, "hex") else str(val)
    else:
        return str(val)


def csv_parse_impl(blob: EastBlob, config: EastStruct) -> EastArray:
    """Parse CSV data from a binary blob."""
    try:
        # Get config options
        columns_opt = config["columns"]
        columns_map = None
        if columns_opt.type == "some":
            columns_map = columns_opt.value

        delimiter_opt = config["delimiter"]
        delimiter = delimiter_opt.value if delimiter_opt.type == "some" else ","

        quote_char_opt = config["quoteChar"]
        quote_char = quote_char_opt.value if quote_char_opt.type == "some" else '"'

        escape_char_opt = config["escapeChar"]
        escape_char = escape_char_opt.value if escape_char_opt.type == "some" else None

        has_header = config["hasHeader"]

        null_string_opt = config["nullString"]
        null_string = null_string_opt.value if null_string_opt.type == "some" else ""

        skip_empty = config["skipEmptyLines"]
        trim_fields = config["trimFields"]

        # Decode blob
        text = bytes(blob).decode("utf-8-sig")  # Handles BOM

        # Use strict parser
        rows = parse_csv_strict(text, delimiter, quote_char, escape_char)
        if not rows:
            return EastArray(CsvRowType, [])

        # Get header
        if has_header:
            header = rows[0]
            data_rows = rows[1:]
        else:
            # Generate column names
            max_cols = max(len(row) for row in rows) if rows else 0
            header = [f"column_{i}" for i in range(max_cols)]
            data_rows = rows

        # Convert to East format
        result = EastArray(CsvRowType, [])
        # Row numbering: header is row 1 if present, data starts at row 2 (or row 1 if no header)
        row_offset = 2 if has_header else 1

        expected_cols = len(header)
        for row_idx, row in enumerate(data_rows):
            if skip_empty and not any(cell.strip() for cell in row):
                row_offset += 1  # Still count skipped rows
                continue

            row_number = row_idx + row_offset

            # Strict field count validation (match TypeScript error format)
            if len(row) > expected_cols:
                raise Exception(
                    f"Too many fields in row {row_number}: expected {expected_cols} columns, got {len(row)}"
                )
            if len(row) < expected_cols:
                raise Exception(
                    f"Too few fields in row {row_number}: expected {expected_cols} columns, got {len(row)}"
                )

            row_dict = EastDict(StringType, LiteralValueType)

            for col_idx, col_name in enumerate(header):
                raw_value = row[col_idx]
                if trim_fields:
                    raw_value = raw_value.strip()
                value: str | None = None if raw_value == null_string else raw_value

                # Get column type hint if available (default to String)
                col_type = "String"
                if columns_map and col_name in columns_map:
                    col_type = columns_map[col_name].type

                try:
                    row_dict[col_name] = convert_value_to_east(value, col_type)
                except ValueError as e:
                    # Match TypeScript error format
                    raise Exception(
                        f"Failed to parse value for header {col_name} in row {row_number}, "
                        f"column {col_idx + 1}: {e}"
                    ) from e
            result.append(row_dict)

        return result
    except Exception as e:
        error_str = str(e)
        # Don't wrap already-formatted errors
        if any(
            msg in error_str
            for msg in [
                "Failed to parse value",
                "Unclosed quote",
                "Too many fields",
                "Too few fields",
                "Expected delimiter or newline after closing quote",
                "Invalid escape sequence",
            ]
        ):
            raise
        raise Exception(f"CSV parse failed: {e}") from e


def csv_serialize_impl(data: EastArray, config: EastStruct) -> EastBlob:
    """Serialize data to CSV format."""
    try:
        delimiter = config["delimiter"]
        quote_char = config["quoteChar"]
        newline = config["newline"]
        include_header = config["includeHeader"]
        null_string = config["nullString"]
        always_quote = config["alwaysQuote"]

        # Validation (matching TypeScript)
        if len(quote_char) != 1:
            raise Exception(f'quoteChar must have length 1, got "{quote_char}"')
        if len(delimiter) == 0:
            raise Exception("delimiter must not be empty")

        if len(data) == 0:
            return EastBlob(b"")

        # Get column names from first row
        first_row = data[0]
        columns = list(first_row.keys())

        # Build rows manually to control newlines (Python csv module doesn't handle custom newlines well)
        lines: list[str] = []

        def escape_field(value: str) -> str:
            """Escape a field value for CSV output."""
            needs_quoting = (
                always_quote
                or delimiter in value
                or quote_char in value
                or newline in value
                or value == null_string
            )

            if not needs_quoting:
                return value

            # Escape quotes by doubling them
            escaped = value.replace(quote_char, quote_char + quote_char)
            return quote_char + escaped + quote_char

        if include_header:
            lines.append(delimiter.join(escape_field(col) for col in columns))

        for row in data:
            row_values = []
            for col in columns:
                val = row.get(col)
                if val is None or (hasattr(val, "type") and val.type == "Null"):
                    # Null values output as null_string without escaping (matches TypeScript)
                    row_values.append(null_string)
                else:
                    str_val = convert_east_to_value(val)
                    # Don't escape/quote if value equals null_string - output as-is
                    if str_val == null_string:
                        row_values.append(null_string)
                    else:
                        row_values.append(escape_field(str_val))
            lines.append(delimiter.join(row_values))

        # Join with configured newline and add trailing newline
        result = newline.join(lines) + newline

        return EastBlob(result.encode("utf-8"))
    except Exception as e:
        raise Exception(f"CSV serialize failed: {e}") from e


# Platform function implementations
csv_impl = [
    PlatformFunction(
        name="csv_parse",
        inputs=[BlobType, CsvParseConfigType],
        output=CsvDataType,
        type="sync",
        fn=csv_parse_impl,
    ),
    PlatformFunction(
        name="csv_serialize",
        inputs=[CsvDataType, CsvSerializeConfigType],
        output=BlobType,
        type="sync",
        fn=csv_serialize_impl,
    ),
]

__all__ = [
    "csv_impl",
    "csv_parse_impl",
    "csv_serialize_impl",
]
