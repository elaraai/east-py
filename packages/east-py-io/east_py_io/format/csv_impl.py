"""CSV platform functions for East.

Provides CSV parsing and serialization for East programs.
"""

import csv
import io
from datetime import datetime
from typing import Any, Literal, cast

from east.runtime.platform import PlatformFunction
from east.types.types import BlobType, StringType
from east.types.values import EastArray, EastBlob, EastDict, EastStruct, EastVariant

from .types import (
    CsvDataType,
    CsvParseConfigType,
    CsvRowType,
    CsvSerializeConfigType,
    LiteralValueType,
)


def convert_value_to_east(value: Any, column_type: str | None = None) -> EastVariant:
    """Convert a Python value to East LiteralValueType variant."""
    if value is None or value == "":
        return EastVariant("Null", None)
    elif isinstance(value, bool):
        return EastVariant("Boolean", value)
    elif isinstance(value, int):
        return EastVariant("Integer", value)
    elif isinstance(value, float):
        return EastVariant("Float", value)
    elif isinstance(value, datetime):
        return EastVariant("DateTime", value)
    elif isinstance(value, bytes):
        return EastVariant("Blob", EastBlob(value))
    elif isinstance(value, str):
        # If column type is specified, use that
        if column_type == "Integer":
            try:
                return EastVariant("Integer", int(value))
            except ValueError:
                pass
        elif column_type == "Float":
            try:
                return EastVariant("Float", float(value))
            except ValueError:
                pass
        elif column_type == "Boolean":
            if value.lower() in ("true", "1", "yes"):
                return EastVariant("Boolean", True)
            elif value.lower() in ("false", "0", "no"):
                return EastVariant("Boolean", False)

        # Auto-detect type
        try:
            if "." in value:
                return EastVariant("Float", float(value))
            else:
                return EastVariant("Integer", int(value))
        except ValueError:
            pass
        # Check for boolean
        if value.lower() in ("true", "false"):
            return EastVariant("Boolean", value.lower() == "true")
        return EastVariant("String", value)
    else:
        return EastVariant("String", str(value))


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


async def csv_parse_impl(blob: EastBlob, config: EastStruct) -> EastArray:
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

        has_header = config["hasHeader"]

        null_string_opt = config["nullString"]
        null_string = null_string_opt.value if null_string_opt.type == "some" else ""

        skip_empty = config["skipEmptyLines"]
        trim_fields = config["trimFields"]

        # Decode blob
        text = bytes(blob).decode("utf-8-sig")  # Handles BOM
        reader = csv.reader(io.StringIO(text), delimiter=delimiter, quotechar=quote_char)

        rows = list(reader)
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
        for row in data_rows:
            if skip_empty and not any(cell.strip() for cell in row):
                continue

            row_dict = EastDict(StringType, LiteralValueType)
            for i, col_name in enumerate(header):
                raw_value = row[i] if i < len(row) else ""
                if trim_fields:
                    raw_value = raw_value.strip()
                value: str | None = None if raw_value == null_string else raw_value

                # Get column type hint if available
                col_type = None
                if columns_map and col_name in columns_map:
                    col_type = columns_map[col_name].type

                row_dict[col_name] = convert_value_to_east(value, col_type)
            result.append(row_dict)

        return result
    except Exception as e:
        raise Exception(f"CSV parse failed: {e}") from e


async def csv_serialize_impl(data: EastArray, config: EastStruct) -> EastBlob:
    """Serialize data to CSV format."""
    try:
        delimiter = config["delimiter"]
        quote_char = config["quoteChar"]
        newline = config["newline"]
        include_header = config["includeHeader"]
        null_string = config["nullString"]
        always_quote = config["alwaysQuote"]

        if len(data) == 0:
            return EastBlob(b"")

        # Get column names from first row
        first_row = data[0]
        columns = list(first_row.keys())

        output = io.StringIO(newline=newline if newline != "\n" else None)
        quoting = cast(Literal[0, 1, 2, 3], csv.QUOTE_ALL if always_quote else csv.QUOTE_MINIMAL)
        writer = csv.writer(output, delimiter=delimiter, quotechar=quote_char, quoting=quoting)

        if include_header:
            writer.writerow(columns)

        for row in data:
            row_values = []
            for col in columns:
                val = row.get(col)
                if val is None or (hasattr(val, "type") and val.type == "Null"):
                    row_values.append(null_string)
                else:
                    row_values.append(convert_east_to_value(val))
            writer.writerow(row_values)

        result = output.getvalue()
        # Apply custom newline if needed
        if newline != "\n":
            result = result.replace("\n", newline)

        return EastBlob(result.encode("utf-8"))
    except Exception as e:
        raise Exception(f"CSV serialize failed: {e}") from e


# Platform function implementations
csv_impl = [
    PlatformFunction(
        name="csv_parse",
        inputs=[BlobType, CsvParseConfigType],
        output=CsvDataType,
        type="async",
        fn=csv_parse_impl,
    ),
    PlatformFunction(
        name="csv_serialize",
        inputs=[CsvDataType, CsvSerializeConfigType],
        output=BlobType,
        type="async",
        fn=csv_serialize_impl,
    ),
]

__all__ = [
    "csv_impl",
    "csv_parse_impl",
    "csv_serialize_impl",
]
