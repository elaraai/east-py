#
# Copyright (c) 2025 Elara AI Pty Ltd
# Licensed under the Business Source License 1.1. See LICENSE.md for details.
#
"""Cython-accelerated CSV decode/encode.

Drop-in replacements for decode_csv_for and encode_csv_for from csv.py.
Gains from typed loop variables in byte-level parsing, fast struct
construction via fast_create_struct, and inlined helper checks.
"""

# Import setup helpers from csv.py (called once, not hot path)
from east.serialization.csv import (
    CsvError,
    CsvLocation,
    FieldInfo,
    create_field_decoder,
    create_field_encoder,
    is_option_type,
    is_supported_field_type,
    quote_field,
    resolve_parse_config,
    resolve_serialize_config,
)
from east.types.types import is_struct_type
from east.types.values import EastArray, EastNone, EastStruct, EastVariant

# Import fast struct/variant creation if available
_HAS_CY_STRUCT = False
try:
    from east.types._values_cy import cy_intern_keys, fast_create_struct
    _HAS_CY_STRUCT = True
except ImportError:
    pass

_HAS_CY_VARIANT = False
try:
    from east.types._values_cy import fast_create_variant
    _HAS_CY_VARIANT = True
except ImportError:
    pass


# =============================================================================
# Cython-accelerated parse_row
# =============================================================================


cpdef tuple cy_parse_row(const unsigned char[:] data, int offset,
                         int delim_byte, int quote_byte, int escape_byte):
    """Parse a CSV row into fields — Cython-accelerated.

    Typed loop variables (i, byte, in_quote) eliminate Python bytecode dispatch
    per byte. Memoryview access avoids bounds-check overhead.

    Returns: (fields, new_offset, is_end)
    """
    cdef list fields = []
    cdef bytearray field_buf = bytearray()
    cdef bint in_quote = False
    cdef int i = offset
    cdef int data_len = data.shape[0]
    cdef unsigned char byte

    while i < data_len:
        byte = data[i]

        if in_quote:
            if byte == <unsigned char>escape_byte and i + 1 < data_len and data[i + 1] == <unsigned char>quote_byte:
                field_buf.append(quote_byte)
                i += 2
                continue
            if byte == <unsigned char>quote_byte:
                in_quote = False
                i += 1
                continue
            field_buf.append(byte)
            i += 1
            continue

        if byte == <unsigned char>quote_byte and len(field_buf) == 0:
            in_quote = True
            i += 1
            continue

        if byte == <unsigned char>delim_byte:
            fields.append(field_buf.decode("utf-8"))
            field_buf.clear()
            i += 1
            continue

        if byte == 0x0D:  # CR
            fields.append(field_buf.decode("utf-8"))
            if i + 1 < data_len and data[i + 1] == 0x0A:
                return (fields, i + 2, False)
            return (fields, i + 1, False)

        if byte == 0x0A:  # LF
            fields.append(field_buf.decode("utf-8"))
            return (fields, i + 1, False)

        field_buf.append(byte)
        i += 1

    if in_quote:
        raise CsvError("unclosed quote at end of file")

    fields.append(field_buf.decode("utf-8"))
    return (fields, i, True)


# =============================================================================
# Inlined helpers
# =============================================================================


cdef bint _is_empty_row(list fields):
    """Check if a row is empty (all fields are empty strings)."""
    cdef int n = len(fields)
    if n == 0:
        return True
    if n == 1:
        return len(<str>fields[0]) == 0
    cdef str f
    for f in fields:
        if len(f) > 0:
            return False
    return True


cdef bint _needs_quoting(str value, str delimiter, str quote_char):
    """Check if a string needs quoting."""
    return delimiter in value or quote_char in value or "\r" in value or "\n" in value


# =============================================================================
# Cython-accelerated decode_csv_for
# =============================================================================


def cy_decode_csv_for(struct_type, config=None, _frozen=False):
    """Create a Cython-accelerated CSV decoder for Array<Struct>.

    Same interface as decode_csv_for. Gains from:
    - cy_parse_row with typed byte loop
    - fast_create_struct instead of EastStruct(dict)
    - Inlined is_empty_row check
    """
    if not is_struct_type(struct_type):
        raise ValueError("CSV decode requires a struct type")

    fields = struct_type.value
    for f in fields:
        if not is_supported_field_type(f["type"]):
            raise ValueError(f"CSV field '{f['name']}' has unsupported type")

    resolved = resolve_parse_config(config)

    delim_byte = ord(resolved.delimiter)
    quote_byte = ord(resolved.quote_char)
    escape_byte = ord(resolved.escape_char)

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

    none_variant = EastNone()
    has_header = resolved.has_header
    column_mapping = resolved.column_mapping
    skip_empty_lines = resolved.skip_empty_lines
    strict = resolved.strict

    # Pre-compute for fast struct construction
    _use_fast = _HAS_CY_STRUCT
    if _use_fast:
        _interned_keys, _key_index = cy_intern_keys(field_names)

    def decode(data):
        cdef int offset, num_fields_row, row_num, num_decoders, j
        cdef bint is_end

        # Skip UTF-8 BOM
        offset = 3 if len(data) >= 3 and data[0:3] == b"\xef\xbb\xbf" else 0

        if has_header:
            header_fields, offset, _ = cy_parse_row(data, offset, delim_byte, quote_byte, escape_byte)
            if column_mapping:
                headers = tuple(column_mapping.get(h, h) for h in header_fields)
            else:
                headers = tuple(header_fields)
        else:
            headers = field_names

        header_to_index = {h: i for i, h in enumerate(headers)}

        decoders = []
        for info in field_infos:
            idx = header_to_index.get(info.name)
            if idx is None and not info.is_optional:
                raise CsvError(f"missing required column '{info.name}'")
            decoders.append((info.name, info.is_optional, info.decoder, idx))

        if strict:
            field_name_set = set(field_names)
            for header in headers:
                if header not in field_name_set:
                    raise CsvError(f"unexpected column '{header}' in strict mode")

        cdef list result = []
        num_decoders = len(decoders)
        row_num = 1
        cdef int data_len = len(data)

        while offset < data_len:
            row_fields, offset, is_end = cy_parse_row(
                data, offset, delim_byte, quote_byte, escape_byte
            )

            if skip_empty_lines and _is_empty_row(row_fields):
                if is_end:
                    break
                continue

            num_fields_row = len(row_fields)

            if _use_fast:
                # Fast path: build values list, then fast_create_struct
                values = [None] * num_decoders
                for j in range(num_decoders):
                    name, is_optional, decoder, header_idx = decoders[j]
                    if header_idx is None:
                        values[j] = none_variant
                    elif header_idx >= num_fields_row:
                        if is_optional:
                            values[j] = none_variant
                        else:
                            raise CsvError(
                                f"row has {num_fields_row} fields, expected at least {header_idx + 1}",
                                CsvLocation(row_num, header_idx, name),
                            )
                    else:
                        values[j] = decoder(
                            row_fields[header_idx], CsvLocation(row_num, header_idx, name)
                        )
                result.append(fast_create_struct(_interned_keys, _key_index, tuple(values)))
            else:
                # Fallback: build dict per row
                row = {}
                for name, is_optional, decoder, header_idx in decoders:
                    if header_idx is None:
                        row[name] = none_variant
                    elif header_idx >= num_fields_row:
                        if is_optional:
                            row[name] = none_variant
                        else:
                            raise CsvError(
                                f"row has {num_fields_row} fields, expected at least {header_idx + 1}",
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

        return EastArray(struct_type, result)

    return decode


# =============================================================================
# Cython-accelerated encode_csv_for
# =============================================================================


def cy_encode_csv_for(struct_type, config=None):
    """Create a Cython-accelerated CSV encoder for Array<Struct>.

    Same interface as encode_csv_for. Gains from typed loop variables
    and inlined needs_quoting check.
    """
    if not is_struct_type(struct_type):
        raise ValueError("CSV encode requires a struct type")

    fields = struct_type.value
    for f in fields:
        if not is_supported_field_type(f["type"]):
            raise ValueError(f"CSV field '{f['name']}' has unsupported type")

    resolved = resolve_serialize_config(config)

    field_names = tuple(f["name"] for f in fields)
    num_fields = len(field_names)
    encoders = tuple(create_field_encoder(f["type"], resolved.null_string) for f in fields)

    delimiter = resolved.delimiter
    quote_char = resolved.quote_char
    escape_char = resolved.escape_char
    newline_bytes = resolved.newline.encode("utf-8")
    delimiter_bytes = delimiter.encode("utf-8")
    include_header = resolved.include_header
    always_quote = resolved.always_quote

    def encode(value):
        cdef int i, row_idx, num_rows
        output = bytearray()

        # Write header
        if include_header:
            for i in range(num_fields):
                if i > 0:
                    output.extend(delimiter_bytes)
                name = field_names[i]
                if always_quote or _needs_quoting(name, delimiter, quote_char):
                    output.extend(quote_field(name, quote_char, escape_char).encode("utf-8"))
                else:
                    output.extend(name.encode("utf-8"))
            if value:
                output.extend(newline_bytes)

        # Write data rows
        num_rows = len(value)
        for row_idx in range(num_rows):
            row_data = value[row_idx]

            for i in range(num_fields):
                if i > 0:
                    output.extend(delimiter_bytes)

                field_value = row_data.get(field_names[i])
                encoded = encoders[i](field_value)

                if always_quote or _needs_quoting(encoded, delimiter, quote_char):
                    encoded = quote_field(encoded, quote_char, escape_char)

                output.extend(encoded.encode("utf-8"))

            if row_idx < num_rows - 1:
                output.extend(newline_bytes)

        return bytes(output)

    return encode
