#
# Copyright (c) 2025 Elara AI Pty Ltd
# Licensed under the Business Source License 1.1. See LICENSE.md for details.
#
"""Beast v1 binary format for East types.

Binary format with byte-ordering preservation for database indexing.
Auto-converts old nullable types to Variant on decode.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC
from datetime import datetime as DateTime
from typing import Any

from east.serialization.binary_utils import (
    BufferWriter,
    read_float64_twiddled,
    read_int64_twiddled,
    read_string_utf8_null,
)
from east.types.types import (
    EastType,
    is_array_type,
    is_blob_type,
    is_boolean_type,
    is_datetime_type,
    is_dict_type,
    is_float_type,
    is_function_type,
    is_integer_type,
    is_never_type,
    is_null_type,
    is_recursive_type,
    is_ref_type,
    is_set_type,
    is_string_type,
    is_struct_type,
    is_variant_type,
)
from east.types.values import (
    EastArray,
    EastBlob,
    EastDict,
    EastSet,
    EastStruct,
    EastVariant,
)

# Beast v1 type tags (0-13) with nullable flag at bit 7
BEAST_TYPE_TO_BYTE = {
    "Array": 0,
    "Blob": 1,
    "Boolean": 2,
    "DateTime": 3,
    "Dict": 4,
    "Float": 5,
    "Integer": 6,
    # Node: 7 - Not implemented
    "Null": 8,
    "Set": 9,
    "String": 10,
    "Struct": 11,
    # Tree: 12 - Not implemented
    "Variant": 13,
}

BEAST_BYTE_TO_TYPE = [
    "Array",
    "Blob",
    "Boolean",
    "DateTime",
    "Dict",
    "Float",
    "Integer",
    None,  # 7 - Node (not implemented)
    "Null",
    "Set",
    "String",
    "Struct",
    None,  # 12 - Tree (not implemented)
    "Variant",
]

# Magic bytes: "East" + east_null + magic numbers
MAGIC_BYTES = bytes([69, 97, 115, 116, 0, 234, 87, 255])


def _normalize_beast_type(type_val: EastType) -> EastType:
    """Normalize a Beast-decoded type to standard East ordering.

    This sorts variant cases and struct fields alphabetically by name,
    matching East's canonical type representation.

    Only used for type comparison in Beast decoding - the original
    ordering is preserved for value decoding (to match byte layout).
    """
    from east.types.types import (
        ArrayType,
        DictType,
        SetType,
        StructType,
        VariantType,
        is_array_type,
        is_dict_type,
        is_set_type,
        is_struct_type,
        is_variant_type,
    )

    if is_array_type(type_val):
        return ArrayType(_normalize_beast_type(type_val.value))

    if is_set_type(type_val):
        return SetType(_normalize_beast_type(type_val.value))

    if is_dict_type(type_val):
        return DictType(
            _normalize_beast_type(type_val.value["key"]),
            _normalize_beast_type(type_val.value["value"]),
        )

    if is_struct_type(type_val):
        # Struct fields preserve declaration order (not sorted)
        # Only normalize the field types recursively
        return StructType([(f["name"], _normalize_beast_type(f["type"])) for f in type_val.value])

    if is_variant_type(type_val):
        # Sort cases by name and normalize their types
        sorted_cases = sorted(type_val.value, key=lambda c: c["name"])
        return VariantType([(c["name"], _normalize_beast_type(c["type"])) for c in sorted_cases])

    # Primitives and other types pass through unchanged
    return type_val


def encode_type_to_beast_buffer(type_val: EastType, writer: BufferWriter) -> None:
    """Encode East type schema to Beast binary format.

    Args:
        type_val: East type to encode
        writer: BufferWriter to write to

    Raises:
        RuntimeError: For Recursive types (not supported in Beast v1)
        ValueError: For unsupported types
    """
    if is_recursive_type(type_val):
        raise RuntimeError("Beast v1 format does not support recursive types")

    if is_ref_type(type_val):
        raise RuntimeError("Beast v1 format does not support Ref types")

    type_byte = BEAST_TYPE_TO_BYTE.get(type_val.type)
    if type_byte is None:
        raise ValueError(f"Unsupported type for Beast v1: {type_val.type}")

    writer.write_uint8(type_byte)

    if is_array_type(type_val) or is_set_type(type_val):
        encode_type_to_beast_buffer(type_val.value, writer)
    elif is_dict_type(type_val):
        encode_type_to_beast_buffer(type_val.value["key"], writer)
        encode_type_to_beast_buffer(type_val.value["value"], writer)
    elif is_struct_type(type_val):
        for field in type_val.value:
            writer.write_uint8(1)  # Continuation byte
            writer.write_string_utf8_null(field["name"])
            encode_type_to_beast_buffer(field["type"], writer)
        writer.write_uint8(0)  # Terminator
    elif is_variant_type(type_val):
        for case in type_val.value:
            writer.write_uint8(1)  # Continuation byte
            writer.write_string_utf8_null(case["name"])
            encode_type_to_beast_buffer(case["type"], writer)
        writer.write_uint8(0)  # Terminator


def decode_type_beast(buffer: bytes, offset: int) -> tuple[EastType, int]:
    """Decode East type schema from Beast binary format.

    Args:
        buffer: Binary data to decode from
        offset: Starting offset in buffer

    Returns:
        Tuple of (type, new_offset)

    Note:
        Auto-converts old nullable types to Variant with "notNull"/"null" cases.
    """
    from east.types.types import (
        ArrayType,
        BlobType,
        BooleanType,
        DateTimeType,
        DictType,
        FloatType,
        IntegerType,
        NullType,
        SetType,
        StringType,
        StructType,
        VariantType,
    )

    type_byte = buffer[offset]
    offset += 1

    # Check for nullable flag (bit 7)
    nullable = type_byte >= 128
    actual_type_byte = type_byte - 128 if nullable else type_byte

    type_name = BEAST_BYTE_TO_TYPE[actual_type_byte]
    if type_name is None:
        raise ValueError(f"Invalid type byte 0x{type_byte:02x} at offset {offset - 1}")

    # Decode base type
    if type_name == "Null":
        base_type = NullType
    elif type_name == "Boolean":
        base_type = BooleanType
    elif type_name == "Integer":
        base_type = IntegerType
    elif type_name == "Float":
        base_type = FloatType
    elif type_name == "DateTime":
        base_type = DateTimeType
    elif type_name == "String":
        base_type = StringType
    elif type_name == "Blob":
        base_type = BlobType
    elif type_name == "Array":
        element_type, offset = decode_type_beast(buffer, offset)
        base_type = ArrayType(element_type)
    elif type_name == "Set":
        key_type, offset = decode_type_beast(buffer, offset)
        base_type = SetType(key_type)
    elif type_name == "Dict":
        key_type, offset = decode_type_beast(buffer, offset)
        value_type, offset = decode_type_beast(buffer, offset)
        base_type = DictType(key_type, value_type)
    elif type_name == "Struct":
        fields = []
        while buffer[offset] == 1:
            offset += 1
            field_name, offset = read_string_utf8_null(buffer, offset)
            field_type, offset = decode_type_beast(buffer, offset)
            fields.append((field_name, field_type))
        if buffer[offset] != 0:
            raise ValueError(
                f"Unexpected struct field separator {buffer[offset]} at offset {offset}"
            )
        offset += 1
        base_type = StructType(fields)
    elif type_name == "Variant":
        cases = []
        while buffer[offset] == 1:
            offset += 1
            case_name, offset = read_string_utf8_null(buffer, offset)
            case_type, offset = decode_type_beast(buffer, offset)
            cases.append((case_name, case_type))
        if buffer[offset] != 0:
            raise ValueError(
                f"Unexpected variant case separator {buffer[offset]} at offset {offset}"
            )
        offset += 1
        base_type = VariantType(cases)
    else:
        raise ValueError(f"Unhandled type: {type_name}")

    # If nullable, wrap in Variant with "some" (tag 0) and "none" (tag 1)
    # Old nullable: tag 0 = has value, tag 1 = null
    # So we need: index 0 = some, index 1 = none (to match byte encoding)
    # NOTE: We create EastVariant directly to preserve tag order for value decoding.
    # VariantType() sorts alphabetically which would break the tag-to-case mapping.
    # The type is normalized before comparison with user-expected types.
    if nullable:
        nullable_variant: EastType = EastVariant(
            "Variant",
            [
                {"name": "some", "type": base_type},
                {"name": "none", "type": NullType},
            ],
        )
        return (nullable_variant, offset)

    return (base_type, offset)


def encode_beast_value_to_buffer_for(type_val: EastType) -> Callable[[Any, BufferWriter], None]:
    """Create value encoder for given type.

    Args:
        type_val: East type to create encoder for

    Returns:
        Function that encodes values to BufferWriter
    """
    if is_never_type(type_val):

        def encode_never(_: Any, _writer: BufferWriter) -> None:
            raise RuntimeError("Cannot encode Never type")

        return encode_never

    if is_null_type(type_val):
        return lambda _val, _writer: None  # Null encodes as nothing

    if is_boolean_type(type_val):
        return lambda val, writer: writer.write_uint8(1 if val else 0)

    if is_integer_type(type_val):
        return lambda val, writer: writer.write_int64_twiddled(val)

    if is_float_type(type_val):
        return lambda val, writer: writer.write_float64_twiddled(val)

    if is_string_type(type_val):
        return lambda val, writer: writer.write_string_utf8_null(val)

    if is_datetime_type(type_val):
        return lambda val, writer: writer.write_int64_twiddled(int(val.timestamp() * 1000))

    if is_blob_type(type_val):

        def encode_blob(val: EastBlob, writer: BufferWriter) -> None:
            # 8-byte big-endian length + raw bytes
            import struct

            length_bytes = struct.pack(">Q", len(val.data))
            writer.write_bytes(length_bytes)
            writer.write_bytes(val.data)

        return encode_blob

    if is_array_type(type_val):
        value_encoder = encode_beast_value_to_buffer_for(type_val.value)

        def encode_array(val: Any, writer: BufferWriter) -> None:
            for item in val:
                writer.write_uint8(1)  # Continuation byte
                value_encoder(item, writer)
            writer.write_uint8(0)  # End marker

        return encode_array

    if is_set_type(type_val):
        key_encoder = encode_beast_value_to_buffer_for(type_val.value)

        def encode_set(val: Any, writer: BufferWriter) -> None:
            for key in val:
                writer.write_uint8(1)  # Continuation byte
                key_encoder(key, writer)
            writer.write_uint8(0)  # End marker

        return encode_set

    if is_dict_type(type_val):
        key_encoder = encode_beast_value_to_buffer_for(type_val.value["key"])
        value_encoder = encode_beast_value_to_buffer_for(type_val.value["value"])

        def encode_dict(val: Any, writer: BufferWriter) -> None:
            for k, v in val.items():
                writer.write_uint8(1)  # Continuation byte
                key_encoder(k, writer)
                value_encoder(v, writer)
            writer.write_uint8(0)  # End marker

        return encode_dict

    if is_struct_type(type_val):
        field_encoders = [
            (field["name"], encode_beast_value_to_buffer_for(field["type"]))
            for field in type_val.value
        ]

        def encode_struct(val: Any, writer: BufferWriter) -> None:
            # Handle both dict and EastStruct objects
            for field_name, encoder in field_encoders:
                encoder(val[field_name], writer)

        return encode_struct

    if is_variant_type(type_val):
        case_encoders = {
            case["name"]: encode_beast_value_to_buffer_for(case["type"]) for case in type_val.value
        }
        case_tags = {case["name"]: i for i, case in enumerate(type_val.value)}

        def encode_variant(val: Any, writer: BufferWriter) -> None:
            # Variants use .type and .value properties
            variant_tag = val.type
            variant_value = val.value

            tag_index = case_tags[variant_tag]
            writer.write_uint8(tag_index)
            case_encoders[variant_tag](variant_value, writer)

        return encode_variant

    if is_recursive_type(type_val):
        raise RuntimeError("Beast v1 format does not support recursive types")

    if is_ref_type(type_val):
        raise RuntimeError("Beast v1 format does not support Ref types")

    if is_function_type(type_val):
        raise RuntimeError("Functions cannot be serialized")

    raise ValueError(f"Unhandled type: {type_val.type}")


def decode_beast_value_for(type_val: EastType) -> Callable[[bytes, int], tuple[Any, int]]:
    """Create value decoder for given type.

    Args:
        type_val: East type to create decoder for

    Returns:
        Function that decodes values from bytes at offset
    """
    if is_never_type(type_val):

        def decode_never(_buffer: bytes, _offset: int) -> tuple[Any, int]:
            raise RuntimeError("Cannot decode Never type")

        return decode_never

    if is_null_type(type_val):
        return lambda buffer, offset: (None, offset)

    if is_boolean_type(type_val):

        def decode_bool(buffer: bytes, offset: int) -> tuple[bool, int]:
            return (buffer[offset] != 0, offset + 1)

        return decode_bool

    if is_integer_type(type_val):
        return read_int64_twiddled

    if is_float_type(type_val):
        return read_float64_twiddled

    if is_string_type(type_val):
        return read_string_utf8_null

    if is_datetime_type(type_val):

        def decode_datetime(buffer: bytes, offset: int) -> tuple[DateTime, int]:
            millis, new_offset = read_int64_twiddled(buffer, offset)
            dt = DateTime.fromtimestamp(millis / 1000.0, tz=UTC)
            return (dt, new_offset)

        return decode_datetime

    if is_blob_type(type_val):

        def decode_blob(buffer: bytes, offset: int) -> tuple[EastBlob, int]:
            import struct

            length = struct.unpack_from(">Q", buffer, offset)[0]
            offset += 8
            data = buffer[offset : offset + length]
            return (EastBlob(bytes(data)), offset + length)

        return decode_blob

    if is_array_type(type_val):
        element_type = type_val.value
        value_decoder = decode_beast_value_for(element_type)

        def decode_array(buffer: bytes, offset: int) -> tuple[EastArray, int]:
            items = []
            while buffer[offset] == 1:
                offset += 1
                item, offset = value_decoder(buffer, offset)
                items.append(item)
            if buffer[offset] != 0:
                raise ValueError(f"Invalid continuation byte {buffer[offset]} at offset {offset}")
            offset += 1
            return (EastArray(element_type, items), offset)

        return decode_array

    if is_set_type(type_val):
        element_type = type_val.value
        key_decoder = decode_beast_value_for(element_type)

        def decode_set(buffer: bytes, offset: int) -> tuple[EastSet, int]:
            keys = []
            while buffer[offset] == 1:
                offset += 1
                key, offset = key_decoder(buffer, offset)
                keys.append(key)
            if buffer[offset] != 0:
                raise ValueError(
                    f"Unexpected set continuation byte {buffer[offset]} at offset {offset}"
                )
            offset += 1
            return (EastSet(element_type, keys), offset)

        return decode_set

    if is_dict_type(type_val):
        key_type = type_val.value["key"]
        value_type = type_val.value["value"]
        key_decoder = decode_beast_value_for(key_type)
        value_decoder = decode_beast_value_for(value_type)

        def decode_dict(buffer: bytes, offset: int) -> tuple[EastDict, int]:
            entries = {}
            while buffer[offset] == 1:
                offset += 1
                k, offset = key_decoder(buffer, offset)
                v, offset = value_decoder(buffer, offset)
                entries[k] = v
            if buffer[offset] != 0:
                raise ValueError(
                    f"Unexpected dict continuation byte {buffer[offset]} at offset {offset}"
                )
            offset += 1
            return (EastDict(key_type, value_type, entries), offset)

        return decode_dict

    if is_struct_type(type_val):
        field_decoders = [
            (field["name"], decode_beast_value_for(field["type"])) for field in type_val.value
        ]
        struct_keys: tuple[str, ...] = tuple(name for name, _ in field_decoders)

        def decode_struct(buffer: bytes, offset: int) -> tuple[Any, int]:
            values = []
            for _, decoder in field_decoders:
                value, offset = decoder(buffer, offset)
                values.append(value)
            return (EastStruct._from_tuples(struct_keys, tuple(values)), offset)

        return decode_struct

    if is_variant_type(type_val):
        case_decoders = [
            (case["name"], decode_beast_value_for(case["type"])) for case in type_val.value
        ]

        def decode_variant(buffer: bytes, offset: int) -> tuple[Any, int]:
            tag_index = buffer[offset]
            offset += 1
            if tag_index >= len(case_decoders):
                raise ValueError(f"Invalid variant tag {tag_index} at offset {offset - 1}")
            case_name, decoder = case_decoders[tag_index]
            value, offset = decoder(buffer, offset)

            # Variant values are hashable EastVariant instances
            return (EastVariant(case_name, value), offset)

        return decode_variant

    if is_recursive_type(type_val):
        raise RuntimeError("Beast v1 format does not support recursive types")

    if is_function_type(type_val):
        raise RuntimeError("Functions cannot be deserialized")

    raise ValueError(f"Unhandled type: {type_val.type}")


def encode_beast_for(type_val: EastType) -> Callable[[Any], bytes]:
    """Create encoder for Beast v1 format with header.

    Args:
        type_val: East type to create encoder for

    Returns:
        Function that encodes values to Beast binary format
    """
    value_encoder = encode_beast_value_to_buffer_for(type_val)

    def encode(value: Any) -> bytes:
        writer = BufferWriter()
        writer.write_bytes(MAGIC_BYTES)
        encode_type_to_beast_buffer(type_val, writer)
        value_encoder(value, writer)
        return writer.to_bytes()

    return encode


def decode_beast(data: bytes) -> dict:
    """Decode Beast v1 format without type checking.

    Args:
        data: Binary data to decode

    Returns:
        Dict with 'type' and 'value' keys
    """
    # Verify magic bytes
    if len(data) < len(MAGIC_BYTES):
        raise ValueError(f"Data too short for Beast format: {len(data)} bytes")

    for i in range(len(MAGIC_BYTES)):
        if data[i] != MAGIC_BYTES[i]:
            raise ValueError(
                f"Invalid Beast magic bytes at offset {i}: "
                f"expected 0x{MAGIC_BYTES[i]:02x}, got 0x{data[i]:02x}"
            )

    # Decode type schema
    offset = len(MAGIC_BYTES)
    decoded_type, offset = decode_type_beast(data, offset)

    # Decode value
    value_decoder = decode_beast_value_for(decoded_type)
    value, offset = value_decoder(data, offset)

    # Verify all data consumed
    if offset != len(data):
        raise ValueError(
            f"Unexpected data after Beast value at offset {offset} "
            f"({len(data) - offset} bytes remaining)"
        )

    return {"type": decoded_type, "value": value}


def decode_beast_for(type_val: EastType) -> Callable[[bytes], Any]:
    """Create decoder for Beast v1 format with type validation.

    Args:
        type_val: Expected East type

    Returns:
        Function that decodes and validates values
    """

    def decode(data: bytes) -> Any:
        # Verify magic bytes
        if len(data) < len(MAGIC_BYTES):
            raise ValueError(f"Data too short for Beast format: {len(data)} bytes")

        for i in range(len(MAGIC_BYTES)):
            if data[i] != MAGIC_BYTES[i]:
                raise ValueError(
                    f"Invalid Beast magic bytes at offset {i}: "
                    f"expected 0x{MAGIC_BYTES[i]:02x}, got 0x{data[i]:02x}"
                )

        # Decode type schema
        offset = len(MAGIC_BYTES)
        decoded_type, offset = decode_type_beast(data, offset)

        # Normalize decoded type for comparison (sorts variant cases/struct fields)
        # but keep original for value decoding (preserves byte-level tag ordering)
        normalized_type = _normalize_beast_type(decoded_type)

        # Verify type matches
        from east.types.types import is_type_equal

        if not is_type_equal(normalized_type, type_val):
            from east.serialization.east_printer import print_type

            raise ValueError(
                f"Type mismatch: expected {print_type(type_val)}, "
                f"got {print_type(normalized_type)}"
            )

        # Decode value using original decoded_type (correct tag ordering)
        value_decoder = decode_beast_value_for(decoded_type)
        value, offset = value_decoder(data, offset)

        # Verify all data consumed
        if offset != len(data):
            raise ValueError(
                f"Unexpected data after Beast value at offset {offset} "
                f"({len(data) - offset} bytes remaining)"
            )

        return value

    return decode


__all__ = [
    "MAGIC_BYTES",
    "encode_beast_for",
    "decode_beast",
    "decode_beast_for",
]
