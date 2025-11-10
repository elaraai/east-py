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
from east.types.primitives import Blob

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

# Magic bytes: "East" + null + magic numbers
MAGIC_BYTES = bytes([69, 97, 115, 116, 0, 234, 87, 255])


def encode_type_to_beast_buffer(type_val: Any, writer: BufferWriter) -> None:
    """Encode East type schema to Beast binary format.

    Args:
        type_val: East type to encode
        writer: BufferWriter to write to

    Raises:
        RuntimeError: For Recursive types (not supported in Beast v1)
        ValueError: For unsupported types
    """
    tag = type_val.tag

    if tag == "Recursive":
        raise RuntimeError("Beast v1 format does not support recursive types")

    if tag == "Ref":
        raise RuntimeError("Beast v1 format does not support ref types")

    type_byte = BEAST_TYPE_TO_BYTE.get(tag)
    if type_byte is None:
        raise ValueError(f"Unsupported type for Beast v1: {tag}")

    writer.write_uint8(type_byte)

    if tag == "Array" or tag == "Set":
        encode_type_to_beast_buffer(type_val.value, writer)  # type: ignore[attr-defined]
    elif tag == "Dict":
        dict_struct = type_val.value  # type: ignore[attr-defined]
        encode_type_to_beast_buffer(dict_struct.key, writer)  # type: ignore[attr-defined]
        encode_type_to_beast_buffer(dict_struct.value, writer)  # type: ignore[attr-defined]
    elif tag == "Struct":
        fields = type_val.value  # type: ignore[attr-defined]
        for field in fields:
            writer.write_uint8(1)  # Continuation byte
            writer.write_string_utf8_null(field.name)  # type: ignore[attr-defined]
            encode_type_to_beast_buffer(field.type, writer)  # type: ignore[attr-defined]
        writer.write_uint8(0)  # Terminator
    elif tag == "Variant":
        cases = type_val.value  # type: ignore[attr-defined]
        for case in cases:
            writer.write_uint8(1)  # Continuation byte
            writer.write_string_utf8_null(case.name)  # type: ignore[attr-defined]
            encode_type_to_beast_buffer(case.type, writer)  # type: ignore[attr-defined]
        writer.write_uint8(0)  # Terminator


def decode_type_beast(buffer: bytes, offset: int) -> tuple[Any, int]:
    """Decode East type schema from Beast binary format.

    Args:
        buffer: Binary data to decode from
        offset: Starting offset in buffer

    Returns:
        Tuple of (type, new_offset)

    Note:
        Auto-converts old nullable types to Variant with "notNull"/"null" cases.
    """
    from east.types.type_system import (
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

    # If nullable, wrap in Variant with "notNull" (tag 0) and "null" (tag 1)
    if nullable:
        return (VariantType([("notNull", base_type), ("null", NullType)]), offset)

    return (base_type, offset)


def encode_beast_value_to_buffer_for(type_val: Any) -> Callable[[Any, BufferWriter], None]:
    """Create value encoder for given type.

    Args:
        type_val: East type to create encoder for

    Returns:
        Function that encodes values to BufferWriter
    """

    tag = type_val.tag

    if tag == "Never":

        def encode_never(_: Any, _writer: BufferWriter) -> None:
            raise RuntimeError("Cannot encode Never type")

        return encode_never

    if tag == "Null":
        return lambda _val, _writer: None  # Null encodes as nothing

    if tag == "Boolean":
        return lambda val, writer: writer.write_uint8(1 if val else 0)

    if tag == "Integer":
        return lambda val, writer: writer.write_int64_twiddled(val)

    if tag == "Float":
        return lambda val, writer: writer.write_float64_twiddled(val)

    if tag == "String":
        return lambda val, writer: writer.write_string_utf8_null(val)

    if tag == "DateTime":
        return lambda val, writer: writer.write_int64_twiddled(int(val.timestamp() * 1000))

    if tag == "Blob":

        def encode_blob(val: Blob, writer: BufferWriter) -> None:
            # 8-byte big-endian length + raw bytes
            import struct

            length_bytes = struct.pack(">Q", len(val.data))
            writer.write_bytes(length_bytes)
            writer.write_bytes(val.data)

        return encode_blob

    if tag == "Array":
        value_encoder = encode_beast_value_to_buffer_for(type_val.value)  # type: ignore[attr-defined]

        def encode_array(val: Any, writer: BufferWriter) -> None:
            for item in val:
                writer.write_uint8(1)  # Continuation byte
                value_encoder(item, writer)
            writer.write_uint8(0)  # End marker

        return encode_array

    if tag == "Set":
        key_encoder = encode_beast_value_to_buffer_for(type_val.value)  # type: ignore[attr-defined]

        def encode_set(val: Any, writer: BufferWriter) -> None:
            for key in val:
                writer.write_uint8(1)  # Continuation byte
                key_encoder(key, writer)
            writer.write_uint8(0)  # End marker

        return encode_set

    if tag == "Dict":
        dict_struct = type_val.value  # type: ignore[attr-defined]
        key_encoder = encode_beast_value_to_buffer_for(dict_struct.key)  # type: ignore[attr-defined]
        value_encoder = encode_beast_value_to_buffer_for(dict_struct.value)  # type: ignore[attr-defined]

        def encode_dict(val: Any, writer: BufferWriter) -> None:
            for k, v in val.items():
                writer.write_uint8(1)  # Continuation byte
                key_encoder(k, writer)
                value_encoder(v, writer)
            writer.write_uint8(0)  # End marker

        return encode_dict

    if tag == "Struct":
        fields = type_val.value  # type: ignore[attr-defined]
        field_encoders = [
            (field.name, encode_beast_value_to_buffer_for(field.type)) for field in fields
        ]  # type: ignore[attr-defined]

        def encode_struct(val: Any, writer: BufferWriter) -> None:
            # Handle both dict and EastStruct objects
            for field_name, encoder in field_encoders:
                field_value = val[field_name] if isinstance(val, dict) else getattr(val, field_name)
                encoder(field_value, writer)

        return encode_struct

    if tag == "Variant":
        cases = type_val.value  # type: ignore[attr-defined]
        case_encoders = {case.name: encode_beast_value_to_buffer_for(case.type) for case in cases}  # type: ignore[attr-defined]
        case_tags = {case.name: i for i, case in enumerate(cases)}  # type: ignore[attr-defined]

        def encode_variant(val: Any, writer: BufferWriter) -> None:
            # Handle both dict format and EastVariant objects
            from east.types.structural import EastVariant

            if isinstance(val, EastVariant):
                variant_tag = val.tag
                variant_value = val.value
            else:
                # Assume dict format
                variant_tag = val["type"]
                variant_value = val["value"]

            tag_index = case_tags[variant_tag]
            writer.write_uint8(tag_index)
            case_encoders[variant_tag](variant_value, writer)

        return encode_variant

    if tag == "Recursive":
        raise RuntimeError("Beast v1 format does not support recursive types")

    if tag == "Ref":
        raise RuntimeError("Beast v1 format does not support ref types")

    if tag == "Function":
        raise RuntimeError("Functions cannot be serialized")

    raise ValueError(f"Unhandled type: {tag}")


def decode_beast_value_for(type_val: Any) -> Callable[[bytes, int], tuple[Any, int]]:
    """Create value decoder for given type.

    Args:
        type_val: East type to create decoder for

    Returns:
        Function that decodes values from bytes at offset
    """
    from east.types.containers import EastArray, EastDict, EastSet

    tag = type_val.tag

    if tag == "Never":

        def decode_never(_buffer: bytes, _offset: int) -> tuple[Any, int]:
            raise RuntimeError("Cannot decode Never type")

        return decode_never

    if tag == "Null":
        return lambda buffer, offset: (None, offset)

    if tag == "Boolean":

        def decode_bool(buffer: bytes, offset: int) -> tuple[bool, int]:
            return (buffer[offset] != 0, offset + 1)

        return decode_bool

    if tag == "Integer":
        return read_int64_twiddled

    if tag == "Float":
        return read_float64_twiddled

    if tag == "String":
        return read_string_utf8_null

    if tag == "DateTime":

        def decode_datetime(buffer: bytes, offset: int) -> tuple[DateTime, int]:
            millis, new_offset = read_int64_twiddled(buffer, offset)
            dt = DateTime.fromtimestamp(millis / 1000.0, tz=UTC)
            return (dt, new_offset)

        return decode_datetime

    if tag == "Blob":

        def decode_blob(buffer: bytes, offset: int) -> tuple[Blob, int]:
            import struct

            length = struct.unpack_from(">Q", buffer, offset)[0]
            offset += 8
            data = buffer[offset : offset + length]
            return (Blob(bytes(data)), offset + length)

        return decode_blob

    if tag == "Array":
        value_decoder = decode_beast_value_for(type_val.value)  # type: ignore[attr-defined]

        def decode_array(buffer: bytes, offset: int) -> tuple[EastArray, int]:
            items = []
            while buffer[offset] == 1:
                offset += 1
                item, offset = value_decoder(buffer, offset)
                items.append(item)
            if buffer[offset] != 0:
                raise ValueError(f"Invalid continuation byte {buffer[offset]} at offset {offset}")
            offset += 1
            return (EastArray(type_val.value, items), offset)  # type: ignore[attr-defined]

        return decode_array

    if tag == "Set":
        key_decoder = decode_beast_value_for(type_val.value)  # type: ignore[attr-defined]

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
            return (EastSet(type_val.value, keys), offset)  # type: ignore[attr-defined]

        return decode_set

    if tag == "Dict":
        dict_struct = type_val.value  # type: ignore[attr-defined]
        key_decoder = decode_beast_value_for(dict_struct.key)  # type: ignore[attr-defined]
        value_decoder = decode_beast_value_for(dict_struct.value)  # type: ignore[attr-defined]

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
            return (EastDict(dict_struct.key, dict_struct.value, entries), offset)  # type: ignore[attr-defined]

        return decode_dict

    if tag == "Struct":
        fields = type_val.value  # type: ignore[attr-defined]
        field_decoders = [(field.name, decode_beast_value_for(field.type)) for field in fields]  # type: ignore[attr-defined]

        def decode_struct(buffer: bytes, offset: int) -> tuple[Any, int]:
            result = {}
            for field_name, decoder in field_decoders:
                value, offset = decoder(buffer, offset)
                result[field_name] = value

            # Build runtime _StructTypeClass and create EastStruct instance
            from east.types.type_system import _StructTypeClass

            fields_list = [(field.name, field.type) for field in fields]  # type: ignore[attr-defined]
            runtime_type = _StructTypeClass(tuple(fields_list))
            return (runtime_type.create(**result), offset)

        return decode_struct

    if tag == "Variant":
        cases = type_val.value  # type: ignore[attr-defined]
        case_decoders = [(case.name, decode_beast_value_for(case.type)) for case in cases]  # type: ignore[attr-defined]

        def decode_variant(buffer: bytes, offset: int) -> tuple[Any, int]:
            tag_index = buffer[offset]
            offset += 1
            if tag_index >= len(case_decoders):
                raise ValueError(f"Invalid variant tag {tag_index} at offset {offset - 1}")
            case_name, decoder = case_decoders[tag_index]
            value, offset = decoder(buffer, offset)

            # Build runtime _VariantTypeClass and create EastVariant instance
            from east.types.type_system import _VariantTypeClass

            cases_list = [(case.name, case.type) for case in cases]  # type: ignore[attr-defined]
            runtime_type = _VariantTypeClass(tuple(cases_list))
            return (runtime_type.create(case_name, value), offset)

        return decode_variant

    if tag == "Recursive":
        raise RuntimeError("Beast v1 format does not support recursive types")

    if tag == "Function":
        raise RuntimeError("Functions cannot be deserialized")

    raise ValueError(f"Unhandled type: {tag}")


def encode_beast_for(type_val: Any) -> Callable[[Any], bytes]:
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


def decode_beast_for(type_val: Any) -> Callable[[bytes], Any]:
    """Create decoder for Beast v1 format with type validation.

    Args:
        type_val: Expected East type

    Returns:
        Function that decodes and validates values
    """
    value_decoder = decode_beast_value_for(type_val)

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

        # Verify type matches (use direct type equality, not value equality)
        if decoded_type != type_val:
            from east.serialization.east_printer import print_type

            raise ValueError(
                f"Type mismatch: expected {print_type(type_val)}, "
                f"got {print_type(decoded_type)}"
            )

        # Decode value
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
