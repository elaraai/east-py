"""Beast v2 binary format for East types.

Headerless binary format with varint encoding and backreference support.
Supports Ref types for mutable reference cells with aliasing.

Key differences from Beast v1:
- No type schema header (headerless format)
- Varint encoding for integers and lengths
- Zigzag encoding for signed integers
- Little-endian floats
- Backreference support for mutable types (Array, Set, Dict, Ref)
- Full Ref type support
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC
from datetime import datetime as DateTime
from typing import Any

from east.serialization.binary_utils import (
    BufferWriter,
    read_float64_le,
    read_string_utf8_varint,
    read_varint,
    read_zigzag,
)
from east.types.primitives import Blob
from east.types.ref import Ref

# Context types for backreference tracking


class Beast2EncodeContext:
    """Context for tracking mutable references during encoding.

    Attributes:
        refs: Maps object id() to byte offset where inline content begins
    """

    def __init__(self):
        self.refs: dict[int, int] = {}


class Beast2DecodeContext:
    """Context for tracking mutable references during decoding.

    Attributes:
        refs: Maps byte offset to decoded object
    """

    def __init__(self):
        self.refs: dict[int, Any] = {}


def encode_beast2_value_to_buffer_for(
    type_val: Any, type_ctx: list[Callable] | None = None
) -> Callable[[Any, BufferWriter, Beast2EncodeContext], None]:
    """Create value encoder for given type.

    Args:
        type_val: East type to create encoder for
        type_ctx: Stack of encoders for recursive types

    Returns:
        Function that encodes values to BufferWriter with context
    """
    if type_ctx is None:
        type_ctx = []

    tag = type_val.tag

    if tag == "Never":

        def encode_never(_: Any, _writer: BufferWriter, _ctx: Beast2EncodeContext) -> None:
            raise RuntimeError("Cannot encode Never type")

        return encode_never

    if tag == "Null":
        return lambda _val, _writer, _ctx: None  # Null encodes as nothing

    if tag == "Boolean":
        return lambda val, writer, _ctx: writer.write_uint8(1 if val else 0)

    if tag == "Integer":
        return lambda val, writer, _ctx: writer.write_zigzag(val)

    if tag == "Float":
        return lambda val, writer, _ctx: writer.write_float64_le(val)

    if tag == "String":
        return lambda val, writer, _ctx: writer.write_string_utf8_varint(val)

    if tag == "DateTime":
        return lambda val, writer, _ctx: writer.write_zigzag(int(val.timestamp() * 1000))

    if tag == "Blob":

        def encode_blob(val: Blob, writer: BufferWriter, _ctx: Beast2EncodeContext) -> None:
            writer.write_varint(len(val.data))
            writer.write_bytes(val.data)

        return encode_blob

    if tag == "Array":
        value_encoder: Callable[[Any, BufferWriter, Beast2EncodeContext], None] | None = None

        def encode_array(val: Any, writer: BufferWriter, ctx: Beast2EncodeContext) -> None:
            # Check for backreference
            obj_id = id(val)
            if obj_id in ctx.refs:
                offset_diff = writer.current_offset - ctx.refs[obj_id]
                writer.write_varint(offset_diff)
                return

            # Write inline marker and register
            writer.write_varint(0)
            ctx.refs[obj_id] = writer.current_offset

            # Encode contents
            writer.write_varint(len(val))
            for item in val:
                value_encoder(item, writer, ctx)  # type: ignore

        type_ctx.append(encode_array)
        value_encoder = encode_beast2_value_to_buffer_for(type_val.value, type_ctx)  # type: ignore[attr-defined]
        type_ctx.pop()
        return encode_array

    if tag == "Set":
        key_encoder = encode_beast2_value_to_buffer_for(type_val.value, type_ctx)  # type: ignore[attr-defined]

        def encode_set(val: Any, writer: BufferWriter, ctx: Beast2EncodeContext) -> None:
            # Check for backreference
            obj_id = id(val)
            if obj_id in ctx.refs:
                offset_diff = writer.current_offset - ctx.refs[obj_id]
                writer.write_varint(offset_diff)
                return

            # Write inline marker and register
            writer.write_varint(0)
            ctx.refs[obj_id] = writer.current_offset

            # Encode contents
            writer.write_varint(len(val))
            for key in val:
                key_encoder(key, writer, ctx)

        return encode_set

    if tag == "Dict":
        dict_struct = type_val.value  # type: ignore[attr-defined]
        key_encoder = encode_beast2_value_to_buffer_for(dict_struct.key, type_ctx)  # type: ignore[attr-defined]
        value_encoder_dict: Callable[[Any, BufferWriter, Beast2EncodeContext], None] | None = None

        def encode_dict(val: Any, writer: BufferWriter, ctx: Beast2EncodeContext) -> None:
            # Check for backreference
            obj_id = id(val)
            if obj_id in ctx.refs:
                offset_diff = writer.current_offset - ctx.refs[obj_id]
                writer.write_varint(offset_diff)
                return

            # Write inline marker and register
            writer.write_varint(0)
            ctx.refs[obj_id] = writer.current_offset

            # Encode contents
            writer.write_varint(len(val))
            for k, v in val.items():
                key_encoder(k, writer, ctx)
                value_encoder_dict(v, writer, ctx)  # type: ignore

        type_ctx.append(encode_dict)
        value_encoder_dict = encode_beast2_value_to_buffer_for(dict_struct.value, type_ctx)  # type: ignore[attr-defined]
        type_ctx.pop()
        return encode_dict

    if tag == "Ref":
        inner_encoder = encode_beast2_value_to_buffer_for(type_val.value, type_ctx)  # type: ignore[attr-defined]

        def encode_ref(val: Ref, writer: BufferWriter, ctx: Beast2EncodeContext) -> None:
            # Check for backreference
            obj_id = id(val)
            if obj_id in ctx.refs:
                offset_diff = writer.current_offset - ctx.refs[obj_id]
                writer.write_varint(offset_diff)
                return

            # Write inline marker and register
            writer.write_varint(0)
            ctx.refs[obj_id] = writer.current_offset

            # Encode the referenced value
            inner_encoder(val.value, writer, ctx)

        return encode_ref

    if tag == "Struct":
        fields = type_val.value  # type: ignore[attr-defined]
        field_encoders = [
            (field.name, encode_beast2_value_to_buffer_for(field.type, type_ctx))
            for field in fields
        ]  # type: ignore[attr-defined]

        def encode_struct(val: Any, writer: BufferWriter, ctx: Beast2EncodeContext) -> None:
            # Handle both dict and EastStruct objects
            for field_name, encoder in field_encoders:
                field_value = val[field_name] if isinstance(val, dict) else getattr(val, field_name)
                encoder(field_value, writer, ctx)

        return encode_struct

    if tag == "Variant":
        cases = type_val.value  # type: ignore[attr-defined]
        case_encoders: dict[str, Callable] = {}
        case_tags: dict[str, int] = {}

        def encode_variant(val: Any, writer: BufferWriter, ctx: Beast2EncodeContext) -> None:
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
            writer.write_varint(tag_index)
            case_encoders[variant_tag](variant_value, writer, ctx)

        type_ctx.append(encode_variant)
        for i, case in enumerate(cases):  # type: ignore[attr-defined]
            case_name = case.name  # type: ignore[attr-defined]
            case_tags[case_name] = i
            case_encoders[case_name] = encode_beast2_value_to_buffer_for(case.type, type_ctx)  # type: ignore[attr-defined]
        type_ctx.pop()
        return encode_variant

    if tag == "Recursive":
        # Look up encoder from type context stack
        depth = int(type_val.value)  # type: ignore[attr-defined]
        ret = type_ctx[len(type_ctx) - depth]
        if ret is None:
            raise RuntimeError("Internal error: Recursive type context not found")
        return ret

    if tag == "Function":
        raise RuntimeError("Functions cannot be serialized")

    raise ValueError(f"Unhandled type: {tag}")


def decode_beast2_value_for(
    type_val: Any, type_ctx: list[Callable] | None = None
) -> Callable[[bytes, int, Beast2DecodeContext], tuple[Any, int]]:
    """Create value decoder for given type.

    Args:
        type_val: East type to create decoder for
        type_ctx: Stack of decoders for recursive types

    Returns:
        Function that decodes values from bytes at offset with context
    """
    if type_ctx is None:
        type_ctx = []

    from east.types.containers import EastArray, EastDict, EastSet

    tag = type_val.tag

    if tag == "Never":

        def decode_never(
            _buffer: bytes, _offset: int, _ctx: Beast2DecodeContext
        ) -> tuple[Any, int]:
            raise RuntimeError("Cannot decode Never type")

        return decode_never

    if tag == "Null":
        return lambda buffer, offset, ctx: (None, offset)

    if tag == "Boolean":

        def decode_bool(buffer: bytes, offset: int, _ctx: Beast2DecodeContext) -> tuple[bool, int]:
            return (buffer[offset] != 0, offset + 1)

        return decode_bool

    if tag == "Integer":

        def decode_int(buffer: bytes, offset: int, _ctx: Beast2DecodeContext) -> tuple[int, int]:
            return read_zigzag(buffer, offset)

        return decode_int

    if tag == "Float":

        def decode_float(
            buffer: bytes, offset: int, _ctx: Beast2DecodeContext
        ) -> tuple[float, int]:
            return read_float64_le(buffer, offset)

        return decode_float

    if tag == "String":

        def decode_string(buffer: bytes, offset: int, _ctx: Beast2DecodeContext) -> tuple[str, int]:
            return read_string_utf8_varint(buffer, offset)

        return decode_string

    if tag == "DateTime":

        def decode_datetime(
            buffer: bytes, offset: int, _ctx: Beast2DecodeContext
        ) -> tuple[DateTime, int]:
            millis, new_offset = read_zigzag(buffer, offset)
            dt = DateTime.fromtimestamp(millis / 1000.0, tz=UTC)
            return (dt, new_offset)

        return decode_datetime

    if tag == "Blob":

        def decode_blob(buffer: bytes, offset: int, _ctx: Beast2DecodeContext) -> tuple[Blob, int]:
            length, new_offset = read_varint(buffer, offset)
            if new_offset + length > len(buffer):
                raise ValueError(
                    f"Buffer underflow reading blob at offset {offset}, length {length}"
                )
            data = buffer[new_offset : new_offset + length]
            return (Blob(bytes(data)), new_offset + length)

        return decode_blob

    if tag == "Array":
        value_decoder: Callable[[bytes, int, Beast2DecodeContext], tuple[Any, int]] | None = None

        def decode_array(
            buffer: bytes, offset: int, ctx: Beast2DecodeContext
        ) -> tuple[EastArray, int]:
            ref_or_inline, new_offset = read_varint(buffer, offset)

            # Check if this is a backreference
            if ref_or_inline > 0:
                target_offset = offset - ref_or_inline
                if target_offset not in ctx.refs:
                    raise ValueError(
                        f"Undefined backreference at offset {offset}, target {target_offset}"
                    )
                return (ctx.refs[target_offset], new_offset)

            # Inline array - register at offset after varint(0)
            result = EastArray(type_val.value)  # type: ignore[attr-defined]
            ctx.refs[new_offset] = result

            # Decode contents
            length, length_offset = read_varint(buffer, new_offset)
            current_offset = length_offset
            for _ in range(length):
                item, current_offset = value_decoder(buffer, current_offset, ctx)  # type: ignore
                result.append(item)

            return (result, current_offset)

        type_ctx.append(decode_array)
        value_decoder = decode_beast2_value_for(type_val.value, type_ctx)  # type: ignore[attr-defined]
        type_ctx.pop()
        return decode_array

    if tag == "Set":
        key_decoder = decode_beast2_value_for(type_val.value, type_ctx)  # type: ignore[attr-defined]

        def decode_set(buffer: bytes, offset: int, ctx: Beast2DecodeContext) -> tuple[EastSet, int]:
            ref_or_inline, new_offset = read_varint(buffer, offset)

            # Check if this is a backreference
            if ref_or_inline > 0:
                target_offset = offset - ref_or_inline
                if target_offset not in ctx.refs:
                    raise ValueError(
                        f"Undefined backreference at offset {offset}, target {target_offset}"
                    )
                return (ctx.refs[target_offset], new_offset)

            # Inline set - register at offset after varint(0)
            result = EastSet(type_val.value)  # type: ignore[attr-defined]
            ctx.refs[new_offset] = result

            # Decode contents
            length, length_offset = read_varint(buffer, new_offset)
            current_offset = length_offset
            for _ in range(length):
                key, current_offset = key_decoder(buffer, current_offset, ctx)
                result.add(key)

            return (result, current_offset)

        return decode_set

    if tag == "Dict":
        dict_struct = type_val.value  # type: ignore[attr-defined]
        key_decoder = decode_beast2_value_for(dict_struct.key, type_ctx)  # type: ignore[attr-defined]
        value_decoder_dict: Callable[[bytes, int, Beast2DecodeContext], tuple[Any, int]] | None = (
            None
        )

        def decode_dict(
            buffer: bytes, offset: int, ctx: Beast2DecodeContext
        ) -> tuple[EastDict, int]:
            ref_or_inline, new_offset = read_varint(buffer, offset)

            # Check if this is a backreference
            if ref_or_inline > 0:
                target_offset = offset - ref_or_inline
                if target_offset not in ctx.refs:
                    raise ValueError(
                        f"Undefined backreference at offset {offset}, target {target_offset}"
                    )
                return (ctx.refs[target_offset], new_offset)

            # Inline dict - register at offset after varint(0)
            result = EastDict(dict_struct.key, dict_struct.value)  # type: ignore[attr-defined]
            ctx.refs[new_offset] = result

            # Decode contents
            length, length_offset = read_varint(buffer, new_offset)
            current_offset = length_offset
            for _ in range(length):
                k, current_offset = key_decoder(buffer, current_offset, ctx)
                v, current_offset = value_decoder_dict(buffer, current_offset, ctx)  # type: ignore
                result[k] = v

            return (result, current_offset)

        type_ctx.append(decode_dict)
        value_decoder_dict = decode_beast2_value_for(dict_struct.value, type_ctx)  # type: ignore[attr-defined]
        type_ctx.pop()
        return decode_dict

    if tag == "Ref":
        inner_decoder = decode_beast2_value_for(type_val.value, type_ctx)  # type: ignore[attr-defined]

        def decode_ref(buffer: bytes, offset: int, ctx: Beast2DecodeContext) -> tuple[Ref, int]:
            ref_or_inline, new_offset = read_varint(buffer, offset)

            # Check if this is a backreference
            if ref_or_inline > 0:
                target_offset = offset - ref_or_inline
                if target_offset not in ctx.refs:
                    raise ValueError(
                        f"Undefined backreference at offset {offset}, target {target_offset}"
                    )
                return (ctx.refs[target_offset], new_offset)

            # Inline ref - create placeholder and register at offset after varint(0)
            result = Ref(None)  # Temporary value
            ctx.refs[new_offset] = result

            # Decode the referenced value
            value, final_offset = inner_decoder(buffer, new_offset, ctx)
            result.value = value

            return (result, final_offset)

        return decode_ref

    if tag == "Struct":
        fields = type_val.value  # type: ignore[attr-defined]
        field_decoders = [
            (field.name, decode_beast2_value_for(field.type, type_ctx)) for field in fields
        ]  # type: ignore[attr-defined]

        def decode_struct(buffer: bytes, offset: int, ctx: Beast2DecodeContext) -> tuple[Any, int]:
            result = {}
            current_offset = offset
            for field_name, decoder in field_decoders:
                value, current_offset = decoder(buffer, current_offset, ctx)
                result[field_name] = value

            # Build runtime _StructTypeClass and create EastStruct instance
            from east.types.type_system import _StructTypeClass

            fields_list = [(field.name, field.type) for field in fields]  # type: ignore[attr-defined]
            runtime_type = _StructTypeClass(tuple(fields_list))
            return (runtime_type.create(**result), current_offset)

        return decode_struct

    if tag == "Variant":
        cases = type_val.value  # type: ignore[attr-defined]
        case_decoders = [
            (case.name, decode_beast2_value_for(case.type, type_ctx)) for case in cases
        ]  # type: ignore[attr-defined]

        def decode_variant(buffer: bytes, offset: int, ctx: Beast2DecodeContext) -> tuple[Any, int]:
            tag_index, tag_offset = read_varint(buffer, offset)
            if tag_index >= len(case_decoders):
                raise ValueError(f"Invalid variant tag {tag_index} at offset {offset}")
            case_name, decoder = case_decoders[tag_index]
            value, final_offset = decoder(buffer, tag_offset, ctx)

            # Build runtime _VariantTypeClass and create EastVariant instance
            from east.types.type_system import _VariantTypeClass

            cases_list = [(case.name, case.type) for case in cases]  # type: ignore[attr-defined]
            runtime_type = _VariantTypeClass(tuple(cases_list))
            return (runtime_type.create(case_name, value), final_offset)

        return decode_variant

    if tag == "Recursive":
        # Look up decoder from type context stack
        depth = int(type_val.value)  # type: ignore[attr-defined]
        ret = type_ctx[len(type_ctx) - depth]
        if ret is None:
            raise RuntimeError("Internal error: Recursive type context not found")
        return ret

    if tag == "Function":
        raise RuntimeError("Functions cannot be deserialized")

    raise ValueError(f"Unhandled type: {tag}")


def encode_beast2_for(type_val: Any) -> Callable[[Any], bytes]:
    """Create encoder for Beast v2 format (headerless).

    Args:
        type_val: East type to create encoder for

    Returns:
        Function that encodes values to Beast v2 binary format
    """
    value_encoder = encode_beast2_value_to_buffer_for(type_val)

    def encode(value: Any) -> bytes:
        writer = BufferWriter()
        ctx = Beast2EncodeContext()
        value_encoder(value, writer, ctx)
        return writer.to_bytes()

    return encode


def decode_beast2_for(type_val: Any) -> Callable[[bytes], Any]:
    """Create decoder for Beast v2 format (headerless).

    Args:
        type_val: Expected East type

    Returns:
        Function that decodes values from Beast v2 binary format
    """
    value_decoder = decode_beast2_value_for(type_val)

    def decode(data: bytes) -> Any:
        ctx = Beast2DecodeContext()
        value, offset = value_decoder(data, 0, ctx)

        # Verify all data consumed
        if offset != len(data):
            raise ValueError(
                f"Unexpected data after Beast v2 value at offset {offset} "
                f"({len(data) - offset} bytes remaining)"
            )

        return value

    return decode


__all__ = [
    "Beast2EncodeContext",
    "Beast2DecodeContext",
    "encode_beast2_value_to_buffer_for",
    "decode_beast2_value_for",
    "encode_beast2_for",
    "decode_beast2_for",
]
