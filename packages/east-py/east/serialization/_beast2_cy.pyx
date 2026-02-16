#
# Copyright (c) 2025 Elara AI Pty Ltd
# Licensed under the Business Source License 1.1. See LICENSE.md for details.
#
"""Cython-accelerated Beast v2 decoder.

Drop-in replacement for decode_beast2_value_for from beast2.py.
Gains from typed locals in byte-parsing closures and fast C-level
imports of binary read functions.
"""

from datetime import UTC
from datetime import datetime as DateTime

import numpy as np_

from east.runtime.compiler import EAST_CAPTURES_ATTR, EAST_IR_ATTR
from east.types.types import (
    is_array_type,
    is_async_function_type,
    is_blob_type,
    is_boolean_type,
    is_datetime_type,
    is_dict_type,
    is_float_type,
    is_function_type,
    is_integer_type,
    is_matrix_type,
    is_never_type,
    is_null_type,
    is_recursive_type,
    is_ref_type,
    is_set_type,
    is_string_type,
    is_struct_type,
    is_variant_type,
    is_vector_type,
)
from east.types.values import (
    EAST_ELEMENT_TO_DTYPE,
    EastArray,
    EastBlob,
    EastDict,
    EastMatrix,
    EastRef,
    EastSet,
    EastStruct,
    EastVariant,
    EastVector,
)

# Try to import Cython-accelerated types + fast construction
_HAS_CY_STRUCT = False
_HAS_CY_VARIANT = False
try:
    from east.types._values_cy import CyEastStruct, cy_intern_keys, fast_create_struct
    _HAS_CY_STRUCT = True
except ImportError:
    pass
try:
    from east.types._values_cy import CyEastVariant, fast_create_variant
    _HAS_CY_VARIANT = True
except ImportError:
    pass

# Try to import Cython-accelerated binary utils, fall back to pure Python
try:
    from east.serialization._binary_utils_cy import (
        read_float64_le,
        read_string_utf8_varint,
        read_varint,
        read_zigzag,
    )
except ImportError:
    from east.serialization.binary_utils import (
        _py_read_float64_le as read_float64_le,
        _py_read_string_utf8_varint as read_string_utf8_varint,
        _py_read_varint as read_varint,
        _py_read_zigzag as read_zigzag,
    )


def decode_beast2_value_for(type_val, type_ctx=None, options=None):
    """Create value decoder for given type.

    Cython-accelerated version — structurally identical to the pure-Python
    implementation in beast2.py but benefits from typed local variables
    in the inner decoding closures and C-level binary read functions.

    Args:
        type_val: East type to create decoder for
        type_ctx: Stack of decoders for recursive types
        options: Decode options (platform functions for function compilation)

    Returns:
        Function that decodes values from bytes at offset with context
    """
    if type_ctx is None:
        type_ctx = []
    if options is None:
        options = {}

    if is_never_type(type_val):

        def decode_never(_buffer, _offset, _ctx):
            raise RuntimeError("Cannot decode Never type")

        return decode_never

    if is_null_type(type_val):
        return lambda buffer, offset, ctx: (None, offset)

    if is_boolean_type(type_val):

        def decode_bool(buffer, int offset, _ctx):
            return (buffer[offset] != 0, offset + 1)

        return decode_bool

    if is_integer_type(type_val):

        def decode_int(buffer, int offset, _ctx):
            return read_zigzag(buffer, offset)

        return decode_int

    if is_float_type(type_val):

        def decode_float(buffer, int offset, _ctx):
            return read_float64_le(buffer, offset)

        return decode_float

    if is_string_type(type_val):

        def decode_string(buffer, int offset, _ctx):
            return read_string_utf8_varint(buffer, offset)

        return decode_string

    if is_datetime_type(type_val):

        def decode_datetime(buffer, int offset, _ctx):
            cdef long long millis
            millis, new_offset = read_zigzag(buffer, offset)
            dt = DateTime.fromtimestamp(millis / 1000.0, tz=UTC)
            return (dt, new_offset)

        return decode_datetime

    if is_blob_type(type_val):

        def decode_blob(buffer, int offset, _ctx):
            cdef int length
            cdef int new_offset
            length, new_offset = read_varint(buffer, offset)
            if new_offset + length > len(buffer):
                raise ValueError(
                    f"Buffer underflow reading blob at offset {offset}, length {length}"
                )
            data = buffer[new_offset : new_offset + length]
            return (EastBlob(bytes(data)), new_offset + length)

        return decode_blob

    if is_vector_type(type_val):
        element_type = type_val.value
        dtype = np_.dtype(EAST_ELEMENT_TO_DTYPE[element_type.type])

        def decode_vector(buffer, int offset, _ctx):
            cdef int length
            cdef int new_offset
            cdef int byte_count
            length, new_offset = read_varint(buffer, offset)
            byte_count = length * dtype.itemsize
            if new_offset + byte_count > len(buffer):
                raise ValueError(
                    f"Buffer underflow reading vector at offset {offset}, length {length}"
                )
            data = np_.frombuffer(buffer, dtype=dtype, count=length, offset=new_offset).copy()
            return (EastVector(element_type, data), new_offset + byte_count)

        return decode_vector

    if is_matrix_type(type_val):
        element_type = type_val.value
        dtype = np_.dtype(EAST_ELEMENT_TO_DTYPE[element_type.type])

        def decode_matrix(buffer, int offset, _ctx):
            cdef int rows, cols, count, byte_count, new_offset
            rows, new_offset = read_varint(buffer, offset)
            cols, new_offset = read_varint(buffer, new_offset)
            count = rows * cols
            byte_count = count * dtype.itemsize
            if new_offset + byte_count > len(buffer):
                raise ValueError(
                    f"Buffer underflow reading matrix at offset {offset}, {rows}x{cols}"
                )
            data = np_.frombuffer(
                buffer, dtype=dtype, count=count, offset=new_offset
            ).copy().reshape(rows, cols)
            return (EastMatrix(element_type, data, rows, cols), new_offset + byte_count)

        return decode_matrix

    if is_array_type(type_val):
        element_type = type_val.value
        value_decoder = [None]  # Mutable container for recursive reference

        def decode_array(buffer, int offset, ctx):
            cdef int ref_or_inline, new_offset, length, current_offset, target_offset
            ref_or_inline, new_offset = read_varint(buffer, offset)

            # Check if this is a backreference
            if ref_or_inline > 0:
                target_offset = offset - ref_or_inline
                if target_offset not in ctx.refs:
                    raise ValueError(
                        f"Undefined backreference at offset {offset}, target {target_offset}"
                    )
                return (ctx.refs[target_offset], new_offset)

            # Inline array - create placeholder for backreferences
            result = EastArray(element_type)
            ctx.refs[new_offset] = result

            # Decode contents into a list first (avoids _check_not_iterating overhead)
            length, current_offset = read_varint(buffer, new_offset)
            items = []
            items_append = items.append
            _value_decoder = value_decoder[0]
            for _ in range(length):
                item, current_offset = _value_decoder(buffer, current_offset, ctx)
                items_append(item)

            # Extend the array with all items at once
            list.extend(result, items)

            return (result, current_offset)

        type_ctx.append(decode_array)
        value_decoder[0] = decode_beast2_value_for(element_type, type_ctx, options)
        type_ctx.pop()
        return decode_array

    if is_set_type(type_val):
        element_type = type_val.value
        key_decoder = [None]

        def decode_set(buffer, int offset, ctx):
            cdef int ref_or_inline, new_offset, length, current_offset, target_offset
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
            result = EastSet(element_type)
            ctx.refs[new_offset] = result

            # Decode contents
            length, current_offset = read_varint(buffer, new_offset)
            _key_decoder = key_decoder[0]
            for _ in range(length):
                key, current_offset = _key_decoder(buffer, current_offset, ctx)
                result.add(key)

            return (result, current_offset)

        # Push decoder onto stack before building element decoder
        type_ctx.append(decode_set)
        key_decoder[0] = decode_beast2_value_for(element_type, type_ctx, options)
        type_ctx.pop()

        return decode_set

    if is_dict_type(type_val):
        key_type = type_val.value["key"]
        value_type = type_val.value["value"]
        _key_decoder_fn = decode_beast2_value_for(key_type, type_ctx, options)
        value_decoder_dict = [None]

        def decode_dict(buffer, int offset, ctx):
            cdef int ref_or_inline, new_offset, length, current_offset, target_offset
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
            result = EastDict(key_type, value_type)
            ctx.refs[new_offset] = result

            # Decode contents
            length, current_offset = read_varint(buffer, new_offset)
            _vd = value_decoder_dict[0]
            for _ in range(length):
                k, current_offset = _key_decoder_fn(buffer, current_offset, ctx)
                v, current_offset = _vd(buffer, current_offset, ctx)
                result[k] = v

            return (result, current_offset)

        type_ctx.append(decode_dict)
        value_decoder_dict[0] = decode_beast2_value_for(value_type, type_ctx, options)
        type_ctx.pop()
        return decode_dict

    if is_ref_type(type_val):
        inner_type = type_val.value
        inner_decoder = [None]

        def decode_ref(buffer, int offset, ctx):
            cdef int ref_or_inline, new_offset, target_offset
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
            result = EastRef(None)
            ctx.refs[new_offset] = result

            # Decode the referenced value
            value, final_offset = inner_decoder[0](buffer, new_offset, ctx)
            result.value = value

            return (result, final_offset)

        # Push decoder onto stack before building inner decoder
        type_ctx.append(decode_ref)
        inner_decoder[0] = decode_beast2_value_for(inner_type, type_ctx, options)
        type_ctx.pop()

        return decode_ref

    if is_struct_type(type_val):
        field_decoders = []
        struct_keys = tuple(field["name"] for field in type_val.value)

        # Pre-compute interned keys once at decoder creation time
        if _HAS_CY_STRUCT:
            _interned_keys, _key_index = cy_intern_keys(struct_keys)

            def decode_struct(buffer, int offset, ctx):
                cdef int current_offset = offset
                values = []
                values_append = values.append
                for _, decoder in field_decoders:
                    value, current_offset = decoder(buffer, current_offset, ctx)
                    values_append(value)
                return (fast_create_struct(_interned_keys, _key_index, tuple(values)), current_offset)
        else:
            def decode_struct(buffer, int offset, ctx):
                cdef int current_offset = offset
                values = []
                values_append = values.append
                for _, decoder in field_decoders:
                    value, current_offset = decoder(buffer, current_offset, ctx)
                    values_append(value)
                return (EastStruct._from_tuples(struct_keys, tuple(values)), current_offset)

        # Push decoder onto stack before building field decoders
        type_ctx.append(decode_struct)

        # Build field decoders
        for field in type_val.value:
            field_decoders.append(
                (field["name"], decode_beast2_value_for(field["type"], type_ctx, options))
            )

        # Pop from stack after building
        type_ctx.pop()

        return decode_struct

    if is_variant_type(type_val):
        # Use a mutable container for recursive reference
        decoder_ref = [None]

        def decode_variant_recursive(buffer, offset, ctx):
            return decoder_ref[0](buffer, offset, ctx)

        # Add wrapper to type_ctx before processing cases (for recursive types)
        type_ctx.append(decode_variant_recursive)

        case_decoders = [
            (case["name"], decode_beast2_value_for(case["type"], type_ctx, options))
            for case in type_val.value
        ]

        # Pop from type_ctx after processing cases
        type_ctx.pop()

        if _HAS_CY_VARIANT:
            def decode_variant(buffer, int offset, ctx):
                cdef int tag_index, tag_offset
                tag_index, tag_offset = read_varint(buffer, offset)
                if tag_index >= len(case_decoders):
                    raise ValueError(f"Invalid variant tag {tag_index} at offset {offset}")
                case_name, decoder = case_decoders[tag_index]
                value, final_offset = decoder(buffer, tag_offset, ctx)
                return (fast_create_variant(case_name, value), final_offset)
        else:
            def decode_variant(buffer, int offset, ctx):
                cdef int tag_index, tag_offset
                tag_index, tag_offset = read_varint(buffer, offset)
                if tag_index >= len(case_decoders):
                    raise ValueError(f"Invalid variant tag {tag_index} at offset {offset}")
                case_name, decoder = case_decoders[tag_index]
                value, final_offset = decoder(buffer, tag_offset, ctx)
                return (EastVariant(case_name, value), final_offset)

        # Store actual decoder in the mutable container
        decoder_ref[0] = decode_variant

        return decode_variant

    if is_recursive_type(type_val):
        # Look up decoder from type context stack
        depth = type_val.value
        ret = type_ctx[len(type_ctx) - depth]
        if ret is None:
            raise RuntimeError("Internal error: Recursive type context not found")
        return ret

    if is_function_type(type_val):
        # Lazy import to avoid circular dependency
        from east.types.type_of_type import IRType

        ir_decoder = decode_beast2_value_for(IRType, type_ctx, options)

        # Get platform from options
        platform = options.get("platform", [])

        def decode_function(buffer, int offset, ctx):
            # Decode the IR
            ir, new_offset = ir_decoder(buffer, offset, ctx)

            # Validate it's a Function IR
            if ir["type"] != "Function":
                raise RuntimeError(f"Expected Function IR, got {ir['type']} at offset {offset}")

            # Decode capture count
            captures = ir["value"]["captures"]
            cdef int capture_count
            capture_count, new_offset = read_varint(buffer, new_offset)

            if capture_count != len(captures):
                raise RuntimeError(
                    f"Capture count mismatch: IR has {len(captures)}, data has {capture_count}"
                )

            # Decode capture values and build environment
            capture_env = {}
            for cap_var in captures:
                name = cap_var["value"]["name"]
                cap_type = cap_var["value"]["type"]

                cap_decoder = decode_beast2_value_for(cap_type, type_ctx, options)
                cap_value, new_offset = cap_decoder(buffer, new_offset, ctx)
                capture_env[name] = cap_value

            # Compile the function with capture environment
            from east.runtime.compiler import FunctionFactory, _compile_ir

            platform_list = platform or []
            platform_fns = {}
            async_platform_fns = set()

            if platform_list:
                platform_fns = {pf["name"]: pf["fn"] for pf in platform_list}

            try:
                compiled, _ = _compile_ir(ir, platform_fns, async_platform_fns, platform_list)
                if isinstance(compiled, FunctionFactory):
                    fn = compiled.make(capture_env)
                else:
                    fn = compiled
            except Exception as e:
                raise RuntimeError(f"Failed to compile decoded function: {e}") from e

            return (fn, new_offset)

        return decode_function

    if is_async_function_type(type_val):
        # Lazy import to avoid circular dependency
        from east.types.type_of_type import IRType

        ir_decoder = decode_beast2_value_for(IRType, type_ctx, options)

        # Get platform from options
        platform = options.get("platform", [])

        def decode_async_function(buffer, int offset, ctx):
            # Decode the IR
            ir, new_offset = ir_decoder(buffer, offset, ctx)

            # Validate it's an AsyncFunction IR
            if ir["type"] != "AsyncFunction":
                raise RuntimeError(
                    f"Expected AsyncFunction IR, got {ir['type']} at offset {offset}"
                )

            # Decode capture count
            captures = ir["value"]["captures"]
            cdef int capture_count
            capture_count, new_offset = read_varint(buffer, new_offset)

            if capture_count != len(captures):
                raise RuntimeError(
                    f"Capture count mismatch: IR has {len(captures)}, data has {capture_count}"
                )

            # Decode capture values and build environment
            capture_env = {}
            for cap_var in captures:
                name = cap_var["value"]["name"]
                cap_type = cap_var["value"]["type"]

                cap_decoder = decode_beast2_value_for(cap_type, type_ctx, options)
                cap_value, new_offset = cap_decoder(buffer, new_offset, ctx)
                capture_env[name] = cap_value

            # Compile the async function with capture environment
            from east.runtime.compiler import FunctionFactory, _compile_ir

            platform_list = platform or []
            platform_fns = {}
            async_platform_fns = set()

            if platform_list:
                platform_fns = {pf["name"]: pf["fn"] for pf in platform_list}
                async_platform_fns = {pf["name"] for pf in platform_list if pf["type"] == "async"}

            try:
                compiled, _ = _compile_ir(ir, platform_fns, async_platform_fns, platform_list)
                if isinstance(compiled, FunctionFactory):
                    fn = compiled.make(capture_env)
                else:
                    fn = compiled
            except Exception as e:
                raise RuntimeError(f"Failed to compile decoded async function: {e}") from e

            return (fn, new_offset)

        return decode_async_function

    raise ValueError(f"Unhandled type: {type_val.type}")
