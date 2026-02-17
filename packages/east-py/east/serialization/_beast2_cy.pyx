# cython: boundscheck=False, wraparound=False, cdivision=True
#
# Copyright (c) 2025 Elara AI Pty Ltd
# Licensed under the Business Source License 1.1. See LICENSE.md for details.
#
"""Monolithic C-level Beast v2 decoder.

Drop-in replacement for decode_beast2_value_for from beast2.py.
Uses a single recursive cdef function with int* offset to eliminate
Python function call overhead and tuple allocation in the decode path.
"""

from datetime import UTC
from datetime import datetime as DateTime

from cpython.unicode cimport PyUnicode_DecodeUTF8
from libc.string cimport memcpy

import numpy as np_

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


# ─── Caches ───────────────────────────────────────────────────────────────

cdef dict _struct_cache = {}
cdef dict _variant_cache = {}
cdef dict _dict_cache = {}


# ─── Inline binary read functions ─────────────────────────────────────────

cdef inline long long read_varint_fast(const unsigned char[:] buf, int* offset) except? -1:
    """Read unsigned varint, advance offset in place."""
    cdef unsigned long long result = 0
    cdef int shift = 0
    cdef int off = offset[0]
    cdef int buf_len = buf.shape[0]
    cdef unsigned char byte

    while off < buf_len:
        byte = buf[off]
        off += 1
        result |= (<unsigned long long>(byte & 0x7F)) << shift
        if byte < 0x80:
            offset[0] = off
            return <long long>result
        shift += 7

    raise ValueError(f"Buffer underflow reading varint at offset {offset[0]}")


cdef inline long long read_zigzag_fast(const unsigned char[:] buf, int* offset) except? -1:
    """Read zigzag-encoded signed varint, advance offset in place."""
    cdef unsigned long long result = 0
    cdef int shift = 0
    cdef int off = offset[0]
    cdef int buf_len = buf.shape[0]
    cdef unsigned char byte

    while off < buf_len:
        byte = buf[off]
        off += 1
        result |= (<unsigned long long>(byte & 0x7F)) << shift
        if byte < 0x80:
            offset[0] = off
            return <long long>(result >> 1) ^ -<long long>(result & 1)
        shift += 7

    raise ValueError(f"Buffer underflow reading zigzag at offset {offset[0]}")


cdef inline double read_float64_fast(const unsigned char[:] buf, int* offset) except? -1.0:
    """Read 8-byte LE double, advance offset in place.

    Uses memcpy for unaligned-safe read. Assumes little-endian platform.
    """
    cdef int off = offset[0]
    if off + 8 > buf.shape[0]:
        raise ValueError(f"Buffer underflow reading float64 at offset {off}")
    cdef double value
    memcpy(&value, &buf[off], 8)
    offset[0] = off + 8
    return value


cdef inline str read_string_fast(const unsigned char[:] buf, int* offset):
    """Read varint-prefixed UTF-8 string, advance offset in place."""
    cdef unsigned long long length = 0
    cdef int shift = 0
    cdef int off = offset[0]
    cdef int buf_len = buf.shape[0]
    cdef unsigned char byte
    cdef int end

    # Inline varint for length
    while off < buf_len:
        byte = buf[off]
        off += 1
        length |= (<unsigned long long>(byte & 0x7F)) << shift
        if byte < 0x80:
            break
        shift += 7
    else:
        raise ValueError(f"Buffer underflow reading string length at offset {offset[0]}")

    end = off + <int>length
    if end > buf_len:
        raise ValueError(f"Buffer underflow reading string, length {length}")

    # Decode UTF-8 directly from buffer pointer — avoids intermediate bytes copy
    cdef str s = PyUnicode_DecodeUTF8(<const char*>&buf[off], <Py_ssize_t>length, NULL)
    offset[0] = end
    return s


# ─── Type-specific decoders ──────────────────────────────────────────────

cdef object decode_struct_value(const unsigned char[:] buf, int* offset,
                                object type_val, list type_ctx, object ctx, dict options):
    cdef object cache_key = id(type_val)
    cached = _struct_cache.get(cache_key)
    cdef tuple field_types
    cdef int i, n

    if cached is None:
        fields = type_val.value
        struct_keys = tuple(f["name"] for f in fields)
        field_types = tuple(f["type"] for f in fields)
        if _HAS_CY_STRUCT:
            interned_keys, key_index = cy_intern_keys(struct_keys)
            cached = (interned_keys, key_index, field_types)
        else:
            cached = (struct_keys, None, field_types)
        _struct_cache[cache_key] = cached

    keys_or_interned, key_index, field_types = cached

    type_ctx.append(type_val)

    n = len(field_types)
    cdef list values = [None] * n
    for i in range(n):
        values[i] = decode_value(buf, offset, field_types[i], type_ctx, ctx, options)

    type_ctx.pop()

    if _HAS_CY_STRUCT:
        return fast_create_struct(keys_or_interned, key_index, tuple(values))
    else:
        return EastStruct._from_tuples(keys_or_interned, tuple(values))


cdef object decode_variant_value(const unsigned char[:] buf, int* offset,
                                 object type_val, list type_ctx, object ctx, dict options):
    cdef object cache_key = id(type_val)
    cached = _variant_cache.get(cache_key)
    cdef tuple case_names, case_types

    if cached is None:
        cases = type_val.value
        case_names = tuple(c["name"] for c in cases)
        case_types = tuple(c["type"] for c in cases)
        cached = (case_names, case_types)
        _variant_cache[cache_key] = cached

    case_names, case_types = cached

    cdef long long tag_index = read_varint_fast(buf, offset)

    if tag_index < 0 or tag_index >= len(case_names):
        raise ValueError(f"Invalid variant tag {tag_index}")

    cdef str case_name = case_names[tag_index]
    case_type = case_types[tag_index]

    type_ctx.append(type_val)
    value = decode_value(buf, offset, case_type, type_ctx, ctx, options)
    type_ctx.pop()

    if _HAS_CY_VARIANT:
        return fast_create_variant(case_name, value)
    else:
        return EastVariant(case_name, value)


cdef object decode_array_value(const unsigned char[:] buf, int* offset,
                               object type_val, list type_ctx, object ctx, dict options):
    cdef int start_offset = offset[0]
    cdef long long ref_or_inline = read_varint_fast(buf, offset)
    cdef int target_offset
    cdef long long length
    cdef int i

    if ref_or_inline > 0:
        target_offset = start_offset - <int>ref_or_inline
        if target_offset not in ctx.refs:
            raise ValueError(
                f"Undefined backreference at offset {start_offset}, target {target_offset}"
            )
        return ctx.refs[target_offset]

    element_type = type_val.value
    result = EastArray(element_type)
    cdef int ref_offset = offset[0]
    ctx.refs[ref_offset] = result

    length = read_varint_fast(buf, offset)

    type_ctx.append(type_val)

    cdef list items = []
    items_append = items.append
    for i in range(<int>length):
        items_append(decode_value(buf, offset, element_type, type_ctx, ctx, options))

    type_ctx.pop()

    list.extend(result, items)

    return result


cdef object decode_set_value(const unsigned char[:] buf, int* offset,
                             object type_val, list type_ctx, object ctx, dict options):
    cdef int start_offset = offset[0]
    cdef long long ref_or_inline = read_varint_fast(buf, offset)
    cdef int target_offset
    cdef long long length
    cdef int i

    if ref_or_inline > 0:
        target_offset = start_offset - <int>ref_or_inline
        if target_offset not in ctx.refs:
            raise ValueError(
                f"Undefined backreference at offset {start_offset}, target {target_offset}"
            )
        return ctx.refs[target_offset]

    element_type = type_val.value
    result = EastSet(element_type)
    cdef int ref_offset = offset[0]
    ctx.refs[ref_offset] = result

    length = read_varint_fast(buf, offset)

    type_ctx.append(type_val)

    for i in range(<int>length):
        result.add(decode_value(buf, offset, element_type, type_ctx, ctx, options))

    type_ctx.pop()

    return result


cdef object decode_dict_value(const unsigned char[:] buf, int* offset,
                              object type_val, list type_ctx, object ctx, dict options):
    cdef int start_offset = offset[0]
    cdef long long ref_or_inline = read_varint_fast(buf, offset)
    cdef int target_offset
    cdef long long length
    cdef int i

    if ref_or_inline > 0:
        target_offset = start_offset - <int>ref_or_inline
        if target_offset not in ctx.refs:
            raise ValueError(
                f"Undefined backreference at offset {start_offset}, target {target_offset}"
            )
        return ctx.refs[target_offset]

    # Cache key_type and value_type
    cdef object cache_key = id(type_val)
    cached = _dict_cache.get(cache_key)
    if cached is None:
        cached = (type_val.value["key"], type_val.value["value"])
        _dict_cache[cache_key] = cached
    key_type, value_type = cached

    result = EastDict(key_type, value_type)
    cdef int ref_offset = offset[0]
    ctx.refs[ref_offset] = result

    length = read_varint_fast(buf, offset)

    # Match current behavior: Dict is NOT on type_ctx during key decoding,
    # but IS on type_ctx during value decoding
    for i in range(<int>length):
        k = decode_value(buf, offset, key_type, type_ctx, ctx, options)
        type_ctx.append(type_val)
        v = decode_value(buf, offset, value_type, type_ctx, ctx, options)
        type_ctx.pop()
        result[k] = v

    return result


cdef object decode_ref_value(const unsigned char[:] buf, int* offset,
                             object type_val, list type_ctx, object ctx, dict options):
    cdef int start_offset = offset[0]
    cdef long long ref_or_inline = read_varint_fast(buf, offset)
    cdef int target_offset

    if ref_or_inline > 0:
        target_offset = start_offset - <int>ref_or_inline
        if target_offset not in ctx.refs:
            raise ValueError(
                f"Undefined backreference at offset {start_offset}, target {target_offset}"
            )
        return ctx.refs[target_offset]

    inner_type = type_val.value
    result = EastRef(None)
    cdef int ref_offset = offset[0]
    ctx.refs[ref_offset] = result

    type_ctx.append(type_val)
    value = decode_value(buf, offset, inner_type, type_ctx, ctx, options)
    type_ctx.pop()

    result.value = value

    return result


cdef object decode_blob_value(const unsigned char[:] buf, int* offset):
    cdef long long length = read_varint_fast(buf, offset)
    cdef int off = offset[0]
    if off + <int>length > buf.shape[0]:
        raise ValueError(
            f"Buffer underflow reading blob at offset {off}, length {length}"
        )
    data = bytes(buf[off:off + <int>length])
    offset[0] = off + <int>length
    return EastBlob(data)


cdef object decode_vector_value(const unsigned char[:] buf, int* offset, object type_val):
    element_type = type_val.value
    dtype = np_.dtype(EAST_ELEMENT_TO_DTYPE[element_type.type])
    cdef long long length = read_varint_fast(buf, offset)
    cdef int byte_count = <int>length * dtype.itemsize
    cdef int off = offset[0]
    if off + byte_count > buf.shape[0]:
        raise ValueError(
            f"Buffer underflow reading vector at offset {off}, length {length}"
        )
    data = np_.frombuffer(bytes(buf[off:off + byte_count]), dtype=dtype, count=<int>length).copy()
    offset[0] = off + byte_count
    return EastVector(element_type, data)


cdef object decode_matrix_value(const unsigned char[:] buf, int* offset, object type_val):
    element_type = type_val.value
    dtype = np_.dtype(EAST_ELEMENT_TO_DTYPE[element_type.type])
    cdef long long rows = read_varint_fast(buf, offset)
    cdef long long cols = read_varint_fast(buf, offset)
    cdef int count = <int>rows * <int>cols
    cdef int byte_count = count * dtype.itemsize
    cdef int off = offset[0]
    if off + byte_count > buf.shape[0]:
        raise ValueError(
            f"Buffer underflow reading matrix at offset {off}, {rows}x{cols}"
        )
    data = np_.frombuffer(
        bytes(buf[off:off + byte_count]), dtype=dtype, count=count
    ).copy().reshape(<int>rows, <int>cols)
    offset[0] = off + byte_count
    return EastMatrix(element_type, data, <int>rows, <int>cols)


cdef object decode_function_value(const unsigned char[:] buf, int* offset,
                                  object type_val, list type_ctx, object ctx, dict options):
    from east.types.type_of_type import IRType

    ir = decode_value(buf, offset, IRType, type_ctx, ctx, options)

    if ir["type"] != "Function":
        raise RuntimeError(f"Expected Function IR, got {ir['type']}")

    captures = ir["value"]["captures"]
    cdef long long capture_count = read_varint_fast(buf, offset)

    if capture_count != len(captures):
        raise RuntimeError(
            f"Capture count mismatch: IR has {len(captures)}, data has {capture_count}"
        )

    capture_env = {}
    for cap_var in captures:
        name = cap_var["value"]["name"]
        cap_type = cap_var["value"]["type"]
        cap_value = decode_value(buf, offset, cap_type, type_ctx, ctx, options)
        capture_env[name] = cap_value

    from east.runtime.compiler import FunctionFactory, _compile_ir

    platform = options.get("platform", [])
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

    return fn


cdef object decode_async_function_value(const unsigned char[:] buf, int* offset,
                                        object type_val, list type_ctx, object ctx, dict options):
    from east.types.type_of_type import IRType

    ir = decode_value(buf, offset, IRType, type_ctx, ctx, options)

    if ir["type"] != "AsyncFunction":
        raise RuntimeError(f"Expected AsyncFunction IR, got {ir['type']}")

    captures = ir["value"]["captures"]
    cdef long long capture_count = read_varint_fast(buf, offset)

    if capture_count != len(captures):
        raise RuntimeError(
            f"Capture count mismatch: IR has {len(captures)}, data has {capture_count}"
        )

    capture_env = {}
    for cap_var in captures:
        name = cap_var["value"]["name"]
        cap_type = cap_var["value"]["type"]
        cap_value = decode_value(buf, offset, cap_type, type_ctx, ctx, options)
        capture_env[name] = cap_value

    from east.runtime.compiler import FunctionFactory, _compile_ir

    platform = options.get("platform", [])
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

    return fn


# ─── Main dispatch ────────────────────────────────────────────────────────

cdef object decode_value(const unsigned char[:] buf, int* offset,
                         object type_val, list type_ctx, object ctx, dict options):
    """Monolithic decoder: dispatch on type tag, recurse via C calls."""
    cdef str tag = type_val.type

    # Ordered by expected frequency in IR payloads
    if tag == "Struct":
        return decode_struct_value(buf, offset, type_val, type_ctx, ctx, options)
    elif tag == "Variant":
        return decode_variant_value(buf, offset, type_val, type_ctx, ctx, options)
    elif tag == "String":
        return read_string_fast(buf, offset)
    elif tag == "Integer":
        return read_zigzag_fast(buf, offset)
    elif tag == "Boolean":
        if buf[offset[0]] != 0:
            offset[0] += 1
            return True
        else:
            offset[0] += 1
            return False
    elif tag == "Null":
        return None
    elif tag == "Float":
        return read_float64_fast(buf, offset)
    elif tag == "Array":
        return decode_array_value(buf, offset, type_val, type_ctx, ctx, options)
    elif tag == "Recursive":
        depth = type_val.value
        actual_type = type_ctx[len(type_ctx) - depth]
        return decode_value(buf, offset, actual_type, type_ctx, ctx, options)
    elif tag == "DateTime":
        millis = read_zigzag_fast(buf, offset)
        return DateTime.fromtimestamp(millis / 1000.0, tz=UTC)
    elif tag == "Set":
        return decode_set_value(buf, offset, type_val, type_ctx, ctx, options)
    elif tag == "Dict":
        return decode_dict_value(buf, offset, type_val, type_ctx, ctx, options)
    elif tag == "Ref":
        return decode_ref_value(buf, offset, type_val, type_ctx, ctx, options)
    elif tag == "Blob":
        return decode_blob_value(buf, offset)
    elif tag == "Vector":
        return decode_vector_value(buf, offset, type_val)
    elif tag == "Matrix":
        return decode_matrix_value(buf, offset, type_val)
    elif tag == "Function":
        return decode_function_value(buf, offset, type_val, type_ctx, ctx, options)
    elif tag == "AsyncFunction":
        return decode_async_function_value(buf, offset, type_val, type_ctx, ctx, options)
    elif tag == "Never":
        raise RuntimeError("Cannot decode Never type")
    else:
        raise ValueError(f"Unhandled type: {tag}")


# ─── Entry point ──────────────────────────────────────────────────────────

def decode_beast2_value_for(type_val, type_ctx=None, options=None):
    """Create value decoder for given type.

    Drop-in replacement — same API, monolithic C internals.
    The type_ctx parameter is accepted for API compatibility but not used;
    recursive type resolution happens at decode time via an internal stack.

    Args:
        type_val: East type to create decoder for
        type_ctx: Ignored (kept for API compatibility)
        options: Decode options (platform functions for function compilation)

    Returns:
        Function that decodes values from bytes at offset with context
    """
    if options is None:
        options = {}

    def decode(buffer, int offset, ctx):
        cdef const unsigned char[:] buf = buffer
        cdef int c_offset = offset
        cdef list decode_type_ctx = []
        result = decode_value(buf, &c_offset, type_val, decode_type_ctx, ctx, options)
        return (result, c_offset)

    return decode
