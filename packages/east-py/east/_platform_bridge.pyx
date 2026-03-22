# cython: boundscheck=False, wraparound=False, cdivision=True
# cython: language_level=3
# eastc: true
#
# Copyright (c) 2025 Elara AI Pty Ltd
# Licensed under the Business Source License 1.1. See LICENSE.md for details.
#
"""Platform function bridge: enables east-c to call Python platform functions.

Uses the same trampoline pattern as east-c's WASM bridge (wasm_api.c):
1. A pre_call hook stores the platform function name before each invocation
2. A single shared C callback dispatches to the correct Python function
3. Generic platform functions are specialized lazily on first call

The pre_call hook is set on the PlatformRegistry via inline C (since the
struct is declared opaque in _eastc.pxd to break a circular dependency).
"""

from cpython.ref cimport PyObject, Py_INCREF, Py_DECREF
from libc.stddef cimport size_t
from libc.stdint cimport int64_t, uint8_t, uintptr_t
from libc.stdlib cimport malloc, free
from libc.string cimport strdup

from east cimport _eastc
from east._eastc_bridge cimport py_type_to_c, c_value_to_py, py_value_to_c, _c_type_tag_to_py_type

import asyncio

# ─── Inline C for accessing PlatformRegistry.pre_call ─────────────────────
# PlatformRegistry is opaque in _eastc.pxd, so we use inline C to set the
# pre_call hook field.

cdef extern from *:
    """
    #include "east/platform.h"

    static void _set_pre_call_hook(PlatformRegistry *reg,
                                   void (*hook)(const char*, EastType**, size_t)) {
        reg->pre_call = hook;
    }
    """
    void _set_pre_call_hook(_eastc.PlatformRegistry *reg,
                            void (*hook)(const char*, _eastc.EastType**, size_t))


# ─── Module-level state ──────────────────────────────────────────────────

# Registry of Python platform function implementations.
# Non-generic: name → (py_fn, is_async, [input_c_type_ptrs], output_c_type_ptr)
# Generic factory: name → (py_factory_fn, is_async)
cdef dict _py_platform_fns = {}
cdef dict _py_generic_factories = {}

# Cache for specialized generic functions: (name, type_param_tuple) → py_fn
cdef dict _specialized_cache = {}

# The full platform_list (needed by generic factories that receive it as first arg)
cdef list _stored_platform_list = []

# Stashed event loop for async platform function support
cdef object _stashed_event_loop = None

# Context set by pre_call hook before each platform function invocation
cdef const char* _current_name = NULL
cdef _eastc.EastType** _current_type_params = NULL
cdef size_t _current_num_type_params = 0


# ─── c_value_to_py_auto: type-inferred conversion ───────────────────────
# Platform callbacks receive EastValue** without explicit types.  Fortunately
# C container values embed their element/key/value types, so we can reconstruct
# the EastType* from the value itself.

cdef _eastc.EastType* _type_from_value(_eastc.EastValue *val):
    """Reconstruct a non-owned EastType* from a C value's embedded type info.

    Returns a pointer that is NOT retained — caller must NOT release it.
    For primitives, returns the global singleton address.
    """
    cdef _eastc.EastValueKind kind = val.kind

    if kind == _eastc.EAST_VAL_NULL:
        return &_eastc.east_null_type
    elif kind == _eastc.EAST_VAL_BOOLEAN:
        return &_eastc.east_boolean_type
    elif kind == _eastc.EAST_VAL_INTEGER:
        return &_eastc.east_integer_type
    elif kind == _eastc.EAST_VAL_FLOAT:
        return &_eastc.east_float_type
    elif kind == _eastc.EAST_VAL_STRING:
        return &_eastc.east_string_type
    elif kind == _eastc.EAST_VAL_DATETIME:
        return &_eastc.east_datetime_type
    elif kind == _eastc.EAST_VAL_BLOB:
        return &_eastc.east_blob_type
    elif kind == _eastc.EAST_VAL_ARRAY:
        return val.data.array.elem_type
    elif kind == _eastc.EAST_VAL_SET:
        return val.data.set.elem_type
    elif kind == _eastc.EAST_VAL_DICT:
        # Need to construct a dict type — but we don't have one handy.
        # Fall back to using key_type/val_type from the value.
        return NULL  # handled specially in _c_value_to_py_auto
    elif kind == _eastc.EAST_VAL_STRUCT:
        return val.data.struct_.type
    elif kind == _eastc.EAST_VAL_VARIANT:
        return val.data.variant.type
    elif kind == _eastc.EAST_VAL_FUNCTION:
        return NULL  # functions are handled specially
    else:
        return NULL


cdef object _c_value_to_py_auto(_eastc.EastValue *val):
    """Convert a C value to Python, inferring the type from the value itself."""
    cdef _eastc.EastValueKind kind = val.kind

    # Primitives — direct conversion without needing a type
    if kind == _eastc.EAST_VAL_NULL:
        return None
    elif kind == _eastc.EAST_VAL_BOOLEAN:
        return val.data.boolean
    elif kind == _eastc.EAST_VAL_INTEGER:
        return val.data.integer
    elif kind == _eastc.EAST_VAL_FLOAT:
        return val.data.float64
    elif kind == _eastc.EAST_VAL_STRING:
        return val.data.string.data[:val.data.string.len].decode("utf-8")
    elif kind == _eastc.EAST_VAL_DATETIME:
        from datetime import UTC, datetime as DateTime
        return DateTime.fromtimestamp(val.data.datetime / 1000.0, tz=UTC)
    elif kind == _eastc.EAST_VAL_BLOB:
        from east.types.values import EastBlob
        return EastBlob((<char*>val.data.blob.data)[:val.data.blob.len])

    # Containers — use embedded type info via c_value_to_py
    elif kind == _eastc.EAST_VAL_ARRAY:
        c_type = _eastc.east_array_type(val.data.array.elem_type)
        try:
            return c_value_to_py(val, c_type)
        finally:
            _eastc.east_type_release(c_type)
    elif kind == _eastc.EAST_VAL_SET:
        c_type = _eastc.east_set_type(val.data.set.elem_type)
        try:
            return c_value_to_py(val, c_type)
        finally:
            _eastc.east_type_release(c_type)
    elif kind == _eastc.EAST_VAL_DICT:
        c_type = _eastc.east_dict_type(val.data.dict.key_type, val.data.dict.val_type)
        try:
            return c_value_to_py(val, c_type)
        finally:
            _eastc.east_type_release(c_type)
    elif kind == _eastc.EAST_VAL_STRUCT:
        return c_value_to_py(val, val.data.struct_.type)
    elif kind == _eastc.EAST_VAL_VARIANT:
        return c_value_to_py(val, val.data.variant.type)
    elif kind == _eastc.EAST_VAL_REF:
        # Refs: we don't have type info, but the inner value does
        inner_val = _c_value_to_py_auto(val.data.ref.value)
        from east.types.values import east_ref
        return east_ref(inner_val)
    elif kind == _eastc.EAST_VAL_VECTOR:
        c_type = _eastc.east_vector_type(val.data.vector.elem_type)
        try:
            return c_value_to_py(val, c_type)
        finally:
            _eastc.east_type_release(c_type)
    elif kind == _eastc.EAST_VAL_MATRIX:
        c_type = _eastc.east_matrix_type(val.data.matrix.elem_type)
        try:
            return c_value_to_py(val, c_type)
        finally:
            _eastc.east_type_release(c_type)
    elif kind == _eastc.EAST_VAL_FUNCTION:
        # For function values passed as args to platform functions,
        # wrap them so Python can call them via east_call
        fn_type = val.data.function.compiled
        if fn_type == NULL:
            raise ValueError("NULL compiled function in platform arg")
        # Use the function's IR type to build a proper callable
        # For now, return a wrapper that calls east_call
        return _wrap_c_function_for_python(val)
    else:
        raise ValueError(f"Unknown C value kind in platform arg: {kind}")


cdef object _wrap_c_function_for_python(_eastc.EastValue *val):
    """Wrap a C function value as a Python callable.

    Delegates actual east_call to _eastc_bridge.so to avoid _Thread_local
    mismatch (each .so has its own copy of static thread-locals like
    g_builtin_error from the statically-linked libeast-c.a).
    """
    _eastc.east_value_retain(val)
    cdef _eastc.EastCompiledFn *compiled = val.data.function.compiled

    # Build input/output type pointer lists for _invoke_c_function
    # The function type is NOT on compiled.ir.type (that's the body type).
    # We need to get it from the _c_function_to_py path in the bridge.
    # For now, use the function value's compiled fn to get the type from
    # the original IR node's type field.
    cdef _eastc.EastType *fn_type = NULL
    cdef size_t num_inputs = 0

    # compiled.ir is the function body. We don't have the function node type.
    # Pass empty input types and null output type — _invoke_c_function handles
    # the null output case, and 0-arg functions (most callbacks) need no input types.
    py_val_ptr = <uintptr_t>val
    py_input_type_ptrs = []
    py_output_type_ptr = <uintptr_t>0  # null = auto-detect

    def call_c_fn(*args):
        from east.runtime._compiler_eastc import _invoke_c_function_py
        return _invoke_c_function_py(py_val_ptr, py_input_type_ptrs, py_output_type_ptr, args)

    return call_c_fn


# ─── Pre-call hook ───────────────────────────────────────────────────────

cdef void _pre_call_hook(const char *name,
                         _eastc.EastType **type_params,
                         size_t num_type_params) noexcept nogil:
    """Called by east-c just before each platform function invocation.

    Stores the function name and type params so the shared callback knows
    which Python function to dispatch to.
    """
    global _current_name, _current_type_params, _current_num_type_params
    _current_name = name
    _current_type_params = type_params
    _current_num_type_params = num_type_params


# ─── Shared C callback ──────────────────────────────────────────────────

cdef _eastc.EvalResult _python_platform_fn(_eastc.EastValue **args,
                                            size_t num_args) noexcept with gil:
    """Shared C callback for all Python platform functions.

    The pre_call hook has already set _current_name so we know which
    Python function to call.
    """
    cdef _eastc.EvalResult err_result
    cdef _eastc.EastValue *c_result
    cdef _eastc.EastType *out_type
    cdef uintptr_t output_c_type_ptr

    err_result.status = _eastc.EVAL_ERROR
    err_result.value = NULL
    err_result.label = NULL
    err_result.error_message = NULL
    err_result.locations = NULL
    err_result.num_locations = 0

    try:
        name = _current_name.decode("utf-8") if _current_name != NULL else ""

        # Look up the Python callable
        py_fn = None
        is_async = False
        output_c_type_ptr = <uintptr_t>0

        entry = _py_platform_fns.get(name)
        if entry is not None:
            py_fn = entry[0]
            is_async = entry[1]
            output_c_type_ptr = <uintptr_t>entry[3] if len(entry) > 3 else 0
        elif name in _py_generic_factories:
            # Lazily specialize the generic function
            py_fn = _specialize_generic(name)
            is_async = _py_generic_factories[name][1]
        else:
            err_result.error_message = strdup(
                f"Platform function '{name}' not found".encode("utf-8"))
            return err_result

        # Convert C args to Python
        py_args = []
        for i in range(num_args):
            py_args.append(_c_value_to_py_auto(args[i]))

        # Call the Python function
        result = py_fn(*py_args)

        # Handle async results
        if asyncio.iscoroutine(result):
            result = _run_async(result)

        # Convert result back to C
        if result is None:
            c_result = _eastc.east_null()
        elif output_c_type_ptr != 0:
            out_type = <_eastc.EastType*>output_c_type_ptr
            c_result = py_value_to_c(result, out_type)
        else:
            # Infer output type from the result value
            c_result = _py_value_to_c_auto(result)

        return _eastc.eval_ok(c_result)

    except Exception as e:
        msg = str(e).encode("utf-8")
        err_result.error_message = strdup(<const char*>msg)
        return err_result


cdef object _specialize_generic(str name):
    """Lazily specialize a generic platform function using current type params."""
    # Build cache key from current type params
    py_type_params = []
    for i in range(_current_num_type_params):
        py_tp = _c_type_tag_to_py_type(_current_type_params[i])
        py_type_params.append(py_tp)

    cache_key = (name, tuple(str(tp) for tp in py_type_params))
    cached = _specialized_cache.get(cache_key)
    if cached is not None:
        return cached

    factory_entry = _py_generic_factories[name]
    py_factory = factory_entry[0]

    # Call factory: fn(platform_list, *type_params) → specialized_fn
    specialized = py_factory(_stored_platform_list, *py_type_params)
    _specialized_cache[cache_key] = specialized
    return specialized


cdef object _run_async(object coro):
    """Run an async coroutine synchronously.

    Since east-c platform callbacks only produce coroutines that ultimately
    call synchronous C functions (not real I/O), we can step through the
    coroutine manually without an event loop.
    """
    # For simple coroutines that only await other coroutines (no I/O),
    # we can drive them manually.
    result = None
    try:
        while True:
            result = coro.send(result)
            # If yielded value is a coroutine, run it recursively
            if asyncio.iscoroutine(result):
                result = _run_async(result)
            elif hasattr(result, '__await__'):
                # Handle awaitable objects (e.g. from asyncio.sleep(0))
                sub = result.__await__()
                try:
                    sub_result = None
                    while True:
                        sub_result = sub.send(sub_result)
                        if asyncio.iscoroutine(sub_result):
                            sub_result = _run_async(sub_result)
                except StopIteration as e:
                    result = e.value
            else:
                # Unknown yield — just pass it back
                pass
    except StopIteration as e:
        return e.value


cdef _eastc.EastValue* _py_value_to_c_auto(object val) except NULL:
    """Convert a Python value to C, inferring the type."""
    if val is None:
        return _eastc.east_null()
    elif isinstance(val, bool):
        return _eastc.east_boolean(<bint>val)
    elif isinstance(val, int):
        return _eastc.east_integer(<int64_t>val)
    elif isinstance(val, float):
        return _eastc.east_float(<double>val)
    elif isinstance(val, str):
        b = val.encode("utf-8")
        return _eastc.east_string_len(<const char*>b, len(b))
    else:
        # For complex types, we need the type. Try to get it from the value.
        from east.types.values import EastStruct, EastVariant, EastArray, EastBlob
        if isinstance(val, EastStruct):
            # Build a C struct type from the value's fields
            # This is complex — for now, use the declared output type
            raise RuntimeError(
                "Cannot auto-convert struct result without output type. "
                "Ensure platform function has declared output type.")
        elif isinstance(val, EastVariant):
            raise RuntimeError(
                "Cannot auto-convert variant result without output type. "
                "Ensure platform function has declared output type.")
        elif isinstance(val, EastBlob):
            return _eastc.east_blob(
                <const uint8_t*><char*><bytes>val,
                <size_t>len(val))
        elif isinstance(val, bytes):
            return _eastc.east_blob(
                <const uint8_t*><char*>val,
                <size_t>len(val))
        else:
            raise RuntimeError(
                f"Cannot auto-convert {type(val).__name__} result without output type.")


# ─── Shared generic factory callback ─────────────────────────────────────

cdef _eastc.PlatformFn _python_generic_factory(
        _eastc.EastType **type_params, size_t num_type_params) noexcept with gil:
    """Shared GenericPlatformFactory callback.

    Returns the shared platform callback. Actual specialization is done
    lazily in _python_platform_fn using pre_call context.
    """
    return _python_platform_fn


# ─── Public API ──────────────────────────────────────────────────────────

cdef void register_platform_functions(_eastc.PlatformRegistry *reg,
                                       list platform_list) except *:
    """Register Python platform functions in a C PlatformRegistry.

    Iterates over PlatformFunction/GenericPlatformFunction dicts and registers
    each in both the Python dispatch table and the C PlatformRegistry.
    """
    global _stored_platform_list
    _py_platform_fns.clear()
    _py_generic_factories.clear()
    _specialized_cache.clear()
    _stored_platform_list = platform_list

    # Set the pre_call hook so we know which function to dispatch to
    _set_pre_call_hook(reg, _pre_call_hook)

    for pf in platform_list:
        name = pf["name"]
        name_bytes = name.encode("utf-8")
        is_async = pf.get("type", "sync") == "async"

        if "type_parameters" in pf:
            # Generic platform function
            _py_generic_factories[name] = (pf["fn"], is_async)
            _eastc.platform_registry_add_generic(
                reg,
                strdup(<const char*>name_bytes),
                _python_generic_factory,
                <bint>is_async,
            )
        else:
            # Non-generic platform function — store with input/output types
            output_c_type_ptr = <uintptr_t>0
            input_c_type_ptrs = []
            if "output" in pf:
                out_type = py_type_to_c(pf["output"])
                output_c_type_ptr = <uintptr_t>out_type
                # Don't release — keep retained for the lifetime of the registry
            if "inputs" in pf:
                for inp in pf["inputs"]:
                    inp_type = py_type_to_c(inp)
                    input_c_type_ptrs.append(<uintptr_t>inp_type)

            _py_platform_fns[name] = (pf["fn"], is_async, input_c_type_ptrs, output_c_type_ptr)
            _eastc.platform_registry_add(
                reg,
                strdup(<const char*>name_bytes),
                _python_platform_fn,
                <bint>is_async,
            )


def stash_event_loop(loop):
    """Stash an event loop for async platform function support."""
    global _stashed_event_loop
    _stashed_event_loop = loop


def clear_platform_state():
    """Clear all platform function state. For testing."""
    _py_platform_fns.clear()
    _py_generic_factories.clear()
    _specialized_cache.clear()
    _stored_platform_list.clear()
