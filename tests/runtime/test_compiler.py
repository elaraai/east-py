"""Tests for the IR compiler."""

import asyncio
import json

import pytest

import east.builtins  # noqa: F401 - Import to register builtins
from east.ir.builders import (
    ir_builtin,
    ir_function,
    ir_new_ref,
    ir_platform,
    ir_value,
    ir_variable,
    location,
)
from east.runtime.compiler import compile, compile_async
from east.runtime.platform import PlatformFunction
from east.serialization.json import decode_json_for
from east.types.types import (
    FunctionType,
    IntegerType,
    IRType,
    NullType,
    RefType,
    StringType,
)


class TestCompiler:
    """Test IR compilation."""

    def test_compile_simple_value(self):
        """Test compiling a simple value."""
        loc = location("test", 1, 1)
        ir = ir_value(IntegerType, loc, 42)

        compiled = compile(ir)
        result = compiled({})
        assert result == 42

    def test_compile_variable(self):
        """Test compiling a variable reference."""
        loc = location("test", 1, 1)
        ir = ir_variable(IntegerType, "x", loc, mutable=False, captured=False)

        compiled = compile(ir)
        result = compiled({"x": 100})
        assert result == 100

    def test_compile_builtin_add(self):
        """Test compiling a builtin call."""
        loc = location("test", 1, 1)

        # Create: IntegerAdd(5, 3)
        arg1 = ir_value(IntegerType, loc, 5)
        arg2 = ir_value(IntegerType, loc, 3)
        ir = ir_builtin(IntegerType, loc, "IntegerAdd", [], [arg1, arg2])

        compiled = compile(ir)
        result = compiled({})
        assert result == 8

    def test_compile_simple_function(self):
        """Test compiling a simple function: (x) -> x + 1"""
        loc = location("test", 1, 1)

        # Create parameter
        param_x = ir_variable(IntegerType, "x", loc, mutable=False, captured=False)

        # Create body: IntegerAdd(x, 1)
        value_1 = ir_value(IntegerType, loc, 1)
        body = ir_builtin(IntegerType, loc, "IntegerAdd", [], [param_x, value_1])

        # Create function type
        func_type = FunctionType([IntegerType], IntegerType, [])

        # Create function: (x: Integer) -> x + 1
        func_ir = ir_function(func_type, loc, [], [param_x], body)

        # Compile and test
        compiled_fn = compile(func_ir)
        assert compiled_fn(5) == 6
        assert compiled_fn(10) == 11
        assert compiled_fn(0) == 1

    def test_compile_function_multiple_params(self):
        """Test compiling a function with multiple parameters: (x, y) -> x + y"""
        loc = location("test", 1, 1)

        # Create parameters
        param_x = ir_variable(IntegerType, "x", loc, mutable=False, captured=False)
        param_y = ir_variable(IntegerType, "y", loc, mutable=False, captured=False)

        # Create body: IntegerAdd(x, y)
        body = ir_builtin(IntegerType, loc, "IntegerAdd", [], [param_x, param_y])

        # Create function type
        func_type = FunctionType([IntegerType, IntegerType], IntegerType, [])

        # Create function
        func_ir = ir_function(func_type, loc, [], [param_x, param_y], body)

        # Compile and test
        compiled_fn = compile(func_ir)
        assert compiled_fn(5, 3) == 8
        assert compiled_fn(10, 20) == 30

    def test_compile_increment_from_json(self):
        """Test compiling increment function decoded from JSON IR."""
        increment_ir_json = {
            "type": "Function",
            "value": {
                "type": {
                    "type": "Function",
                    "value": {
                        "inputs": [{"type": "Integer", "value": None}],
                        "output": {"type": "Integer", "value": None},
                        "platforms": [],
                    },
                },
                "location": {
                    "filename": "node:internal/modules/esm/loader",
                    "line": "651",
                    "column": "26",
                },
                "captures": [],
                "parameters": [
                    {
                        "type": "Variable",
                        "value": {
                            "type": {"type": "Integer", "value": None},
                            "name": "_0",
                            "location": {
                                "filename": "node:internal/modules/esm/loader",
                                "line": "651",
                                "column": "26",
                            },
                            "mutable": False,
                            "captured": False,
                        },
                    }
                ],
                "body": {
                    "type": "Builtin",
                    "value": {
                        "type": {"type": "Integer", "value": None},
                        "location": {
                            "filename": "node:internal/modules/esm/loader",
                            "line": "651",
                            "column": "26",
                        },
                        "builtin": "IntegerAdd",
                        "type_parameters": [],
                        "arguments": [
                            {
                                "type": "Variable",
                                "value": {
                                    "type": {"type": "Integer", "value": None},
                                    "name": "_0",
                                    "location": {
                                        "filename": "node:internal/modules/esm/loader",
                                        "line": "651",
                                        "column": "26",
                                    },
                                    "mutable": False,
                                    "captured": False,
                                },
                            },
                            {
                                "type": "Value",
                                "value": {
                                    "type": {"type": "Integer", "value": None},
                                    "location": {
                                        "filename": "node:internal/modules/esm/loader",
                                        "line": "651",
                                        "column": "26",
                                    },
                                    "value": {"type": "Integer", "value": "1"},
                                },
                            },
                        ],
                    },
                },
            },
        }

        # Decode JSON to IR
        decoder = decode_json_for(IRType)
        json_bytes = json.dumps(increment_ir_json).encode("utf-8")
        ir = decoder(json_bytes)

        # Compile the IR
        compiled_fn = compile(ir)

        # Test that it works as an increment function
        assert compiled_fn(0) == 1
        assert compiled_fn(5) == 6
        assert compiled_fn(10) == 11
        assert compiled_fn(-1) == 0
        assert compiled_fn(999) == 1000

    def test_compile_with_sync_platform(self):
        """Test compile() with a synchronous platform function."""
        loc = location("test", 1, 1)

        # Define a sync platform function: add_one(x) -> x + 1
        def add_one_impl(x: int) -> int:
            return x + 1

        platform = [
            PlatformFunction(
                name="add_one",
                inputs=[IntegerType],
                output=IntegerType,
                type="sync",
                fn=add_one_impl,
            )
        ]

        # Build IR: (x: Integer) -> add_one(x)
        param_x = ir_variable(IntegerType, "x", loc, mutable=False, captured=False)
        body = ir_platform(IntegerType, loc, "add_one", [param_x])
        func_type = FunctionType([IntegerType], IntegerType, ["add_one"])
        func_ir = ir_function(func_type, loc, [], [param_x], body)

        # Compile with platform
        compiled_fn = compile(func_ir, platform)

        # Test
        assert compiled_fn(5) == 6
        assert compiled_fn(10) == 11
        assert compiled_fn(0) == 1

    def test_compile_rejects_async_platform(self):
        """Test that compile() rejects async platform functions."""

        async def async_func(x: int) -> int:
            await asyncio.sleep(0)
            return x + 1

        platform = [
            PlatformFunction(
                name="async_add",
                inputs=[IntegerType],
                output=IntegerType,
                type="async",
                fn=async_func,
            )
        ]

        loc = location("test", 1, 1)
        param_x = ir_variable(IntegerType, "x", loc, mutable=False, captured=False)
        body = ir_platform(IntegerType, loc, "async_add", [param_x])
        func_type = FunctionType([IntegerType], IntegerType, ["async_add"])
        func_ir = ir_function(func_type, loc, [], [param_x], body)

        # Should raise ValueError
        with pytest.raises(
            ValueError,
            match="Cannot use compile\\(\\) with async platform functions.*Use compile_async\\(\\)",
        ):
            compile(func_ir, platform)

    def test_compile_async_with_async_platform(self):
        """Test compile_async() with an async platform function."""

        async def delay_and_add(x: int) -> int:
            await asyncio.sleep(0.001)  # Small delay
            return x + 10

        platform = [
            PlatformFunction(
                name="delay_add",
                inputs=[IntegerType],
                output=IntegerType,
                type="async",
                fn=delay_and_add,
            )
        ]

        loc = location("test", 1, 1)
        param_x = ir_variable(IntegerType, "x", loc, mutable=False, captured=False)
        body = ir_platform(IntegerType, loc, "delay_add", [param_x])
        func_type = FunctionType([IntegerType], IntegerType, ["delay_add"])
        func_ir = ir_function(func_type, loc, [], [param_x], body)

        # Compile with async
        compiled_fn = compile_async(func_ir, platform)

        # Test with asyncio.run
        result = asyncio.run(compiled_fn(5))
        assert result == 15

        result = asyncio.run(compiled_fn(0))
        assert result == 10

    def test_compile_async_rejects_sync_only(self):
        """Test that compile_async() rejects sync-only platform functions."""

        def sync_func(x: int) -> int:
            return x + 1

        platform = [
            PlatformFunction(
                name="sync_add",
                inputs=[IntegerType],
                output=IntegerType,
                type="sync",
                fn=sync_func,
            )
        ]

        loc = location("test", 1, 1)
        param_x = ir_variable(IntegerType, "x", loc, mutable=False, captured=False)
        body = ir_platform(IntegerType, loc, "sync_add", [param_x])
        func_type = FunctionType([IntegerType], IntegerType, ["sync_add"])
        func_ir = ir_function(func_type, loc, [], [param_x], body)

        # Should raise ValueError
        with pytest.raises(
            ValueError,
            match="No async platform functions found.*Use compile\\(\\) instead",
        ):
            compile_async(func_ir, platform)

    def test_platform_function_not_found(self):
        """Test that compilation raises error for unknown platform function."""
        loc = location("test", 1, 1)

        # Build IR calling a platform function that doesn't exist
        param_x = ir_variable(IntegerType, "x", loc, mutable=False, captured=False)
        body = ir_platform(IntegerType, loc, "unknown_function", [param_x])
        func_type = FunctionType([IntegerType], IntegerType, ["unknown_function"])
        func_ir = ir_function(func_type, loc, [], [param_x], body)

        # Empty platform list
        platform: list[PlatformFunction] = []

        # Should raise ValueError during analysis
        with pytest.raises(ValueError, match="Platform function 'unknown_function' not found"):
            compile(func_ir, platform)

    def test_compile_new_ref(self):
        """Test compiling NewRef IR node."""
        from east.types.ref import Ref, deref

        # Create IR for: ref(42)
        loc = location("test.east", 1, 1)
        value_ir = ir_value(IntegerType, loc, 42)
        ref_ir = ir_new_ref(RefType(IntegerType), loc, value_ir)

        # Compile
        compiled = compile(ref_ir)

        # Execute
        result = compiled({})

        # Verify
        assert isinstance(result, Ref)
        assert deref(result) == 42

    def test_compile_ref_get(self):
        """Test compiling Ref.Get builtin."""
        # Create IR for: ref(42).get()
        loc = location("test.east", 1, 1)
        ref_value = ir_value(IntegerType, loc, 42)
        ref_ir = ir_new_ref(RefType(IntegerType), loc, ref_value)

        # Call Ref.Get on the ref
        get_ir = ir_builtin(IntegerType, loc, "RefGet", [IntegerType], [ref_ir])

        # Compile
        compiled = compile(get_ir)

        # Execute
        result = compiled({})

        # Verify
        assert result == 42

    def test_compile_ref_update(self):
        """Test compiling Ref.Update builtin."""
        # Create IR for: r = ref(0); r.update(100)
        loc = location("test.east", 1, 1)

        # ref(0)
        ref_value = ir_value(IntegerType, loc, 0)
        ref_ir = ir_new_ref(RefType(IntegerType), loc, ref_value)

        # update(100)
        new_value = ir_value(IntegerType, loc, 100)
        update_ir = ir_builtin(NullType, loc, "RefUpdate", [IntegerType], [ref_ir, new_value])

        # Compile
        compiled = compile(update_ir)

        # Execute
        result = compiled({})

        # Verify update returns None
        assert result is None

    def test_compile_ref_merge(self):
        """Test compiling Ref.Merge builtin.

        Note: This test verifies the ref merge mechanism works but
        doesn't test with an IR function since that requires more complex
        setup with Let/Assign to preserve the ref across calls.
        """
        # Test the builtin directly works - this validates the implementation
        from east.builtins.registry import get_builtin
        from east.types.ref import deref, ref

        ref_merge = get_builtin("RefMerge")(IntegerType, IntegerType)
        r = ref(10)
        result = ref_merge(r, 5, lambda cur, delta: cur + delta)
        assert result is None
        assert deref(r) == 15

    def test_ref_builtins_directly(self):
        """Test ref builtins work correctly."""
        from east.builtins.registry import get_builtin
        from east.types.ref import deref, ref

        # Test Ref.Get
        ref_get = get_builtin("RefGet")(IntegerType)
        r = ref(42)
        assert ref_get(r) == 42

        # Test Ref.Update
        ref_update = get_builtin("RefUpdate")(IntegerType)
        r = ref(0)
        result = ref_update(r, 100)
        assert result is None
        assert deref(r) == 100

        # Test Ref.Merge
        ref_merge = get_builtin("RefMerge")(IntegerType, IntegerType)
        r = ref(10)
        result = ref_merge(r, 5, lambda cur, delta: cur + delta)
        assert result is None
        assert deref(r) == 15

        # Test Ref.Merge with string
        ref_merge_str = get_builtin("RefMerge")(StringType, StringType)
        r_str = ref("hello")
        ref_merge_str(r_str, " world", lambda cur, new: cur + new)
        assert deref(r_str) == "hello world"
