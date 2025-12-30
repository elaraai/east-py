#
# Copyright (c) 2025 Elara AI Pty Ltd
# Licensed under the Business Source License 1.1. See LICENSE.md for details.
#
"""Tests for error handling and IR location display."""

import pytest
from east.ir.builders import (
    ir_block,
    ir_builtin,
    ir_function,
    ir_value,
    location,
)
from east.runtime.compiler import EastError, _wrap_exception_with_location, compile
from east.types.types import FloatType, FunctionType
from east.types.values import EastArray


class TestEastError:
    """Tests for EastError exception class."""

    def test_east_error_single_location(self):
        """EastError with single location shows filename:line:column format."""
        loc = {"filename": "test.ts", "line": 42, "column": 10}
        err = EastError("division by zero", loc)

        result = str(err)
        assert result == "test.ts:42:10: division by zero"

    def test_east_error_with_stack_trace(self):
        """EastError with multiple locations shows stack trace."""
        loc1 = {"filename": "inner.ts", "line": 10, "column": 5}
        loc2 = {"filename": "outer.ts", "line": 20, "column": 15}
        loc3 = {"filename": "main.ts", "line": 30, "column": 1}

        err = EastError("error message", loc1)
        err.push_location(loc2)
        err.push_location(loc3)

        result = str(err)
        lines = result.split("\n")

        assert lines[0] == "inner.ts:10:5: error message"
        assert lines[1] == "Stack trace:"
        # Stack is shown in reverse order (outer first)
        assert "  at main.ts:30:1" in lines[2]
        assert "  at outer.ts:20:15" in lines[3]

    def test_push_location_adds_to_stack(self):
        """push_location adds entries to ir_stack."""
        loc1 = {"filename": "a.ts", "line": 1, "column": 1}
        loc2 = {"filename": "b.ts", "line": 2, "column": 2}

        err = EastError("msg", loc1)
        assert len(err.ir_stack) == 1

        err.push_location(loc2)
        assert len(err.ir_stack) == 2
        assert err.ir_stack[1] == loc2


class TestWrapExceptionWithLocation:
    """Tests for _wrap_exception_with_location helper."""

    def test_wrap_regular_exception(self):
        """Wraps regular exception in EastError."""
        loc = {"filename": "test.ts", "line": 5, "column": 3}
        original = ZeroDivisionError("float division by zero")

        result = _wrap_exception_with_location(original, loc)

        assert isinstance(result, EastError)
        assert result.message == "float division by zero"
        assert result.location == loc

    def test_wrap_east_error_pushes_location(self):
        """Wrapping EastError pushes location to stack instead of wrapping."""
        loc1 = {"filename": "inner.ts", "line": 10, "column": 5}
        loc2 = {"filename": "outer.ts", "line": 20, "column": 15}

        original = EastError("error", loc1)
        result = _wrap_exception_with_location(original, loc2)

        # Should return the same EastError with pushed location
        assert result is original
        assert len(result.ir_stack) == 2
        assert result.ir_stack[1] == loc2


class TestIRErrorLocations:
    """Tests for error locations when running IR that fails."""

    def test_division_by_zero_shows_ir_location(self):
        """Division by zero error includes IR source location."""
        # Build IR: function that does 1.0 / 0.0
        loc = location("myfile.ts", 42, 10)

        # Create 1.0 / 0.0 builtin call
        dividend = ir_value(FloatType, loc, 1.0)
        divisor = ir_value(FloatType, loc, 0.0)
        divide_ir = ir_builtin(
            FloatType,
            loc,
            "FloatDivide",
            EastArray(None, []),  # no type params
            [dividend, divisor],
        )

        # Wrap in a function
        func_type = FunctionType([], FloatType)
        body = ir_block(FloatType, loc, [divide_ir])
        func_ir = ir_function(func_type, loc, [], [], body)

        # Compile and run - should raise EastError with our location
        compiled = compile(func_ir, [])

        with pytest.raises(EastError) as exc_info:
            compiled()

        err = exc_info.value
        assert err.location["filename"] == "myfile.ts"
        assert err.location["line"] == 42
        assert err.location["column"] == 10
        assert "division" in err.message.lower() or "divide" in err.message.lower()

    def test_error_message_format(self):
        """Error message includes filename:line:column format."""
        loc = location("src/app.ts", 100, 25)

        dividend = ir_value(FloatType, loc, 5.0)
        divisor = ir_value(FloatType, loc, 0.0)
        divide_ir = ir_builtin(
            FloatType,
            loc,
            "FloatDivide",
            EastArray(None, []),
            [dividend, divisor],
        )

        func_type = FunctionType([], FloatType)
        body = ir_block(FloatType, loc, [divide_ir])
        func_ir = ir_function(func_type, loc, [], [], body)

        compiled = compile(func_ir, [])

        with pytest.raises(EastError) as exc_info:
            compiled()

        error_str = str(exc_info.value)
        assert "src/app.ts:100:25:" in error_str
