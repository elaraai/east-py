"""Tests for try-catch-finally functionality."""

from east.ir.builders import (
    ir_builtin,
    ir_function,
    ir_trycatch,
    ir_value,
    ir_variable,
    location,
)
from east.runtime.compiler import compile
from east.types.type_system import (
    ArrayType,
    FunctionType,
    IntegerType,
    StringType,
    StructType,
)


def test_try_catch_no_error():
    """Try-catch with no error should execute try block only and return try result."""
    # Build IR for:
    # function() {
    #   try {
    #     return 42
    #   } catch (msg, stack) {
    #     return -1
    #   }
    # }

    loc = location("<test>", 1, 0)

    # Variables for catch clause
    msg_var = ir_variable(StringType, "msg", loc)
    stack_var = ir_variable(
        ArrayType(
            StructType([("filename", StringType), ("line", IntegerType), ("column", IntegerType)])
        ),
        "stack",
        loc,
    )

    # Try body: return 42
    try_body = ir_value(IntegerType, loc, 42)

    # Catch body: return -1
    catch_body = ir_value(IntegerType, loc, -1)

    # TryCatch - type is Integer (union of try and catch return types)
    trycatch = ir_trycatch(
        IntegerType,
        loc,
        try_body,
        catch_body,
        msg_var,
        stack_var,
    )

    # Function
    func_ir = ir_function(
        FunctionType([], IntegerType, []),
        loc,
        [],  # no captures
        [],  # no parameters
        trycatch,
    )

    # Compile and test
    func = compile(func_ir)
    result = func()

    assert result == 42


def test_try_catch_with_error():
    """Try-catch with error should execute catch block."""
    # Build IR for:
    # function() {
    #   try {
    #     1 / 0  # Division by zero error
    #   } catch (msg, stack) {
    #     return -1
    #   }
    # }

    loc = location("<test>", 1, 0)

    # Variables
    msg_var = ir_variable(StringType, "msg", loc)
    stack_var = ir_variable(
        ArrayType(
            StructType([("filename", StringType), ("line", IntegerType), ("column", IntegerType)])
        ),
        "stack",
        loc,
    )

    # Try body: 1 / 0 (will raise exception)
    try_body = ir_builtin(
        IntegerType,
        loc,
        "IntegerDivide",
        [],
        [ir_value(IntegerType, loc, 1), ir_value(IntegerType, loc, 0)],
    )

    # Catch body: return -1
    catch_body = ir_value(IntegerType, loc, -1)

    # TryCatch
    trycatch = ir_trycatch(
        IntegerType,
        loc,
        try_body,
        catch_body,
        msg_var,
        stack_var,
    )

    # Function
    func_ir = ir_function(
        FunctionType([], IntegerType, []),
        loc,
        [],
        [],
        trycatch,
    )

    # Compile and test
    func = compile(func_ir)
    result = func()

    assert result == -1


def test_try_finally_executes_correctly():
    """Try-finally should not interfere with return value."""
    # Build IR for:
    # function() {
    #   try {
    #     return 42
    #   } catch (msg, stack) {
    #     return -1
    #   } finally {
    #     # Finally block that doesn't affect return value
    #     1 + 1  # side effect-free expression
    #   }
    # }

    loc = location("<test>", 1, 0)

    # Variables
    msg_var = ir_variable(StringType, "msg", loc)
    stack_var = ir_variable(
        ArrayType(
            StructType([("filename", StringType), ("line", IntegerType), ("column", IntegerType)])
        ),
        "stack",
        loc,
    )

    # Try body: return 42
    try_body = ir_value(IntegerType, loc, 42)

    # Catch body: return -1
    catch_body = ir_value(IntegerType, loc, -1)

    # Finally body: 1 + 1 (doesn't affect return)
    finally_body = ir_builtin(
        IntegerType,
        loc,
        "IntegerAdd",
        [],
        [ir_value(IntegerType, loc, 1), ir_value(IntegerType, loc, 1)],
    )

    # TryCatch with finally
    trycatch = ir_trycatch(
        IntegerType,
        loc,
        try_body,
        catch_body,
        msg_var,
        stack_var,
        finally_body,
    )

    # Function
    func_ir = ir_function(
        FunctionType([], IntegerType, []),
        loc,
        [],
        [],
        trycatch,
    )

    # Compile and test
    func = compile(func_ir)
    result = func()

    # Result should be 42 from try block, not affected by finally
    assert result == 42


def test_try_catch_finally_with_error():
    """Try-catch-finally should execute catch and finally on error."""
    # Build IR for:
    # function() {
    #   try {
    #     1 / 0
    #   } catch (msg, stack) {
    #     return -1
    #   } finally {
    #     2 + 2
    #   }
    # }

    loc = location("<test>", 1, 0)

    # Variables
    msg_var = ir_variable(StringType, "msg", loc)
    stack_var = ir_variable(
        ArrayType(
            StructType([("filename", StringType), ("line", IntegerType), ("column", IntegerType)])
        ),
        "stack",
        loc,
    )

    # Try body: 1 / 0 (error)
    try_body = ir_builtin(
        IntegerType,
        loc,
        "IntegerDivide",
        [],
        [ir_value(IntegerType, loc, 1), ir_value(IntegerType, loc, 0)],
    )

    # Catch body: return -1
    catch_body = ir_value(IntegerType, loc, -1)

    # Finally body: 2 + 2
    finally_body = ir_builtin(
        IntegerType,
        loc,
        "IntegerAdd",
        [],
        [ir_value(IntegerType, loc, 2), ir_value(IntegerType, loc, 2)],
    )

    # TryCatch with finally
    trycatch = ir_trycatch(
        IntegerType,
        loc,
        try_body,
        catch_body,
        msg_var,
        stack_var,
        finally_body,
    )

    # Function
    func_ir = ir_function(
        FunctionType([], IntegerType, []),
        loc,
        [],
        [],
        trycatch,
    )

    # Compile and test
    func = compile(func_ir)
    result = func()

    # Result should be -1 from catch block
    assert result == -1


if __name__ == "__main__":
    test_try_catch_no_error()
    print("✓ test_try_catch_no_error passed")

    test_try_catch_with_error()
    print("✓ test_try_catch_with_error passed")

    test_try_finally_executes_correctly()
    print("✓ test_try_finally_executes_correctly passed")

    test_try_catch_finally_with_error()
    print("✓ test_try_catch_finally_with_error passed")

    print("\nAll tests passed!")
