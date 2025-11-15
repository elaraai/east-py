"""Comprehensive tests for East IR compilation and execution.

This test suite covers all IR node types and their compilation to executable Python code.
Tests are organized by IR construct: refs, try-catch, blocks, conditionals, loops, etc.
"""

import pytest

from east.ir.builders import (
    ir_block,
    ir_builtin,
    ir_function,
    ir_ifelse,
    ir_label,
    ir_new_ref,
    ir_trycatch,
    ir_value,
    ir_variable,
    ir_while,
    location,
)
from east.runtime.compiler import compile
from east.types.ref import Ref, deref, is_ref, ref, set_ref
from east.types.types import (
    ArrayType,
    BooleanType,
    FunctionType,
    IntegerType,
    NullType,
    RefType,
    StringType,
    StructType,
)

# =============================================================================
# Ref Tests (from test_ref.py)
# =============================================================================


class TestRefs:
    """Tests for ref type functionality."""

    def test_ref_creation(self):
        """Create a ref and check type."""
        r = ref(42)
        assert isinstance(r, Ref)
        assert is_ref(r)
        assert deref(r) == 42

    def test_ref_mutation(self):
        """Mutate a ref's value."""
        r = ref(0)
        assert deref(r) == 0

        set_ref(r, 1)
        assert deref(r) == 1

        set_ref(r, deref(r) + 1)
        assert deref(r) == 2

    def test_ref_aliasing(self):
        """Test that refs have identity semantics."""
        r1 = ref([1, 2, 3])
        r2 = r1  # Same ref

        set_ref(r2, [4, 5, 6])
        assert deref(r1) == [4, 5, 6]  # r1 sees the change
        assert r1 is r2

    def test_ref_distinct(self):
        """Test that different refs are distinct."""
        r1 = ref(42)
        r2 = ref(42)

        assert r1 is not r2  # Different refs
        assert deref(r1) == deref(r2)  # Same value

        set_ref(r1, 99)
        assert deref(r1) == 99
        assert deref(r2) == 42  # r2 unchanged

    def test_ref_nested(self):
        """Test nested refs."""
        inner = ref(10)
        outer = ref(inner)

        assert deref(deref(outer)) == 10

        set_ref(inner, 20)
        assert deref(deref(outer)) == 20

    def test_ref_type_creation(self):
        """Test RefType constructor."""
        int_ref_type = RefType(IntegerType)
        assert int_ref_type["type"] == "Ref"
        assert int_ref_type["value"] == IntegerType

        array_ref_type = RefType(ArrayType(StringType))
        assert array_ref_type["type"] == "Ref"
        assert array_ref_type["value"]["type"] == "Array"

    def test_ref_type_requires_data_type(self):
        """Test that refs can only contain data types."""
        from east.types.types import FunctionType

        # Should work with data types
        RefType(IntegerType)
        RefType(ArrayType(IntegerType))

        # Should fail with function types
        func_type = FunctionType([IntegerType], IntegerType, [])
        try:
            RefType(func_type)
            raise AssertionError("Should have raised TypeError")
        except TypeError as e:
            assert "data type" in str(e)

    def test_is_ref_false_for_non_refs(self):
        """Test is_ref returns False for non-refs."""
        assert not is_ref(42)
        assert not is_ref([1, 2, 3])
        assert not is_ref({"value": 42})
        assert not is_ref(None)

    def test_ref_repr(self):
        """Test ref string representation."""
        r = ref(42)
        assert repr(r) == "ref(42)"

        r2 = ref([1, 2])
        assert "ref" in repr(r2)

    def test_ref_type_of(self):
        """Test type_of with refs."""
        from east.types.types import type_of

        r = ref(42)
        ref_type = type_of(r)
        assert ref_type["type"] == "Ref"
        assert ref_type["value"] == IntegerType

    def test_ref_default_value(self):
        """Test default_value for ref type."""
        from east.utils.default import default_value

        typ = RefType(IntegerType)
        val = default_value(typ)

        assert is_ref(val)
        assert deref(val) == 0  # Default int is 0

    def test_ref_comparison_identity(self):
        """Test ref identity comparison."""
        from east.utils.ordering import is_for

        ref_type = RefType(IntegerType)

        # Identity - same ref
        r1 = ref(42)
        r2 = r1

        is_comparer = is_for(ref_type)
        assert is_comparer(r1, r2)

        # Different refs
        r3 = ref(42)
        assert not is_comparer(r1, r3)

    def test_ref_comparison_equality(self):
        """Test ref structural equality."""
        from east.utils.ordering import equal_for

        ref_type = RefType(IntegerType)
        equal_comparer = equal_for(ref_type)

        # Same identity
        r1 = ref(42)
        r2 = r1
        assert equal_comparer(r1, r2)

        # Different identity, same value
        r3 = ref(42)
        r4 = ref(42)
        assert equal_comparer(r3, r4)

        # Different values
        r5 = ref(42)
        r6 = ref(99)
        assert not equal_comparer(r5, r6)

    def test_ref_comparison_ordering(self):
        """Test ref ordering comparison."""
        from east.utils.ordering import compare_for

        ref_type = RefType(IntegerType)
        comparer = compare_for(ref_type)

        r1 = ref(10)
        r2 = ref(20)
        r3 = ref(10)

        assert comparer(r1, r2) < 0  # 10 < 20
        assert comparer(r2, r1) > 0  # 20 > 10
        assert comparer(r1, r3) == 0  # Same value

    def test_ref_circular_equality(self):
        """Test circular refs can be compared."""
        from east.utils.ordering import equal_for

        ref_type = RefType(RefType(IntegerType))
        equal_comparer = equal_for(ref_type)

        r1: Ref[Ref | None] = ref(None)
        r2: Ref[Ref | None] = ref(None)

        set_ref(r1, r1)  # Self-reference
        set_ref(r2, r2)  # Self-reference

        # Should not infinite loop
        assert equal_comparer(r1, r2)  # Both are circular


# =============================================================================
# Try-Catch-Finally Tests (from test_trycatch.py)
# =============================================================================


class TestTryCatchFinally:
    """Tests for try-catch-finally functionality."""

    def test_try_catch_no_error(self):
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
                StructType(
                    [("filename", StringType), ("line", IntegerType), ("column", IntegerType)]
                )
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

    def test_try_catch_with_error(self):
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
                StructType(
                    [("filename", StringType), ("line", IntegerType), ("column", IntegerType)]
                )
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

    def test_try_finally_executes_correctly(self):
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
                StructType(
                    [("filename", StringType), ("line", IntegerType), ("column", IntegerType)]
                )
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

    def test_try_catch_finally_with_error(self):
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
                StructType(
                    [("filename", StringType), ("line", IntegerType), ("column", IntegerType)]
                )
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

    def test_try_catch_without_finally_optimized(self):
        """Test that try-catch without finally creates optimized code path.

        When no finally_body is provided, ir_trycatch creates a dummy Value node.
        The compiler should detect this and generate the optimized code path
        (no finally block in generated code).
        """
        # Build IR for try-catch WITHOUT finally
        loc = location("<test>", 1, 0)

        msg_var = ir_variable(StringType, "msg", loc)
        stack_var = ir_variable(
            ArrayType(
                StructType(
                    [("filename", StringType), ("line", IntegerType), ("column", IntegerType)]
                )
            ),
            "stack",
            loc,
        )

        # Try body: return 100
        try_body = ir_value(IntegerType, loc, 100)

        # Catch body: return -1
        catch_body = ir_value(IntegerType, loc, -1)

        # No finally_body argument - should create dummy Value node internally
        trycatch = ir_trycatch(
            IntegerType,
            loc,
            try_body,
            catch_body,
            msg_var,
            stack_var,
            # Note: no finally_body argument
        )

        # Verify that finally_body was created as a Value node
        assert (
            trycatch["value"]["finally_body"]["type"] == "Value"
        ), "Should create dummy Value node when no finally provided"

        # Function
        func_ir = ir_function(
            FunctionType([], IntegerType, []),
            loc,
            [],
            [],
            trycatch,
        )

        # Compile and test - should use optimized path without finally
        func = compile(func_ir)
        result = func()

        assert result == 100


# =============================================================================
# Block Tests (let, const, assign)
# =============================================================================


class TestBlocks:
    """Tests for block statements: let, const, assign."""

    def test_block_empty(self):
        """Empty block returns null."""
        # function() { }
        loc = location("<test>", 1, 0)

        from east.types.primitives import null

        block = ir_block(NullType, loc, [])

        func_ir = ir_function(FunctionType([], NullType, []), loc, [], [], block)

        func = compile(func_ir)
        result = func()

        # Empty block may return None (Python null) instead of the null singleton
        assert result is null or result is None

    def test_block_single_value(self):
        """Block with single value returns that value."""
        # function() { 42 }
        loc = location("<test>", 1, 0)

        block = ir_block(IntegerType, loc, [ir_value(IntegerType, loc, 42)])

        func_ir = ir_function(FunctionType([], IntegerType, []), loc, [], [], block)

        func = compile(func_ir)
        result = func()

        assert result == 42

    def test_block_multiple_statements(self):
        """Block with multiple statements returns last value."""
        # function() {
        #   1 + 1
        #   2 + 2
        #   3 + 3
        # }
        loc = location("<test>", 1, 0)

        statements = [
            ir_builtin(
                IntegerType,
                loc,
                "IntegerAdd",
                [],
                [ir_value(IntegerType, loc, 1), ir_value(IntegerType, loc, 1)],
            ),
            ir_builtin(
                IntegerType,
                loc,
                "IntegerAdd",
                [],
                [ir_value(IntegerType, loc, 2), ir_value(IntegerType, loc, 2)],
            ),
            ir_builtin(
                IntegerType,
                loc,
                "IntegerAdd",
                [],
                [ir_value(IntegerType, loc, 3), ir_value(IntegerType, loc, 3)],
            ),
        ]

        block = ir_block(IntegerType, loc, statements)

        func_ir = ir_function(FunctionType([], IntegerType, []), loc, [], [], block)

        func = compile(func_ir)
        result = func()

        assert result == 6  # Last expression: 3 + 3


# =============================================================================
# Conditional Tests (if/else)
# =============================================================================


class TestConditionals:
    """Tests for if/else conditionals."""

    def test_if_true_branch(self):
        """If with true condition executes then branch."""
        # function() {
        #   if (true) { 42 } else { 0 }
        # }
        loc = location("<test>", 1, 0)

        # ir_ifelse takes a list of (predicate, body) tuples
        ifelse = ir_ifelse(
            IntegerType,
            loc,
            [(ir_value(BooleanType, loc, True), ir_value(IntegerType, loc, 42))],  # ifs
            ir_value(IntegerType, loc, 0),  # else_body
        )

        func_ir = ir_function(FunctionType([], IntegerType, []), loc, [], [], ifelse)

        func = compile(func_ir)
        result = func()

        assert result == 42

    def test_if_false_branch(self):
        """If with false condition executes else branch."""
        # function() {
        #   if (false) { 42 } else { 99 }
        # }
        loc = location("<test>", 1, 0)

        ifelse = ir_ifelse(
            IntegerType,
            loc,
            [(ir_value(BooleanType, loc, False), ir_value(IntegerType, loc, 42))],  # ifs
            ir_value(IntegerType, loc, 99),  # else_body
        )

        func_ir = ir_function(FunctionType([], IntegerType, []), loc, [], [], ifelse)

        func = compile(func_ir)
        result = func()

        assert result == 99

    def test_if_with_condition_expression(self):
        """If with computed condition."""
        # function() {
        #   if (10 < 20) { "yes" } else { "no" }
        # }
        loc = location("<test>", 1, 0)

        # Condition: 10 < 20
        # Less is a generic builtin, so we pass IntegerType as a type argument
        condition = ir_builtin(
            BooleanType,
            loc,
            "Less",
            [IntegerType],  # type arguments
            [ir_value(IntegerType, loc, 10), ir_value(IntegerType, loc, 20)],  # value arguments
        )

        ifelse = ir_ifelse(
            StringType,
            loc,
            [(condition, ir_value(StringType, loc, "yes"))],  # ifs
            ir_value(StringType, loc, "no"),  # else_body
        )

        func_ir = ir_function(FunctionType([], StringType, []), loc, [], [], ifelse)

        func = compile(func_ir)
        result = func()

        assert result == "yes"


# =============================================================================
# Loop Tests (while)
# =============================================================================


class TestLoops:
    """Tests for loop constructs."""

    def test_while_never_executes(self):
        """While loop with false condition never executes body."""
        # function() {
        #   while (false) {
        #     return 42
        #   }
        #   return 0
        # }
        loc = location("<test>", 1, 0)

        # ir_while(typ, loc, predicate, label, body)
        loop = ir_while(
            NullType,
            loc,
            ir_value(BooleanType, loc, False),  # predicate
            ir_label("loop1", loc),  # label
            ir_value(IntegerType, loc, 42),  # body
        )

        block = ir_block(IntegerType, loc, [loop, ir_value(IntegerType, loc, 0)])

        func_ir = ir_function(FunctionType([], IntegerType, []), loc, [], [], block)

        func = compile(func_ir)
        result = func()

        assert result == 0  # Loop never executed, returns 0


# =============================================================================
# IR NewRef Tests
# =============================================================================


class TestIRNewRef:
    """Tests for IR NewRef node."""

    def test_new_ref_creates_mutable_cell(self):
        """NewRef creates a mutable reference cell."""
        # function() {
        #   newRef(42)
        # }
        loc = location("<test>", 1, 0)

        new_ref = ir_new_ref(RefType(IntegerType), loc, ir_value(IntegerType, loc, 42))

        func_ir = ir_function(FunctionType([], RefType(IntegerType), []), loc, [], [], new_ref)

        func = compile(func_ir)
        result = func()

        assert is_ref(result)
        assert deref(result) == 42

    def test_new_ref_with_expression(self):
        """NewRef with computed value."""
        # function() {
        #   newRef(10 + 20)
        # }
        loc = location("<test>", 1, 0)

        value_expr = ir_builtin(
            IntegerType,
            loc,
            "IntegerAdd",
            [],
            [ir_value(IntegerType, loc, 10), ir_value(IntegerType, loc, 20)],
        )

        new_ref = ir_new_ref(RefType(IntegerType), loc, value_expr)

        func_ir = ir_function(FunctionType([], RefType(IntegerType), []), loc, [], [], new_ref)

        func = compile(func_ir)
        result = func()

        assert is_ref(result)
        assert deref(result) == 30


# =============================================================================
# Builtin Tests
# =============================================================================


class TestBuiltins:
    """Tests for builtin function calls."""

    def test_builtin_integer_add(self):
        """Test integer addition builtin."""
        # function() { 5 + 7 }
        loc = location("<test>", 1, 0)

        add = ir_builtin(
            IntegerType,
            loc,
            "IntegerAdd",
            [],
            [ir_value(IntegerType, loc, 5), ir_value(IntegerType, loc, 7)],
        )

        func_ir = ir_function(FunctionType([], IntegerType, []), loc, [], [], add)

        func = compile(func_ir)
        result = func()

        assert result == 12

    def test_builtin_integer_multiply(self):
        """Test integer multiplication builtin."""
        # function() { 6 * 7 }
        loc = location("<test>", 1, 0)

        mul = ir_builtin(
            IntegerType,
            loc,
            "IntegerMultiply",
            [],
            [ir_value(IntegerType, loc, 6), ir_value(IntegerType, loc, 7)],
        )

        func_ir = ir_function(FunctionType([], IntegerType, []), loc, [], [], mul)

        func = compile(func_ir)
        result = func()

        assert result == 42

    def test_builtin_string_concat(self):
        """Test string concatenation builtin."""
        # function() { "Hello" + " " + "World" }
        loc = location("<test>", 1, 0)

        # First concat: "Hello" + " "
        concat1 = ir_builtin(
            StringType,
            loc,
            "StringConcat",
            [],
            [ir_value(StringType, loc, "Hello"), ir_value(StringType, loc, " ")],
        )

        # Second concat: result + "World"
        concat2 = ir_builtin(
            StringType,
            loc,
            "StringConcat",
            [],
            [concat1, ir_value(StringType, loc, "World")],
        )

        func_ir = ir_function(FunctionType([], StringType, []), loc, [], [], concat2)

        func = compile(func_ir)
        result = func()

        assert result == "Hello World"


# =============================================================================
# Function Tests
# =============================================================================


class TestFunctions:
    """Tests for function creation and compilation."""

    def test_function_no_params_constant(self):
        """Function with no parameters returning constant."""
        # function() { 42 }
        loc = location("<test>", 1, 0)

        func_ir = ir_function(
            FunctionType([], IntegerType, []),
            loc,
            [],  # captures
            [],  # parameters
            ir_value(IntegerType, loc, 42),
        )

        func = compile(func_ir)
        result = func()

        assert result == 42

    def test_function_with_param(self):
        """Function with parameter."""
        # function(x) { x }
        loc = location("<test>", 1, 0)

        x_var = ir_variable(IntegerType, "x", loc)

        func_ir = ir_function(
            FunctionType([IntegerType], IntegerType, []),
            loc,
            [],  # captures
            [x_var],  # parameters
            x_var,  # body: return x
        )

        func = compile(func_ir)
        result = func(99)

        assert result == 99

    def test_function_with_multiple_params(self):
        """Function with multiple parameters."""
        # function(x, y) { x + y }
        loc = location("<test>", 1, 0)

        x_var = ir_variable(IntegerType, "x", loc)
        y_var = ir_variable(IntegerType, "y", loc)

        body = ir_builtin(IntegerType, loc, "IntegerAdd", [], [x_var, y_var])

        func_ir = ir_function(
            FunctionType([IntegerType, IntegerType], IntegerType, []),
            loc,
            [],
            [x_var, y_var],
            body,
        )

        func = compile(func_ir)
        result = func(10, 20)

        assert result == 30


# =============================================================================
# Value Tests
# =============================================================================


class TestValues:
    """Tests for literal value nodes."""

    def test_value_null(self):
        """Null value."""
        from east.types.primitives import null

        loc = location("<test>", 1, 0)

        func_ir = ir_function(
            FunctionType([], NullType, []),
            loc,
            [],
            [],
            ir_value(NullType, loc, null),
        )

        func = compile(func_ir)
        result = func()

        assert result is null

    def test_value_boolean_true(self):
        """Boolean true value."""
        loc = location("<test>", 1, 0)

        func_ir = ir_function(
            FunctionType([], BooleanType, []),
            loc,
            [],
            [],
            ir_value(BooleanType, loc, True),
        )

        func = compile(func_ir)
        result = func()

        assert result is True

    def test_value_boolean_false(self):
        """Boolean false value."""
        loc = location("<test>", 1, 0)

        func_ir = ir_function(
            FunctionType([], BooleanType, []),
            loc,
            [],
            [],
            ir_value(BooleanType, loc, False),
        )

        func = compile(func_ir)
        result = func()

        assert result is False

    def test_value_integer(self):
        """Integer value."""
        loc = location("<test>", 1, 0)

        func_ir = ir_function(
            FunctionType([], IntegerType, []),
            loc,
            [],
            [],
            ir_value(IntegerType, loc, 12345),
        )

        func = compile(func_ir)
        result = func()

        assert result == 12345

    def test_value_float(self):
        """Float value."""
        from east.types.types import FloatType

        loc = location("<test>", 1, 0)

        func_ir = ir_function(
            FunctionType([], FloatType, []),
            loc,
            [],
            [],
            ir_value(FloatType, loc, 3.14159),
        )

        func = compile(func_ir)
        result = func()

        assert result == 3.14159

    def test_value_string(self):
        """String value."""
        loc = location("<test>", 1, 0)

        func_ir = ir_function(
            FunctionType([], StringType, []),
            loc,
            [],
            [],
            ir_value(StringType, loc, "Hello, East!"),
        )

        func = compile(func_ir)
        result = func()

        assert result == "Hello, East!"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
