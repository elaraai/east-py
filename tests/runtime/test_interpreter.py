"""Tests for the Interpreter class."""

import pytest

from east.ir.nodes import (
    Assign,
    Block,
    Break,
    Call,
    Continue,
    Error,
    ForArray,
    ForDict,
    ForSet,
    Function,
    GetField,
    IfElse,
    Let,
    Location,
    Match,
    MatchCase,
    NewArray,
    NewDict,
    NewSet,
    Return,
    StructNode,
    TryCatch,
    Value,
    Variable,
    VariantNode,
)
from east.runtime.interpreter import (
    BreakException,
    ContinueException,
    EastError,
    Interpreter,
    ReturnException,
)
from east.types.primitives import null
from east.types.type_system import (
    BooleanType,
    IntegerType,
    NullType,
    StringType,
    StructTypeFromFields,
    VariantTypeFromCases,
)

# Test location for all nodes
loc = Location("<test>", 1, 1)


class TestInterpreterBasics:
    """Test basic interpreter operations."""

    def test_eval_value_node(self):
        """Test evaluating Value node."""
        interp = Interpreter()
        node = Value(42, IntegerType, loc)
        result = interp.eval(node)
        assert result == 42

    def test_eval_null_value(self):
        """Test evaluating null value."""
        interp = Interpreter()
        node = Value(null, NullType, loc)
        result = interp.eval(node)
        assert result is null

    def test_eval_boolean_value(self):
        """Test evaluating boolean values."""
        interp = Interpreter()
        assert interp.eval(Value(True, BooleanType, loc)) is True
        assert interp.eval(Value(False, BooleanType, loc)) is False

    def test_eval_string_value(self):
        """Test evaluating string value."""
        interp = Interpreter()
        node = Value("hello", StringType, loc)
        result = interp.eval(node)
        assert result == "hello"


class TestInterpreterVariables:
    """Test variable operations."""

    def test_eval_let_and_variable(self):
        """Test let binding and variable lookup."""
        interp = Interpreter()
        let_node = Let("x", False, Value(42, IntegerType, loc), loc)
        interp.eval(let_node)

        var_node = Variable("x", loc)
        result = interp.eval(var_node)
        assert result == 42

    def test_eval_let_returns_null(self):
        """Test that let returns null."""
        interp = Interpreter()
        let_node = Let("x", False, Value(42, IntegerType, loc), loc)
        result = interp.eval(let_node)
        assert result is null

    def test_eval_mutable_let_and_assign(self):
        """Test mutable variable and assignment."""
        interp = Interpreter()
        let_node = Let("x", True, Value(42, IntegerType, loc), loc)
        interp.eval(let_node)

        assign_node = Assign("x", Value(100, IntegerType, loc), loc)
        interp.eval(assign_node)

        var_node = Variable("x", loc)
        result = interp.eval(var_node)
        assert result == 100

    def test_eval_assign_returns_null(self):
        """Test that assign returns null."""
        interp = Interpreter()
        interp.eval(Let("x", True, Value(42, IntegerType, loc), loc))
        assign_node = Assign("x", Value(100, IntegerType, loc), loc)
        result = interp.eval(assign_node)
        assert result is null

    def test_eval_undefined_variable_raises_error(self):
        """Test accessing undefined variable raises error."""
        interp = Interpreter()
        var_node = Variable("x", loc)
        with pytest.raises(EastError, match="not found"):
            interp.eval(var_node)


class TestInterpreterBlocks:
    """Test block evaluation."""

    def test_eval_empty_block(self):
        """Test empty block returns null."""
        interp = Interpreter()
        block = Block((), loc)
        result = interp.eval(block)
        assert result is null

    def test_eval_block_returns_last_value(self):
        """Test block returns last statement value."""
        interp = Interpreter()
        block = Block(
            (
                Value(1, IntegerType, loc),
                Value(2, IntegerType, loc),
                Value(3, IntegerType, loc),
            ),
            loc,
        )
        result = interp.eval(block)
        assert result == 3

    def test_eval_block_with_let(self):
        """Test block with variable binding."""
        interp = Interpreter()
        block = Block(
            (
                Let("x", False, Value(42, IntegerType, loc), loc),
                Variable("x", loc),
            ),
            loc,
        )
        result = interp.eval(block)
        assert result == 42


class TestInterpreterControlFlow:
    """Test control flow constructs."""

    def test_eval_if_else_true_branch(self):
        """Test if-else with true condition."""
        interp = Interpreter()
        node = IfElse(
            Value(True, BooleanType, loc),
            Value(1, IntegerType, loc),
            Value(2, IntegerType, loc),
            loc,
        )
        result = interp.eval(node)
        assert result == 1

    def test_eval_if_else_false_branch(self):
        """Test if-else with false condition."""
        interp = Interpreter()
        node = IfElse(
            Value(False, BooleanType, loc),
            Value(1, IntegerType, loc),
            Value(2, IntegerType, loc),
            loc,
        )
        result = interp.eval(node)
        assert result == 2

    def test_eval_while_loop(self):
        """Test while loop."""
        # This test needs builtin functions to work properly
        # For now, just test that while nodes are accepted
        # TODO: Add proper while loop test once builtins are implemented
        pass

    def test_eval_return_raises_exception(self):
        """Test return raises ReturnException."""
        interp = Interpreter()
        node = Return(Value(42, IntegerType, loc), loc)
        with pytest.raises(ReturnException) as exc_info:
            interp.eval(node)
        assert exc_info.value.value == 42

    def test_eval_break_raises_exception(self):
        """Test break raises BreakException."""
        interp = Interpreter()
        node = Break(None, loc)
        with pytest.raises(BreakException):
            interp.eval(node)

    def test_eval_continue_raises_exception(self):
        """Test continue raises ContinueException."""
        interp = Interpreter()
        node = Continue(None, loc)
        with pytest.raises(ContinueException):
            interp.eval(node)


class TestInterpreterCollections:
    """Test collection operations."""

    def test_eval_new_array(self):
        """Test array construction."""
        interp = Interpreter()
        from east.types.containers import EastArray

        node = NewArray(
            IntegerType,
            (
                Value(1, IntegerType, loc),
                Value(2, IntegerType, loc),
                Value(3, IntegerType, loc),
            ),
            loc,
        )
        result = interp.eval(node)
        assert isinstance(result, EastArray)
        assert list(result) == [1, 2, 3]

    def test_eval_new_set(self):
        """Test set construction."""
        interp = Interpreter()
        from east.types.containers import EastSet

        node = NewSet(
            IntegerType,
            (
                Value(3, IntegerType, loc),
                Value(1, IntegerType, loc),
                Value(2, IntegerType, loc),
            ),
            loc,
        )
        result = interp.eval(node)
        assert isinstance(result, EastSet)
        assert list(result) == [1, 2, 3]  # Sorted

    def test_eval_new_dict(self):
        """Test dict construction."""
        interp = Interpreter()
        from east.types.containers import EastDict

        node = NewDict(
            StringType,
            IntegerType,
            (
                (Value("a", StringType, loc), Value(1, IntegerType, loc)),
                (Value("b", StringType, loc), Value(2, IntegerType, loc)),
            ),
            loc,
        )
        result = interp.eval(node)
        assert isinstance(result, EastDict)
        assert dict(result.items()) == {"a": 1, "b": 2}

    def test_eval_for_array(self):
        """Test for loop over array."""
        interp = Interpreter()
        from east.types.containers import EastArray

        # Build array [10, 20, 30]
        array_val = EastArray(IntegerType, [10, 20, 30])
        interp.global_env.bind("arr", array_val, False)

        # for (i, elem) in arr { ... }
        # We'll just return the last element
        loop = ForArray(
            None,
            "i",
            "elem",
            Variable("arr", loc),
            Variable("elem", loc),
            loc,
        )
        result = interp.eval(loop)
        assert result == 30

    def test_eval_for_set(self):
        """Test for loop over set."""
        interp = Interpreter()
        from east.types.containers import EastSet

        # Build set {1, 2, 3}
        set_val = EastSet(IntegerType, [1, 2, 3])
        interp.global_env.bind("s", set_val, False)

        # for elem in s { elem }
        loop = ForSet(
            None,
            "elem",
            Variable("s", loc),
            Variable("elem", loc),
            loc,
        )
        result = interp.eval(loop)
        assert result == 3  # Last element

    def test_eval_for_dict(self):
        """Test for loop over dict."""
        interp = Interpreter()
        from east.types.containers import EastDict

        # Build dict {a: 1, b: 2}
        dict_val = EastDict(StringType, IntegerType, {"a": 1, "b": 2})
        interp.global_env.bind("d", dict_val, False)

        # for (k, v) in d { v }
        loop = ForDict(
            None,
            "k",
            "v",
            Variable("d", loc),
            Variable("v", loc),
            loc,
        )
        result = interp.eval(loop)
        assert result == 2  # Last value


class TestInterpreterStructs:
    """Test struct operations."""

    def test_eval_struct_construction(self):
        """Test struct construction."""
        interp = Interpreter()
        struct_type = StructTypeFromFields(
            [
                ("name", StringType),
                ("age", IntegerType),
            ]
        )

        node = StructNode(
            struct_type,
            (
                ("name", Value("Alice", StringType, loc)),
                ("age", Value(30, IntegerType, loc)),
            ),
            loc,
        )
        result = interp.eval(node)
        assert result.name == "Alice"
        assert result.age == 30

    def test_eval_get_field(self):
        """Test struct field access."""
        interp = Interpreter()
        struct_type = StructTypeFromFields(
            [
                ("name", StringType),
                ("age", IntegerType),
            ]
        )

        struct_node = StructNode(
            struct_type,
            (
                ("name", Value("Alice", StringType, loc)),
                ("age", Value(30, IntegerType, loc)),
            ),
            loc,
        )
        interp.global_env.bind("person", interp.eval(struct_node), False)

        get_name = GetField(Variable("person", loc), "name", loc)
        assert interp.eval(get_name) == "Alice"

        get_age = GetField(Variable("person", loc), "age", loc)
        assert interp.eval(get_age) == 30


class TestInterpreterVariants:
    """Test variant operations."""

    def test_eval_variant_construction(self):
        """Test variant construction."""
        interp = Interpreter()
        variant_type = VariantTypeFromCases(
            [
                ("Some", IntegerType),
                ("None", NullType),
            ]
        )

        node = VariantNode(
            variant_type,
            "Some",
            Value(42, IntegerType, loc),
            loc,
        )
        result = interp.eval(node)
        assert result.tag == "Some"
        assert result.value == 42

    def test_eval_match_first_case(self):
        """Test pattern matching - first case."""
        interp = Interpreter()
        variant_type = VariantTypeFromCases(
            [
                ("Some", IntegerType),
                ("None", NullType),
            ]
        )

        # Create variant
        variant_node = VariantNode(
            variant_type,
            "Some",
            Value(42, IntegerType, loc),
            loc,
        )
        interp.global_env.bind("opt", interp.eval(variant_node), False)

        # Match on it
        match = Match(
            Variable("opt", loc),
            (
                MatchCase("Some", "x", Variable("x", loc)),
                MatchCase("None", "_", Value(0, IntegerType, loc)),
            ),
            loc,
        )
        result = interp.eval(match)
        assert result == 42

    def test_eval_match_second_case(self):
        """Test pattern matching - second case."""
        interp = Interpreter()
        variant_type = VariantTypeFromCases(
            [
                ("Some", IntegerType),
                ("None", NullType),
            ]
        )

        # Create variant
        variant_node = VariantNode(
            variant_type,
            "None",
            Value(null, NullType, loc),
            loc,
        )
        interp.global_env.bind("opt", interp.eval(variant_node), False)

        # Match on it
        match = Match(
            Variable("opt", loc),
            (
                MatchCase("Some", "x", Variable("x", loc)),
                MatchCase("None", "_", Value(0, IntegerType, loc)),
            ),
            loc,
        )
        result = interp.eval(match)
        assert result == 0


class TestInterpreterFunctions:
    """Test function operations."""

    def test_eval_function_creates_closure(self):
        """Test function evaluation creates closure."""
        interp = Interpreter()
        func_node = Function(
            ("x",),
            (IntegerType,),
            IntegerType,
            Variable("x", loc),
            loc,
        )
        result = interp.eval(func_node)
        assert isinstance(result, dict)
        assert result["type"] == "closure"
        assert result["node"] is func_node

    def test_eval_call_simple_function(self):
        """Test calling a simple function."""
        interp = Interpreter()

        # Define: fn(x) { x }
        func_node = Function(
            ("x",),
            (IntegerType,),
            IntegerType,
            Variable("x", loc),
            loc,
        )
        interp.global_env.bind("identity", interp.eval(func_node), False)

        # Call: identity(42)
        call_node = Call(
            Variable("identity", loc),
            (Value(42, IntegerType, loc),),
            loc,
        )
        result = interp.eval(call_node)
        assert result == 42

    def test_eval_call_with_return(self):
        """Test calling function with explicit return."""
        interp = Interpreter()

        # Define: fn(x) { return x + 1; }  (fake with just return x)
        func_node = Function(
            ("x",),
            (IntegerType,),
            IntegerType,
            Return(Variable("x", loc), loc),
            loc,
        )
        interp.global_env.bind("func", interp.eval(func_node), False)

        # Call: func(42)
        call_node = Call(
            Variable("func", loc),
            (Value(42, IntegerType, loc),),
            loc,
        )
        result = interp.eval(call_node)
        assert result == 42

    def test_eval_call_no_return_gives_null(self):
        """Test calling function without return gives null."""
        interp = Interpreter()

        # Define: fn(x) { let y = x; }
        func_node = Function(
            ("x",),
            (IntegerType,),
            NullType,
            Let("y", False, Variable("x", loc), loc),
            loc,
        )
        interp.global_env.bind("func", interp.eval(func_node), False)

        # Call: func(42)
        call_node = Call(
            Variable("func", loc),
            (Value(42, IntegerType, loc),),
            loc,
        )
        result = interp.eval(call_node)
        assert result is null

    def test_eval_closure_captures_environment(self):
        """Test closure captures environment."""
        interp = Interpreter()

        # let x = 10; fn(y) { x }
        interp.eval(Let("x", False, Value(10, IntegerType, loc), loc))

        func_node = Function(
            ("y",),
            (IntegerType,),
            IntegerType,
            Variable("x", loc),  # References outer x
            loc,
        )
        interp.global_env.bind("func", interp.eval(func_node), False)

        # Call: func(99) -> should return 10 (captured x)
        call_node = Call(
            Variable("func", loc),
            (Value(99, IntegerType, loc),),
            loc,
        )
        result = interp.eval(call_node)
        assert result == 10


class TestInterpreterErrors:
    """Test error handling."""

    def test_eval_error_throws(self):
        """Test error node throws EastError."""
        interp = Interpreter()
        node = Error(Value("oops", StringType, loc), loc)
        with pytest.raises(EastError, match="oops"):
            interp.eval(node)

    def test_eval_try_catch_catches_error(self):
        """Test try-catch catches error."""
        interp = Interpreter()

        # try { error "oops"; } catch (e) { e }
        node = TryCatch(
            Error(Value("oops", StringType, loc), loc),
            "e",
            Variable("e", loc),
            loc,
        )
        result = interp.eval(node)
        assert result == "oops"

    def test_eval_try_catch_no_error(self):
        """Test try-catch with no error."""
        interp = Interpreter()

        # try { 42 } catch (e) { 0 }
        node = TryCatch(
            Value(42, IntegerType, loc),
            "e",
            Value(0, IntegerType, loc),
            loc,
        )
        result = interp.eval(node)
        assert result == 42

    def test_error_includes_location(self):
        """Test error includes location."""
        interp = Interpreter()
        error_loc = Location("test.east", 10, 5)
        node = Error(Value("fail", StringType, error_loc), error_loc)
        with pytest.raises(EastError) as exc_info:
            interp.eval(node)
        assert exc_info.value.location == error_loc
