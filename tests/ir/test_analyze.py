"""Tests for IR analysis (async propagation and validation)."""

import pytest

from east.ir.analyze import analyze_ir
from east.ir.builders import (
    ir_block,
    ir_builtin,
    ir_function,
    ir_ifelse,
    ir_label,
    ir_platform,
    ir_value,
    ir_variable,
    ir_while,
    location,
)
from east.runtime.platform import PlatformFunction
from east.types.types import (
    ArrayType,
    BooleanType,
    FunctionType,
    IntegerType,
    NullType,
    StringType,
    StructType,
)


class TestBasicValidation:
    """Test basic IR validation."""

    def test_should_accept_valid_value_ir_node(self):
        """Should accept valid Value IR node."""
        loc = location("test", 1, 1)
        ir = ir_value(IntegerType, loc, 42)

        # Should not raise
        analyzed_ir, is_async_map = analyze_ir(ir, [], {})

        # Value should be marked as sync
        assert is_async_map[id(ir)] is False


class TestIsAsyncPropagation:
    """Test is_async metadata propagation through IR nodes."""

    def test_value_expressions_should_be_synchronous(self):
        """Value expressions should be synchronous (is_async=False)."""
        loc = location("test", 1, 1)
        ir = ir_value(IntegerType, loc, 42)

        _, is_async_map = analyze_ir(ir, [], {})

        assert is_async_map[id(ir)] is False

    def test_async_platform_function_call_should_be_async(self):
        """Async platform function call should be async (is_async=True)."""
        loc = location("test", 1, 1)

        # Define async platform function
        async def async_fetch(url: str) -> str:
            return "data"

        platform = [
            PlatformFunction(
                name="fetch",
                inputs=[StringType],
                output=StringType,
                type="async",
                fn=async_fetch,
            )
        ]

        # Create IR: fetch("https://example.com")
        url_arg = ir_value(StringType, loc, "https://example.com")
        ir = ir_platform(StringType, loc, "fetch", [url_arg])

        _, is_async_map = analyze_ir(ir, platform, {})

        # Platform call should be async
        assert is_async_map[id(ir)] is True

    def test_sync_platform_function_call_should_be_synchronous(self):
        """Sync platform function call should be synchronous (is_async=False)."""
        loc = location("test", 1, 1)

        # Define sync platform function
        def log(message: str) -> None:
            print(message)

        platform = [
            PlatformFunction(
                name="log",
                inputs=[StringType],
                output=NullType,
                type="sync",
                fn=log,
            )
        ]

        # Create IR: log("hello")
        msg_arg = ir_value(StringType, loc, "hello")
        ir = ir_platform(NullType, loc, "log", [msg_arg])

        _, is_async_map = analyze_ir(ir, platform, {})

        # Platform call should be sync (sync function, sync arguments)
        assert is_async_map[id(ir)] is False

    def test_platform_function_with_async_argument_should_be_async(self):
        """Platform function with async argument should be async."""
        loc = location("test", 1, 1)

        # Define platform functions
        async def async_fetch(url: str) -> str:
            return "data"

        def log(message: str) -> None:
            print(message)

        platform = [
            PlatformFunction(
                name="fetch",
                inputs=[StringType],
                output=StringType,
                type="async",
                fn=async_fetch,
            ),
            PlatformFunction(
                name="log",
                inputs=[StringType],
                output=NullType,
                type="sync",
                fn=log,
            ),
        ]

        # Create IR: log(fetch("https://example.com"))
        url_arg = ir_value(StringType, loc, "https://example.com")
        fetch_call = ir_platform(StringType, loc, "fetch", [url_arg])
        log_call = ir_platform(NullType, loc, "log", [fetch_call])

        _, is_async_map = analyze_ir(log_call, platform, {})

        # log call has async argument (fetch), so it should be async
        assert is_async_map[id(log_call)] is True
        assert is_async_map[id(fetch_call)] is True

    def test_block_with_async_statement_should_be_async(self):
        """Block with async statement should be async."""
        loc = location("test", 1, 1)

        # Define async platform function
        async def async_fetch(url: str) -> str:
            return "data"

        platform = [
            PlatformFunction(
                name="fetch",
                inputs=[StringType],
                output=StringType,
                type="async",
                fn=async_fetch,
            )
        ]

        # Create IR: { fetch("url"); 42 }
        url_arg = ir_value(StringType, loc, "https://example.com")
        fetch_call = ir_platform(StringType, loc, "fetch", [url_arg])
        value_stmt = ir_value(IntegerType, loc, 42)
        ir = ir_block(IntegerType, loc, [fetch_call, value_stmt])

        _, is_async_map = analyze_ir(ir, platform, {})

        # Block should be async (contains async statement)
        assert is_async_map[id(ir)] is True

    def test_block_with_only_sync_statements_should_be_sync(self):
        """Block with only sync statements should be sync."""
        loc = location("test", 1, 1)

        # Create IR: { 42; 100 }
        stmt1 = ir_value(IntegerType, loc, 42)
        stmt2 = ir_value(IntegerType, loc, 100)
        ir = ir_block(IntegerType, loc, [stmt1, stmt2])

        _, is_async_map = analyze_ir(ir, [], {})

        # Block should be sync
        assert is_async_map[id(ir)] is False

    def test_ifelse_with_async_predicate_should_be_async(self):
        """IfElse with async predicate should be async."""
        loc = location("test", 1, 1)

        # Define async platform function
        async def async_check() -> bool:
            return True

        platform = [
            PlatformFunction(
                name="check",
                inputs=[],
                output=BooleanType,
                type="async",
                fn=async_check,
            )
        ]

        # Create IR: if (check()) { 1 } else { 2 }
        predicate = ir_platform(BooleanType, loc, "check", [])
        if_body = ir_value(IntegerType, loc, 1)
        else_body = ir_value(IntegerType, loc, 2)
        ir = ir_ifelse(IntegerType, loc, [(predicate, if_body)], else_body)

        _, is_async_map = analyze_ir(ir, platform, {})

        # IfElse should be async (async predicate)
        assert is_async_map[id(ir)] is True

    def test_ifelse_with_async_if_body_should_be_async(self):
        """IfElse with async if_body should be async."""
        loc = location("test", 1, 1)

        # Define async platform function
        async def async_fetch(url: str) -> str:
            return "data"

        platform = [
            PlatformFunction(
                name="fetch",
                inputs=[StringType],
                output=StringType,
                type="async",
                fn=async_fetch,
            )
        ]

        # Create IR: if (true) { fetch("url") } else { "default" }
        predicate = ir_value(BooleanType, loc, True)
        url_arg = ir_value(StringType, loc, "https://example.com")
        if_body = ir_platform(StringType, loc, "fetch", [url_arg])
        else_body = ir_value(StringType, loc, "default")
        ir = ir_ifelse(StringType, loc, [(predicate, if_body)], else_body)

        _, is_async_map = analyze_ir(ir, platform, {})

        # IfElse should be async (async if_body)
        assert is_async_map[id(ir)] is True

    def test_ifelse_with_async_else_body_should_be_async(self):
        """IfElse with async else_body should be async."""
        loc = location("test", 1, 1)

        # Define async platform function
        async def async_fetch(url: str) -> str:
            return "data"

        platform = [
            PlatformFunction(
                name="fetch",
                inputs=[StringType],
                output=StringType,
                type="async",
                fn=async_fetch,
            )
        ]

        # Create IR: if (false) { "default" } else { fetch("url") }
        predicate = ir_value(BooleanType, loc, False)
        if_body = ir_value(StringType, loc, "default")
        url_arg = ir_value(StringType, loc, "https://example.com")
        else_body = ir_platform(StringType, loc, "fetch", [url_arg])
        ir = ir_ifelse(StringType, loc, [(predicate, if_body)], else_body)

        _, is_async_map = analyze_ir(ir, platform, {})

        # IfElse should be async (async else_body)
        assert is_async_map[id(ir)] is True

    def test_ifelse_with_all_sync_branches_should_be_sync(self):
        """IfElse with all sync branches should be sync."""
        loc = location("test", 1, 1)

        # Create IR: if (true) { 1 } else { 2 }
        predicate = ir_value(BooleanType, loc, True)
        if_body = ir_value(IntegerType, loc, 1)
        else_body = ir_value(IntegerType, loc, 2)
        ir = ir_ifelse(IntegerType, loc, [(predicate, if_body)], else_body)

        _, is_async_map = analyze_ir(ir, [], {})

        # IfElse should be sync
        assert is_async_map[id(ir)] is False

    def test_while_with_async_predicate_should_be_async(self):
        """While with async predicate should be async."""
        loc = location("test", 1, 1)
        label = ir_label("loop", loc)

        # Define async platform function
        async def async_check() -> bool:
            return True

        platform = [
            PlatformFunction(
                name="check",
                inputs=[],
                output=BooleanType,
                type="async",
                fn=async_check,
            )
        ]

        # Create IR: while (check()) { 42 }
        predicate = ir_platform(BooleanType, loc, "check", [])
        body = ir_value(IntegerType, loc, 42)
        ir = ir_while(NullType, loc, predicate, label, body)

        _, is_async_map = analyze_ir(ir, platform, {})

        # While should be async (async predicate)
        assert is_async_map[id(ir)] is True

    def test_while_with_async_body_should_be_async(self):
        """While with async body should be async."""
        loc = location("test", 1, 1)
        label = ir_label("loop", loc)

        # Define async platform function
        async def async_fetch(url: str) -> str:
            return "data"

        platform = [
            PlatformFunction(
                name="fetch",
                inputs=[StringType],
                output=StringType,
                type="async",
                fn=async_fetch,
            )
        ]

        # Create IR: while (true) { fetch("url") }
        predicate = ir_value(BooleanType, loc, True)
        url_arg = ir_value(StringType, loc, "https://example.com")
        body = ir_platform(StringType, loc, "fetch", [url_arg])
        ir = ir_while(NullType, loc, predicate, label, body)

        _, is_async_map = analyze_ir(ir, platform, {})

        # While should be async (async body)
        assert is_async_map[id(ir)] is True

    def test_function_node_should_track_async_in_body(self):
        """Function node should track async in body."""
        loc = location("test", 1, 1)

        # Define async platform function
        async def async_fetch(url: str) -> str:
            return "data"

        platform = [
            PlatformFunction(
                name="fetch",
                inputs=[StringType],
                output=StringType,
                type="async",
                fn=async_fetch,
            )
        ]

        # Create IR: (url: String) -> fetch(url)
        param_url = ir_variable(StringType, "url", loc, mutable=False, captured=False)
        body = ir_platform(StringType, loc, "fetch", [param_url])
        func_type = FunctionType([StringType], StringType, ["fetch"])
        ir = ir_function(func_type, loc, [], [param_url], body)

        _, is_async_map = analyze_ir(ir, platform, {})

        # Function node itself is sync (always), but body is async
        assert is_async_map[id(ir)] is False  # Function node
        assert is_async_map[id(body)] is True  # Body is async

    def test_builtin_with_async_argument_should_be_async(self):
        """Builtin with async argument should be async."""
        loc = location("test", 1, 1)

        # Define async platform function
        async def async_fetch(url: str) -> str:
            return "data"

        platform = [
            PlatformFunction(
                name="fetch",
                inputs=[StringType],
                output=StringType,
                type="async",
                fn=async_fetch,
            )
        ]

        # Create IR: IntegerAdd(5, async_result)
        # First get an async value
        url_arg = ir_value(StringType, loc, "url")
        fetch_call = ir_platform(StringType, loc, "fetch", [url_arg])

        # Use it in a builtin (conceptually - would need StringLength or similar)
        # For simplicity, let's use a block with builtin
        value_5 = ir_value(IntegerType, loc, 5)
        value_3 = ir_value(IntegerType, loc, 3)
        builtin_call = ir_builtin(IntegerType, loc, "IntegerAdd", [], [value_5, value_3])

        # Block with async fetch then sync builtin
        block = ir_block(IntegerType, loc, [fetch_call, builtin_call])

        _, is_async_map = analyze_ir(block, platform, {})

        # Builtin itself is sync (args are sync values)
        assert is_async_map[id(builtin_call)] is False
        # But block is async due to fetch
        assert is_async_map[id(block)] is True

    def test_nested_async_propagation_through_multiple_levels(self):
        """Nested async propagation through multiple levels."""
        loc = location("test", 1, 1)

        # Define async platform function
        async def async_fetch(url: str) -> str:
            return "data"

        platform = [
            PlatformFunction(
                name="fetch",
                inputs=[StringType],
                output=StringType,
                type="async",
                fn=async_fetch,
            )
        ]

        # Create nested IR:
        # {
        #   if (true) {
        #     {
        #       fetch("url")
        #     }
        #   } else {
        #     "default"
        #   }
        # }
        url_arg = ir_value(StringType, loc, "https://example.com")
        fetch_call = ir_platform(StringType, loc, "fetch", [url_arg])
        inner_block = ir_block(StringType, loc, [fetch_call])
        predicate = ir_value(BooleanType, loc, True)
        else_body = ir_value(StringType, loc, "default")
        ifelse = ir_ifelse(StringType, loc, [(predicate, inner_block)], else_body)
        outer_block = ir_block(StringType, loc, [ifelse])

        _, is_async_map = analyze_ir(outer_block, platform, {})

        # Async should propagate from fetch -> inner_block -> ifelse -> outer_block
        assert is_async_map[id(fetch_call)] is True
        assert is_async_map[id(inner_block)] is True
        assert is_async_map[id(ifelse)] is True
        assert is_async_map[id(outer_block)] is True


class TestErrorCases:
    """Test error cases in IR analysis."""

    def test_should_reject_unknown_platform_function_name(self):
        """Should reject unknown platform function name."""
        loc = location("test", 1, 1)

        # Create IR calling unknown platform function
        url_arg = ir_value(StringType, loc, "https://example.com")
        ir = ir_platform(StringType, loc, "unknown_function", [url_arg])

        # Should raise ValueError
        with pytest.raises(ValueError, match="Platform function 'unknown_function' not found"):
            analyze_ir(ir, [], {})

    def test_function_requiring_missing_platform(self):
        """Function requiring platform that isn't provided should raise error."""
        loc = location("test", 1, 1)

        # Define only log platform
        def log(message: str) -> None:
            print(message)

        platform = [
            PlatformFunction(
                name="log",
                inputs=[StringType],
                output=NullType,
                type="sync",
                fn=log,
            )
        ]

        # Create function that requires "fetch" platform (not provided)
        param_url = ir_variable(StringType, "url", loc, mutable=False, captured=False)
        body = ir_platform(StringType, loc, "log", [param_url])  # Body uses log (available)
        func_type = FunctionType(
            [StringType], StringType, ["fetch"]
        )  # But type claims we need fetch!
        ir = ir_function(func_type, loc, [], [param_url], body)

        # Should raise ValueError about missing "fetch" platform
        with pytest.raises(
            ValueError, match=r"requires platform function\(s\) \['fetch'\].*not available"
        ):
            analyze_ir(ir, platform, {})

    def test_function_requiring_multiple_missing_platforms(self):
        """Function requiring multiple missing platforms should list all missing."""
        loc = location("test", 1, 1)

        # Define only log platform
        def log(message: str) -> None:
            print(message)

        platform = [
            PlatformFunction(
                name="log",
                inputs=[StringType],
                output=NullType,
                type="sync",
                fn=log,
            )
        ]

        # Create function requiring 3 platforms, only 1 provided
        param_x = ir_variable(IntegerType, "x", loc, mutable=False, captured=False)
        body = ir_value(IntegerType, loc, 42)
        func_type = FunctionType(
            [IntegerType],
            IntegerType,
            ["fetch", "database", "log"],  # fetch and database missing
        )
        ir = ir_function(func_type, loc, [], [param_x], body)

        # Should raise ValueError listing both missing platforms
        with pytest.raises(
            ValueError, match=r"requires platform function\(s\) \['fetch', 'database'\]"
        ):
            analyze_ir(ir, platform, {})

    def test_nested_functions_with_different_platform_requirements(self):
        """Nested functions with different platform requirements should validate correctly."""
        loc = location("test", 1, 1)

        # Define platform A
        def platform_a() -> int:
            return 1

        platform = [
            PlatformFunction(
                name="platformA",
                inputs=[],
                output=IntegerType,
                type="sync",
                fn=platform_a,
            )
        ]

        # Create outer function requiring platformA
        # Body: inner function requiring platformB (not provided)
        inner_param = ir_variable(IntegerType, "y", loc, mutable=False, captured=False)
        inner_body = ir_value(IntegerType, loc, 10)
        inner_func_type = FunctionType([IntegerType], IntegerType, ["platformB"])  # Missing!
        inner_func = ir_function(inner_func_type, loc, [], [inner_param], inner_body)

        outer_param = ir_variable(IntegerType, "x", loc, mutable=False, captured=False)
        outer_body = inner_func  # Outer body is the inner function
        outer_func_type = FunctionType(
            [IntegerType], FunctionType([IntegerType], IntegerType, ["platformB"]), ["platformA"]
        )
        outer_func = ir_function(outer_func_type, loc, [], [outer_param], outer_body)

        # Should fail because inner function requires platformB
        with pytest.raises(ValueError, match=r"requires platform function\(s\) \['platformB'\]"):
            analyze_ir(outer_func, platform, {})

    def test_multiple_platform_errors_in_ir_tree(self):
        """Multiple platform errors should report first error encountered."""
        loc = location("test", 1, 1)

        # No platforms provided
        platform: list[PlatformFunction] = []

        # Create block with multiple unknown platform calls
        call1 = ir_platform(StringType, loc, "unknown1", [])
        call2 = ir_platform(StringType, loc, "unknown2", [])
        block = ir_block(StringType, loc, [call1, call2])

        # Should raise error for first unknown platform encountered
        with pytest.raises(ValueError, match="Platform function 'unknown1' not found"):
            analyze_ir(block, platform, {})

    def test_platform_validation_with_correct_platforms(self):
        """Function with correct platforms should succeed."""
        loc = location("test", 1, 1)

        # Define both platforms
        async def fetch(url: str) -> str:
            return "data"

        def log(message: str) -> None:
            print(message)

        platform = [
            PlatformFunction(
                name="fetch",
                inputs=[StringType],
                output=StringType,
                type="async",
                fn=fetch,
            ),
            PlatformFunction(
                name="log",
                inputs=[StringType],
                output=NullType,
                type="sync",
                fn=log,
            ),
        ]

        # Create function requiring both platforms
        param_url = ir_variable(StringType, "url", loc, mutable=False, captured=False)
        fetch_call = ir_platform(StringType, loc, "fetch", [param_url])
        log_call = ir_platform(NullType, loc, "log", [fetch_call])
        body = ir_block(NullType, loc, [fetch_call, log_call])
        func_type = FunctionType([StringType], NullType, ["fetch", "log"])
        ir = ir_function(func_type, loc, [], [param_url], body)

        # Should succeed - all required platforms provided
        analyzed_ir, is_async_map = analyze_ir(ir, platform, {})
        assert analyzed_ir is not None
        assert isinstance(is_async_map, dict)


class TestVariableTracking:
    """Test variable context tracking and validation."""

    def test_undefined_variable_should_raise_error(self):
        """Undefined variable should raise NameError."""
        loc = location("test", 1, 1)

        # Create Variable IR referencing "x" without defining it
        ir = ir_variable(IntegerType, "x", loc, mutable=False, captured=False)

        # Should raise NameError
        with pytest.raises(NameError, match="Variable 'x' is not defined"):
            analyze_ir(ir, [], {})

    def test_variable_used_before_definition_should_raise_error(self):
        """Variable used before definition should raise NameError."""
        # TODO: Need ir_let builder to test variable usage before definition
        # This test requires creating a Block with Variable("x") followed by Let(x = 5)
        # which is invalid - can't use x before it's defined
        pass

    def test_function_parameter_available_in_body(self):
        """Function parameter should be available in body."""
        loc = location("test", 1, 1)

        # Create function: (x: Integer) -> x
        param_x = ir_variable(IntegerType, "x", loc, mutable=False, captured=False)
        body = param_x  # Body just returns the parameter
        func_type = FunctionType([IntegerType], IntegerType, [])
        ir = ir_function(func_type, loc, [], [param_x], body)

        # Should succeed - parameter x is available in body
        analyzed_ir, is_async_map = analyze_ir(ir, [], {})
        assert analyzed_ir is not None

    def test_multiple_variables_in_scope(self):
        """Multiple variables should all be tracked."""
        loc = location("test", 1, 1)

        # Create function with two parameters: (x: Integer, y: Integer) -> x
        param_x = ir_variable(IntegerType, "x", loc, mutable=False, captured=False)
        param_y = ir_variable(IntegerType, "y", loc, mutable=False, captured=False)
        body = param_x  # Body references x
        func_type = FunctionType([IntegerType, IntegerType], IntegerType, [])
        ir = ir_function(func_type, loc, [], [param_x, param_y], body)

        # Should succeed - both parameters available
        analyzed_ir, is_async_map = analyze_ir(ir, [], {})
        assert analyzed_ir is not None

    def test_variable_shadowing_in_nested_scope(self):
        """Variable shadowing in nested scope should work correctly."""
        loc = location("test", 1, 1)

        # Outer function: (x: Integer) -> (y: Integer) -> y
        # Inner function shadows by having parameter y
        outer_param_x = ir_variable(IntegerType, "x", loc, mutable=False, captured=False)

        # Inner function
        inner_param_y = ir_variable(IntegerType, "y", loc, mutable=False, captured=False)
        inner_body = inner_param_y  # References inner y (shadowing)
        inner_func_type = FunctionType([IntegerType], IntegerType, [])
        inner_func = ir_function(inner_func_type, loc, [], [inner_param_y], inner_body)

        # Outer function
        outer_body = inner_func
        outer_func_type = FunctionType(
            [IntegerType], FunctionType([IntegerType], IntegerType, []), []
        )
        outer_func = ir_function(outer_func_type, loc, [], [outer_param_x], outer_body)

        # Should succeed - shadowing is allowed
        analyzed_ir, is_async_map = analyze_ir(outer_func, [], {})
        assert analyzed_ir is not None

    def test_try_catch_variables_available_in_catch_body(self):
        """TryCatch message and stack variables should be available in catch body."""
        from east.ir.builders import ir_trycatch

        loc = location("test", 1, 1)

        # Try body: 42
        try_body = ir_value(IntegerType, loc, 42)

        # Catch variables
        msg_var = ir_variable(StringType, "msg", loc, mutable=False, captured=False)
        stack_var = ir_variable(
            ArrayType(
                StructType(
                    [("filename", StringType), ("line", IntegerType), ("column", IntegerType)]
                )
            ),
            "stack",
            loc,
            mutable=False,
            captured=False,
        )

        # Catch body references msg variable
        catch_body = msg_var

        # TryCatch
        trycatch = ir_trycatch(StringType, loc, try_body, catch_body, msg_var, stack_var)

        # Should succeed - msg is available in catch body
        analyzed_ir, is_async_map = analyze_ir(trycatch, [], {})
        assert analyzed_ir is not None

    def test_try_catch_undefined_variable_in_catch_body_should_fail(self):
        """TryCatch with undefined variable in catch body should raise NameError."""
        from east.ir.builders import ir_trycatch

        loc = location("test", 1, 1)

        # Try body: 42
        try_body = ir_value(IntegerType, loc, 42)

        # Catch variables
        msg_var = ir_variable(StringType, "msg", loc, mutable=False, captured=False)
        stack_var = ir_variable(
            ArrayType(
                StructType(
                    [("filename", StringType), ("line", IntegerType), ("column", IntegerType)]
                )
            ),
            "stack",
            loc,
            mutable=False,
            captured=False,
        )

        # Catch body references undefined variable
        undefined_var = ir_variable(
            IntegerType, "undefined_var", loc, mutable=False, captured=False
        )
        catch_body = undefined_var

        # TryCatch
        trycatch = ir_trycatch(IntegerType, loc, try_body, catch_body, msg_var, stack_var)

        # Should fail - undefined_var is not defined
        with pytest.raises(NameError, match="Variable 'undefined_var' is not defined"):
            analyze_ir(trycatch, [], {})


class TestMissingIRNodes:
    """Test IR nodes that require manual construction (no builder yet)."""

    def test_let_with_async_value_should_be_async(self):
        """Let with async value should be async."""
        from east.types.containers import EastArray

        loc = location("test", 1, 1)

        # Define async platform function
        async def async_fetch(url: str) -> str:
            return "data"

        platform = [
            PlatformFunction(
                name="fetch",
                inputs=[StringType],
                output=StringType,
                type="async",
                fn=async_fetch,
            )
        ]

        # Create IR manually: let x = fetch("url"); return x
        url_arg = ir_value(StringType, loc, "https://example.com")
        fetch_call = ir_platform(StringType, loc, "fetch", [url_arg])

        # Manually construct Let IR node
        var_x = ir_variable(StringType, "x", loc, mutable=True, captured=False)
        let_node = {
            "type": "Let",
            "value": {
                "type": StringType,
                "location": loc,
                "variable": var_x,
                "value": fetch_call,
            },
        }

        # Return uses the variable
        return_node = {
            "type": "Return",
            "value": {"type": StringType, "location": loc, "value": var_x},
        }

        # Block with let and return
        block_statements = EastArray(
            {"type": "Variant", "value": []},  # IRType placeholder
            [let_node, return_node],
        )
        block = {
            "type": "Block",
            "value": {
                "type": StringType,
                "location": loc,
                "statements": block_statements,
            },
        }

        # Wrap in function
        func_type = FunctionType([], StringType, ["fetch"])
        func = ir_function(func_type, loc, [], [], block)

        _, is_async_map = analyze_ir(func, platform, {})

        # Let should be async
        assert is_async_map[id(let_node)] is True
        assert is_async_map[id(fetch_call)] is True

    def test_assign_with_async_value_should_be_async(self):
        """Assign with async value should be async."""
        from east.types.containers import EastArray

        loc = location("test", 1, 1)

        async def async_fetch(url: str) -> str:
            return "data"

        platform = [
            PlatformFunction(
                name="fetch",
                inputs=[StringType],
                output=StringType,
                type="async",
                fn=async_fetch,
            )
        ]

        # Create IR: let x = "init"; x = fetch("url"); return x
        var_x = ir_variable(StringType, "x", loc, mutable=True, captured=False)
        init_value = ir_value(StringType, loc, "init")
        let_node = {
            "type": "Let",
            "value": {
                "type": StringType,
                "location": loc,
                "variable": var_x,
                "value": init_value,
            },
        }

        url_arg = ir_value(StringType, loc, "url")
        fetch_call = ir_platform(StringType, loc, "fetch", [url_arg])

        # Manually construct Assign IR node
        assign_node = {
            "type": "Assign",
            "value": {
                "type": StringType,
                "location": loc,
                "variable": var_x,
                "value": fetch_call,
            },
        }

        return_node = {
            "type": "Return",
            "value": {"type": StringType, "location": loc, "value": var_x},
        }

        block_statements = EastArray(
            {"type": "Variant", "value": []},
            [let_node, assign_node, return_node],
        )
        block = {
            "type": "Block",
            "value": {
                "type": StringType,
                "location": loc,
                "statements": block_statements,
            },
        }

        func_type = FunctionType([], StringType, ["fetch"])
        func = ir_function(func_type, loc, [], [], block)

        _, is_async_map = analyze_ir(func, platform, {})

        # Assign should be async
        assert is_async_map[id(assign_node)] is True

    def test_for_array_with_async_body_should_be_async(self):
        """ForArray with async body should be async."""
        from east.types.containers import EastArray

        loc = location("test", 1, 1)

        async def async_process(x: int) -> None:
            pass

        platform = [
            PlatformFunction(
                name="process",
                inputs=[IntegerType],
                output=NullType,
                type="async",
                fn=async_process,
            )
        ]

        # Create IR: let arr = [1, 2, 3]; for (i, x in arr) { process(x) }
        # Create Value IR node manually with EastArray as LiteralValue variant
        array_val = EastArray(IntegerType, [1, 2, 3])
        array_value = {
            "type": "Value",
            "value": {
                "type": ArrayType(IntegerType),
                "location": loc,
                "value": {"type": "Array", "value": array_val},  # LiteralValue variant
            },
        }
        var_arr = ir_variable(ArrayType(IntegerType), "arr", loc, mutable=False, captured=False)
        let_node = {
            "type": "Let",
            "value": {
                "type": ArrayType(IntegerType),
                "location": loc,
                "variable": var_arr,
                "value": array_value,
            },
        }

        # ForArray variables
        var_i = ir_variable(IntegerType, "i", loc, mutable=False, captured=False)
        var_x = ir_variable(IntegerType, "x", loc, mutable=False, captured=False)

        # Body: process(x)
        process_call = ir_platform(NullType, loc, "process", [var_x])

        label = ir_label("for1", loc)
        forarray_node = {
            "type": "ForArray",
            "value": {
                "type": NullType,
                "location": loc,
                "array": var_arr,
                "key": var_i,
                "value": var_x,
                "body": process_call,
                "label": label,
            },
        }

        block_statements = EastArray(
            {"type": "Variant", "value": []},
            [let_node, forarray_node],
        )
        block = {
            "type": "Block",
            "value": {
                "type": NullType,
                "location": loc,
                "statements": block_statements,
            },
        }

        func_type = FunctionType([], NullType, ["process"])
        func = ir_function(func_type, loc, [], [], block)

        _, is_async_map = analyze_ir(func, platform, {})

        # ForArray should be async (async body)
        assert is_async_map[id(forarray_node)] is True

    def test_call_with_async_function_should_be_async(self):
        """Call with function that has async body should be async."""
        from east.types.containers import EastArray

        loc = location("test", 1, 1)

        async def async_work() -> int:
            return 42

        platform = [
            PlatformFunction(
                name="work",
                inputs=[],
                output=IntegerType,
                type="async",
                fn=async_work,
            )
        ]

        # Create inner function that calls async platform
        work_call = ir_platform(IntegerType, loc, "work", [])
        return_work = {
            "type": "Return",
            "value": {"type": IntegerType, "location": loc, "value": work_call},
        }

        inner_func_type = FunctionType([], IntegerType, ["work"])
        inner_func = ir_function(inner_func_type, loc, [], [], return_work)

        # Create Call IR node
        call_node = {
            "type": "Call",
            "value": {
                "type": IntegerType,
                "location": loc,
                "function": inner_func,
                "arguments": EastArray(
                    {"type": "Variant", "value": []},
                    [],
                ),
            },
        }

        # Outer function calls inner
        return_call = {
            "type": "Return",
            "value": {"type": IntegerType, "location": loc, "value": call_node},
        }

        outer_func_type = FunctionType([], IntegerType, ["work"])
        outer_func = ir_function(outer_func_type, loc, [], [], return_call)

        _, is_async_map = analyze_ir(outer_func, platform, {})

        # Call should be async (function body is async)
        assert is_async_map[id(call_node)] is True

    def test_trycatch_with_async_try_body_should_be_async(self):
        """TryCatch with async try body should be async."""
        from east.ir.builders import ir_trycatch

        loc = location("test", 1, 1)

        async def async_fetch(url: str) -> str:
            return "data"

        platform = [
            PlatformFunction(
                name="fetch",
                inputs=[StringType],
                output=StringType,
                type="async",
                fn=async_fetch,
            )
        ]

        # Try body: fetch("url")
        url_arg = ir_value(StringType, loc, "url")
        try_body = ir_platform(StringType, loc, "fetch", [url_arg])

        # Catch body: "error"
        catch_body = ir_value(StringType, loc, "error")

        # Catch variables
        msg_var = ir_variable(StringType, "msg", loc, mutable=False, captured=False)
        stack_var = ir_variable(
            ArrayType(
                StructType(
                    [("filename", StringType), ("line", IntegerType), ("column", IntegerType)]
                )
            ),
            "stack",
            loc,
            mutable=False,
            captured=False,
        )

        # TryCatch
        trycatch = ir_trycatch(StringType, loc, try_body, catch_body, msg_var, stack_var)

        func_type = FunctionType([], StringType, ["fetch"])
        func = ir_function(func_type, loc, [], [], trycatch)

        _, is_async_map = analyze_ir(func, platform, {})

        # TryCatch should be async (try body is async)
        assert is_async_map[id(trycatch)] is True

    def test_trycatch_with_async_catch_body_should_be_async(self):
        """TryCatch with async catch body should be async."""
        from east.ir.builders import ir_trycatch

        loc = location("test", 1, 1)

        async def async_log(msg: str) -> None:
            print(msg)

        platform = [
            PlatformFunction(
                name="log",
                inputs=[StringType],
                output=NullType,
                type="async",
                fn=async_log,
            )
        ]

        # Try body: 42 / 0 (will error)
        try_body = ir_builtin(
            IntegerType,
            loc,
            "IntegerDivide",
            [],
            [ir_value(IntegerType, loc, 1), ir_value(IntegerType, loc, 0)],
        )

        # Catch variables
        msg_var = ir_variable(StringType, "msg", loc, mutable=False, captured=False)
        stack_var = ir_variable(
            ArrayType(
                StructType(
                    [("filename", StringType), ("line", IntegerType), ("column", IntegerType)]
                )
            ),
            "stack",
            loc,
            mutable=False,
            captured=False,
        )

        # Catch body: log(msg) (async)
        catch_body = ir_platform(NullType, loc, "log", [msg_var])

        # TryCatch
        trycatch = ir_trycatch(NullType, loc, try_body, catch_body, msg_var, stack_var)

        func_type = FunctionType([], NullType, ["log"])
        func = ir_function(func_type, loc, [], [], trycatch)

        _, is_async_map = analyze_ir(func, platform, {})

        # TryCatch should be async (catch body is async)
        assert is_async_map[id(trycatch)] is True

    def test_trycatch_with_async_finally_body_should_be_async(self):
        """TryCatch with async finally body should be async."""
        from east.ir.builders import ir_trycatch

        loc = location("test", 1, 1)

        async def async_cleanup() -> None:
            pass

        platform = [
            PlatformFunction(
                name="cleanup",
                inputs=[],
                output=NullType,
                type="async",
                fn=async_cleanup,
            )
        ]

        # Try body: 42
        try_body = ir_value(IntegerType, loc, 42)

        # Catch body: -1
        catch_body = ir_value(IntegerType, loc, -1)

        # Catch variables
        msg_var = ir_variable(StringType, "msg", loc, mutable=False, captured=False)
        stack_var = ir_variable(
            ArrayType(
                StructType(
                    [("filename", StringType), ("line", IntegerType), ("column", IntegerType)]
                )
            ),
            "stack",
            loc,
            mutable=False,
            captured=False,
        )

        # Finally body: cleanup() (async)
        finally_body = ir_platform(NullType, loc, "cleanup", [])

        # TryCatch with finally
        trycatch = ir_trycatch(
            IntegerType, loc, try_body, catch_body, msg_var, stack_var, finally_body
        )

        func_type = FunctionType([], IntegerType, ["cleanup"])
        func = ir_function(func_type, loc, [], [], trycatch)

        _, is_async_map = analyze_ir(func, platform, {})

        # TryCatch should be async (finally body is async)
        assert is_async_map[id(trycatch)] is True

    def test_return_with_async_value_should_be_async(self):
        """Return with async value should be async."""
        loc = location("test", 1, 1)

        async def async_fetch(url: str) -> str:
            return "data"

        platform = [
            PlatformFunction(
                name="fetch",
                inputs=[StringType],
                output=StringType,
                type="async",
                fn=async_fetch,
            )
        ]

        # Return fetch("url")
        url_arg = ir_value(StringType, loc, "url")
        fetch_call = ir_platform(StringType, loc, "fetch", [url_arg])

        return_node = {
            "type": "Return",
            "value": {"type": StringType, "location": loc, "value": fetch_call},
        }

        func_type = FunctionType([], StringType, ["fetch"])
        func = ir_function(func_type, loc, [], [], return_node)

        _, is_async_map = analyze_ir(func, platform, {})

        # Return should be async
        assert is_async_map[id(return_node)] is True

    def test_multiple_if_else_cases_with_async_should_be_async(self):
        """IfElse with multiple cases (else-if chains) with async should be async."""
        loc = location("test", 1, 1)

        async def async_check1() -> bool:
            return False

        async def async_check2() -> bool:
            return True

        platform = [
            PlatformFunction(
                name="check1",
                inputs=[],
                output=BooleanType,
                type="async",
                fn=async_check1,
            ),
            PlatformFunction(
                name="check2",
                inputs=[],
                output=BooleanType,
                type="async",
                fn=async_check2,
            ),
        ]

        # if (check1()) { 1 } else if (check2()) { 2 } else { 3 }
        check1 = ir_platform(BooleanType, loc, "check1", [])
        check2 = ir_platform(BooleanType, loc, "check2", [])
        body1 = ir_value(IntegerType, loc, 1)
        body2 = ir_value(IntegerType, loc, 2)
        else_body = ir_value(IntegerType, loc, 3)

        # Multiple if cases
        ifelse = ir_ifelse(
            IntegerType,
            loc,
            [(check1, body1), (check2, body2)],  # Two if cases
            else_body,
        )

        func_type = FunctionType([], IntegerType, ["check1", "check2"])
        func = ir_function(func_type, loc, [], [], ifelse)

        _, is_async_map = analyze_ir(func, platform, {})

        # IfElse should be async (async predicates)
        assert is_async_map[id(ifelse)] is True


class TestCompileIntegration:
    """Test compile/compileAsync integration with analyze_ir."""

    def test_compile_rejects_async_platform(self):
        """compile() should throw when given async platform functions."""
        from east.runtime.compiler import compile

        loc = location("test", 1, 1)

        async def async_fetch(url: str) -> str:
            return "data"

        platform = [
            PlatformFunction(
                name="fetch",
                inputs=[StringType],
                output=StringType,
                type="async",
                fn=async_fetch,
            )
        ]

        # Create function using async platform
        url_arg = ir_value(StringType, loc, "url")
        fetch_call = ir_platform(StringType, loc, "fetch", [url_arg])
        return_node = {
            "type": "Return",
            "value": {"type": StringType, "location": loc, "value": fetch_call},
        }

        func_type = FunctionType([], StringType, ["fetch"])
        func_ir = ir_function(func_type, loc, [], [], return_node)

        # compile() should reject async platforms
        with pytest.raises(
            ValueError, match=r"Cannot use compile\(\) with async platform functions"
        ):
            compile(func_ir, platform)

    def test_compile_async_rejects_sync_only(self):
        """compileAsync() should throw when no async platform functions."""
        from east.runtime.compiler import compile_async

        loc = location("test", 1, 1)

        def log(msg: str) -> None:
            print(msg)

        platform = [
            PlatformFunction(
                name="log",
                inputs=[StringType],
                output=NullType,
                type="sync",
                fn=log,
            )
        ]

        # Create function using sync platform only
        msg_arg = ir_value(StringType, loc, "hello")
        log_call = ir_platform(NullType, loc, "log", [msg_arg])

        func_type = FunctionType([], NullType, ["log"])
        func_ir = ir_function(func_type, loc, [], [], log_call)

        # compileAsync() should reject sync-only platforms
        with pytest.raises(ValueError, match=r"No async platform functions found"):
            compile_async(func_ir, platform)

    def test_compile_succeeds_with_sync_platforms(self):
        """compile() should succeed with only sync platform functions."""
        from east.runtime.compiler import compile

        loc = location("test", 1, 1)

        def log(msg: str) -> None:
            print(msg)

        platform = [
            PlatformFunction(
                name="log",
                inputs=[StringType],
                output=NullType,
                type="sync",
                fn=log,
            )
        ]

        # Create function using sync platform
        msg_arg = ir_value(StringType, loc, "hello")
        log_call = ir_platform(NullType, loc, "log", [msg_arg])

        func_type = FunctionType([], NullType, ["log"])
        func_ir = ir_function(func_type, loc, [], [], log_call)

        # Should succeed
        compiled = compile(func_ir, platform)
        result = compiled()
        assert result is None or result.__class__.__name__ == "Null"

    def test_compile_async_succeeds_with_async_platforms(self):
        """compileAsync() should succeed with async platform functions."""
        import asyncio

        from east.runtime.compiler import compile_async

        loc = location("test", 1, 1)

        async def async_fetch(url: str) -> str:
            return url

        platform = [
            PlatformFunction(
                name="fetch",
                inputs=[StringType],
                output=StringType,
                type="async",
                fn=async_fetch,
            )
        ]

        # Create function using async platform
        url_arg = ir_value(StringType, loc, "test")
        fetch_call = ir_platform(StringType, loc, "fetch", [url_arg])
        return_node = {
            "type": "Return",
            "value": {"type": StringType, "location": loc, "value": fetch_call},
        }

        func_type = FunctionType([], StringType, ["fetch"])
        func_ir = ir_function(func_type, loc, [], [], return_node)

        # Should succeed
        compiled = compile_async(func_ir, platform)
        result = asyncio.run(compiled())
        assert result == "test"
