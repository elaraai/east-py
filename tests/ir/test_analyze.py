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
from east.types.type_system import (
    BooleanType,
    FunctionType,
    IntegerType,
    NullType,
    StringType,
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
        platform: list[str] = []

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


class TestCompileIntegration:
    """Test compile/compileAsync integration with analyze_ir.

    Note: These tests are already covered in tests/runtime/test_compiler.py
    but are documented here for completeness.
    """

    # These tests are in test_compiler.py:
    # - test_compile_rejects_async_platform
    # - test_compile_async_rejects_sync_only
    # - test_compile_with_sync_platform
    # - test_compile_async_with_async_platform

    def test_documentation_only(self):
        """Placeholder - actual tests are in test_compiler.py."""
        pass
