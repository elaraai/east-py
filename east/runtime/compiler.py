"""East IR compiler - compiles IR to native Python callables.

This compiler converts East IR into native Python functions, similar to how
the TypeScript and Julia implementations work. This allows builtins to work
with native Python callables instead of IR-level closures.
"""

from collections.abc import Callable
from typing import Any

from east.builtins import get_builtin
from east.ir.analyze import analyze_ir
from east.runtime.platform import PlatformFunction
from east.types.structural import EastVariant


class ReturnException(Exception):
    """Exception used for early return from functions."""

    def __init__(self, value: Any):
        self.value = value
        super().__init__()


def compile(ir: EastVariant, platform: list[PlatformFunction] | None = None) -> Callable:
    """Compile East IR to a native Python callable (synchronous).

    Args:
        ir: IR node (EastVariant) to compile (typically a Function node)
        platform: List of platform functions available to the IR (optional)

    Returns:
        Native Python callable

    Raises:
        ValueError: If any platform function is async (use compile_async instead)

    Example:
        >>> # Compile a Function IR that adds 1 to its input
        >>> func = compile(function_ir)
        >>> func(5)  # Returns 6
    """
    # Build platform function lookup dict and async set
    platform_fns: dict[str, Callable[..., Any]] = {}
    async_platform_fns: set[str] = set()
    platform_list = platform or []

    if platform_list:
        # Validate no async platform functions
        async_fns = [pf["name"] for pf in platform_list if pf["type"] == "async"]
        if async_fns:
            raise ValueError(
                f"Cannot use compile() with async platform functions: {', '.join(async_fns)}. "
                "Use compile_async() instead."
            )
        platform_fns = {pf["name"]: pf["fn"] for pf in platform_list}

        # Analyze IR before compilation (computes is_async metadata)
        ir, is_async_map = analyze_ir(ir, platform_list, {})
    else:
        # No platform functions - all nodes are sync
        is_async_map = {}

    return _compile_ir(ir, platform_fns, async_platform_fns, is_async_map)


def compile_async(ir: EastVariant, platform: list[PlatformFunction] | None = None) -> Callable:
    """Compile East IR to a native Python async callable.

    Args:
        ir: IR node (EastVariant) to compile (typically a Function node)
        platform: List of platform functions available to the IR (must include at least one async)

    Returns:
        Native Python async callable (coroutine function)

    Raises:
        ValueError: If no platform functions are async (use compile instead)

    Example:
        >>> import asyncio
        >>> # Compile a Function IR that calls async platform functions
        >>> func = compile_async(function_ir, platform)
        >>> asyncio.run(func(5))
    """
    # Build platform function lookup dict and async set
    platform_fns: dict[str, Callable[..., Any]] = {}
    async_platform_fns: set[str] = set()
    platform_list = platform or []

    if platform_list:
        # Validate at least one async platform function
        async_fns = [pf["name"] for pf in platform_list if pf["type"] == "async"]
        if not async_fns:
            raise ValueError(
                "No async platform functions found. "
                "Use compile() instead of compile_async() for better performance."
            )
        platform_fns = {pf["name"]: pf["fn"] for pf in platform_list}
        async_platform_fns = set(async_fns)

        # Analyze IR before compilation (computes is_async metadata)
        ir, is_async_map = analyze_ir(ir, platform_list, {})
    else:
        # No platform functions - all nodes are sync
        is_async_map = {}

    # Compile the IR with async support
    compiled = _compile_ir(ir, platform_fns, async_platform_fns, is_async_map)

    # If this is a Function node, the compiled result is already async-aware
    # Just return it directly
    return compiled


def _compile_ir(
    ir: EastVariant,
    platform_fns: dict[str, Callable[..., Any]],
    async_platform_fns: set[str],
    is_async_map: dict[int, bool],
) -> Callable:
    """Internal helper to compile IR nodes recursively.

    Args:
        ir: IR node to compile
        platform_fns: Dictionary mapping platform function names to implementations
        async_platform_fns: Set of platform function names that are async
        is_async_map: Dictionary mapping id(node) -> is_async bool

    Returns:
        Compiled callable that takes an environment dict and returns a value
    """
    # Dispatch based on IR variant tag
    tag = ir.tag

    if tag == "Function":
        return _compile_function(ir, platform_fns, async_platform_fns, is_async_map)
    if tag == "Value":
        return _compile_value(ir, platform_fns, async_platform_fns, is_async_map)
    if tag == "Variable":
        return _compile_variable(ir, platform_fns, async_platform_fns, is_async_map)
    if tag == "Builtin":
        return _compile_builtin(ir, platform_fns, async_platform_fns, is_async_map)
    if tag == "Block":
        return _compile_block(ir, platform_fns, async_platform_fns, is_async_map)
    if tag == "IfElse":
        return _compile_ifelse(ir, platform_fns, async_platform_fns, is_async_map)
    if tag == "While":
        return _compile_while(ir, platform_fns, async_platform_fns, is_async_map)
    if tag == "Let":
        return _compile_let(ir, platform_fns, async_platform_fns, is_async_map)
    if tag == "Platform":
        return _compile_platform(ir, platform_fns, async_platform_fns, is_async_map)
    if tag == "TryCatch":
        return _compile_trycatch(ir, platform_fns, async_platform_fns, is_async_map)
    if tag == "NewRef":
        return _compile_new_ref(ir, platform_fns, async_platform_fns, is_async_map)
    raise NotImplementedError(f"Compilation for {tag} not yet implemented")


def _compile_function(
    node: EastVariant,
    platform_fns: dict[str, Callable[..., Any]],
    async_platform_fns: set[str],
    is_async_map: dict[int, bool],
) -> Callable:
    """Compile a Function IR node to a Python callable.

    The function captures its environment and returns a closure that can be called.
    """
    # Extract struct from variant
    func_struct = node.value

    # Compile the body
    body_compiled = _compile_ir(func_struct.body, platform_fns, async_platform_fns, is_async_map)

    # Get parameter names from parameter Variable IR nodes
    param_names = [param.value.name for param in func_struct.parameters]

    # Check if body is async using static analysis
    body_is_async = is_async_map.get(id(func_struct.body), False)

    if body_is_async:
        # Create async Python function
        async def compiled_fn_async(*args):
            if len(args) != len(param_names):
                raise TypeError(f"Function expects {len(param_names)} arguments, got {len(args)}")

            # Create environment mapping parameter names to values
            env = dict(zip(param_names, args, strict=False))

            # Execute body with environment - body returns a coroutine
            return await body_compiled(env)

        return compiled_fn_async

    # Create sync Python function
    def compiled_fn_sync(*args):
        if len(args) != len(param_names):
            raise TypeError(f"Function expects {len(param_names)} arguments, got {len(args)}")

        env = dict(zip(param_names, args, strict=False))
        return body_compiled(env)

    return compiled_fn_sync


def _compile_value(
    node: EastVariant,
    platform_fns: dict[str, Callable[..., Any]],
    async_platform_fns: set[str],
    is_async_map: dict[int, bool],
) -> Callable:
    """Compile a Value IR node (literal constant)."""
    # Extract the LiteralValue variant from the Value struct
    lit_val_variant = node.value.value
    # Extract the actual value from the LiteralValue variant
    value = lit_val_variant.value
    # Values are always sync
    return lambda env: value


def _compile_variable(
    node: EastVariant,
    platform_fns: dict[str, Callable[..., Any]],
    async_platform_fns: set[str],
    is_async_map: dict[int, bool],
) -> Callable:
    """Compile a Variable IR node (variable reference)."""
    name = node.value.name
    # Variables are always sync
    return lambda env: env[name]


def _compile_builtin(
    node: EastVariant,
    platform_fns: dict[str, Callable[..., Any]],
    async_platform_fns: set[str],
    is_async_map: dict[int, bool],
) -> Callable:
    """Compile a Builtin IR node (builtin function call)."""
    builtin_struct = node.value
    builtin_name = builtin_struct.builtin
    builtin_fn = get_builtin(builtin_name)

    # Extract type parameters (needed for builtins like Print, Parse, etc.)
    type_params = builtin_struct.type_parameters

    # Compile all arguments and check if any are async
    arg_compiled = []
    any_arg_async = False
    for arg in builtin_struct.arguments:
        arg_fn = _compile_ir(arg, platform_fns, async_platform_fns, is_async_map)
        arg_compiled.append(arg_fn)
        if is_async_map.get(id(arg), False):
            any_arg_async = True

    # If any argument is async, return async version
    if any_arg_async:

        async def call_builtin_async(env):
            # Evaluate all arguments, awaiting async ones
            args = []
            for arg_fn, arg_ir in zip(arg_compiled, builtin_struct.arguments, strict=False):
                if is_async_map.get(id(arg_ir), False):
                    # Argument is async, await it
                    args.append(await arg_fn(env))
                else:
                    # Argument is sync
                    args.append(arg_fn(env))
            # Add type parameters at the end
            all_args = args + type_params
            # Call the builtin
            return builtin_fn(*all_args)

        return call_builtin_async

    # Otherwise return sync version
    def call_builtin_sync(env):
        # Evaluate all arguments
        args = [arg_fn(env) for arg_fn in arg_compiled]
        # Add type parameters at the end
        all_args = args + type_params
        # Call the builtin
        return builtin_fn(*all_args)

    return call_builtin_sync


def _compile_block(
    node: EastVariant,
    platform_fns: dict[str, Callable[..., Any]],
    async_platform_fns: set[str],
    is_async_map: dict[int, bool],
) -> Callable:
    """Compile a Block IR node (sequence of statements)."""
    block_struct = node.value

    # Compile all statements and track which are async
    stmt_info = []
    any_stmt_async = False
    for stmt in block_struct.statements:
        stmt_fn = _compile_ir(stmt, platform_fns, async_platform_fns, is_async_map)
        stmt_is_async = is_async_map.get(id(stmt), False)
        stmt_info.append((stmt_fn, stmt_is_async))
        if stmt_is_async:
            any_stmt_async = True

    # If any statement is async, return async version
    if any_stmt_async:

        async def execute_block_async(env):
            result = None
            for stmt_fn, is_async in stmt_info:
                if is_async:
                    # Statement is async, await it
                    result = await stmt_fn(env)
                else:
                    # Statement is sync
                    result = stmt_fn(env)
            return result

        return execute_block_async

    # Otherwise return sync version
    def execute_block_sync(env):
        result = None
        for stmt_fn, _ in stmt_info:
            result = stmt_fn(env)
        return result

    return execute_block_sync


def _compile_ifelse(
    node: EastVariant,
    platform_fns: dict[str, Callable[..., Any]],
    async_platform_fns: set[str],
    is_async_map: dict[int, bool],
) -> Callable:
    """Compile an IfElse IR node."""
    ifelse_struct = node.value

    # Compile each if case and track if any are async
    if_cases_info = []
    any_async = False
    for case in ifelse_struct.ifs:
        pred_fn = _compile_ir(case.predicate, platform_fns, async_platform_fns, is_async_map)
        body_fn = _compile_ir(case.body, platform_fns, async_platform_fns, is_async_map)
        pred_is_async = is_async_map.get(id(case.predicate), False)
        body_is_async = is_async_map.get(id(case.body), False)
        if_cases_info.append((pred_fn, pred_is_async, body_fn, body_is_async))
        if pred_is_async or body_is_async:
            any_async = True

    else_fn = _compile_ir(ifelse_struct.else_body, platform_fns, async_platform_fns, is_async_map)
    else_is_async = is_async_map.get(id(ifelse_struct.else_body), False)
    if else_is_async:
        any_async = True

    # If any part is async, return async version
    if any_async:

        async def execute_ifelse_async(env):
            for pred_fn, pred_is_async, body_fn, body_is_async in if_cases_info:
                if pred_is_async:
                    predicate_result = await pred_fn(env)
                else:
                    predicate_result = pred_fn(env)

                if predicate_result:
                    if body_is_async:
                        return await body_fn(env)
                    return body_fn(env)

            if else_is_async:
                return await else_fn(env)
            return else_fn(env)

        return execute_ifelse_async

    # Otherwise return sync version
    def execute_ifelse_sync(env):
        for pred_fn, _, body_fn, _ in if_cases_info:
            if pred_fn(env):
                return body_fn(env)
        return else_fn(env)

    return execute_ifelse_sync


def _compile_while(
    node: EastVariant,
    platform_fns: dict[str, Callable[..., Any]],
    async_platform_fns: set[str],
    is_async_map: dict[int, bool],
) -> Callable:
    """Compile a While IR node."""
    while_struct = node.value
    predicate_compiled = _compile_ir(
        while_struct.predicate, platform_fns, async_platform_fns, is_async_map
    )
    body_compiled = _compile_ir(while_struct.body, platform_fns, async_platform_fns, is_async_map)

    # Check if predicate or body is async using static analysis
    pred_is_async = is_async_map.get(id(while_struct.predicate), False)
    body_is_async = is_async_map.get(id(while_struct.body), False)

    # If any part is async, return async version
    if pred_is_async or body_is_async:

        async def execute_while_async(env):
            from east.types.primitives import null

            while True:
                if pred_is_async:
                    predicate_result = await predicate_compiled(env)
                else:
                    predicate_result = predicate_compiled(env)

                if not predicate_result:
                    break

                if body_is_async:
                    await body_compiled(env)
                else:
                    body_compiled(env)

            return null

        return execute_while_async

    # Otherwise return sync version
    def execute_while_sync(env):
        from east.types.primitives import null

        while predicate_compiled(env):
            body_compiled(env)
        return null

    return execute_while_sync


def _compile_let(
    node: EastVariant,
    platform_fns: dict[str, Callable[..., Any]],
    async_platform_fns: set[str],
    is_async_map: dict[int, bool],
) -> Callable:
    """Compile a Let IR node (variable binding)."""
    let_struct = node.value
    # Get the variable name from the Variable IR node
    var_name = let_struct.variable.value.name
    # Compile the value expression
    value_compiled = _compile_ir(let_struct.value, platform_fns, async_platform_fns, is_async_map)

    # Check if value expression is async using static analysis
    value_is_async = is_async_map.get(id(let_struct.value), False)

    # If value is async, return async version
    if value_is_async:

        async def execute_let_async(env):
            from east.types.primitives import null

            # Evaluate the value and bind it to the variable name in the environment
            value = await value_compiled(env)
            env[var_name] = value
            # Let statements return null
            return null

        return execute_let_async

    # Otherwise return sync version
    def execute_let_sync(env):
        from east.types.primitives import null

        value = value_compiled(env)
        env[var_name] = value
        return null

    return execute_let_sync


def _compile_platform(
    node: EastVariant,
    platform_fns: dict[str, Callable[..., Any]],
    async_platform_fns: set[str],
    is_async_map: dict[int, bool],
) -> Callable:
    """Compile a Platform IR node (platform function call)."""
    platform_struct = node.value
    platform_name = platform_struct.name

    # Look up the platform function
    if platform_name not in platform_fns:
        raise ValueError(
            f"Platform function '{platform_name}' not found. "
            f"Available platform functions: {', '.join(platform_fns.keys())}"
        )

    platform_fn = platform_fns[platform_name]

    # Check if this specific platform function is async
    is_async_fn = platform_name in async_platform_fns

    # Compile all arguments and track which are async
    arg_info = []
    any_arg_async = False
    for arg in platform_struct.arguments:
        arg_fn = _compile_ir(arg, platform_fns, async_platform_fns, is_async_map)
        arg_is_async = is_async_map.get(id(arg), False)
        arg_info.append((arg_fn, arg_is_async))
        if arg_is_async:
            any_arg_async = True

    # If platform function is async OR any argument is async, return async version
    if is_async_fn or any_arg_async:

        async def call_platform_async(env):
            # Evaluate all arguments, selectively awaiting async ones
            args = []
            for arg_fn, arg_is_async in arg_info:
                if arg_is_async:
                    args.append(await arg_fn(env))
                else:
                    args.append(arg_fn(env))

            # Call the platform function
            if is_async_fn:
                # Platform function itself is async, await it
                return await platform_fn(*args)
            # Platform function is sync, just call it
            return platform_fn(*args)

        return call_platform_async

    # Otherwise return sync version
    def call_platform_sync(env):
        # Evaluate all arguments
        args = [arg_fn(env) for arg_fn, _ in arg_info]
        # Call the platform function
        return platform_fn(*args)

    return call_platform_sync


def _compile_new_ref(
    node: EastVariant,
    platform_fns: dict[str, Callable[..., Any]],
    async_platform_fns: set[str],
    is_async_map: dict[int, bool],
) -> Callable:
    """Compile a NewRef IR node (creates a reference cell).

    Args:
        node: NewRef IR variant
        platform_fns: Available platform functions
        async_platform_fns: Set of async platform function names
        is_async_map: Map from IR node id to async status

    Returns:
        Compiled function that creates a ref cell
    """
    from east.types.ref import ref

    newref_struct = node.value

    # Compile the value
    value_fn = _compile_ir(newref_struct.value, platform_fns, async_platform_fns, is_async_map)

    # Check if async
    value_is_async = is_async_map.get(id(newref_struct.value), False)

    if value_is_async:

        async def execute_new_ref_async(env):
            val = await value_fn(env)
            return ref(val)

        return execute_new_ref_async

    def execute_new_ref_sync(env):
        val = value_fn(env)
        return ref(val)

    return execute_new_ref_sync


def _extract_stack_trace(exception: Exception):
    """Extract stack trace from exception and convert to East format.

    Returns:
        List of structs with {filename: str, line: int, column: int}
    """
    from east.ir.builders import location
    from east.types.containers import EastArray
    from east.types.type_system import IntegerType, StringType, StructType

    # Stack entry type: {filename: String, line: Integer, column: Integer}
    # Note: Python doesn't track column numbers in tracebacks
    stack_frames = []
    tb = exception.__traceback__

    while tb is not None:
        frame = tb.tb_frame
        filename = frame.f_code.co_filename
        line = tb.tb_lineno
        column = 0  # Python doesn't track column numbers

        # Create location struct
        loc_struct = location(filename, line, column)
        stack_frames.append(loc_struct)

        tb = tb.tb_next

    # Create array type for stack
    stack_type = StructType(
        [("filename", StringType), ("line", IntegerType), ("column", IntegerType)]
    )

    return EastArray(stack_type, stack_frames)


def _compile_trycatch(
    node: EastVariant,
    platform_fns: dict[str, Callable[..., Any]],
    async_platform_fns: set[str],
    is_async_map: dict[int, bool],
) -> Callable:
    """Compile a TryCatch IR node (try-catch-finally error handling)."""
    trycatch_struct = node.value

    # Compile try body
    try_body_fn = _compile_ir(
        trycatch_struct.try_body, platform_fns, async_platform_fns, is_async_map
    )

    # Extract message and stack variable names
    message_var = trycatch_struct.message.value
    message_name = message_var.name
    stack_var = trycatch_struct.stack.value
    stack_name = stack_var.name

    # Compile catch body
    catch_body_fn = _compile_ir(
        trycatch_struct.catch_body, platform_fns, async_platform_fns, is_async_map
    )

    # Compile finally body if present (not null)
    from east.types.primitives import Null

    finally_body_fn = None
    finally_is_async = False
    if (
        hasattr(trycatch_struct, "finally_body")
        and trycatch_struct.finally_body is not None
        and not isinstance(trycatch_struct.finally_body, Null)
    ):
        finally_body_fn = _compile_ir(
            trycatch_struct.finally_body, platform_fns, async_platform_fns, is_async_map
        )
        finally_is_async = is_async_map.get(id(trycatch_struct.finally_body), False)

    # Check if any component is async
    try_is_async = is_async_map.get(id(trycatch_struct.try_body), False)
    catch_is_async = is_async_map.get(id(trycatch_struct.catch_body), False)
    is_async = try_is_async or catch_is_async or finally_is_async

    if is_async:

        async def execute_trycatch_async(env):
            try:
                # Execute try body
                if try_is_async:
                    result = await try_body_fn(env)
                else:
                    result = try_body_fn(env)
            except Exception as e:
                # Execute catch body with error info
                # Create child environment with message and stack
                catch_env = {**env}
                catch_env[message_name] = str(e)
                # Extract stack trace and convert to East array format
                catch_env[stack_name] = _extract_stack_trace(e)

                if catch_is_async:
                    result = await catch_body_fn(catch_env)
                else:
                    result = catch_body_fn(catch_env)
            finally:
                # Execute finally block if present
                if finally_body_fn is not None:
                    if finally_is_async:
                        await finally_body_fn(env)
                    else:
                        finally_body_fn(env)

            return result

        return execute_trycatch_async

    def execute_trycatch_sync(env):
        try:
            # Execute try body
            result = try_body_fn(env)
        except Exception as e:
            # Execute catch body with error info
            catch_env = {**env}
            catch_env[message_name] = str(e)
            # Extract stack trace and convert to East array format
            catch_env[stack_name] = _extract_stack_trace(e)

            result = catch_body_fn(catch_env)
        finally:
            # Execute finally block if present
            if finally_body_fn is not None:
                finally_body_fn(env)

        return result

    return execute_trycatch_sync
