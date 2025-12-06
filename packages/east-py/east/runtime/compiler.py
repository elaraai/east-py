"""East IR compiler - compiles IR to native Python callables.

This compiler converts East IR into native Python functions, similar to how
the TypeScript and Julia implementations work. This allows builtins to work
with native Python callables instead of IR-level closures.
"""

from collections.abc import Callable
from typing import Any

from east.builtins import get_builtin
from east.runtime.platform import PlatformFunction
from east.types.ir import IR
from east.types.values import EastNull, EastStruct, EastVariant


class ReturnException(Exception):
    """Exception used for early return from functions."""

    def __init__(self, value: Any):
        self.value = value
        super().__init__()


class BreakException(Exception):
    """Exception used to implement break in loops."""

    def __init__(self, label: str):
        self.label = label
        super().__init__()


class ContinueException(Exception):
    """Exception used to implement continue in loops."""

    def __init__(self, label: str):
        self.label = label
        super().__init__()


class FunctionFactory:
    """Wrapper for function factories that need parent environment to create callables.

    This allows functions to flow through the IR (stored in variables, passed as arguments,
    returned, etc.) without being called prematurely. When the function is actually needed,
    we call factory.make(env) to get the actual callable with captured environment.

    FunctionFactory itself is callable - calling it with env unwraps it to the actual function.
    """

    def __init__(self, factory_fn):
        self.factory_fn = factory_fn

    def make(self, env):
        """Create the actual callable with the given environment."""
        return self.factory_fn(env)

    def __call__(self, env):
        """Make FunctionFactory callable - calling it with env unwraps to actual function."""
        return self.make(env)


class CaptureAwareEnv(dict):
    """Environment that delegates captured variable lookups to parent scope.

    Used by compiled functions to implement closure semantics. Local variables
    are stored in the dict, while captured variables are looked up in the parent
    environment.
    """

    def __init__(self, local_vars, parent, captures):
        super().__init__(local_vars)
        self._parent = parent
        self._captures = set(captures)

    def __getitem__(self, key):
        if key in dict.keys(self):
            return dict.__getitem__(self, key)
        if key in self._captures:
            return self._parent[key]
        raise KeyError(key)

    def __setitem__(self, key, value):
        if key in self._captures:
            self._parent[key] = value
        else:
            dict.__setitem__(self, key, value)

    def __contains__(self, key):
        return dict.__contains__(self, key) or (key in self._captures and key in self._parent)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def make_child(self, extra_vars):
        """Create a child environment that preserves the capture chain.

        Used by loops and try-catch to create scoped environments without
        breaking closure variable access.
        """
        child = CaptureAwareEnv(dict(self), self._parent, self._captures)
        child.update(extra_vars)
        return child


def _make_child_env(env, extra_vars):
    """Create a child environment, preserving CaptureAwareEnv if present."""
    if isinstance(env, CaptureAwareEnv):
        return env.make_child(extra_vars)
    return {**env, **extra_vars}


def compile(ir: IR, platform: list[PlatformFunction] | None = None) -> Callable:
    """Compile East IR to a native Python callable (synchronous).

    Args:
        ir: IR node to compile (typically a Function node)
        platform: List of platform functions available to the IR (optional).
            Async platform functions are allowed but will not be used by sync IR.

    Returns:
        Native Python callable

    Example:
        >>> # Compile a Function IR that adds 1 to its input
        >>> func = compile(function_ir)
        >>> func(5)  # Returns 6
    """
    platform_fns: dict[str, Callable[..., Any]] = {}
    async_platform_fns: set[str] = set()
    platform_list = platform or []

    if platform_list:
        # Include all platform functions - sync IR won't call async ones
        platform_fns = {pf["name"]: pf["fn"] for pf in platform_list}

    compiled, _ = _compile_ir(ir, platform_fns, async_platform_fns)

    # If compiled is a FunctionFactory, unwrap it with empty environment
    if isinstance(compiled, FunctionFactory):
        return compiled.make({})

    return compiled


def compile_async(ir: IR, platform: list[PlatformFunction] | None = None) -> Callable:
    """Compile East IR to a native Python async callable.

    Args:
        ir: IR node to compile (typically an AsyncFunction node)
        platform: List of platform functions available to the IR

    Returns:
        Native Python async callable (coroutine function)

    Example:
        >>> import asyncio
        >>> # Compile an AsyncFunction IR that calls async platform functions
        >>> func = compile_async(function_ir, platform)
        >>> asyncio.run(func(5))
    """
    platform_fns: dict[str, Callable[..., Any]] = {}
    async_platform_fns: set[str] = set()
    platform_list = platform or []

    if platform_list:
        platform_fns = {pf["name"]: pf["fn"] for pf in platform_list}
        async_platform_fns = {pf["name"] for pf in platform_list if pf["type"] == "async"}

    compiled, _ = _compile_ir(ir, platform_fns, async_platform_fns)

    # If compiled is a FunctionFactory, unwrap it with empty environment
    if isinstance(compiled, FunctionFactory):
        return compiled.make({})

    return compiled


def _compile_ir(
    ir: IR,
    platform_fns: dict[str, Callable[..., Any]],
    async_platform_fns: set[str],
) -> tuple[Callable, bool]:
    """Internal helper to compile IR nodes recursively.

    Args:
        ir: IR node to compile
        platform_fns: Dictionary mapping platform function names to implementations
        async_platform_fns: Set of platform function names that are async

    Returns:
        Tuple of (compiled callable, is_async)
    """
    tag = ir["type"]

    if tag == "Function":
        return _compile_function(ir, platform_fns, async_platform_fns)
    if tag == "AsyncFunction":
        return _compile_async_function(ir, platform_fns, async_platform_fns)
    if tag == "Value":
        return _compile_value(ir, platform_fns, async_platform_fns)
    if tag == "Variable":
        return _compile_variable(ir, platform_fns, async_platform_fns)
    if tag == "Builtin":
        return _compile_builtin(ir, platform_fns, async_platform_fns)
    if tag == "Block":
        return _compile_block(ir, platform_fns, async_platform_fns)
    if tag == "IfElse":
        return _compile_ifelse(ir, platform_fns, async_platform_fns)
    if tag == "While":
        return _compile_while(ir, platform_fns, async_platform_fns)
    if tag == "Let":
        return _compile_let(ir, platform_fns, async_platform_fns)
    if tag == "Platform":
        return _compile_platform(ir, platform_fns, async_platform_fns)
    if tag == "TryCatch":
        return _compile_trycatch(ir, platform_fns, async_platform_fns)
    if tag == "NewRef":
        return _compile_new_ref(ir, platform_fns, async_platform_fns)
    if tag == "Call":
        return _compile_call(ir, platform_fns, async_platform_fns)
    if tag == "CallAsync":
        return _compile_call_async(ir, platform_fns, async_platform_fns)
    if tag == "As":
        return _compile_as(ir, platform_fns, async_platform_fns)
    if tag == "Return":
        return _compile_return(ir, platform_fns, async_platform_fns)
    if tag == "Assign":
        return _compile_assign(ir, platform_fns, async_platform_fns)
    if tag == "Struct":
        return _compile_struct(ir, platform_fns, async_platform_fns)
    if tag == "GetField":
        return _compile_getfield(ir, platform_fns, async_platform_fns)
    if tag == "Variant":
        return _compile_variant(ir, platform_fns, async_platform_fns)
    if tag == "Match":
        return _compile_match(ir, platform_fns, async_platform_fns)
    if tag == "NewArray":
        return _compile_newarray(ir, platform_fns, async_platform_fns)
    if tag == "NewDict":
        return _compile_newdict(ir, platform_fns, async_platform_fns)
    if tag == "NewSet":
        return _compile_newset(ir, platform_fns, async_platform_fns)
    if tag == "ForArray":
        return _compile_forarray(ir, platform_fns, async_platform_fns)
    if tag == "ForSet":
        return _compile_forset(ir, platform_fns, async_platform_fns)
    if tag == "ForDict":
        return _compile_fordict(ir, platform_fns, async_platform_fns)
    if tag == "Break":
        return _compile_break(ir, platform_fns, async_platform_fns)
    if tag == "Continue":
        return _compile_continue(ir, platform_fns, async_platform_fns)
    if tag == "Error":
        return _compile_error(ir, platform_fns, async_platform_fns)
    if tag == "UnwrapRecursive":
        return _compile_unwraprecursive(ir, platform_fns, async_platform_fns)
    if tag == "WrapRecursive":
        return _compile_wraprecursive(ir, platform_fns, async_platform_fns)
    raise NotImplementedError(f"Compilation for {tag} not yet implemented")


def _compile_function(
    node: IR,
    platform_fns: dict[str, Callable[..., Any]],
    async_platform_fns: set[str],
) -> tuple[Callable, bool]:
    """Compile a Function IR node to a Python callable.

    The function captures its environment and returns a closure that can be called.
    """
    func_struct = node["value"]

    # Compile the body
    body_compiled, body_is_async = _compile_ir(
        func_struct["body"], platform_fns, async_platform_fns
    )

    # Get parameter names from parameter Variable IR nodes
    param_names = [param["value"]["name"] for param in func_struct["parameters"]]

    # Get captured variable names
    capture_names = [cap["value"]["name"] for cap in func_struct["captures"]]

    if body_is_async:
        # Return a function that takes the parent environment and returns the actual callable
        def make_async_fn(parent_env):
            # Create async Python function
            async def compiled_fn_async(*args):
                if len(args) != len(param_names):
                    raise TypeError(
                        f"Function expects {len(param_names)} arguments, got {len(args)}"
                    )

                # Create environment with parameters
                local_env = dict(zip(param_names, args, strict=False))

                # Use capture-aware environment if there are captures
                if capture_names:
                    env = CaptureAwareEnv(local_env, parent_env, capture_names)
                else:
                    env = local_env

                # Execute body with environment - body returns a coroutine
                try:
                    return await body_compiled(env)
                except ReturnException as e:
                    return e.value

            return compiled_fn_async

        return FunctionFactory(make_async_fn), False  # Function definition itself is not async

    # Return a function that takes the parent environment and returns the actual callable
    def make_sync_fn(parent_env):
        # Create sync Python function
        def compiled_fn_sync(*args):
            if len(args) != len(param_names):
                raise TypeError(f"Function expects {len(param_names)} arguments, got {len(args)}")

            # Create environment with parameters
            local_env = dict(zip(param_names, args, strict=False))

            # Use capture-aware environment if there are captures
            if capture_names:
                env = CaptureAwareEnv(local_env, parent_env, capture_names)
            else:
                env = local_env

            try:
                return body_compiled(env)
            except ReturnException as e:
                return e.value

        return compiled_fn_sync

    return FunctionFactory(make_sync_fn), False


def _compile_async_function(
    node: IR,
    platform_fns: dict[str, Callable[..., Any]],
    async_platform_fns: set[str],
) -> tuple[Callable, bool]:
    """Compile an AsyncFunction IR node to a Python async callable.

    AsyncFunction nodes always produce async functions.
    """
    func_struct = node["value"]

    # Compile the body
    body_compiled, _ = _compile_ir(func_struct["body"], platform_fns, async_platform_fns)

    # Get parameter names from parameter Variable IR nodes
    param_names = [param["value"]["name"] for param in func_struct["parameters"]]

    # Get captured variable names
    capture_names = [cap["value"]["name"] for cap in func_struct["captures"]]

    # AsyncFunction always creates an async callable
    def make_async_fn(parent_env):
        # Create async Python function
        async def compiled_fn_async(*args):
            if len(args) != len(param_names):
                raise TypeError(f"Function expects {len(param_names)} arguments, got {len(args)}")

            # Create environment with parameters
            local_env = dict(zip(param_names, args, strict=False))

            # Use capture-aware environment if there are captures
            if capture_names:
                env = CaptureAwareEnv(local_env, parent_env, capture_names)
            else:
                env = local_env

            try:
                result = body_compiled(env)
                # Always await in async function if body returns a coroutine
                if hasattr(result, "__await__"):
                    result = await result
                return result
            except ReturnException as e:
                return e.value

        return compiled_fn_async

    # Creating an async function is NOT async (isAsync: false in TS)
    return FunctionFactory(make_async_fn), False


def _compile_value(
    node: IR,
    platform_fns: dict[str, Callable[..., Any]],
    async_platform_fns: set[str],
) -> tuple[Callable, bool]:
    """Compile a Value IR node (literal constant)."""
    lit_val_variant = node["value"]["value"]
    value = lit_val_variant["value"]
    return lambda env: value, False


def _compile_variable(
    node: IR,
    platform_fns: dict[str, Callable[..., Any]],
    async_platform_fns: set[str],
) -> tuple[Callable, bool]:
    """Compile a Variable IR node (variable reference)."""
    name = node["value"]["name"]
    return lambda env: env[name], False


def _compile_builtin(
    node: IR,
    platform_fns: dict[str, Callable[..., Any]],
    async_platform_fns: set[str],
) -> tuple[Callable, bool]:
    """Compile a Builtin IR node (builtin function call)."""
    builtin_struct = node["value"]
    builtin_name = builtin_struct["builtin"]
    builtin_factory = get_builtin(builtin_name)
    type_params = builtin_struct["type_parameters"]
    specialized_fn = builtin_factory(*type_params)

    # Compile all arguments and track async
    arg_info = []
    any_arg_async = False
    for arg in builtin_struct["arguments"]:
        arg_fn, arg_is_async = _compile_ir(arg, platform_fns, async_platform_fns)
        arg_info.append((arg_fn, arg_is_async))
        if arg_is_async:
            any_arg_async = True

    if any_arg_async:

        async def call_builtin_async(env):
            args = []
            for arg_fn, arg_is_async in arg_info:
                if arg_is_async:
                    args.append(await arg_fn(env))
                else:
                    args.append(arg_fn(env))
            return specialized_fn(*args)

        return call_builtin_async, True

    def call_builtin_sync(env):
        args = [arg_fn for arg_fn, _ in arg_info]
        return specialized_fn(*[f(env) for f in args])

    return call_builtin_sync, False


def _compile_block(
    node: IR,
    platform_fns: dict[str, Callable[..., Any]],
    async_platform_fns: set[str],
) -> tuple[Callable, bool]:
    """Compile a Block IR node (sequence of statements)."""
    block_struct = node["value"]

    stmt_info = []
    any_stmt_async = False
    for stmt in block_struct["statements"]:
        stmt_fn, stmt_is_async = _compile_ir(stmt, platform_fns, async_platform_fns)
        stmt_info.append((stmt_fn, stmt_is_async))
        if stmt_is_async:
            any_stmt_async = True

    if any_stmt_async:

        async def execute_block_async(env):
            result = None
            for stmt_fn, is_async in stmt_info:
                if is_async:
                    result = await stmt_fn(env)
                else:
                    result = stmt_fn(env)
            return result

        return execute_block_async, True

    def execute_block_sync(env):
        result = None
        for stmt_fn, _ in stmt_info:
            result = stmt_fn(env)
        return result

    return execute_block_sync, False


def _compile_ifelse(
    node: IR,
    platform_fns: dict[str, Callable[..., Any]],
    async_platform_fns: set[str],
) -> tuple[Callable, bool]:
    """Compile an IfElse IR node."""
    ifelse_struct = node["value"]

    if_cases_info = []
    any_async = False
    for case in ifelse_struct["ifs"]:
        pred_fn, pred_is_async = _compile_ir(case["predicate"], platform_fns, async_platform_fns)
        body_fn, body_is_async = _compile_ir(case["body"], platform_fns, async_platform_fns)
        if_cases_info.append((pred_fn, pred_is_async, body_fn, body_is_async))
        if pred_is_async or body_is_async:
            any_async = True

    else_fn, else_is_async = _compile_ir(
        ifelse_struct["else_body"], platform_fns, async_platform_fns
    )
    if else_is_async:
        any_async = True

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

        return execute_ifelse_async, True

    def execute_ifelse_sync(env):
        for pred_fn, _, body_fn, _ in if_cases_info:
            if pred_fn(env):
                return body_fn(env)
        return else_fn(env)

    return execute_ifelse_sync, False


def _compile_while(
    node: IR,
    platform_fns: dict[str, Callable[..., Any]],
    async_platform_fns: set[str],
) -> tuple[Callable, bool]:
    """Compile a While IR node."""
    while_struct = node["value"]
    predicate_compiled, pred_is_async = _compile_ir(
        while_struct["predicate"], platform_fns, async_platform_fns
    )
    body_compiled, body_is_async = _compile_ir(
        while_struct["body"], platform_fns, async_platform_fns
    )
    label = while_struct["label"]["name"]

    if pred_is_async or body_is_async:

        async def execute_while_async(env):
            from east.types.values import east_null

            while True:
                if pred_is_async:
                    predicate_result = await predicate_compiled(env)
                else:
                    predicate_result = predicate_compiled(env)

                if not predicate_result:
                    break

                try:
                    if body_is_async:
                        await body_compiled(env)
                    else:
                        body_compiled(env)
                except ContinueException as e:
                    if e.label == label:
                        continue
                    raise
                except BreakException as e:
                    if e.label == label:
                        break
                    raise

            return east_null

        return execute_while_async, True

    def execute_while_sync(env):
        from east.types.values import east_null

        while predicate_compiled(env):
            try:
                body_compiled(env)
            except ContinueException as e:
                if e.label == label:
                    continue
                raise
            except BreakException as e:
                if e.label == label:
                    break
                raise

        return east_null

    return execute_while_sync, False


def _compile_let(
    node: IR,
    platform_fns: dict[str, Callable[..., Any]],
    async_platform_fns: set[str],
) -> tuple[Callable, bool]:
    """Compile a Let IR node (variable binding)."""
    let_struct = node["value"]
    var_name = let_struct["variable"]["value"]["name"]
    value_compiled, value_is_async = _compile_ir(
        let_struct["value"], platform_fns, async_platform_fns
    )

    if value_is_async:

        async def execute_let_async(env):
            from east.types.values import east_null

            value = await value_compiled(env)
            env[var_name] = value
            return east_null

        return execute_let_async, True

    def execute_let_sync(env):
        from east.types.values import east_null

        value = value_compiled(env)
        env[var_name] = value
        return east_null

    return execute_let_sync, False


def _compile_platform(
    node: IR,
    platform_fns: dict[str, Callable[..., Any]],
    async_platform_fns: set[str],
) -> tuple[Callable, bool]:
    """Compile a Platform IR node (platform function call)."""
    platform_struct = node["value"]
    platform_name = platform_struct["name"]

    if platform_name not in platform_fns:
        raise ValueError(
            f"Platform function '{platform_name}' not found. "
            f"Available platform functions: {', '.join(platform_fns.keys())}"
        )

    platform_fn = platform_fns[platform_name]
    # Use the async field from the IR node (new design)
    is_async_fn = platform_struct.get("async", platform_name in async_platform_fns)

    arg_info = []
    any_arg_async = False
    for arg in platform_struct["arguments"]:
        arg_fn, arg_is_async = _compile_ir(arg, platform_fns, async_platform_fns)
        arg_info.append((arg_fn, arg_is_async))
        if arg_is_async:
            any_arg_async = True

    if is_async_fn or any_arg_async:

        async def call_platform_async(env):
            args = []
            for arg_fn, arg_is_async in arg_info:
                if isinstance(arg_fn, FunctionFactory):
                    arg = arg_fn.make(env)
                elif arg_is_async:
                    arg = await arg_fn(env)
                else:
                    arg = arg_fn(env)
                if isinstance(arg, FunctionFactory):
                    arg = arg.make(env)
                args.append(arg)

            if is_async_fn:
                return await platform_fn(*args)
            return platform_fn(*args)

        return call_platform_async, True

    def call_platform_sync(env):
        args = []
        for arg_fn, _ in arg_info:
            if isinstance(arg_fn, FunctionFactory):
                arg = arg_fn.make(env)
            else:
                arg = arg_fn(env)
                if isinstance(arg, FunctionFactory):
                    arg = arg.make(env)
            args.append(arg)
        return platform_fn(*args)

    return call_platform_sync, False


def _compile_new_ref(
    node: IR,
    platform_fns: dict[str, Callable[..., Any]],
    async_platform_fns: set[str],
) -> tuple[Callable, bool]:
    """Compile a NewRef IR node (creates a reference cell)."""
    from east.types.values import east_ref

    newref_struct = node["value"]
    value_fn, value_is_async = _compile_ir(newref_struct["value"], platform_fns, async_platform_fns)

    if value_is_async:

        async def execute_new_ref_async(env):
            val = await value_fn(env)
            return east_ref(val)

        return execute_new_ref_async, True

    def execute_new_ref_sync(env):
        val = value_fn(env)
        return east_ref(val)

    return execute_new_ref_sync, False


def _extract_stack_trace(exception: Exception):
    """Extract stack trace from exception and convert to East format.

    Returns:
        List of structs with {filename: str, line: int, column: int}
    """
    from east.ir.builders import location
    from east.types.types import IntegerType, StringType, StructType
    from east.types.values import EastArray

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
    node: IR,
    platform_fns: dict[str, Callable[..., Any]],
    async_platform_fns: set[str],
) -> tuple[Callable, bool]:
    """Compile a TryCatch IR node (try-catch-finally error handling)."""
    trycatch_struct = node["value"]

    try_body_fn, try_is_async = _compile_ir(
        trycatch_struct["try_body"], platform_fns, async_platform_fns
    )

    message_var = trycatch_struct["message"]["value"]
    message_name = message_var["name"]
    stack_var = trycatch_struct["stack"]["value"]
    stack_name = stack_var["name"]

    catch_body_fn, catch_is_async = _compile_ir(
        trycatch_struct["catch_body"], platform_fns, async_platform_fns
    )

    finally_body_fn = None
    finally_is_async = False

    if trycatch_struct["finally_body"]["type"] != "Value":
        finally_body_fn, finally_is_async = _compile_ir(
            trycatch_struct["finally_body"], platform_fns, async_platform_fns
        )

    is_async = try_is_async or catch_is_async or finally_is_async

    if is_async:
        if finally_body_fn is None:

            async def execute_trycatch_async(env):
                try:
                    if try_is_async:
                        return await try_body_fn(env)
                    return try_body_fn(env)
                except Exception as e:
                    catch_env = _make_child_env(
                        env,
                        {
                            message_name: str(e),
                            stack_name: _extract_stack_trace(e),
                        },
                    )

                    if catch_is_async:
                        result = await catch_body_fn(catch_env)
                    else:
                        result = catch_body_fn(catch_env)

                    for key, value in catch_env.items():
                        if key not in (message_name, stack_name):
                            env[key] = value

                    return result

            return execute_trycatch_async, True

        async def execute_trycatch_async_finally(env):
            try:
                if try_is_async:
                    result = await try_body_fn(env)
                else:
                    result = try_body_fn(env)
            except ReturnException:
                if finally_is_async:
                    await finally_body_fn(env)
                else:
                    finally_body_fn(env)
                raise
            except Exception as e:
                catch_env = _make_child_env(
                    env,
                    {
                        message_name: str(e),
                        stack_name: _extract_stack_trace(e),
                    },
                )

                if catch_is_async:
                    result = await catch_body_fn(catch_env)
                else:
                    result = catch_body_fn(catch_env)

                for key, value in catch_env.items():
                    if key not in (message_name, stack_name):
                        env[key] = value
            finally:
                if finally_is_async:
                    await finally_body_fn(env)
                else:
                    finally_body_fn(env)

            return result

        return execute_trycatch_async_finally, True

    if finally_body_fn is None:

        def execute_trycatch_sync(env):
            try:
                return try_body_fn(env)
            except Exception as e:
                catch_env = _make_child_env(
                    env,
                    {
                        message_name: str(e),
                        stack_name: _extract_stack_trace(e),
                    },
                )
                result = catch_body_fn(catch_env)

                for key, value in catch_env.items():
                    if key not in (message_name, stack_name):
                        env[key] = value

                return result

        return execute_trycatch_sync, False

    def execute_trycatch_sync_finally(env):
        try:
            result = try_body_fn(env)
        except ReturnException:
            finally_body_fn(env)
            raise
        except Exception as e:
            catch_env = _make_child_env(
                env,
                {
                    message_name: str(e),
                    stack_name: _extract_stack_trace(e),
                },
            )
            result = catch_body_fn(catch_env)

            for key, value in catch_env.items():
                if key not in (message_name, stack_name):
                    env[key] = value
        finally:
            finally_body_fn(env)

        return result

    return execute_trycatch_sync_finally, False


def _compile_call(
    node: IR,
    platform_fns: dict[str, Callable[..., Any]],
    async_platform_fns: set[str],
) -> tuple[Callable, bool]:
    """Compile function call IR node (for sync functions).

    Call is used for sync function calls. The result is only async if the
    function expression or arguments are async.
    """
    func_compiled, func_is_async = _compile_ir(
        node["value"]["function"], platform_fns, async_platform_fns
    )

    args_info = []
    any_arg_async = False
    for arg in node["value"]["arguments"]:
        arg_fn, arg_is_async = _compile_ir(arg, platform_fns, async_platform_fns)
        args_info.append((arg_fn, arg_is_async))
        if arg_is_async:
            any_arg_async = True

    is_async = func_is_async or any_arg_async

    if is_async:

        async def call_async(env):
            if func_is_async:
                func = await func_compiled(env)
            else:
                func = func_compiled(env)
            if isinstance(func, FunctionFactory):
                func = func.make(env)
            args = []
            for arg_fn, arg_is_async in args_info:
                if arg_is_async:
                    arg = await arg_fn(env)
                else:
                    arg = arg_fn(env)
                if isinstance(arg, FunctionFactory):
                    arg = arg.make(env)
                args.append(arg)
            # Call sync function (no await on result)
            return func(*args)

        return call_async, True

    def call_sync(env):
        func = func_compiled(env)
        if isinstance(func, FunctionFactory):
            func = func.make(env)
        args = []
        for arg_fn, _ in args_info:
            arg = arg_fn(env)
            if isinstance(arg, FunctionFactory):
                arg = arg.make(env)
            args.append(arg)
        return func(*args)

    return call_sync, False


def _compile_call_async(
    node: IR,
    platform_fns: dict[str, Callable[..., Any]],
    async_platform_fns: set[str],
) -> tuple[Callable, bool]:
    """Compile async function call IR node (CallAsync).

    CallAsync is used to call async functions and always awaits the result.
    """
    func_compiled, func_is_async = _compile_ir(
        node["value"]["function"], platform_fns, async_platform_fns
    )

    args_info = []
    for arg in node["value"]["arguments"]:
        arg_fn, arg_is_async = _compile_ir(arg, platform_fns, async_platform_fns)
        args_info.append((arg_fn, arg_is_async))

    # CallAsync is always async
    async def call_async_exec(env):
        if func_is_async:
            func = await func_compiled(env)
        else:
            func = func_compiled(env)
        if isinstance(func, FunctionFactory):
            func = func.make(env)

        # Evaluate and await all arguments if needed
        args = []
        for arg_fn, arg_is_async in args_info:
            if arg_is_async:
                arg = await arg_fn(env)
            else:
                arg = arg_fn(env)
            if isinstance(arg, FunctionFactory):
                arg = arg.make(env)
            args.append(arg)

        # Call the async function and await the result
        result = func(*args)
        if hasattr(result, "__await__"):
            result = await result
        return result

    return call_async_exec, True


def _compile_as(
    node: IR,
    platform_fns: dict[str, Callable[..., Any]],
    async_platform_fns: set[str],
) -> tuple[Callable, bool]:
    """Compile type assertion (As)."""
    return _compile_ir(node["value"]["value"], platform_fns, async_platform_fns)


def _compile_return(
    node: IR,
    platform_fns: dict[str, Callable[..., Any]],
    async_platform_fns: set[str],
) -> tuple[Callable, bool]:
    """Compile return statement."""
    value_compiled, value_is_async = _compile_ir(
        node["value"]["value"], platform_fns, async_platform_fns
    )

    if value_is_async:

        async def return_async(env):
            value = await value_compiled(env)
            raise ReturnException(value)

        return return_async, True

    def return_sync(env):
        value = value_compiled(env)
        raise ReturnException(value)

    return return_sync, False


def _compile_assign(
    node: IR,
    platform_fns: dict[str, Callable[..., Any]],
    async_platform_fns: set[str],
) -> tuple[Callable, bool]:
    """Compile variable assignment."""
    value_compiled, value_is_async = _compile_ir(
        node["value"]["value"], platform_fns, async_platform_fns
    )
    variable_name = node["value"]["variable"]["value"]["name"]

    if value_is_async:

        async def assign_async(env):
            env[variable_name] = await value_compiled(env)
            return EastNull()

        return assign_async, True

    def assign_sync(env):
        env[variable_name] = value_compiled(env)
        return EastNull()

    return assign_sync, False


def _compile_struct(
    node: IR,
    platform_fns: dict[str, Callable[..., Any]],
    async_platform_fns: set[str],
) -> tuple[Callable, bool]:
    """Compile struct literal."""
    field_names = [field["name"] for field in node["value"]["fields"]]
    fields_info = []
    any_async = False
    for field in node["value"]["fields"]:
        fn, is_async = _compile_ir(field["value"], platform_fns, async_platform_fns)
        fields_info.append((fn, is_async))
        if is_async:
            any_async = True

    if any_async:

        async def struct_async(env):
            struct = {}
            for name, (field_fn, field_is_async) in zip(field_names, fields_info, strict=True):
                if field_is_async:
                    struct[name] = await field_fn(env)
                else:
                    struct[name] = field_fn(env)
            return EastStruct(struct)

        return struct_async, True

    def struct_sync(env):
        struct = {}
        for name, (field_fn, _) in zip(field_names, fields_info, strict=True):
            struct[name] = field_fn(env)
        return EastStruct(struct)

    return struct_sync, False


def _compile_getfield(
    node: IR,
    platform_fns: dict[str, Callable[..., Any]],
    async_platform_fns: set[str],
) -> tuple[Callable, bool]:
    """Compile struct field access."""
    struct_compiled, struct_is_async = _compile_ir(
        node["value"]["struct"], platform_fns, async_platform_fns
    )
    field_name = node["value"]["field"]

    if struct_is_async:

        async def getfield_async(env):
            struct = await struct_compiled(env)
            return struct[field_name]

        return getfield_async, True

    def getfield_sync(env):
        struct = struct_compiled(env)
        return struct[field_name]

    return getfield_sync, False


def _compile_variant(
    node: IR,
    platform_fns: dict[str, Callable[..., Any]],
    async_platform_fns: set[str],
) -> tuple[Callable, bool]:
    """Compile variant constructor."""
    value_compiled, value_is_async = _compile_ir(
        node["value"]["value"], platform_fns, async_platform_fns
    )
    case_name = node["value"]["case"]

    if value_is_async:

        async def variant_async(env):
            value = await value_compiled(env)
            return EastVariant(case_name, value)

        return variant_async, True

    def variant_sync(env):
        value = value_compiled(env)
        return EastVariant(case_name, value)

    return variant_sync, False


def _compile_match(
    node: IR,
    platform_fns: dict[str, Callable[..., Any]],
    async_platform_fns: set[str],
) -> tuple[Callable, bool]:
    """Compile match (pattern matching on variants)."""
    variant_compiled, variant_is_async = _compile_ir(
        node["value"]["variant"], platform_fns, async_platform_fns
    )

    cases_compiled = {}
    any_body_async = False
    for case in node["value"]["cases"]:
        case_name = case["case"]
        variable_name = case["variable"]["value"]["name"]
        body_compiled, body_is_async = _compile_ir(case["body"], platform_fns, async_platform_fns)
        cases_compiled[case_name] = (variable_name, body_compiled, body_is_async)
        if body_is_async:
            any_body_async = True

    is_async = variant_is_async or any_body_async

    if is_async:

        async def match_async(env):
            if variant_is_async:
                variant = await variant_compiled(env)
            else:
                variant = variant_compiled(env)
            case_name = variant["type"]
            case_value = variant["value"]

            variable_name, body_compiled, body_is_async = cases_compiled[case_name]
            env[variable_name] = case_value
            try:
                if body_is_async:
                    return await body_compiled(env)
                return body_compiled(env)
            finally:
                del env[variable_name]

        return match_async, True

    def match_sync(env):
        variant = variant_compiled(env)
        case_name = variant["type"]
        case_value = variant["value"]

        variable_name, body_compiled, _ = cases_compiled[case_name]
        env[variable_name] = case_value
        try:
            return body_compiled(env)
        finally:
            del env[variable_name]

    return match_sync, False


def _compile_newarray(
    node: IR,
    platform_fns: dict[str, Callable[..., Any]],
    async_platform_fns: set[str],
) -> tuple[Callable, bool]:
    """Compile array literal."""
    from east.types.values import EastArray

    elements_info = []
    any_async = False
    for elem in node["value"]["values"]:
        fn, is_async = _compile_ir(elem, platform_fns, async_platform_fns)
        elements_info.append((fn, is_async))
        if is_async:
            any_async = True

    element_type = node["value"]["type"]["value"]

    if any_async:

        async def newarray_async(env):
            elements = []
            for elem_fn, elem_is_async in elements_info:
                if elem_is_async:
                    elements.append(await elem_fn(env))
                else:
                    elements.append(elem_fn(env))
            return EastArray(element_type, elements)

        return newarray_async, True

    def newarray_sync(env):
        elements = [fn(env) for fn, _ in elements_info]
        return EastArray(element_type, elements)

    return newarray_sync, False


def _compile_newset(
    node: IR,
    platform_fns: dict[str, Callable[..., Any]],
    async_platform_fns: set[str],
) -> tuple[Callable, bool]:
    """Compile set literal."""
    from east.types.values import EastSet

    elements_info = []
    any_async = False
    for elem in node["value"]["values"]:
        fn, is_async = _compile_ir(elem, platform_fns, async_platform_fns)
        elements_info.append((fn, is_async))
        if is_async:
            any_async = True

    element_type = node["value"]["type"]["value"]

    if any_async:

        async def newset_async(env):
            elements = []
            for elem_fn, elem_is_async in elements_info:
                if elem_is_async:
                    elements.append(await elem_fn(env))
                else:
                    elements.append(elem_fn(env))
            return EastSet(element_type, elements)

        return newset_async, True

    def newset_sync(env):
        elements = [fn(env) for fn, _ in elements_info]
        return EastSet(element_type, elements)

    return newset_sync, False


def _compile_newdict(
    node: IR,
    platform_fns: dict[str, Callable[..., Any]],
    async_platform_fns: set[str],
) -> tuple[Callable, bool]:
    """Compile dictionary literal."""
    from east.types.values import EastDict

    entries_info = []
    any_async = False
    for entry in node["value"]["values"]:
        key_fn, key_is_async = _compile_ir(entry["key"], platform_fns, async_platform_fns)
        val_fn, val_is_async = _compile_ir(entry["value"], platform_fns, async_platform_fns)
        entries_info.append((key_fn, key_is_async, val_fn, val_is_async))
        if key_is_async or val_is_async:
            any_async = True

    key_type = node["value"]["type"]["value"]["key"]
    value_type = node["value"]["type"]["value"]["value"]

    if any_async:

        async def newdict_async(env):
            entries = {}
            for key_fn, key_is_async, val_fn, val_is_async in entries_info:
                if key_is_async:
                    key = await key_fn(env)
                else:
                    key = key_fn(env)
                if val_is_async:
                    value = await val_fn(env)
                else:
                    value = val_fn(env)
                entries[key] = value
            return EastDict(key_type, value_type, entries)

        return newdict_async, True

    def newdict_sync(env):
        entries = {}
        for key_fn, _, val_fn, _ in entries_info:
            entries[key_fn(env)] = val_fn(env)
        return EastDict(key_type, value_type, entries)

    return newdict_sync, False


def _compile_forarray(
    node: IR,
    platform_fns: dict[str, Callable[..., Any]],
    async_platform_fns: set[str],
) -> tuple[Callable, bool]:
    """Compile for-array loop."""
    array_compiled, array_is_async = _compile_ir(
        node["value"]["array"], platform_fns, async_platform_fns
    )
    body_compiled, body_is_async = _compile_ir(
        node["value"]["body"], platform_fns, async_platform_fns
    )

    key_var_name = node["value"]["key"]["value"]["name"]
    element_var_name = node["value"]["value"]["value"]["name"]

    is_async = array_is_async or body_is_async

    if is_async:

        async def forarray_async(env):
            if array_is_async:
                array = await array_compiled(env)
            else:
                array = array_compiled(env)
            array._lock_for_iteration()
            try:
                for i, elem in enumerate(array):
                    child_env = _make_child_env(env, {key_var_name: i, element_var_name: elem})
                    try:
                        if body_is_async:
                            await body_compiled(child_env)
                        else:
                            body_compiled(child_env)
                        for key, value in child_env.items():
                            if key not in (key_var_name, element_var_name):
                                env[key] = value
                    except BreakException:
                        break
                    except ContinueException:
                        continue
                return EastNull()
            finally:
                array._unlock_for_iteration()

        return forarray_async, True

    def forarray_sync(env):
        array = array_compiled(env)
        array._lock_for_iteration()
        try:
            for i, elem in enumerate(array):
                child_env = _make_child_env(env, {key_var_name: i, element_var_name: elem})
                try:
                    body_compiled(child_env)
                    for key, value in child_env.items():
                        if key not in (key_var_name, element_var_name):
                            env[key] = value
                except BreakException:
                    break
                except ContinueException:
                    continue
            return EastNull()
        finally:
            array._unlock_for_iteration()

    return forarray_sync, False


def _compile_forset(
    node: IR,
    platform_fns: dict[str, Callable[..., Any]],
    async_platform_fns: set[str],
) -> tuple[Callable, bool]:
    """Compile for-set loop."""
    set_compiled, set_is_async = _compile_ir(node["value"]["set"], platform_fns, async_platform_fns)
    body_compiled, body_is_async = _compile_ir(
        node["value"]["body"], platform_fns, async_platform_fns
    )

    element_var_name = node["value"]["key"]["value"]["name"]

    is_async = set_is_async or body_is_async

    if is_async:

        async def forset_async(env):
            if set_is_async:
                east_set = await set_compiled(env)
            else:
                east_set = set_compiled(env)
            east_set._lock_for_iteration()
            try:
                for elem in east_set:
                    child_env = _make_child_env(env, {element_var_name: elem})
                    try:
                        if body_is_async:
                            await body_compiled(child_env)
                        else:
                            body_compiled(child_env)
                        for key, value in child_env.items():
                            if key != element_var_name:
                                env[key] = value
                    except BreakException:
                        break
                    except ContinueException:
                        continue
                return EastNull()
            finally:
                east_set._unlock_for_iteration()

        return forset_async, True

    def forset_sync(env):
        east_set = set_compiled(env)
        east_set._lock_for_iteration()
        try:
            for elem in east_set:
                child_env = _make_child_env(env, {element_var_name: elem})
                try:
                    body_compiled(child_env)
                    for key, value in child_env.items():
                        if key != element_var_name:
                            env[key] = value
                except BreakException:
                    break
                except ContinueException:
                    continue
            return EastNull()
        finally:
            east_set._unlock_for_iteration()

    return forset_sync, False


def _compile_fordict(
    node: IR,
    platform_fns: dict[str, Callable[..., Any]],
    async_platform_fns: set[str],
) -> tuple[Callable, bool]:
    """Compile for-dict loop."""
    dict_compiled, dict_is_async = _compile_ir(
        node["value"]["dict"], platform_fns, async_platform_fns
    )
    body_compiled, body_is_async = _compile_ir(
        node["value"]["body"], platform_fns, async_platform_fns
    )

    key_var_name = node["value"]["key"]["value"]["name"]
    value_var_name = node["value"]["value"]["value"]["name"]

    is_async = dict_is_async or body_is_async

    if is_async:

        async def fordict_async(env):
            if dict_is_async:
                east_dict = await dict_compiled(env)
            else:
                east_dict = dict_compiled(env)
            east_dict._lock_for_iteration()
            try:
                for key, value in east_dict.items():
                    child_env = _make_child_env(env, {key_var_name: key, value_var_name: value})
                    try:
                        if body_is_async:
                            await body_compiled(child_env)
                        else:
                            body_compiled(child_env)
                        for k, v in child_env.items():
                            if k not in (key_var_name, value_var_name):
                                env[k] = v
                    except BreakException:
                        break
                    except ContinueException:
                        continue
                return EastNull()
            finally:
                east_dict._unlock_for_iteration()

        return fordict_async, True

    def fordict_sync(env):
        east_dict = dict_compiled(env)
        east_dict._lock_for_iteration()
        try:
            for key, value in east_dict.items():
                child_env = _make_child_env(env, {key_var_name: key, value_var_name: value})
                try:
                    body_compiled(child_env)
                    for k, v in child_env.items():
                        if k not in (key_var_name, value_var_name):
                            env[k] = v
                except BreakException:
                    break
                except ContinueException:
                    continue
            return EastNull()
        finally:
            east_dict._unlock_for_iteration()

    return fordict_sync, False


def _compile_break(
    node: IR,
    platform_fns: dict[str, Callable[..., Any]],
    async_platform_fns: set[str],
) -> tuple[Callable, bool]:
    """Compile break statement."""
    label = node["value"]["label"]["name"]

    def break_sync(env):
        raise BreakException(label)

    return break_sync, False


def _compile_continue(
    node: IR,
    platform_fns: dict[str, Callable[..., Any]],
    async_platform_fns: set[str],
) -> tuple[Callable, bool]:
    """Compile continue statement."""
    label = node["value"]["label"]["name"]

    def continue_sync(env):
        raise ContinueException(label)

    return continue_sync, False


def _compile_error(
    node: IR,
    platform_fns: dict[str, Callable[..., Any]],
    async_platform_fns: set[str],
) -> tuple[Callable, bool]:
    """Compile error (throw exception)."""
    message_compiled, message_is_async = _compile_ir(
        node["value"]["message"], platform_fns, async_platform_fns
    )

    if message_is_async:

        async def error_async(env):
            message = await message_compiled(env)
            raise RuntimeError(message)

        return error_async, True

    def error_sync(env):
        message = message_compiled(env)
        raise RuntimeError(message)

    return error_sync, False


def _compile_unwraprecursive(
    node: IR,
    platform_fns: dict[str, Callable[..., Any]],
    async_platform_fns: set[str],
) -> tuple[Callable, bool]:
    """Compile unwrap of recursive type."""
    return _compile_ir(node["value"]["value"], platform_fns, async_platform_fns)


def _compile_wraprecursive(
    node: IR,
    platform_fns: dict[str, Callable[..., Any]],
    async_platform_fns: set[str],
) -> tuple[Callable, bool]:
    """Compile wrap in recursive type."""
    return _compile_ir(node["value"]["value"], platform_fns, async_platform_fns)
