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
from east.types.primitives import Null
from east.types.structural import EastStruct, EastVariant
from east.types.types import IRNode


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


def compile(ir: IRNode, platform: list[PlatformFunction] | None = None) -> Callable:
    """Compile East IR to a native Python callable (synchronous).

    Args:
        ir: IR node (IRNode) to compile (typically a Function node)
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

    compiled = _compile_ir(ir, platform_fns, async_platform_fns, is_async_map)

    # If compiled is a FunctionFactory, unwrap it with empty environment
    if isinstance(compiled, FunctionFactory):
        return compiled.make({})

    return compiled


def compile_async(ir: IRNode, platform: list[PlatformFunction] | None = None) -> Callable:
    """Compile East IR to a native Python async callable.

    Args:
        ir: IR node (IRNode) to compile (typically a Function node)
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

    # If compiled is a FunctionFactory, unwrap it with empty environment
    if isinstance(compiled, FunctionFactory):
        return compiled.make({})

    return compiled


def _compile_ir(
    ir: IRNode,
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
    tag = ir["type"]

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
    if tag == "Call":
        return _compile_call(ir, platform_fns, async_platform_fns, is_async_map)
    if tag == "As":
        return _compile_as(ir, platform_fns, async_platform_fns, is_async_map)
    if tag == "Return":
        return _compile_return(ir, platform_fns, async_platform_fns, is_async_map)
    if tag == "Assign":
        return _compile_assign(ir, platform_fns, async_platform_fns, is_async_map)
    if tag == "Struct":
        return _compile_struct(ir, platform_fns, async_platform_fns, is_async_map)
    if tag == "GetField":
        return _compile_getfield(ir, platform_fns, async_platform_fns, is_async_map)
    if tag == "Variant":
        return _compile_variant(ir, platform_fns, async_platform_fns, is_async_map)
    if tag == "Match":
        return _compile_match(ir, platform_fns, async_platform_fns, is_async_map)
    if tag == "NewArray":
        return _compile_newarray(ir, platform_fns, async_platform_fns, is_async_map)
    if tag == "NewDict":
        return _compile_newdict(ir, platform_fns, async_platform_fns, is_async_map)
    if tag == "NewSet":
        return _compile_newset(ir, platform_fns, async_platform_fns, is_async_map)
    if tag == "ForArray":
        return _compile_forarray(ir, platform_fns, async_platform_fns, is_async_map)
    if tag == "ForSet":
        return _compile_forset(ir, platform_fns, async_platform_fns, is_async_map)
    if tag == "ForDict":
        return _compile_fordict(ir, platform_fns, async_platform_fns, is_async_map)
    if tag == "Break":
        return _compile_break(ir, platform_fns, async_platform_fns, is_async_map)
    if tag == "Continue":
        return _compile_continue(ir, platform_fns, async_platform_fns, is_async_map)
    if tag == "Error":
        return _compile_error(ir, platform_fns, async_platform_fns, is_async_map)
    if tag == "UnwrapRecursive":
        return _compile_unwraprecursive(ir, platform_fns, async_platform_fns, is_async_map)
    if tag == "WrapRecursive":
        return _compile_wraprecursive(ir, platform_fns, async_platform_fns, is_async_map)
    raise NotImplementedError(f"Compilation for {tag} not yet implemented")


def _compile_function(
    node: IRNode,
    platform_fns: dict[str, Callable[..., Any]],
    async_platform_fns: set[str],
    is_async_map: dict[int, bool],
) -> Callable:
    """Compile a Function IR node to a Python callable.

    The function captures its environment and returns a closure that can be called.
    """
    # Extract struct from variant
    func_struct = node["value"]

    # Compile the body
    body_compiled = _compile_ir(func_struct["body"], platform_fns, async_platform_fns, is_async_map)

    # Get parameter names from parameter Variable IR nodes
    param_names = [param["value"]["name"] for param in func_struct["parameters"]]

    # Get captured variable names
    capture_names = [cap["value"]["name"] for cap in func_struct["captures"]]

    # Check if body is async using static analysis
    body_is_async = is_async_map.get(id(func_struct["body"]), False)

    if body_is_async:
        # Return a function that takes the parent environment and returns the actual callable
        def make_async_fn(parent_env):
            # Create custom environment that delegates captured var assignments to parent
            class CaptureAwareEnv(dict):
                def __init__(self, local_vars, parent, captures):
                    super().__init__(local_vars)
                    self._parent = parent
                    self._captures = set(captures)

                def __getitem__(self, key):
                    # Check local first, then parent
                    if key in dict.keys(self):
                        return dict.__getitem__(self, key)
                    if key in self._captures:
                        return self._parent[key]
                    raise KeyError(key)

                def __setitem__(self, key, value):
                    # If captured variable, write to parent; otherwise write locally
                    if key in self._captures:
                        self._parent[key] = value
                    else:
                        dict.__setitem__(self, key, value)

                def __contains__(self, key):
                    return dict.__contains__(self, key) or (
                        key in self._captures and key in self._parent
                    )

                def get(self, key, default=None):
                    try:
                        return self[key]
                    except KeyError:
                        return default

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

        return FunctionFactory(make_async_fn)

    # Return a function that takes the parent environment and returns the actual callable
    def make_sync_fn(parent_env):
        # Create custom environment that delegates captured var assignments to parent
        class CaptureAwareEnv(dict):
            def __init__(self, local_vars, parent, captures):
                super().__init__(local_vars)
                self._parent = parent
                self._captures = set(captures)

            def __getitem__(self, key):
                # Check local first, then parent
                if key in dict.keys(self):
                    return dict.__getitem__(self, key)
                if key in self._captures:
                    return self._parent[key]
                raise KeyError(key)

            def __setitem__(self, key, value):
                # If captured variable, write to parent; otherwise write locally
                if key in self._captures:
                    self._parent[key] = value
                else:
                    dict.__setitem__(self, key, value)

            def __contains__(self, key):
                return dict.__contains__(self, key) or (
                    key in self._captures and key in self._parent
                )

            def get(self, key, default=None):
                try:
                    return self[key]
                except KeyError:
                    return default

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

    return FunctionFactory(make_sync_fn)


def _compile_value(
    node: IRNode,
    platform_fns: dict[str, Callable[..., Any]],
    async_platform_fns: set[str],
    is_async_map: dict[int, bool],
) -> Callable:
    """Compile a Value IR node (literal constant)."""
    # Extract the LiteralValue variant from the Value struct
    lit_val_variant = node["value"]["value"]
    # Extract the actual value from the LiteralValue variant
    value = lit_val_variant["value"]
    # Values are always sync
    return lambda env: value


def _compile_variable(
    node: IRNode,
    platform_fns: dict[str, Callable[..., Any]],
    async_platform_fns: set[str],
    is_async_map: dict[int, bool],
) -> Callable:
    """Compile a Variable IR node (variable reference)."""
    name = node["value"]["name"]
    # Variables are always sync
    return lambda env: env[name]


def _compile_builtin(
    node: IRNode,
    platform_fns: dict[str, Callable[..., Any]],
    async_platform_fns: set[str],
    is_async_map: dict[int, bool],
) -> Callable:
    """Compile a Builtin IR node (builtin function call).

    All builtins are factory functions that are called at compile time with
    type parameters to produce specialized implementations.
    """
    builtin_struct = node["value"]
    builtin_name = builtin_struct["builtin"]
    builtin_factory = get_builtin(builtin_name)

    # Extract type parameters
    type_params = builtin_struct["type_parameters"]

    # Call factory at compile time with type parameters to get specialized function
    specialized_fn = builtin_factory(*type_params)

    # Compile all arguments and check if any are async
    arg_compiled = []
    any_arg_async = False
    for arg in builtin_struct["arguments"]:
        arg_fn = _compile_ir(arg, platform_fns, async_platform_fns, is_async_map)
        arg_compiled.append(arg_fn)
        if is_async_map.get(id(arg), False):
            any_arg_async = True

    # If any argument is async, return async version
    if any_arg_async:

        async def call_builtin_async(env):
            # Evaluate all arguments, awaiting async ones
            args = []
            for arg_fn, arg_ir in zip(arg_compiled, builtin_struct["arguments"], strict=False):
                if is_async_map.get(id(arg_ir), False):
                    args.append(await arg_fn(env))
                else:
                    args.append(arg_fn(env))
            # Call the specialized function
            return specialized_fn(*args)

        return call_builtin_async

    # Otherwise return sync version
    def call_builtin_sync(env):
        args = [arg_fn(env) for arg_fn in arg_compiled]
        return specialized_fn(*args)

    return call_builtin_sync


def _compile_block(
    node: IRNode,
    platform_fns: dict[str, Callable[..., Any]],
    async_platform_fns: set[str],
    is_async_map: dict[int, bool],
) -> Callable:
    """Compile a Block IR node (sequence of statements)."""
    block_struct = node["value"]

    # Compile all statements and track which are async
    stmt_info = []
    any_stmt_async = False
    for stmt in block_struct["statements"]:
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
    node: IRNode,
    platform_fns: dict[str, Callable[..., Any]],
    async_platform_fns: set[str],
    is_async_map: dict[int, bool],
) -> Callable:
    """Compile an IfElse IR node."""
    ifelse_struct = node["value"]

    # Compile each if case and track if any are async
    if_cases_info = []
    any_async = False
    for case in ifelse_struct["ifs"]:
        pred_fn = _compile_ir(case["predicate"], platform_fns, async_platform_fns, is_async_map)
        body_fn = _compile_ir(case["body"], platform_fns, async_platform_fns, is_async_map)
        pred_is_async = is_async_map.get(id(case["predicate"]), False)
        body_is_async = is_async_map.get(id(case["body"]), False)
        if_cases_info.append((pred_fn, pred_is_async, body_fn, body_is_async))
        if pred_is_async or body_is_async:
            any_async = True

    else_fn = _compile_ir(
        ifelse_struct["else_body"], platform_fns, async_platform_fns, is_async_map
    )
    else_is_async = is_async_map.get(id(ifelse_struct["else_body"]), False)
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
    node: IRNode,
    platform_fns: dict[str, Callable[..., Any]],
    async_platform_fns: set[str],
    is_async_map: dict[int, bool],
) -> Callable:
    """Compile a While IR node."""
    while_struct = node["value"]
    predicate_compiled = _compile_ir(
        while_struct["predicate"], platform_fns, async_platform_fns, is_async_map
    )
    body_compiled = _compile_ir(
        while_struct["body"], platform_fns, async_platform_fns, is_async_map
    )
    label = while_struct["label"]["name"]

    # Check if predicate or body is async using static analysis
    pred_is_async = is_async_map.get(id(while_struct["predicate"]), False)
    body_is_async = is_async_map.get(id(while_struct["body"]), False)

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

            return null

        return execute_while_async

    # Otherwise return sync version
    def execute_while_sync(env):
        from east.types.primitives import null

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

        return null

    return execute_while_sync


def _compile_let(
    node: IRNode,
    platform_fns: dict[str, Callable[..., Any]],
    async_platform_fns: set[str],
    is_async_map: dict[int, bool],
) -> Callable:
    """Compile a Let IR node (variable binding)."""
    let_struct = node["value"]
    # Get the variable name from the Variable IR node
    var_name = let_struct["variable"]["value"]["name"]
    # Compile the value expression
    value_compiled = _compile_ir(
        let_struct["value"], platform_fns, async_platform_fns, is_async_map
    )

    # Check if value expression is async using static analysis
    value_is_async = is_async_map.get(id(let_struct["value"]), False)

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
    node: IRNode,
    platform_fns: dict[str, Callable[..., Any]],
    async_platform_fns: set[str],
    is_async_map: dict[int, bool],
) -> Callable:
    """Compile a Platform IR node (platform function call)."""
    platform_struct = node["value"]
    platform_name = platform_struct["name"]

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
    for arg in platform_struct["arguments"]:
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
                # Check if arg_fn is a FunctionFactory (from compiling Function IR node)
                if isinstance(arg_fn, FunctionFactory):
                    arg = arg_fn.make(env)
                elif arg_is_async:
                    arg = await arg_fn(env)
                else:
                    arg = arg_fn(env)
                # Unwrap FunctionFactory if the result is one
                if isinstance(arg, FunctionFactory):
                    arg = arg.make(env)
                args.append(arg)

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
        args = []
        for arg_fn, _ in arg_info:
            # Check if arg_fn is a FunctionFactory (from compiling Function IR node)
            if isinstance(arg_fn, FunctionFactory):
                arg = arg_fn.make(env)
            else:
                arg = arg_fn(env)
                # Unwrap FunctionFactory if the result is one
                if isinstance(arg, FunctionFactory):
                    arg = arg.make(env)
            args.append(arg)
        # Call the platform function
        return platform_fn(*args)

    return call_platform_sync


def _compile_new_ref(
    node: IRNode,
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

    newref_struct = node["value"]

    # Compile the value
    value_fn = _compile_ir(newref_struct["value"], platform_fns, async_platform_fns, is_async_map)

    # Check if async
    value_is_async = is_async_map.get(id(newref_struct["value"]), False)

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
    from east.types.types import IntegerType, StringType, StructType

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
    node: IRNode,
    platform_fns: dict[str, Callable[..., Any]],
    async_platform_fns: set[str],
    is_async_map: dict[int, bool],
) -> Callable:
    """Compile a TryCatch IR node (try-catch-finally error handling).

    Optimized to skip trivial finally blocks (Value nodes) at compile-time and
    generate separate code paths for with/without finally to avoid runtime checks.
    """
    trycatch_struct = node["value"]

    # Compile try body
    try_body_fn = _compile_ir(
        trycatch_struct["try_body"], platform_fns, async_platform_fns, is_async_map
    )

    # Extract message and stack variable names
    message_var = trycatch_struct["message"]["value"]
    message_name = message_var["name"]
    stack_var = trycatch_struct["stack"]["value"]
    stack_name = stack_var["name"]

    # Compile catch body
    catch_body_fn = _compile_ir(
        trycatch_struct["catch_body"], platform_fns, async_platform_fns, is_async_map
    )

    # Don't compile finally_body if it's just a Value node (effect-free)
    # This optimizes away trivial finally blocks at compile-time
    finally_body_fn = None
    finally_is_async = False

    if trycatch_struct["finally_body"]["type"] != "Value":
        finally_body_fn = _compile_ir(
            trycatch_struct["finally_body"], platform_fns, async_platform_fns, is_async_map
        )
        finally_is_async = is_async_map.get(id(trycatch_struct["finally_body"]), False)

    # Check if any component is async
    try_is_async = is_async_map.get(id(trycatch_struct["try_body"]), False)
    catch_is_async = is_async_map.get(id(trycatch_struct["catch_body"]), False)
    is_async = try_is_async or catch_is_async or finally_is_async

    # Split into 4 variants for optimal code generation (no runtime checks)
    if is_async:
        if finally_body_fn is None:
            # Async without finally

            async def execute_trycatch_async(env):
                try:
                    if try_is_async:
                        return await try_body_fn(env)
                    return try_body_fn(env)
                except Exception as e:
                    catch_env = {**env}
                    catch_env[message_name] = str(e)
                    catch_env[stack_name] = _extract_stack_trace(e)

                    if catch_is_async:
                        result = await catch_body_fn(catch_env)
                    else:
                        result = catch_body_fn(catch_env)

                    # Propagate changes back (except error variables)
                    for key, value in catch_env.items():
                        if key not in (message_name, stack_name):
                            env[key] = value

                    return result

            return execute_trycatch_async

        # Async with finally

        async def execute_trycatch_async_finally(env):
            from east.runtime.compiler import ReturnException

            try:
                if try_is_async:
                    result = await try_body_fn(env)
                else:
                    result = try_body_fn(env)
            except ReturnException:
                # Let ReturnException propagate through finally
                if finally_is_async:
                    await finally_body_fn(env)
                else:
                    finally_body_fn(env)
                raise
            except Exception as e:
                catch_env = {**env}
                catch_env[message_name] = str(e)
                catch_env[stack_name] = _extract_stack_trace(e)

                if catch_is_async:
                    result = await catch_body_fn(catch_env)
                else:
                    result = catch_body_fn(catch_env)

                # Propagate changes back (except error variables)
                for key, value in catch_env.items():
                    if key not in (message_name, stack_name):
                        env[key] = value
            finally:
                if finally_is_async:
                    await finally_body_fn(env)
                else:
                    finally_body_fn(env)

            return result

        return execute_trycatch_async_finally

    # Sync variants
    if finally_body_fn is None:
        # Sync without finally

        def execute_trycatch_sync(env):
            try:
                return try_body_fn(env)
            except Exception as e:
                catch_env = {**env}
                catch_env[message_name] = str(e)
                catch_env[stack_name] = _extract_stack_trace(e)
                result = catch_body_fn(catch_env)

                # Propagate changes back (except error variables)
                for key, value in catch_env.items():
                    if key not in (message_name, stack_name):
                        env[key] = value

                return result

        return execute_trycatch_sync

    # Sync with finally

    def execute_trycatch_sync_finally(env):
        from east.runtime.compiler import ReturnException

        try:
            result = try_body_fn(env)
        except ReturnException:
            # Let ReturnException propagate through finally
            finally_body_fn(env)
            raise
        except Exception as e:
            catch_env = {**env}
            catch_env[message_name] = str(e)
            catch_env[stack_name] = _extract_stack_trace(e)
            result = catch_body_fn(catch_env)

            # Propagate changes back (except error variables)
            for key, value in catch_env.items():
                if key not in (message_name, stack_name):
                    env[key] = value
        finally:
            finally_body_fn(env)

        return result

    return execute_trycatch_sync_finally


def _compile_call(
    node,
    platform_fns: dict[str, Callable[..., Any]],
    async_platform_fns: set[str],
    is_async_map: dict[int, bool],
) -> Callable:
    """Compile function call IR node.

    Compiles: function(...arguments)
    """
    from east.runtime.compiler import FunctionFactory
    from east.runtime.compiler import _compile_ir as _compile

    is_async = is_async_map[id(node)]

    # Compile function expression and arguments
    func_compiled = _compile(
        node["value"]["function"], platform_fns, async_platform_fns, is_async_map
    )
    args_compiled = [
        _compile(arg, platform_fns, async_platform_fns, is_async_map)
        for arg in node["value"]["arguments"]
    ]

    if is_async:

        async def call_async(env):
            func = await func_compiled(env)
            # Unwrap function if it's a FunctionFactory
            if isinstance(func, FunctionFactory):
                func = func.make(env)
            args = []
            for arg_compiled in args_compiled:
                arg = await arg_compiled(env)
                # Unwrap arg if it's a FunctionFactory
                if isinstance(arg, FunctionFactory):
                    arg = arg.make(env)
                args.append(arg)
            return await func(*args)

        return call_async

    def call_sync(env):
        func = func_compiled(env)
        # Unwrap function if it's a FunctionFactory
        if isinstance(func, FunctionFactory):
            func = func.make(env)
        args = []
        for arg_compiled in args_compiled:
            arg = arg_compiled(env)
            # Unwrap arg if it's a FunctionFactory
            if isinstance(arg, FunctionFactory):
                arg = arg.make(env)
            args.append(arg)
        return func(*args)

    return call_sync


def _compile_as(
    node,
    platform_fns: dict[str, Callable[..., Any]],
    async_platform_fns: set[str],
    is_async_map: dict[int, bool],
) -> Callable:
    """Compile type assertion (As).

    Type assertions are compile-time only - runtime just passes value through.
    """
    from east.runtime.compiler import _compile_ir as _compile

    # Simply compile the inner value and pass it through
    return _compile(node["value"]["value"], platform_fns, async_platform_fns, is_async_map)


def _compile_return(
    node,
    platform_fns: dict[str, Callable[..., Any]],
    async_platform_fns: set[str],
    is_async_map: dict[int, bool],
) -> Callable:
    """Compile return statement.

    Throws ReturnException with the return value.
    """
    from east.runtime.compiler import ReturnException
    from east.runtime.compiler import _compile_ir as _compile

    is_async = is_async_map[id(node)]
    value_compiled = _compile(
        node["value"]["value"], platform_fns, async_platform_fns, is_async_map
    )

    if is_async:

        async def return_async(env):
            value = await value_compiled(env)
            raise ReturnException(value)

        return return_async

    def return_sync(env):
        value = value_compiled(env)
        raise ReturnException(value)

    return return_sync


def _compile_assign(
    node,
    platform_fns: dict[str, Callable[..., Any]],
    async_platform_fns: set[str],
    is_async_map: dict[int, bool],
) -> Callable:
    """Compile variable assignment.

    Compiles: variable = value
    """
    from east.runtime.compiler import _compile_ir as _compile

    is_async = is_async_map[id(node)]
    value_compiled = _compile(
        node["value"]["value"], platform_fns, async_platform_fns, is_async_map
    )
    variable_name = node["value"]["variable"]["value"]["name"]

    if is_async:

        async def assign_async(env):
            env[variable_name] = await value_compiled(env)
            return Null()

        return assign_async

    def assign_sync(env):
        env[variable_name] = value_compiled(env)
        return Null()

    return assign_sync


def _compile_struct(
    node,
    platform_fns: dict[str, Callable[..., Any]],
    async_platform_fns: set[str],
    is_async_map: dict[int, bool],
) -> Callable:
    """Compile struct literal.

    Compiles: { field1: value1, field2: value2, ... }
    """
    from east.runtime.compiler import _compile_ir as _compile

    is_async = is_async_map[id(node)]

    # Get field names and compile field values from IR fields
    # Each field is a StructField with name and value
    field_names = [field["name"] for field in node["value"]["fields"]]
    fields_compiled = [
        _compile(field["value"], platform_fns, async_platform_fns, is_async_map)
        for field in node["value"]["fields"]
    ]

    if is_async:

        async def struct_async(env):
            struct = {}
            for name, field_compiled in zip(field_names, fields_compiled, strict=True):
                struct[name] = await field_compiled(env)
            return EastStruct(struct)

        return struct_async

    def struct_sync(env):
        struct = {}
        for name, field_compiled in zip(field_names, fields_compiled, strict=True):
            struct[name] = field_compiled(env)
        return EastStruct(struct)

    return struct_sync


def _compile_getfield(
    node,
    platform_fns: dict[str, Callable[..., Any]],
    async_platform_fns: set[str],
    is_async_map: dict[int, bool],
) -> Callable:
    """Compile struct field access.

    Compiles: struct.field
    """
    from east.runtime.compiler import _compile_ir as _compile

    is_async = is_async_map[id(node)]
    struct_compiled = _compile(
        node["value"]["struct"], platform_fns, async_platform_fns, is_async_map
    )
    field_name = node["value"]["field"]

    if is_async:

        async def getfield_async(env):
            struct = await struct_compiled(env)
            return struct[field_name]

        return getfield_async

    def getfield_sync(env):
        struct = struct_compiled(env)
        return struct[field_name]

    return getfield_sync


def _compile_variant(
    node,
    platform_fns: dict[str, Callable[..., Any]],
    async_platform_fns: set[str],
    is_async_map: dict[int, bool],
) -> Callable:
    """Compile variant constructor.

    Compiles: SomeCase(value)
    """
    from east.runtime.compiler import _compile_ir as _compile

    is_async = is_async_map[id(node)]
    value_compiled = _compile(
        node["value"]["value"], platform_fns, async_platform_fns, is_async_map
    )
    case_name = node["value"]["case"]

    if is_async:

        async def variant_async(env):
            value = await value_compiled(env)
            return EastVariant(case_name, value)

        return variant_async

    def variant_sync(env):
        value = value_compiled(env)
        return EastVariant(case_name, value)

    return variant_sync


def _compile_match(
    node,
    platform_fns: dict[str, Callable[..., Any]],
    async_platform_fns: set[str],
    is_async_map: dict[int, bool],
) -> Callable:
    """Compile match (pattern matching on variants).

    Compiles: match variant { Case1(x) => body1, Case2(y) => body2, ... }
    """
    from east.runtime.compiler import _compile_ir as _compile

    is_async = is_async_map[id(node)]
    variant_compiled = _compile(
        node["value"]["variant"], platform_fns, async_platform_fns, is_async_map
    )

    # Compile each case
    cases_compiled = {}
    for case in node["value"]["cases"]:
        case_name = case["case"]
        variable_name = case["variable"]["value"]["name"]
        body_compiled = _compile(case["body"], platform_fns, async_platform_fns, is_async_map)
        cases_compiled[case_name] = (variable_name, body_compiled)

    if is_async:

        async def match_async(env):
            variant = await variant_compiled(env)
            case_name = variant["type"]
            case_value = variant["value"]

            variable_name, body_compiled = cases_compiled[case_name]
            # Add variable to environment (mutate in place to preserve assignments)
            env[variable_name] = case_value
            try:
                return await body_compiled(env)
            finally:
                # Clean up the variable after match body executes
                del env[variable_name]

        return match_async

    def match_sync(env):
        variant = variant_compiled(env)
        case_name = variant["type"]
        case_value = variant["value"]

        variable_name, body_compiled = cases_compiled[case_name]
        # Add variable to environment (mutate in place to preserve assignments)
        env[variable_name] = case_value
        try:
            return body_compiled(env)
        finally:
            # Clean up the variable after match body executes
            del env[variable_name]

    return match_sync


def _compile_newarray(
    node,
    platform_fns: dict[str, Callable[..., Any]],
    async_platform_fns: set[str],
    is_async_map: dict[int, bool],
) -> Callable:
    """Compile array literal.

    Compiles: [element1, element2, ...]
    """
    from east.runtime.compiler import _compile_ir as _compile
    from east.types.containers import EastArray

    is_async = is_async_map[id(node)]
    elements_compiled = [
        _compile(elem, platform_fns, async_platform_fns, is_async_map)
        for elem in node["value"]["values"]
    ]

    # Get element type from the Array type
    element_type = node["value"]["type"]["value"]

    if is_async:

        async def newarray_async(env):
            elements = []
            for elem_compiled in elements_compiled:
                elements.append(await elem_compiled(env))
            return EastArray(element_type, elements)

        return newarray_async

    def newarray_sync(env):
        elements = [elem_compiled(env) for elem_compiled in elements_compiled]
        return EastArray(element_type, elements)

    return newarray_sync


def _compile_newset(
    node,
    platform_fns: dict[str, Callable[..., Any]],
    async_platform_fns: set[str],
    is_async_map: dict[int, bool],
) -> Callable:
    """Compile set literal.

    Compiles: { element1, element2, ... }
    """
    from east.runtime.compiler import _compile_ir as _compile
    from east.types.containers import EastSet

    is_async = is_async_map[id(node)]
    elements_compiled = [
        _compile(elem, platform_fns, async_platform_fns, is_async_map)
        for elem in node["value"]["values"]
    ]

    # Get element type from the Set type
    element_type = node["value"]["type"]["value"]

    if is_async:

        async def newset_async(env):
            elements = []
            for elem_compiled in elements_compiled:
                elements.append(await elem_compiled(env))
            return EastSet(element_type, elements)

        return newset_async

    def newset_sync(env):
        elements = [elem_compiled(env) for elem_compiled in elements_compiled]
        return EastSet(element_type, elements)

    return newset_sync


def _compile_newdict(
    node,
    platform_fns: dict[str, Callable[..., Any]],
    async_platform_fns: set[str],
    is_async_map: dict[int, bool],
) -> Callable:
    """Compile dictionary literal.

    Compiles: { key1: value1, key2: value2, ... }
    """
    from east.runtime.compiler import _compile_ir as _compile
    from east.types.containers import EastDict

    is_async = is_async_map[id(node)]
    entries_compiled = [
        (
            _compile(entry["key"], platform_fns, async_platform_fns, is_async_map),
            _compile(entry["value"], platform_fns, async_platform_fns, is_async_map),
        )
        for entry in node["value"]["values"]
    ]

    # Get key and value types from the Dict type
    key_type = node["value"]["type"]["value"]["key"]
    value_type = node["value"]["type"]["value"]["value"]

    if is_async:

        async def newdict_async(env):
            entries = {}
            for key_compiled, value_compiled in entries_compiled:
                key = await key_compiled(env)
                value = await value_compiled(env)
                entries[key] = value
            return EastDict(key_type, value_type, entries)

        return newdict_async

    def newdict_sync(env):
        entries = {}
        for key_compiled, value_compiled in entries_compiled:
            key = key_compiled(env)
            value = value_compiled(env)
            entries[key] = value
        return EastDict(key_type, value_type, entries)

    return newdict_sync


def _compile_forarray(
    node,
    platform_fns: dict[str, Callable[..., Any]],
    async_platform_fns: set[str],
    is_async_map: dict[int, bool],
) -> Callable:
    """Compile for-array loop.

    Compiles: for (key, element) in array { body }
    """
    from east.runtime.compiler import BreakException, ContinueException
    from east.runtime.compiler import _compile_ir as _compile

    is_async = is_async_map[id(node)]
    array_compiled = _compile(
        node["value"]["array"], platform_fns, async_platform_fns, is_async_map
    )
    body_compiled = _compile(node["value"]["body"], platform_fns, async_platform_fns, is_async_map)

    key_var_name = node["value"]["key"]["value"]["name"]
    element_var_name = node["value"]["value"]["value"]["name"]

    if is_async:

        async def forarray_async(env):
            array = await array_compiled(env)
            array._lock_for_iteration()
            try:
                for i, elem in enumerate(array):
                    # Create child context with loop variables
                    child_env = {**env, key_var_name: i, element_var_name: elem}
                    try:
                        await body_compiled(child_env)
                        # Propagate changes back to parent env (except loop variables)
                        for key, value in child_env.items():
                            if key not in (key_var_name, element_var_name):
                                env[key] = value
                    except BreakException:
                        break
                    except ContinueException:
                        continue
                return Null()
            finally:
                array._unlock_for_iteration()

        return forarray_async

    def forarray_sync(env):
        array = array_compiled(env)
        array._lock_for_iteration()
        try:
            for i, elem in enumerate(array):
                child_env = {**env, key_var_name: i, element_var_name: elem}
                try:
                    body_compiled(child_env)
                    # Propagate changes back to parent env (except loop variables)
                    for key, value in child_env.items():
                        if key not in (key_var_name, element_var_name):
                            env[key] = value
                except BreakException:
                    break
                except ContinueException:
                    continue
            return Null()
        finally:
            array._unlock_for_iteration()

    return forarray_sync


def _compile_forset(
    node,
    platform_fns: dict[str, Callable[..., Any]],
    async_platform_fns: set[str],
    is_async_map: dict[int, bool],
) -> Callable:
    """Compile for-set loop.

    Compiles: for element in set { body }
    """
    from east.runtime.compiler import BreakException, ContinueException
    from east.runtime.compiler import _compile_ir as _compile

    is_async = is_async_map[id(node)]
    set_compiled = _compile(node["value"]["set"], platform_fns, async_platform_fns, is_async_map)
    body_compiled = _compile(node["value"]["body"], platform_fns, async_platform_fns, is_async_map)

    element_var_name = node["value"]["key"]["value"]["name"]

    if is_async:

        async def forset_async(env):
            east_set = await set_compiled(env)
            east_set._lock_for_iteration()
            try:
                for elem in east_set:
                    child_env = {**env, element_var_name: elem}
                    try:
                        await body_compiled(child_env)
                        # Propagate changes back to parent env (except loop variable)
                        for key, value in child_env.items():
                            if key != element_var_name:
                                env[key] = value
                    except BreakException:
                        break
                    except ContinueException:
                        continue
                return Null()
            finally:
                east_set._unlock_for_iteration()

        return forset_async

    def forset_sync(env):
        east_set = set_compiled(env)
        east_set._lock_for_iteration()
        try:
            for elem in east_set:
                child_env = {**env, element_var_name: elem}
                try:
                    body_compiled(child_env)
                    # Propagate changes back to parent env (except loop variable)
                    for key, value in child_env.items():
                        if key != element_var_name:
                            env[key] = value
                except BreakException:
                    break
                except ContinueException:
                    continue
            return Null()
        finally:
            east_set._unlock_for_iteration()

    return forset_sync


def _compile_fordict(
    node,
    platform_fns: dict[str, Callable[..., Any]],
    async_platform_fns: set[str],
    is_async_map: dict[int, bool],
) -> Callable:
    """Compile for-dict loop.

    Compiles: for (key, value) in dict { body }
    """
    from east.runtime.compiler import BreakException, ContinueException
    from east.runtime.compiler import _compile_ir as _compile

    is_async = is_async_map[id(node)]
    dict_compiled = _compile(node["value"]["dict"], platform_fns, async_platform_fns, is_async_map)
    body_compiled = _compile(node["value"]["body"], platform_fns, async_platform_fns, is_async_map)

    key_var_name = node["value"]["key"]["value"]["name"]
    value_var_name = node["value"]["value"]["value"]["name"]

    if is_async:

        async def fordict_async(env):
            east_dict = await dict_compiled(env)
            east_dict._lock_for_iteration()
            try:
                for key, value in east_dict.items():
                    child_env = {**env, key_var_name: key, value_var_name: value}
                    try:
                        await body_compiled(child_env)
                        # Propagate changes back to parent env (except loop variables)
                        for k, v in child_env.items():
                            if k not in (key_var_name, value_var_name):
                                env[k] = v
                    except BreakException:
                        break
                    except ContinueException:
                        continue
                return Null()
            finally:
                east_dict._unlock_for_iteration()

        return fordict_async

    def fordict_sync(env):
        east_dict = dict_compiled(env)
        east_dict._lock_for_iteration()
        try:
            for key, value in east_dict.items():
                child_env = {**env, key_var_name: key, value_var_name: value}
                try:
                    body_compiled(child_env)
                    # Propagate changes back to parent env (except loop variables)
                    for k, v in child_env.items():
                        if k not in (key_var_name, value_var_name):
                            env[k] = v
                except BreakException:
                    break
                except ContinueException:
                    continue
            return Null()
        finally:
            east_dict._unlock_for_iteration()

    return fordict_sync


def _compile_break(
    node,
    platform_fns: dict[str, Callable[..., Any]],
    async_platform_fns: set[str],
    is_async_map: dict[int, bool],
) -> Callable:
    """Compile break statement."""
    from east.runtime.compiler import BreakException

    label = node["value"]["label"]["name"]
    is_async = is_async_map[id(node)]

    if is_async:

        async def break_async(env):
            raise BreakException(label)

        return break_async

    def break_sync(env):
        raise BreakException(label)

    return break_sync


def _compile_continue(
    node,
    platform_fns: dict[str, Callable[..., Any]],
    async_platform_fns: set[str],
    is_async_map: dict[int, bool],
) -> Callable:
    """Compile continue statement."""
    from east.runtime.compiler import ContinueException

    label = node["value"]["label"]["name"]
    is_async = is_async_map[id(node)]

    if is_async:

        async def continue_async(env):
            raise ContinueException(label)

        return continue_async

    def continue_sync(env):
        raise ContinueException(label)

    return continue_sync


def _compile_error(
    node,
    platform_fns: dict[str, Callable[..., Any]],
    async_platform_fns: set[str],
    is_async_map: dict[int, bool],
) -> Callable:
    """Compile error (throw exception).

    Compiles: throw error_message
    """
    from east.runtime.compiler import _compile_ir as _compile

    is_async = is_async_map[id(node)]
    message_compiled = _compile(
        node["value"]["message"], platform_fns, async_platform_fns, is_async_map
    )

    if is_async:

        async def error_async(env):
            message = await message_compiled(env)
            raise RuntimeError(message)

        return error_async

    def error_sync(env):
        message = message_compiled(env)
        raise RuntimeError(message)

    return error_sync


def _compile_unwraprecursive(
    node,
    platform_fns: dict[str, Callable[..., Any]],
    async_platform_fns: set[str],
    is_async_map: dict[int, bool],
) -> Callable:
    """Compile unwrap of recursive type.

    Simply passes through the value (recursive types are transparent at runtime).
    """
    from east.runtime.compiler import _compile_ir as _compile

    return _compile(node["value"]["value"], platform_fns, async_platform_fns, is_async_map)


def _compile_wraprecursive(
    node,
    platform_fns: dict[str, Callable[..., Any]],
    async_platform_fns: set[str],
    is_async_map: dict[int, bool],
) -> Callable:
    """Compile wrap in recursive type.

    Simply passes through the value (recursive types are transparent at runtime).
    """
    from east.runtime.compiler import _compile_ir as _compile

    return _compile(node["value"]["value"], platform_fns, async_platform_fns, is_async_map)
