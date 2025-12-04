"""IR compilation and execution."""

import asyncio
from pathlib import Path
from typing import Any

from east.runtime.compiler import compile as compile_sync
from east.runtime.compiler import compile_async
from east.runtime.platform import PlatformFunction
from east.serialization.east_printer import print_type
from east.types.types import EastType

from east_py_cli.loader import load_value, save_value


def get_function_signature(ir: Any) -> tuple[list[EastType], EastType, bool]:
    """Extract function signature from IR.

    IR nodes are variants where:
    - ir.type = "Function" or "AsyncFunction" (variant tag)
    - ir.value["type"] = the FunctionType/AsyncFunctionType (East type)
    - ir.value["type"].value["inputs"] = input parameter types
    - ir.value["type"].value["output"] = return type

    Args:
        ir: Parsed IR value (a Function or AsyncFunction IR node)

    Returns:
        Tuple of (param_types, return_type, is_async)

    Raises:
        ValueError: If IR is not a function IR node
    """
    ir_tag = ir.type if hasattr(ir, "type") else None

    if ir_tag == "Function":
        fn_type = ir.value["type"]
        param_types = fn_type.value["inputs"]
        return_type = fn_type.value["output"]
        return (list(param_types), return_type, False)

    elif ir_tag == "AsyncFunction":
        fn_type = ir.value["type"]
        param_types = fn_type.value["inputs"]
        return_type = fn_type.value["output"]
        return (list(param_types), return_type, True)

    else:
        raise ValueError(
            f"IR must be a Function or AsyncFunction node, got: {ir_tag}\n"
            f"The IR file should contain compiled function IR."
        )


def format_signature(param_types: list[EastType], return_type: EastType) -> str:
    """Format function signature for display.

    Args:
        param_types: List of parameter types
        return_type: Return type

    Returns:
        Human-readable signature string
    """
    params = ", ".join(print_type(t) for t in param_types)
    ret = print_type(return_type)
    return f"({params}) -> {ret}"


def run_program(
    ir: Any,
    platform_fns: list[PlatformFunction],
    input_files: list[Path],
    output_file: Path | None = None,
    verbose: bool = False,
) -> Any:
    """Run an East IR program.

    Args:
        ir: Parsed IR value (must be a function type)
        platform_fns: Platform functions to use
        input_files: Input data files (order matches function parameters)
        output_file: Optional output file path
        verbose: Enable verbose output

    Returns:
        The function's return value

    Raises:
        ValueError: If IR is invalid or inputs don't match
    """
    # Validate IR is a function and get signature
    param_types, return_type, is_async = get_function_signature(ir)

    if verbose:
        sig = format_signature(param_types, return_type)
        print(f"Function signature: {sig}")
        print(f"Async: {is_async}")

    # Validate input count
    if len(input_files) != len(param_types):
        sig = format_signature(param_types, return_type)
        raise ValueError(
            f"Function expects {len(param_types)} inputs, got {len(input_files)}\n"
            f"Signature: {sig}"
        )

    # Load inputs with type-directed parsing
    inputs = []
    for i, (file_path, param_type) in enumerate(zip(input_files, param_types, strict=False)):
        if verbose:
            type_str = print_type(param_type)
            print(f"Loading input {i}: {file_path} as {type_str}")
        try:
            value = load_value(file_path, param_type)
            inputs.append(value)
        except Exception as e:
            type_str = print_type(param_type)
            raise ValueError(f"Failed to parse input {i} ({file_path}) as {type_str}: {e}") from e

    # Compile IR
    if verbose:
        print(f"Compiling IR with {len(platform_fns)} platform functions...")

    compiled = compile_async(ir, platform_fns) if is_async else compile_sync(ir, platform_fns)

    # Execute
    if verbose:
        print("Executing...")

    if is_async:
        # Run async function
        result = asyncio.run(compiled(*inputs)) if inputs else asyncio.run(compiled())
    else:
        # Run sync function
        result = compiled(*inputs) if inputs else compiled()

    # Save output if requested
    if output_file is not None:
        if verbose:
            type_str = print_type(return_type)
            print(f"Saving output to {output_file} as {type_str}")
        save_value(output_file, result, return_type)

    return result
