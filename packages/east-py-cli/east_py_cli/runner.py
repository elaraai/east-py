#
# Copyright (c) 2025 Elara AI Pty Ltd
# Licensed under the Business Source License 1.1. See LICENSE.md for details.
#
"""IR compilation and execution."""

import asyncio
from pathlib import Path

from east.runtime.compiler import compile as compile_sync
from east.runtime.compiler import compile_async
from east.runtime.platform import PlatformFunction
from east.serialization.east_printer import print_east, print_type
from east.types.ir import AsyncFunctionIR, FunctionIR
from east.types.types import EastType

from east_py_cli.loader import load_value, save_value


def get_function_signature(
    ir: FunctionIR | AsyncFunctionIR,
) -> tuple[list[EastType], EastType, bool]:
    """Extract function signature from IR.

    Args:
        ir: Parsed IR value (a Function or AsyncFunction IR node)

    Returns:
        Tuple of (param_types, return_type, is_async)

    Raises:
        ValueError: If IR is not a function IR node
    """
    if ir.type == "Function":
        fn_type = ir.value["type"]
        param_types = fn_type.value["inputs"]
        return_type = fn_type.value["output"]
        return (list(param_types), return_type, False)

    elif ir.type == "AsyncFunction":
        fn_type = ir.value["type"]
        param_types = fn_type.value["inputs"]
        return_type = fn_type.value["output"]
        return (list(param_types), return_type, True)

    else:
        raise ValueError(
            f"IR must be a Function or AsyncFunction node, got: {ir.type}\n"
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
    ir: FunctionIR | AsyncFunctionIR,
    platform_fns: list[PlatformFunction],
    input_files: list[Path],
    symbol_irs: dict | None = None,
    output_file: Path | None = None,
    verbose: bool = False,
) -> object:
    """Run an East IR program.

    Args:
        ir: Parsed IR (Function or AsyncFunction node)
        platform_fns: Platform functions to use
        input_files: Input data files (order matches function parameters)
        symbol_irs: Symbol IR definitions from linked modules
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

    # Use is_async from IR - the analyzer already determined if the function needs async
    use_async = is_async

    # Compile IR
    if verbose:
        print(f"Compiling IR with {len(platform_fns)} platform functions...")
        if use_async:
            print("  (async function)")

    sym_irs = symbol_irs or {}
    symbol_values: dict = {}
    compiled = (
        compile_async(ir, sym_irs, symbol_values, platform_fns)
        if use_async
        else compile_sync(ir, sym_irs, symbol_values, platform_fns)
    )

    # Execute
    if verbose:
        print("Executing...")

    if use_async:
        # Run async function
        result = asyncio.run(compiled(*inputs)) if inputs else asyncio.run(compiled())
    else:
        # Run sync function
        result = compiled(*inputs) if inputs else compiled()

    # Save output if requested, otherwise print as .east
    if output_file is not None:
        if verbose:
            type_str = print_type(return_type)
            print(f"Saving output to {output_file} as {type_str}")
        save_value(output_file, result, return_type)
    else:
        # Print result as .east format to stdout
        print(print_east(result, return_type))

    return result
