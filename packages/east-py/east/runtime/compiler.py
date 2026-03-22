#
# Copyright (c) 2025 Elara AI Pty Ltd
# Licensed under the Business Source License 1.1. See LICENSE.md for details.
#
"""East IR compiler — delegates to east-c for compilation and execution.

All IR compilation, builtin resolution, and execution happens in east-c.
Python only provides platform function callbacks via the platform bridge.
"""

from collections.abc import Callable

from east.runtime.platform import PlatformFunction
from east.types.ir import IR

# Re-export constants used by serialization (bridge references these)
EAST_IR_ATTR = "_east_ir"
EAST_CAPTURES_ATTR = "_east_captures"


def compile(ir: IR, platform: list[PlatformFunction] | None = None) -> Callable:
    """Compile East IR to a native Python callable (synchronous)."""
    from east.runtime._compiler_eastc import compile_eastc

    return compile_eastc(ir, platform or [])


def compile_async(ir: IR, platform: list[PlatformFunction] | None = None) -> Callable:
    """Compile East IR to a native Python async callable."""
    from east.runtime._compiler_eastc import compile_eastc

    return compile_eastc(ir, platform or [])


def compile_from_json(
    json_data: bytes,
    platform: list[PlatformFunction] | None = None,
    is_async: bool = False,
) -> Callable:
    """Compile East IR directly from JSON bytes — fast path, no Python round-trip.

    JSON → C value → IRNode → compile → callable, entirely in C.
    """
    from east.runtime._compiler_eastc import compile_eastc_from_json

    return compile_eastc_from_json(json_data, platform or [], is_async)
