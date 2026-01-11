#
# Copyright (c) 2025 Elara AI Pty Ltd
# Licensed under the Business Source License 1.1. See LICENSE.md for details.
#
"""Patch builtin functions.

These are factory builtins - they take type parameters at compile time and
return specialized patch functions that are called at runtime.

ConflictError from patch operations will propagate up and be wrapped
by the compiler with proper location info.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from east.runtime.platform import PlatformFunction

from east.builtins.registry import register_builtin
from east.patch.apply import apply_for as _apply_for
from east.patch.compose import compose_for as _compose_for
from east.patch.diff import diff_for as _diff_for
from east.patch.invert import invert_for as _invert_for
from east.types.types import EastType


def diff(
    _platform: "list[PlatformFunction]",
    T: EastType,
    _P: EastType,  # PatchType - computed by caller, unused here
) -> Callable[[Any, Any], Any]:
    """Factory for diff operation.

    Args:
        _platform: Platform functions (unused)
        T: The East type of values being diffed
        _P: The patch type (computed by caller, unused here)

    Returns:
        A function (before, after) -> patch
    """
    return _diff_for(T)


def apply_patch(
    _platform: "list[PlatformFunction]",
    T: EastType,
    _P: EastType,
) -> Callable[[Any, Any], Any]:
    """Factory for apply operation.

    Args:
        _platform: Platform functions (unused)
        T: The East type of the base value
        _P: The patch type (computed by caller, unused here)

    Returns:
        A function (base, patch) -> patched_value

    Raises:
        ConflictError: If patch conflicts with base value
    """
    return _apply_for(T)


def compose_patch(
    _platform: "list[PlatformFunction]",
    T: EastType,
    _P: EastType,
) -> Callable[[Any, Any], Any]:
    """Factory for compose operation.

    Args:
        _platform: Platform functions (unused)
        T: The East type
        _P: The patch type (computed by caller, unused here)

    Returns:
        A function (first, second) -> combined_patch

    Raises:
        ConflictError: If patches cannot be composed
    """
    return _compose_for(T)


def invert_patch(
    _platform: "list[PlatformFunction]",
    T: EastType,
    _P: EastType,
) -> Callable[[Any], Any]:
    """Factory for invert operation.

    Args:
        _platform: Platform functions (unused)
        T: The East type
        _P: The patch type (computed by caller, unused here)

    Returns:
        A function (patch) -> inverted_patch
    """
    return _invert_for(T)


# Register builtins
register_builtin("Diff", diff)
register_builtin("ApplyPatch", apply_patch)
register_builtin("ComposePatch", compose_patch)
register_builtin("InvertPatch", invert_patch)
