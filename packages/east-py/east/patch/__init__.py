#
# Copyright (c) 2025 Elara AI Pty Ltd
# Licensed under the Business Source License 1.1. See LICENSE.md for details.
#
"""East patch system for computing and applying differences between values.

This module provides four core operations:

- diff_for(type): Create a diff function (before, after) -> patch
- apply_for(type): Create an apply function (base, patch) -> value
- compose_for(type): Create a compose function (first, second) -> combined
- invert_for(type): Create an invert function (patch) -> inverse

Along with:

- PatchType(type): Compute the patch type for a given type
- ConflictError: Exception raised on patch conflicts
"""

from east.patch.apply import apply_for
from east.patch.compose import compose_for
from east.patch.diff import diff_for
from east.patch.invert import invert_for
from east.patch.type_of_patch import PatchType
from east.patch.types import (
    ApplyContext,
    ComposeContext,
    ConflictError,
    DiffContext,
    InvertContext,
    compute_lcs,
)

__all__ = [
    # Core functions
    "diff_for",
    "apply_for",
    "compose_for",
    "invert_for",
    # Type constructor
    "PatchType",
    # Exception
    "ConflictError",
    # Context classes (for internal/advanced use)
    "DiffContext",
    "ApplyContext",
    "ComposeContext",
    "InvertContext",
    "compute_lcs",
]
