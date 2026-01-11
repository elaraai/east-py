#
# Copyright (c) 2025 Elara AI Pty Ltd
# Licensed under the Business Source License 1.1. See LICENSE.md for details.
#
"""Patch system types and utilities.

This module provides:
- Context dataclasses for recursive type handling
- ConflictError exception for patch conflicts
- LCS algorithm for array diffing
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from east.types.types import EastType


class ConflictError(Exception):
    """Raised when patch operations encounter conflicts."""

    pass


@dataclass
class DiffContext:
    """Context for building diff handlers with recursive type support.

    Handlers are built in parallel so that when we encounter a `.Recursive n`
    back-reference, we can look up the appropriate handler from any array.
    """

    diff: list[Callable[[Any, Any], Any]] = field(default_factory=list)
    types: list[EastType] = field(default_factory=list)
    equal: list[Callable[[Any, Any], bool]] = field(default_factory=list)


@dataclass
class ApplyContext:
    """Context for building apply handlers with recursive type support."""

    apply: list[Callable[[Any, Any], Any]] = field(default_factory=list)
    types: list[EastType] = field(default_factory=list)
    equal: list[Callable[[Any, Any], bool]] = field(default_factory=list)
    print: list[Callable[[Any], str]] = field(default_factory=list)


@dataclass
class ComposeContext:
    """Context for building compose handlers with recursive type support."""

    compose: list[Callable[[Any, Any], Any]] = field(default_factory=list)
    apply: list[Callable[[Any, Any], Any]] = field(default_factory=list)
    invert: list[Callable[[Any], Any]] = field(default_factory=list)
    types: list[EastType] = field(default_factory=list)
    equal: list[Callable[[Any, Any], bool]] = field(default_factory=list)
    print: list[Callable[[Any], str]] = field(default_factory=list)


@dataclass
class InvertContext:
    """Context for building invert handlers with recursive type support."""

    invert: list[Callable[[Any], Any]] = field(default_factory=list)
    types: list[EastType] = field(default_factory=list)
    equal: list[Callable[[Any, Any], bool]] = field(default_factory=list)


def compute_lcs(
    before: list[Any],
    after: list[Any],
    equal: Callable[[Any, Any], bool],
) -> tuple[list[int], list[int]]:
    """Compute Longest Common Subsequence indices.

    Args:
        before: First array
        after: Second array
        equal: Equality function for elements

    Returns:
        Tuple of (before_indices, after_indices) for matching elements.
    """
    m, n = len(before), len(after)

    # Build DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if equal(before[i - 1], after[j - 1]):
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # Backtrack to find indices
    before_indices: list[int] = []
    after_indices: list[int] = []
    i, j = m, n

    while i > 0 and j > 0:
        if equal(before[i - 1], after[j - 1]):
            before_indices.insert(0, i - 1)
            after_indices.insert(0, j - 1)
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    return before_indices, after_indices


__all__ = [
    "ConflictError",
    "DiffContext",
    "ApplyContext",
    "ComposeContext",
    "InvertContext",
    "compute_lcs",
]
