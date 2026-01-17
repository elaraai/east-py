#
# Copyright (c) 2025 Elara AI Pty Ltd
# Licensed under the Business Source License 1.1. See LICENSE.md for details.
#
"""Parallel platform functions for East.

Provides parallel computation capabilities for East programs running in Python
using concurrent.futures for parallelism.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from east.runtime.platform import GenericPlatformFunction
from east.types.values import EastArray

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine


def _make_parallel_map(
    input_type: Any,
    output_type: Any,
) -> Callable[[EastArray, Callable[[Any], Any]], Coroutine[Any, Any, EastArray]]:
    """Create parallel_map implementation for given types.

    Args:
        input_type: Input element type (T)
        output_type: Output element type (R)

    Returns:
        Async function that maps over array in parallel
    """

    async def impl(array: EastArray, fn: Callable[[Any], Any]) -> EastArray:
        """Map a function over an array in parallel using thread pool.

        For small arrays (≤4 elements), runs sequentially to avoid overhead.
        Otherwise, distributes work across multiple threads.

        Args:
            array: Input array to map over
            fn: East function to apply to each element (T -> R)

        Returns:
            EastArray of results with the same length as input
        """
        # For small arrays, just run sequentially to avoid overhead
        if len(array) <= 4:
            results = [fn(item) for item in array]
            return EastArray(output_type, results)

        # Use thread pool for larger arrays
        num_workers = min(os.cpu_count() or 4, len(array))

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(fn, array))

        return EastArray(output_type, results)

    return impl


# Platform function registration
# Generic over T (input element type) and R (output element type)
parallel_impl = [
    GenericPlatformFunction(
        name="parallel_map",
        type_parameters=["T", "R"],
        type="async",
        fn=lambda t, r: _make_parallel_map(t, r),
    ),
]


__all__ = ["parallel_impl"]
