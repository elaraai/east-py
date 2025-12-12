#
# Copyright (c) 2025 Elara AI Pty Ltd
# Licensed under the Business Source License 1.1. See LICENSE.md for details.
#
"""Time platform functions for East.

Provides time-related operations for East programs running in Python.
"""

import asyncio
import time

from east.runtime.platform import PlatformFunction
from east.types.types import IntegerType, NullType


def time_now_impl() -> int:
    """Get current Unix timestamp in milliseconds.

    Returns:
        Current time as milliseconds since Unix epoch (January 1, 1970 UTC)
    """
    return int(time.time() * 1000)


async def time_sleep_impl(ms: int) -> None:
    """Sleep for specified number of milliseconds.

    Args:
        ms: Number of milliseconds to sleep (must be non-negative)

    Raises:
        ValueError: If ms is negative
    """
    if ms < 0:
        raise ValueError(f"Sleep duration must be non-negative, got {ms}")
    await asyncio.sleep(ms / 1000.0)


# Platform function implementations
time_impl = [
    PlatformFunction(
        name="time_now",
        inputs=[],
        output=IntegerType,
        type="sync",
        fn=time_now_impl,
    ),
    PlatformFunction(
        name="time_sleep",
        inputs=[IntegerType],
        output=NullType,
        type="async",
        fn=time_sleep_impl,
    ),
]


__all__ = ["time_impl"]
