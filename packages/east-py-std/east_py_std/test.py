"""Test platform functions for East.

Provides test assertion and organization operations for East programs running in Python.
These functions mirror the test utilities in east-node for running East tests.
"""

import asyncio
from typing import Any

from east.runtime.platform import PlatformFunction
from east.types.types import FunctionType, NullType, StringType


def test_pass_impl() -> None:
    """Signal that a test assertion passed.

    This is a no-op - when an assertion passes, execution continues normally.
    """
    pass


def test_fail_impl(message: str) -> None:
    """Signal that a test assertion failed with a message.

    Args:
        message: The error message describing why the assertion failed

    Raises:
        AssertionError: Always raises with the provided message
    """
    raise AssertionError(message)


async def test_impl_fn(name: str, body: Any) -> None:
    """Run a single test case.

    Args:
        name: The name/description of the test
        body: The test function to execute (may be async)
    """
    if callable(body):
        result = body()
        if asyncio.iscoroutine(result):
            await result


async def describe_impl(name: str, body: Any) -> None:
    """Define a test suite/group.

    Args:
        name: The name/description of the test suite
        body: The function containing test definitions (may be async)
    """
    if callable(body):
        result = body()
        if asyncio.iscoroutine(result):
            await result


# Platform function implementations
test_impl = [
    PlatformFunction(
        name="testPass",
        inputs=[],
        output=NullType,
        type="sync",
        fn=test_pass_impl,
    ),
    PlatformFunction(
        name="testFail",
        inputs=[StringType],
        output=NullType,
        type="sync",
        fn=test_fail_impl,
    ),
    PlatformFunction(
        name="test",
        inputs=[StringType, FunctionType([], NullType)],
        output=NullType,
        type="async",
        fn=test_impl_fn,
    ),
    PlatformFunction(
        name="describe",
        inputs=[StringType, FunctionType([], NullType)],
        output=NullType,
        type="async",
        fn=describe_impl,
    ),
]


__all__ = ["test_impl"]
