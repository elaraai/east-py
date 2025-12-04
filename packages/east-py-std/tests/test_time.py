"""Tests for time platform functions."""

import time

import pytest

from east_py_std.time import time_now_impl, time_sleep_impl


def test_time_now():
    """Test time_now returns current timestamp."""
    before = int(time.time() * 1000)
    result = time_now_impl()
    after = int(time.time() * 1000)

    assert isinstance(result, int)
    assert before <= result <= after


def test_time_now_increases():
    """Test time_now returns increasing values."""
    time1 = time_now_impl()
    time.sleep(0.01)  # Sleep 10ms
    time2 = time_now_impl()

    assert time2 > time1


@pytest.mark.asyncio
async def test_time_sleep():
    """Test time_sleep pauses execution."""
    start = time.time()
    await time_sleep_impl(100)  # Sleep 100ms
    elapsed = time.time() - start

    # Allow some tolerance for timing
    assert 0.09 < elapsed < 0.15  # Should be around 0.1 seconds


@pytest.mark.asyncio
async def test_time_sleep_zero():
    """Test time_sleep with zero duration."""
    start = time.time()
    await time_sleep_impl(0)
    elapsed = time.time() - start

    assert elapsed < 0.01  # Should be very fast


@pytest.mark.asyncio
async def test_time_sleep_negative():
    """Test time_sleep rejects negative duration."""
    with pytest.raises(ValueError, match="non-negative"):
        await time_sleep_impl(-100)
