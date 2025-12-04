"""Tests for fetch platform functions.

Uses httpbin.org for real HTTP request testing.
"""

import pytest
from east.types.types import StringType
from east.types.values import EastDict, EastStruct, EastVariant

from east_py_std.fetch import (
    FetchRequestConfig,
    fetch_get_impl,
    fetch_post_impl,
    fetch_request_impl,
)

# Test endpoint
HTTPBIN = "https://httpbin.org"


@pytest.mark.asyncio
async def test_fetch_get_success():
    """Test fetch_get with successful response."""
    result = await fetch_get_impl(f"{HTTPBIN}/get")

    # httpbin returns JSON, should contain "url" field
    assert "httpbin.org/get" in result
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_fetch_get_error_status():
    """Test fetch_get with error status code."""
    with pytest.raises(Exception, match="HTTP.*404"):
        await fetch_get_impl(f"{HTTPBIN}/status/404")


@pytest.mark.asyncio
async def test_fetch_post_success():
    """Test fetch_post with successful response."""
    result = await fetch_post_impl(f"{HTTPBIN}/post", "test data")

    # httpbin echoes back the data
    assert "test data" in result
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_fetch_post_error_status():
    """Test fetch_post with error status code."""
    with pytest.raises(Exception, match="HTTP.*500"):
        await fetch_post_impl(f"{HTTPBIN}/status/500", "test")


@pytest.mark.asyncio
async def test_fetch_request_get():
    """Test fetch_request with GET method."""
    config: EastStruct[FetchRequestConfig] = EastStruct(
        {
            "url": f"{HTTPBIN}/get",
            "method": EastVariant("GET", None),
            "headers": EastDict(StringType, StringType, {}),
            "body": EastVariant("none", None),
        }
    )

    result = await fetch_request_impl(config)

    assert isinstance(result, EastStruct)
    assert result["status"] == 200
    assert "httpbin.org/get" in result["body"]
    assert result["ok"] is True


@pytest.mark.asyncio
async def test_fetch_request_post_with_body():
    """Test fetch_request with POST method and body."""
    config: EastStruct[FetchRequestConfig] = EastStruct(
        {
            "url": f"{HTTPBIN}/post",
            "method": EastVariant("POST", None),
            "headers": EastDict(StringType, StringType, {"Content-Type": "application/json"}),
            "body": EastVariant("some", '{"key": "value"}'),
        }
    )

    result = await fetch_request_impl(config)

    assert result["status"] == 200
    assert result["ok"] is True
    # httpbin echoes the data back
    assert "key" in result["body"]
    assert "value" in result["body"]


@pytest.mark.asyncio
async def test_fetch_request_error_response():
    """Test fetch_request with HTTP error."""
    config: EastStruct[FetchRequestConfig] = EastStruct(
        {
            "url": f"{HTTPBIN}/status/404",
            "method": EastVariant("GET", None),
            "headers": EastDict(StringType, StringType, {}),
            "body": EastVariant("none", None),
        }
    )

    result = await fetch_request_impl(config)

    assert result["status"] == 404
    assert result["ok"] is False


@pytest.mark.asyncio
async def test_fetch_request_with_custom_headers():
    """Test fetch_request with custom headers."""
    config: EastStruct[FetchRequestConfig] = EastStruct(
        {
            "url": f"{HTTPBIN}/headers",
            "method": EastVariant("GET", None),
            "headers": EastDict(
                StringType,
                StringType,
                {"X-Custom-Header": "test-value", "Accept": "application/json"},
            ),
            "body": EastVariant("none", None),
        }
    )

    result = await fetch_request_impl(config)

    assert result["status"] == 200
    assert result["ok"] is True
    # httpbin echoes headers back
    assert "X-Custom-Header" in result["body"]


@pytest.mark.asyncio
async def test_fetch_request_none_body():
    """Test fetch_request with none body option."""
    config: EastStruct[FetchRequestConfig] = EastStruct(
        {
            "url": f"{HTTPBIN}/put",
            "method": EastVariant("PUT", None),
            "headers": EastDict(StringType, StringType, {}),
            "body": EastVariant("none", None),
        }
    )

    result = await fetch_request_impl(config)

    assert result["status"] == 200
    assert result["ok"] is True
