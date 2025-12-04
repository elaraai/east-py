"""HTTP fetch platform functions for East.

Provides HTTP request operations for East programs running in Python.
"""

import urllib.error
import urllib.request
from typing import TypedDict, cast

from east.runtime.platform import PlatformFunction
from east.types.types import (
    BooleanType,
    DictType,
    IntegerType,
    NullType,
    OptionType,
    StringType,
    StructType,
    VariantType,
)
from east.types.values import EastDict, EastStruct, EastVariant


class HttpMethodVariant(TypedDict):
    """HTTP method variant structure."""

    type: str  # "GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"
    value: None


class OptionStringVariant(TypedDict):
    """Option variant structure for optional string."""

    type: str  # "some" or "none"
    value: str | None


class FetchRequestConfig(TypedDict):
    """HTTP request configuration structure."""

    url: str
    method: EastVariant[HttpMethodVariant]
    headers: EastDict
    body: EastVariant[OptionStringVariant]


class FetchResponse(TypedDict):
    """HTTP response structure."""

    status: int
    statusText: str
    headers: EastDict
    body: str
    ok: bool


# HTTP method variant type
fetch_method_type = VariantType(
    [
        ("GET", NullType),
        ("POST", NullType),
        ("PUT", NullType),
        ("DELETE", NullType),
        ("PATCH", NullType),
        ("HEAD", NullType),
        ("OPTIONS", NullType),
    ]
)

# HTTP request configuration structure
fetch_request_config_type = StructType(
    [
        ("url", StringType),
        ("method", fetch_method_type),
        ("headers", DictType(StringType, StringType)),
        ("body", OptionType(StringType)),
    ]
)

# HTTP response structure
fetch_response_type = StructType(
    [
        ("status", IntegerType),
        ("statusText", StringType),
        ("headers", DictType(StringType, StringType)),
        ("body", StringType),
        ("ok", BooleanType),
    ]
)


async def fetch_get_impl(url: str) -> str:
    """Perform HTTP GET request.

    Args:
        url: URL to fetch

    Returns:
        Response body as string

    Raises:
        Exception: If request fails or status is not 2xx
    """
    import asyncio

    loop = asyncio.get_event_loop()

    def _fetch():
        response = urllib.request.urlopen(url)
        if response.status < 200 or response.status >= 300:
            raise Exception(f"HTTP {response.status}: {response.msg}")
        return response.read().decode("utf-8")

    return await loop.run_in_executor(None, _fetch)


async def fetch_post_impl(url: str, body: str) -> str:
    """Perform HTTP POST request.

    Args:
        url: URL to post to
        body: Request body as string

    Returns:
        Response body as string

    Raises:
        Exception: If request fails or status is not 2xx
    """
    import asyncio

    loop = asyncio.get_event_loop()

    def _fetch():
        req = urllib.request.Request(
            url, data=body.encode("utf-8"), headers={"Content-Type": "text/plain"}, method="POST"
        )
        response = urllib.request.urlopen(req)
        if response.status < 200 or response.status >= 300:
            raise Exception(f"HTTP {response.status}: {response.msg}")
        return response.read().decode("utf-8")

    return await loop.run_in_executor(None, _fetch)


async def fetch_request_impl(config: EastStruct[FetchRequestConfig]) -> EastStruct[FetchResponse]:
    """Perform HTTP request with custom configuration.

    Args:
        config: Request configuration (url, method, headers, body)

    Returns:
        HTTP response (status, statusText, headers, body, ok)
    """
    import asyncio

    loop = asyncio.get_event_loop()

    def _fetch() -> EastStruct[FetchResponse]:
        # Cast to TypedDict for type inference (workaround until EastStruct.__getitem__ is typed)
        cfg = cast(FetchRequestConfig, config)

        # Extract config fields
        url = cfg["url"]
        method_variant = cast(HttpMethodVariant, cfg["method"])
        headers_dict = cfg["headers"]
        body_option = cast(OptionStringVariant, cfg["body"])

        # Convert method variant to string
        method = method_variant["type"]

        # Convert headers dict to Python dict
        headers = dict(headers_dict.items())

        # Get body if present
        data = None
        if body_option["type"] == "some":
            body_str = body_option["value"]
            if body_str is not None:
                data = body_str.encode("utf-8")

        # Create request
        req = urllib.request.Request(url, data=data, headers=headers, method=method)

        # Execute request
        try:
            response = urllib.request.urlopen(req)
            status = response.status
            status_text = response.msg
            body = response.read().decode("utf-8")
            response_headers = dict(response.headers)
        except urllib.error.HTTPError as e:
            status = e.code
            status_text = e.msg
            body = e.read().decode("utf-8") if e.fp else ""
            response_headers = dict(e.headers)

        # Convert response headers to EastDict
        east_headers = EastDict(StringType, StringType, response_headers)

        # Create response struct (plain dict wrapped in EastStruct)
        return EastStruct(
            {
                "status": status,
                "statusText": status_text,
                "headers": east_headers,
                "body": body,
                "ok": (200 <= status < 300),
            }
        )

    return await loop.run_in_executor(None, _fetch)


# Platform function implementations
fetch_impl = [
    PlatformFunction(
        name="fetch_get",
        inputs=[StringType],
        output=StringType,
        type="async",
        fn=fetch_get_impl,
    ),
    PlatformFunction(
        name="fetch_post",
        inputs=[StringType, StringType],
        output=StringType,
        type="async",
        fn=fetch_post_impl,
    ),
    PlatformFunction(
        name="fetch_request",
        inputs=[fetch_request_config_type],
        output=fetch_response_type,
        type="async",
        fn=fetch_request_impl,
    ),
]


__all__ = [
    "fetch_impl",
    "fetch_method_type",
    "fetch_request_config_type",
    "fetch_response_type",
]
