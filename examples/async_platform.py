"""Example: Async Platform Functions with East IR

This example demonstrates:
- Decoding IR from JSON
- Defining platform function implementations (sync and async)
- Compiling IR with compile_async()
- Executing compiled async functions

The IR implements a function that:
1. Logs "Fetching URL: {url}"
2. Calls time_ns() to get start time
3. Calls async fetch_status(url) to fetch HTTP status
4. Calls time_ns() again to get end time
5. Logs "Response status: {status} - fetched in {time} ms"
"""

import asyncio
import json
import time

from east.runtime.compiler import compile_async
from east.runtime.platform import PlatformFunction
from east.serialization.json import decode_json_for
from east.types.types import IntegerType, IRType, NullType, StringType

# IR JSON generated from TypeScript East example (/home/crambelsoupy/src/East/contrib/examples/async.ts)
# This is the increment function: (url: String) -> Null
FETCH_STATUS_IR_JSON = {
    "type": "Function",
    "value": {
        "type": {
            "type": "Function",
            "value": {
                "inputs": [{"type": "String", "value": None}],
                "output": {"type": "Null", "value": None},
                "platforms": ["fetch_status", "log", "time_ns"],
            },
        },
        "location": {
            "filename": "node:internal/modules/esm/loader",
            "line": "651",
            "column": "26",
        },
        "captures": [],
        "parameters": [
            {
                "type": "Variable",
                "value": {
                    "type": {"type": "String", "value": None},
                    "name": "_0",
                    "location": {
                        "filename": "node:internal/modules/esm/loader",
                        "line": "651",
                        "column": "26",
                    },
                    "mutable": False,
                    "captured": False,
                },
            }
        ],
        "body": {
            "type": "Block",
            "value": {
                "type": {"type": "Null", "value": None},
                "location": {
                    "filename": "node:internal/modules/esm/loader",
                    "line": "651",
                    "column": "26",
                },
                "statements": [
                    {
                        "type": "Platform",
                        "value": {
                            "type": {"type": "Null", "value": None},
                            "location": {
                                "filename": "node:internal/modules/esm/loader",
                                "line": "651",
                                "column": "26",
                            },
                            "name": "log",
                            "arguments": [
                                {
                                    "type": "Builtin",
                                    "value": {
                                        "type": {"type": "String", "value": None},
                                        "location": {
                                            "filename": "node:internal/modules/esm/loader",
                                            "line": "651",
                                            "column": "26",
                                        },
                                        "builtin": "StringConcat",
                                        "type_parameters": [],
                                        "arguments": [
                                            {
                                                "type": "Value",
                                                "value": {
                                                    "type": {
                                                        "type": "String",
                                                        "value": None,
                                                    },
                                                    "location": {
                                                        "filename": "node:internal/modules/esm/loader",
                                                        "line": "651",
                                                        "column": "26",
                                                    },
                                                    "value": {
                                                        "type": "String",
                                                        "value": "Fetching URL: ",
                                                    },
                                                },
                                            },
                                            {
                                                "type": "Variable",
                                                "value": {
                                                    "type": {
                                                        "type": "String",
                                                        "value": None,
                                                    },
                                                    "name": "_0",
                                                    "location": {
                                                        "filename": "node:internal/modules/esm/loader",
                                                        "line": "651",
                                                        "column": "26",
                                                    },
                                                    "mutable": False,
                                                    "captured": False,
                                                },
                                            },
                                        ],
                                    },
                                }
                            ],
                        },
                    },
                    {
                        "type": "Let",
                        "value": {
                            "type": {"type": "Null", "value": None},
                            "location": {
                                "filename": "node:internal/modules/esm/loader",
                                "line": "651",
                                "column": "26",
                            },
                            "variable": {
                                "type": "Variable",
                                "value": {
                                    "type": {"type": "Integer", "value": None},
                                    "name": "_1",
                                    "location": {
                                        "filename": "node:internal/modules/esm/loader",
                                        "line": "651",
                                        "column": "26",
                                    },
                                    "mutable": True,
                                    "captured": False,
                                },
                            },
                            "value": {
                                "type": "Platform",
                                "value": {
                                    "type": {"type": "Integer", "value": None},
                                    "location": {
                                        "filename": "node:internal/modules/esm/loader",
                                        "line": "651",
                                        "column": "26",
                                    },
                                    "name": "time_ns",
                                    "arguments": [],
                                },
                            },
                        },
                    },
                    {
                        "type": "Platform",
                        "value": {
                            "type": {"type": "String", "value": None},
                            "location": {
                                "filename": "node:internal/modules/esm/loader",
                                "line": "651",
                                "column": "26",
                            },
                            "name": "fetch_status",
                            "arguments": [
                                {
                                    "type": "Variable",
                                    "value": {
                                        "type": {"type": "String", "value": None},
                                        "name": "_0",
                                        "location": {
                                            "filename": "node:internal/modules/esm/loader",
                                            "line": "651",
                                            "column": "26",
                                        },
                                        "mutable": False,
                                        "captured": False,
                                    },
                                }
                            ],
                        },
                    },
                    {
                        "type": "Let",
                        "value": {
                            "type": {"type": "Null", "value": None},
                            "location": {
                                "filename": "node:internal/modules/esm/loader",
                                "line": "651",
                                "column": "26",
                            },
                            "variable": {
                                "type": "Variable",
                                "value": {
                                    "type": {"type": "Integer", "value": None},
                                    "name": "_2",
                                    "location": {
                                        "filename": "node:internal/modules/esm/loader",
                                        "line": "651",
                                        "column": "26",
                                    },
                                    "mutable": True,
                                    "captured": False,
                                },
                            },
                            "value": {
                                "type": "Platform",
                                "value": {
                                    "type": {"type": "Integer", "value": None},
                                    "location": {
                                        "filename": "node:internal/modules/esm/loader",
                                        "line": "651",
                                        "column": "26",
                                    },
                                    "name": "time_ns",
                                    "arguments": [],
                                },
                            },
                        },
                    },
                    {
                        "type": "Platform",
                        "value": {
                            "type": {"type": "Null", "value": None},
                            "location": {
                                "filename": "node:internal/modules/esm/loader",
                                "line": "651",
                                "column": "26",
                            },
                            "name": "log",
                            "arguments": [
                                {
                                    "type": "Builtin",
                                    "value": {
                                        "type": {"type": "String", "value": None},
                                        "location": {
                                            "filename": "node:internal/modules/esm/loader",
                                            "line": "651",
                                            "column": "26",
                                        },
                                        "builtin": "StringConcat",
                                        "type_parameters": [],
                                        "arguments": [
                                            {
                                                "type": "Builtin",
                                                "value": {
                                                    "type": {
                                                        "type": "String",
                                                        "value": None,
                                                    },
                                                    "location": {
                                                        "filename": "node:internal/modules/esm/loader",
                                                        "line": "651",
                                                        "column": "26",
                                                    },
                                                    "builtin": "StringConcat",
                                                    "type_parameters": [],
                                                    "arguments": [
                                                        {
                                                            "type": "Builtin",
                                                            "value": {
                                                                "type": {
                                                                    "type": "String",
                                                                    "value": None,
                                                                },
                                                                "location": {
                                                                    "filename": "node:internal/modules/esm/loader",
                                                                    "line": "651",
                                                                    "column": "26",
                                                                },
                                                                "builtin": "StringConcat",
                                                                "type_parameters": [],
                                                                "arguments": [
                                                                    {
                                                                        "type": "Builtin",
                                                                        "value": {
                                                                            "type": {
                                                                                "type": "String",
                                                                                "value": None,
                                                                            },
                                                                            "location": {
                                                                                "filename": "node:internal/modules/esm/loader",
                                                                                "line": "651",
                                                                                "column": "26",
                                                                            },
                                                                            "builtin": "StringConcat",
                                                                            "type_parameters": [],
                                                                            "arguments": [
                                                                                {
                                                                                    "type": "Value",
                                                                                    "value": {
                                                                                        "type": {
                                                                                            "type": "String",
                                                                                            "value": None,
                                                                                        },
                                                                                        "location": {
                                                                                            "filename": "node:internal/modules/esm/loader",
                                                                                            "line": "651",
                                                                                            "column": "26",
                                                                                        },
                                                                                        "value": {
                                                                                            "type": "String",
                                                                                            "value": "Response status: ",
                                                                                        },
                                                                                    },
                                                                                },
                                                                                {
                                                                                    "type": "Platform",
                                                                                    "value": {
                                                                                        "type": {
                                                                                            "type": "String",
                                                                                            "value": None,
                                                                                        },
                                                                                        "location": {
                                                                                            "filename": "node:internal/modules/esm/loader",
                                                                                            "line": "651",
                                                                                            "column": "26",
                                                                                        },
                                                                                        "name": "fetch_status",
                                                                                        "arguments": [
                                                                                            {
                                                                                                "type": "Variable",
                                                                                                "value": {
                                                                                                    "type": {
                                                                                                        "type": "String",
                                                                                                        "value": None,
                                                                                                    },
                                                                                                    "name": "_0",
                                                                                                    "location": {
                                                                                                        "filename": "node:internal/modules/esm/loader",
                                                                                                        "line": "651",
                                                                                                        "column": "26",
                                                                                                    },
                                                                                                    "mutable": False,
                                                                                                    "captured": False,
                                                                                                },
                                                                                            }
                                                                                        ],
                                                                                    },
                                                                                },
                                                                            ],
                                                                        },
                                                                    },
                                                                    {
                                                                        "type": "Value",
                                                                        "value": {
                                                                            "type": {
                                                                                "type": "String",
                                                                                "value": None,
                                                                            },
                                                                            "location": {
                                                                                "filename": "node:internal/modules/esm/loader",
                                                                                "line": "651",
                                                                                "column": "26",
                                                                            },
                                                                            "value": {
                                                                                "type": "String",
                                                                                "value": " - fetched in ",
                                                                            },
                                                                        },
                                                                    },
                                                                ],
                                                            },
                                                        },
                                                        {
                                                            "type": "Builtin",
                                                            "value": {
                                                                "type": {
                                                                    "type": "String",
                                                                    "value": None,
                                                                },
                                                                "location": {
                                                                    "filename": "node:internal/modules/esm/loader",
                                                                    "line": "651",
                                                                    "column": "26",
                                                                },
                                                                "builtin": "Print",
                                                                "type_parameters": [
                                                                    {
                                                                        "type": "Float",
                                                                        "value": None,
                                                                    }
                                                                ],
                                                                "arguments": [
                                                                    {
                                                                        "type": "Builtin",
                                                                        "value": {
                                                                            "type": {
                                                                                "type": "Float",
                                                                                "value": None,
                                                                            },
                                                                            "location": {
                                                                                "filename": "node:internal/modules/esm/loader",
                                                                                "line": "651",
                                                                                "column": "26",
                                                                            },
                                                                            "builtin": "FloatMultiply",
                                                                            "type_parameters": [],
                                                                            "arguments": [
                                                                                {
                                                                                    "type": "Builtin",
                                                                                    "value": {
                                                                                        "type": {
                                                                                            "type": "Float",
                                                                                            "value": None,
                                                                                        },
                                                                                        "location": {
                                                                                            "filename": "node:internal/modules/esm/loader",
                                                                                            "line": "651",
                                                                                            "column": "26",
                                                                                        },
                                                                                        "builtin": "IntegerToFloat",
                                                                                        "type_parameters": [],
                                                                                        "arguments": [
                                                                                            {
                                                                                                "type": "Builtin",
                                                                                                "value": {
                                                                                                    "type": {
                                                                                                        "type": "Integer",
                                                                                                        "value": None,
                                                                                                    },
                                                                                                    "location": {
                                                                                                        "filename": "node:internal/modules/esm/loader",
                                                                                                        "line": "651",
                                                                                                        "column": "26",
                                                                                                    },
                                                                                                    "builtin": "IntegerSubtract",
                                                                                                    "type_parameters": [],
                                                                                                    "arguments": [
                                                                                                        {
                                                                                                            "type": "Variable",
                                                                                                            "value": {
                                                                                                                "type": {
                                                                                                                    "type": "Integer",
                                                                                                                    "value": None,
                                                                                                                },
                                                                                                                "name": "_2",
                                                                                                                "location": {
                                                                                                                    "filename": "node:internal/modules/esm/loader",
                                                                                                                    "line": "651",
                                                                                                                    "column": "26",
                                                                                                                },
                                                                                                                "mutable": True,
                                                                                                                "captured": False,
                                                                                                            },
                                                                                                        },
                                                                                                        {
                                                                                                            "type": "Variable",
                                                                                                            "value": {
                                                                                                                "type": {
                                                                                                                    "type": "Integer",
                                                                                                                    "value": None,
                                                                                                                },
                                                                                                                "name": "_1",
                                                                                                                "location": {
                                                                                                                    "filename": "node:internal/modules/esm/loader",
                                                                                                                    "line": "651",
                                                                                                                    "column": "26",
                                                                                                                },
                                                                                                                "mutable": True,
                                                                                                                "captured": False,
                                                                                                            },
                                                                                                        },
                                                                                                    ],
                                                                                                },
                                                                                            }
                                                                                        ],
                                                                                    },
                                                                                },
                                                                                {
                                                                                    "type": "Value",
                                                                                    "value": {
                                                                                        "type": {
                                                                                            "type": "Float",
                                                                                            "value": None,
                                                                                        },
                                                                                        "location": {
                                                                                            "filename": "node:internal/modules/esm/loader",
                                                                                            "line": "651",
                                                                                            "column": "26",
                                                                                        },
                                                                                        "value": {
                                                                                            "type": "Float",
                                                                                            "value": 0.000001,
                                                                                        },
                                                                                    },
                                                                                },
                                                                            ],
                                                                        },
                                                                    }
                                                                ],
                                                            },
                                                        },
                                                    ],
                                                },
                                            },
                                            {
                                                "type": "Value",
                                                "value": {
                                                    "type": {
                                                        "type": "String",
                                                        "value": None,
                                                    },
                                                    "location": {
                                                        "filename": "node:internal/modules/esm/loader",
                                                        "line": "651",
                                                        "column": "26",
                                                    },
                                                    "value": {
                                                        "type": "String",
                                                        "value": " ms",
                                                    },
                                                },
                                            },
                                        ],
                                    },
                                }
                            ],
                        },
                    },
                ],
            },
        },
    },
}


# Platform function implementations
def log_impl(message: str) -> None:
    """Sync platform function: log a message to console."""
    print(message)


async def fetch_status_impl(url: str) -> str:
    """Async platform function: fetch HTTP status from URL."""
    # For this example, we'll use a simple implementation without external dependencies
    # In production, you'd use aiohttp or httpx
    import urllib.request

    try:
        # Note: urllib.request is synchronous, but we're wrapping it in async
        # In production, use async libraries like aiohttp
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, urllib.request.urlopen, url)
        status = response.status
        msg = response.msg
        return f"{status} ({msg})"
    except Exception as e:
        return f"Error: {e}"


def time_ns_impl() -> int:
    """Sync platform function: get current time in nanoseconds."""
    return time.time_ns()


# Decode IR from JSON
decoder = decode_json_for(IRType)
ir_bytes = json.dumps(FETCH_STATUS_IR_JSON).encode("utf-8")
fetch_status_ir = decoder(ir_bytes)

# Define platform functions
platform: list[PlatformFunction] = [
    PlatformFunction(name="log", inputs=[StringType], output=NullType, type="sync", fn=log_impl),
    PlatformFunction(
        name="fetch_status",
        inputs=[StringType],
        output=StringType,
        type="async",
        fn=fetch_status_impl,
    ),
    PlatformFunction(name="time_ns", inputs=[], output=IntegerType, type="sync", fn=time_ns_impl),
]

# Compile with async platform functions
compiled_fn = compile_async(fetch_status_ir, platform)


# Execute
async def main():
    """Run the fetch_status function three times."""
    await compiled_fn("https://www.google.com")
    print()
    await compiled_fn("https://www.google.com")
    print()
    await compiled_fn("https://www.google.com")


if __name__ == "__main__":
    asyncio.run(main())
