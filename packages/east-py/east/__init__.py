#
# Copyright (c) 2025 Elara AI Pty Ltd
# Licensed under the Business Source License 1.1. See LICENSE.md for details.
#
"""East.py - Python runtime for the East programming language."""

import contextlib

__version__ = "0.1.0"

# Detect which Cython extensions are available
CYTHON_EXTENSIONS: list[str] = []
with contextlib.suppress(ImportError):
    from east.types._values_cy import CyEastStruct as _  # noqa: F401
    CYTHON_EXTENSIONS.append("values")
with contextlib.suppress(ImportError):
    from east.serialization._beast2_cy import decode_beast2_value_for as _  # noqa: F401
    CYTHON_EXTENSIONS.append("beast2")
with contextlib.suppress(ImportError):
    from east.serialization._binary_utils_cy import read_varint as _  # noqa: F401
    CYTHON_EXTENSIONS.append("binary_utils")
with contextlib.suppress(ImportError):
    from east.serialization._csv_cy import cy_decode_csv_for as _  # noqa: F401
    CYTHON_EXTENSIONS.append("csv")
with contextlib.suppress(ImportError):
    from east.utils._ordering_cy import cy_make_east_key as _  # noqa: F401
    CYTHON_EXTENSIONS.append("ordering")

__all__: list[str] = ["CYTHON_EXTENSIONS"]
