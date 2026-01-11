#
# Copyright (c) 2025 Elara AI Pty Ltd
# Licensed under the Business Source License 1.1. See LICENSE.md for details.
#
"""East builtin functions."""

# Import all builtin modules to trigger registration
from east.builtins import (  # noqa: F401
    array,
    blob,
    boolean,
    comparison,
    datetime_ops,
    dict_ops,
    float_ops,
    integer,
    patch,
    ref_ops,
    set_ops,
    string,
)
from east.builtins.registry import get_builtin, list_builtins, register_builtin

__all__ = ["register_builtin", "get_builtin", "list_builtins"]
