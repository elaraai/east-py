"""East builtin functions."""

# Import all builtin modules to trigger registration
from east.builtins import (  # noqa: F401
    array,
    boolean,
    comparison,
    dict_ops,
    float_ops,
    integer,
    set_ops,
    string,
)
from east.builtins.registry import get_builtin, list_builtins, register_builtin

__all__ = ["register_builtin", "get_builtin", "list_builtins"]
