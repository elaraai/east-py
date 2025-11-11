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
    ref_ops,
    set_ops,
    string,
    type_system,
)
from east.builtins.registry import get_builtin, list_builtins, register_builtin

__all__ = ["register_builtin", "get_builtin", "list_builtins"]
