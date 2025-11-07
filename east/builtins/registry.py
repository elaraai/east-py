"""Builtin function registry."""

from collections.abc import Callable
from typing import Any

# Registry maps builtin name to implementation function
_BUILTINS: dict[str, Callable[..., Any]] = {}


def register_builtin(name: str, func: Callable[..., Any]) -> None:
    """Register a builtin function.

    Args:
        name: Builtin function name
        func: Implementation function
    """
    _BUILTINS[name] = func


def get_builtin(name: str) -> Callable[..., Any]:
    """Get a builtin function by name.

    Args:
        name: Builtin function name

    Returns:
        Implementation function

    Raises:
        KeyError: If builtin not found
    """
    return _BUILTINS[name]


def list_builtins() -> list[str]:
    """List all registered builtin names.

    Returns:
        List of builtin names
    """
    return sorted(_BUILTINS.keys())


__all__ = ["register_builtin", "get_builtin", "list_builtins"]
