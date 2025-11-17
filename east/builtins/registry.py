"""Builtin function registry.

All builtins are factory functions that take type parameters at compile time
and return the actual implementation callable. This matches the TypeScript
architecture and ensures type-dependent operations are optimized at compile time.
"""

from collections.abc import Callable
from typing import Any

# Registry maps builtin name to factory function
# All factories take (*type_params) and return the implementation callable
_BUILTINS: dict[str, Callable[..., Callable[..., Any]]] = {}


def register_builtin(name: str, factory: Callable[..., Callable[..., Any]]) -> None:
    """Register a builtin factory function.

    All builtins are factories that take type parameters at compile time and
    return a callable that takes value arguments at runtime.

    Args:
        name: Builtin function name
        factory: Factory function that takes type parameters and returns implementation
    """
    _BUILTINS[name] = factory


def get_builtin(name: str) -> Callable[..., Callable[..., Any]]:
    """Get a builtin factory function by name.

    Args:
        name: Builtin function name

    Returns:
        Factory function that takes type parameters and returns implementation

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
