"""Environment for variable scoping and binding.

The Environment manages variable lookups, bindings, and nested scopes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class EnvironmentError(Exception):
    """Error in environment operations."""

    pass


@dataclass
class Binding:
    """Variable binding.

    Attributes:
        value: The bound value
        mutable: Whether the variable can be reassigned
    """

    value: Any
    mutable: bool


class Environment:
    """Environment for variable scoping.

    Manages variable bindings with support for nested scopes.
    """

    def __init__(self, parent: Environment | None = None):
        """Initialize environment.

        Args:
            parent: Parent environment for nested scopes
        """
        self.parent = parent
        self.bindings: dict[str, Binding] = {}

    def bind(self, name: str, value: Any, mutable: bool = False) -> None:
        """Bind a variable in this scope.

        Args:
            name: Variable name
            value: Value to bind
            mutable: Whether variable is mutable

        Raises:
            EnvironmentError: If variable already bound in this scope
        """
        if name in self.bindings:
            raise EnvironmentError(f"Variable '{name}' already bound in this scope")
        self.bindings[name] = Binding(value, mutable)

    def lookup(self, name: str) -> Any:
        """Look up a variable value.

        Args:
            name: Variable name

        Returns:
            Variable value

        Raises:
            EnvironmentError: If variable not found
        """
        if name in self.bindings:
            return self.bindings[name].value

        if self.parent is not None:
            return self.parent.lookup(name)

        raise EnvironmentError(f"Variable '{name}' not found")

    def assign(self, name: str, value: Any) -> None:
        """Assign a new value to a mutable variable.

        Args:
            name: Variable name
            value: New value

        Raises:
            EnvironmentError: If variable not found or immutable
        """
        if name in self.bindings:
            binding = self.bindings[name]
            if not binding.mutable:
                raise EnvironmentError(f"Variable '{name}' is not mutable")
            self.bindings[name] = Binding(value, True)
            return

        if self.parent is not None:
            self.parent.assign(name, value)
            return

        raise EnvironmentError(f"Variable '{name}' not found")

    def extend(self) -> Environment:
        """Create a new nested scope.

        Returns:
            New environment with this as parent
        """
        return Environment(parent=self)


__all__: list[str] = ["Environment", "EnvironmentError", "Binding"]
