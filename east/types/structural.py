"""East structural types: Struct and Variant.

Structs are immutable product types with named fields.
Variants are immutable sum types with tagged cases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from east.types.primitives import null

if TYPE_CHECKING:
    from east.types.type_system import EastType


@dataclass(frozen=True)
class EastStruct:
    """Base class for all East struct instances.

    Structs are immutable product types with named fields.
    Each struct tracks its type (field names and types) and values.
    """

    _east_type: EastType
    _values: tuple[Any, ...]

    def __getattr__(self, name: str) -> Any:
        """Get field value by name.

        Args:
            name: Field name

        Returns:
            Field value

        Raises:
            AttributeError: If field doesn't exist
        """
        # Avoid infinite recursion on special attributes
        if name.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        try:
            idx = self._east_type.field_index(name)
            return self._values[idx]
        except (KeyError, IndexError) as e:
            raise AttributeError(f"Struct has no field '{name}'") from e

    def __eq__(self, other: object) -> bool:
        """Structural equality."""
        if not isinstance(other, EastStruct):
            return NotImplemented
        return self._east_type == other._east_type and self._values == other._values

    def __lt__(self, other: object) -> bool:
        """Structural ordering."""
        if not isinstance(other, EastStruct):
            return NotImplemented
        # Compare type first, then values
        if self._east_type != other._east_type:
            return NotImplemented  # Can't compare different struct types
        return self._values < other._values

    def __hash__(self) -> int:
        """Hash for use in sets/dicts."""
        return hash((self._east_type, self._values))

    def __repr__(self) -> str:
        """Return East text format representation."""
        if len(self._values) == 0:
            return "()"
        items = ", ".join(
            f"{name}={repr(value)}"
            for name, value in zip(self._east_type.field_names(), self._values, strict=False)
        )
        return f"({items})"


@dataclass(frozen=True)
class Case:
    """A single case in a variant.

    Represents a tagged value in a sum type.
    """

    tag: str
    value: Any

    def __eq__(self, other: object) -> bool:
        """Cases are equal if tag and value are equal."""
        if not isinstance(other, Case):
            return NotImplemented
        return self.tag == other.tag and self.value == other.value

    def __lt__(self, other: object) -> bool:
        """Cases ordered by tag, then value."""
        if not isinstance(other, Case):
            return NotImplemented
        if self.tag < other.tag:
            return True
        if self.tag > other.tag:
            return False
        return self.value < other.value

    def __hash__(self) -> int:
        """Hash based on tag and value."""
        # Convert lists to tuples for hashing
        value = self.value
        if isinstance(value, list):
            value = tuple(value)
        return hash((self.tag, value))

    def __repr__(self) -> str:
        """Return East text format representation."""
        from east.types.primitives import Null

        if isinstance(self.value, Null):
            return f".{self.tag}"
        return f".{self.tag} {repr(self.value)}"


@dataclass(frozen=True)
class EastVariant:
    """Base class for all East variant instances.

    Variants are immutable sum types with tagged cases.
    Each variant tracks its type (case names and types) and current case.
    """

    _east_type: EastType
    _case: Case

    @property
    def tag(self) -> str:
        """Get the tag of the current case."""
        return self._case.tag

    @property
    def value(self) -> Any:
        """Get the value of the current case."""
        return self._case.value

    def __eq__(self, other: object) -> bool:
        """Structural equality."""
        if not isinstance(other, EastVariant):
            return NotImplemented
        return self._east_type == other._east_type and self._case == other._case

    def __lt__(self, other: object) -> bool:
        """Structural ordering."""
        if not isinstance(other, EastVariant):
            return NotImplemented
        # Compare type first, then case
        if self._east_type != other._east_type:
            return NotImplemented  # Can't compare different variant types
        return self._case < other._case

    def __hash__(self) -> int:
        """Hash for use in sets/dicts."""
        return hash((self._east_type, self._case))

    def __repr__(self) -> str:
        """Return East text format representation."""
        return repr(self._case)


def make_case(tag: str, value: Any = None) -> Case:
    """Helper to create a Case, using null for missing value.

    Args:
        tag: The case tag
        value: The case value (defaults to null)

    Returns:
        A Case instance
    """
    if value is None:
        value = null
    return Case(tag, value)


__all__ = [
    "EastStruct",
    "Case",
    "EastVariant",
    "make_case",
]
