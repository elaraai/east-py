"""Hashable wrappers for East structural types (Struct and Variant).

These classes wrap plain Python dicts to make them hashable for use as keys
in Sets and Dicts, while maintaining dict-like behavior for compatibility.
"""

from __future__ import annotations

from typing import Any, Generic, TypeVar

T = TypeVar("T")
V = TypeVar("V")


class EastStruct(dict, Generic[T]):
    """Hashable, immutable struct wrapper.

    Wraps a plain dict to make it hashable for use in Sets and Dicts.
    Behaves like a dict but implements __hash__() for hashability.

    Generic type parameter T should be a TypedDict describing the structure:

    Example:
        from typing import TypedDict

        class PersonValue(TypedDict):
            name: str
            age: int

        person: EastStruct[PersonValue] = EastStruct({"name": "Alice", "age": 30})
        name = person["name"]  # Type checker knows this is str
    """

    def __init__(self, data: dict[str, Any]):
        """Create an immutable struct from a dict.

        Args:
            data: Dictionary of field names to values
        """
        super().__init__(data)
        self._hash: int | None = None

    def __hash__(self) -> int:
        """Compute hash based on sorted field items."""
        if self._hash is None:
            # Hash based on sorted (key, value) pairs
            # Use id() for unhashable values (e.g., nested dicts, arrays)
            items = []
            for k in sorted(self.keys()):
                v = self[k]
                try:
                    items.append((k, hash(v)))
                except TypeError:
                    # Unhashable value - use id as fallback
                    items.append((k, id(v)))
            self._hash = hash(tuple(items))
        return self._hash

    def __setitem__(self, key: str, value: Any) -> None:
        """Prevent modification after creation."""
        raise TypeError("EastStruct is immutable")

    def __delitem__(self, key: str) -> None:
        """Prevent modification after creation."""
        raise TypeError("EastStruct is immutable")

    def clear(self) -> None:
        """Prevent modification after creation."""
        raise TypeError("EastStruct is immutable")

    def pop(self, *_args: Any) -> Any:
        """Prevent modification after creation."""
        raise TypeError("EastStruct is immutable")

    def popitem(self) -> Any:
        """Prevent modification after creation."""
        raise TypeError("EastStruct is immutable")

    def setdefault(self, _key: str, _default: Any = None) -> Any:
        """Prevent modification after creation."""
        raise TypeError("EastStruct is immutable")

    def update(self, *_args: Any, **_kwargs: Any) -> None:
        """Prevent modification after creation."""
        raise TypeError("EastStruct is immutable")

    def __repr__(self) -> str:
        """Return dict-like representation."""
        return f"EastStruct({dict.__repr__(self)})"


class EastVariant(dict, Generic[V]):
    """Hashable, immutable variant wrapper.

    Wraps a plain dict representing a tagged union to make it hashable
    for use in Sets and Dicts. Behaves like a dict but implements __hash__().

    Generic type parameter V should be a TypedDict describing the variant structure:

    Example:
        from typing import TypedDict, Literal

        class OptionValue(TypedDict):
            type: Literal["some", "none"]
            value: str | None

        opt: EastVariant[OptionValue] = EastVariant("some", "hello")
        tag = opt["type"]  # Type checker knows this is Literal["some", "none"]
    """

    def __init__(self, tag: str, value: Any):
        """Create an immutable variant.

        Args:
            tag: The variant case tag (stored as "type" key)
            value: The value for this case
        """
        super().__init__(type=tag, value=value)
        self._hash: int | None = None

    def __hash__(self) -> int:
        """Compute hash based on type and value."""
        if self._hash is None:
            tag = self["type"]
            value = self["value"]
            try:
                self._hash = hash((tag, value))
            except TypeError:
                # Unhashable value - use id as fallback
                self._hash = hash((tag, id(value)))
        return self._hash

    def __setitem__(self, key: str, value: Any) -> None:
        """Prevent modification after creation."""
        raise TypeError("EastVariant is immutable")

    def __delitem__(self, key: str) -> None:
        """Prevent modification after creation."""
        raise TypeError("EastVariant is immutable")

    def clear(self) -> None:
        """Prevent modification after creation."""
        raise TypeError("EastVariant is immutable")

    def pop(self, *_args: Any) -> Any:
        """Prevent modification after creation."""
        raise TypeError("EastVariant is immutable")

    def popitem(self) -> Any:
        """Prevent modification after creation."""
        raise TypeError("EastVariant is immutable")

    def setdefault(self, _key: str, _default: Any = None) -> Any:
        """Prevent modification after creation."""
        raise TypeError("EastVariant is immutable")

    def update(self, *_args: Any, **_kwargs: Any) -> None:
        """Prevent modification after creation."""
        raise TypeError("EastVariant is immutable")

    def __repr__(self) -> str:
        """Return variant representation."""
        return f"EastVariant(type={self['type']!r}, value={self['value']!r})"
