"""East container types: Array, Set, Dict.

These are mutable containers with type tracking:
- Array: Ordered, indexed collection (like Python list)
- Set: Sorted, unique collection (using East ordering)
- Dict: Sorted key-value collection (keys sorted by East ordering)
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING, Any

from sortedcontainers import SortedDict, SortedSet  # type: ignore[import-untyped]

from east.utils.ordering import make_east_key

if TYPE_CHECKING:
    from east.types.types import EastType


class EastArray(list):
    """East array with element type tracking.

    Arrays are mutable, ordered, 0-indexed collections.
    They behave like Python lists but track the element type.
    """

    def __init__(self, element_type: EastType, items: list | None = None):
        """Create an array with a specific element type.

        Args:
            element_type: The type of elements in this array
            items: Initial items (optional)
        """
        super().__init__(items or [])
        self.element_type = element_type
        self._iteration_lock = 0  # Counter for nested iterations

    def _lock_for_iteration(self) -> None:
        """Lock array for iteration (prevents modifications)."""
        self._iteration_lock += 1

    def _unlock_for_iteration(self) -> None:
        """Unlock array after iteration."""
        self._iteration_lock -= 1

    def _check_not_iterating(self) -> None:
        """Check if array is being iterated and raise error if so."""
        if self._iteration_lock > 0:
            raise RuntimeError("Cannot modify Array during iteration")

    # Override all mutation methods to check for iteration

    def append(self, item: Any) -> None:
        """Add item to end of array."""
        self._check_not_iterating()
        super().append(item)

    def extend(self, items: Any) -> None:
        """Extend array with items."""
        self._check_not_iterating()
        super().extend(items)

    def insert(self, index: int, item: Any) -> None:
        """Insert item at index."""
        self._check_not_iterating()
        super().insert(index, item)

    def remove(self, item: Any) -> None:
        """Remove first occurrence of item."""
        self._check_not_iterating()
        super().remove(item)

    def pop(self, index: int = -1) -> Any:
        """Remove and return item at index."""
        self._check_not_iterating()
        return super().pop(index)

    def clear(self) -> None:
        """Remove all items."""
        self._check_not_iterating()
        super().clear()

    def __setitem__(self, index: Any, value: Any) -> None:
        """Set item at index."""
        self._check_not_iterating()
        super().__setitem__(index, value)

    def __delitem__(self, index: Any) -> None:
        """Delete item at index."""
        self._check_not_iterating()
        super().__delitem__(index)

    def reverse(self) -> None:
        """Reverse array in place."""
        self._check_not_iterating()
        super().reverse()

    def sort(self, *args: Any, **kwargs: Any) -> None:
        """Sort array in place."""
        self._check_not_iterating()
        super().sort(*args, **kwargs)

    def __repr__(self) -> str:
        """Return East text format representation."""
        if len(self) == 0:
            return "[]"
        items = ", ".join(repr(item) for item in self)
        return f"[{items}]"


class EastSet:
    """East set with element type tracking.

    Sets are mutable, sorted collections of unique elements.
    Elements are sorted using East's total ordering.
    """

    def __init__(self, element_type: EastType, items: Iterable[Any] | None = None):
        """Create a set with a specific element type.

        Args:
            element_type: The type of elements in this set
            items: Initial items (optional)
        """
        self.element_type = element_type
        self._data: SortedSet = SortedSet(key=make_east_key(element_type))
        self._iteration_lock = 0  # Counter for nested iterations
        if items is not None:
            for item in items:
                self._data.add(item)

    def _lock_for_iteration(self) -> None:
        """Lock set for iteration (prevents modifications)."""
        self._iteration_lock += 1

    def _unlock_for_iteration(self) -> None:
        """Unlock set after iteration."""
        self._iteration_lock -= 1

    def _check_not_iterating(self) -> None:
        """Check if set is being iterated and raise error if so."""
        if self._iteration_lock > 0:
            raise RuntimeError("Cannot modify Set during iteration")

    def add(self, item: Any) -> None:
        """Add an item to the set.

        Args:
            item: Item to add
        """
        self._check_not_iterating()
        self._data.add(item)

    def remove(self, item: Any) -> None:
        """Remove an item from the set.

        Args:
            item: Item to remove

        Raises:
            KeyError: If item not in set
        """
        self._check_not_iterating()
        self._data.remove(item)

    def discard(self, item: Any) -> None:
        """Remove an item from the set if present.

        Args:
            item: Item to remove
        """
        self._check_not_iterating()
        self._data.discard(item)

    def clear(self) -> None:
        """Remove all items from the set."""
        self._check_not_iterating()
        self._data.clear()

    def __contains__(self, item: Any) -> bool:
        """Check if item is in the set."""
        return item in self._data

    def __len__(self) -> int:
        """Return number of items in the set."""
        return len(self._data)

    def __iter__(self) -> Iterator[Any]:
        """Iterate over items in sorted order."""
        return iter(self._data)

    def __eq__(self, other: object) -> bool:
        """Sets are equal if they contain the same elements."""
        if not isinstance(other, EastSet):
            return NotImplemented
        return self._data == other._data

    def __repr__(self) -> str:
        """Return East text format representation."""
        if len(self) == 0:
            return "{}"
        items = ", ".join(repr(item) for item in self)
        return f"{{{items}}}"


class EastDict:
    """East dict with key and value type tracking.

    Dicts are mutable, sorted collections of key-value pairs.
    Keys are sorted using East's total ordering.
    """

    def __init__(
        self,
        key_type: EastType,
        value_type: EastType,
        items: dict | None = None,
    ):
        """Create a dict with specific key and value types.

        Args:
            key_type: The type of keys in this dict
            value_type: The type of values in this dict
            items: Initial items (optional)
        """
        self.key_type = key_type
        self.value_type = value_type
        self._data: SortedDict = SortedDict(make_east_key(key_type))
        self._iteration_lock = 0  # Counter for nested iterations
        if items is not None:
            for key, value in items.items():
                self._data[key] = value

    def _lock_for_iteration(self) -> None:
        """Lock dict for iteration (prevents modifications)."""
        self._iteration_lock += 1

    def _unlock_for_iteration(self) -> None:
        """Unlock dict after iteration."""
        self._iteration_lock -= 1

    def _check_not_iterating(self) -> None:
        """Check if dict is being iterated and raise error if so."""
        if self._iteration_lock > 0:
            raise RuntimeError("Cannot modify Dict during iteration")

    def __getitem__(self, key: Any) -> Any:
        """Get value for key.

        Args:
            key: The key to look up

        Returns:
            The value for the key

        Raises:
            KeyError: If key not in dict
        """
        return self._data[key]

    def __setitem__(self, key: Any, value: Any) -> None:
        """Set value for key.

        Args:
            key: The key to set
            value: The value to set
        """
        self._check_not_iterating()
        self._data[key] = value

    def __delitem__(self, key: Any) -> None:
        """Delete key from dict.

        Args:
            key: The key to delete

        Raises:
            KeyError: If key not in dict
        """
        self._check_not_iterating()
        del self._data[key]

    def __contains__(self, key: Any) -> bool:
        """Check if key is in the dict."""
        return key in self._data

    def __len__(self) -> int:
        """Return number of key-value pairs."""
        return len(self._data)

    def __iter__(self) -> Iterator[Any]:
        """Iterate over keys in sorted order."""
        return iter(self._data)

    def __eq__(self, other: object) -> bool:
        """Dicts are equal if they have the same key-value pairs."""
        if not isinstance(other, EastDict):
            return NotImplemented
        return self._data == other._data

    def keys(self) -> Iterator[Any]:
        """Return iterator over keys in sorted order."""
        return iter(self._data.keys())

    def values(self) -> Iterator[Any]:
        """Get iterator over values.

        Returns values in key sort order.
        """
        return iter(self._data.values())

    def items(self) -> Iterator[Any]:
        """Return iterator over (key, value) pairs in sorted order."""
        return iter(self._data.items())

    def get(self, key: Any, default: Any = None) -> Any:
        """Get value for key, returning default if not found.

        Args:
            key: The key to look up
            default: Value to return if key not found

        Returns:
            The value for the key, or default
        """
        return self._data.get(key, default)

    def pop(self, key: Any, *args: Any) -> Any:
        """Remove and return value for key.

        Args:
            key: The key to pop
            *args: Optional default value

        Returns:
            The value for the key

        Raises:
            KeyError: If key not in dict and no default provided
        """
        self._check_not_iterating()
        return self._data.pop(key, *args)

    def clear(self) -> None:
        """Remove all key-value pairs."""
        self._check_not_iterating()
        self._data.clear()

    def __repr__(self) -> str:
        """Return East text format representation."""
        if len(self) == 0:
            return "{:}"
        items = ", ".join(f"{repr(k)}: {repr(v)}" for k, v in self.items())
        return f"{{{items}}}"


__all__ = [
    "EastArray",
    "EastSet",
    "EastDict",
]
