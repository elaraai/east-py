#
# Copyright (c) 2025 Elara AI Pty Ltd
# Licensed under the Business Source License 1.1. See LICENSE.md for details.
#
"""East value types - Python representations of East values.

All East value types are prefixed with 'East' for explicit naming:
- EastNull: Unit type (singleton)
- EastBlob: Immutable binary data (extends bytes)
- EastArray: Ordered collection (extends list)
- EastSet: Sorted unique collection
- EastDict: Sorted key-value collection
- EastStruct: Immutable record type (extends dict)
- EastVariant: Tagged union type (extends dict)
- EastOption: Option variant (some/none)
- EastRef: Mutable reference cell

For primitive types (Boolean, Integer, Float, String, DateTime),
Python's built-in types are used directly (bool, int, float, str, datetime).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Generic, SupportsIndex, TypeGuard, TypeVar

from sortedcontainers import SortedDict, SortedSet  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from east.types.types import EastType

T = TypeVar("T")
V = TypeVar("V")
OptionT = TypeVar("OptionT")  # Option inner type


# =============================================================================
# EastNull - Unit type singleton
# =============================================================================


class EastNull:
    """East's canonical unit type.

    Represents the absence of a value, analogous to None but with
    distinct type identity for East's type system.
    """

    _instance: EastNull | None = None

    def __new__(cls) -> EastNull:
        """Ensure EastNull is a singleton."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        """Return East text format representation."""
        return "null"

    def __str__(self) -> str:
        """Return East text format representation."""
        return "null"

    def __eq__(self, other: object) -> bool:
        """EastNull equals only itself."""
        return isinstance(other, EastNull)

    def __hash__(self) -> int:
        """Hash for use in sets/dicts."""
        return hash(None)

    def __lt__(self, other: object) -> bool:
        """EastNull is not less than anything (including itself)."""
        if not isinstance(other, EastNull):
            return NotImplemented
        return False

    def __le__(self, other: object) -> bool:
        """EastNull is less than or equal to itself."""
        if not isinstance(other, EastNull):
            return NotImplemented
        return True

    def __gt__(self, other: object) -> bool:
        """EastNull is not greater than anything."""
        if not isinstance(other, EastNull):
            return NotImplemented
        return False

    def __ge__(self, other: object) -> bool:
        """EastNull is greater than or equal to itself."""
        if not isinstance(other, EastNull):
            return NotImplemented
        return True


# Singleton instance
east_null = EastNull()


# =============================================================================
# EastBlob - Immutable binary data
# =============================================================================


class EastBlob(bytes):
    """East blob type - immutable binary data.

    Extends bytes directly, so it works anywhere bytes is expected.
    Provides East-specific formatting (hexadecimal representation).

    Example:
        data: EastBlob = EastBlob(b"\\x01\\x02\\x03")
        compressed: EastBlob = EastBlob(gzip.compress(data))

        # Works as bytes
        len(data)  # 3
        data[0]    # 1
    """

    def __new__(cls, data: bytes | bytearray | list[int] | EastBlob) -> EastBlob:
        """Create an EastBlob from various byte sources."""
        if isinstance(data, (EastBlob, bytes, bytearray)):
            return super().__new__(cls, data)
        if isinstance(data, list):
            return super().__new__(cls, bytes(data))
        raise TypeError(f"Cannot create EastBlob from {type(data)}")

    @property
    def data(self) -> bytes:
        """Access underlying bytes (for compatibility)."""
        return bytes(self)

    def __hash__(self) -> int:
        """Hash based on bytes content."""
        return super().__hash__()

    def __repr__(self) -> str:
        """Return East hexadecimal format."""
        if len(self) == 0:
            return "0x"
        # Limit display for very large blobs
        if len(self) > 256:
            hex_str = self[:256].hex()
            return f"0x{hex_str}..."
        return f"0x{self.hex()}"

    def __str__(self) -> str:
        """Return East hexadecimal format."""
        return repr(self)


# =============================================================================
# EastArray - Ordered collection
# =============================================================================


def _make_east_key(element_type: EastType) -> Any:
    """Create a key function for East ordering (lazy import to avoid cycles)."""
    from east.utils.ordering import make_east_key

    return make_east_key(element_type)


class EastArray(list, Generic[T]):
    """East array with element type tracking.

    Arrays are mutable, ordered, 0-indexed collections.
    They behave like Python lists but track the element type.

    Generic type parameter T is for static type hints only (e.g., EastArray[float]).
    At runtime, element_type provides the actual East type.
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

    def insert(self, index: SupportsIndex, item: Any) -> None:
        """Insert item at index."""
        self._check_not_iterating()
        super().insert(index, item)

    def remove(self, item: Any) -> None:
        """Remove first occurrence of item."""
        self._check_not_iterating()
        super().remove(item)

    def pop(self, index: SupportsIndex = -1) -> Any:
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


# =============================================================================
# EastSet - Sorted unique collection
# =============================================================================


class EastSet(Generic[T]):
    """East set with element type tracking.

    Sets are mutable, sorted collections of unique elements.
    Elements are sorted using East's total ordering.

    Generic type parameter T is for static type hints only (e.g., EastSet[str]).
    At runtime, element_type provides the actual East type.
    """

    def __init__(self, element_type: EastType, items: Iterable[Any] | None = None):
        """Create a set with a specific element type.

        Args:
            element_type: The type of elements in this set
            items: Initial items (optional)
        """
        self.element_type = element_type
        self._data: SortedSet = SortedSet(key=_make_east_key(element_type))
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
        """Add an item to the set."""
        self._check_not_iterating()
        self._data.add(item)

    def remove(self, item: Any) -> None:
        """Remove an item from the set."""
        self._check_not_iterating()
        self._data.remove(item)

    def discard(self, item: Any) -> None:
        """Remove an item from the set if present."""
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


# =============================================================================
# EastDict - Sorted key-value collection
# =============================================================================


K = TypeVar("K")


class EastDict(Generic[K, V]):
    """East dict with key and value type tracking.

    Dicts are mutable, sorted collections of key-value pairs.
    Keys are sorted using East's total ordering.

    Generic type parameters K and V are for static type hints only
    (e.g., EastDict[str, int]). At runtime, key_type and value_type
    provide the actual East types.
    """

    def __init__(
        self,
        key_type: EastType,
        value_type: EastType,
        items: dict | None = None,
    ):
        """Create a dict with specific key and value types."""
        self.key_type = key_type
        self.value_type = value_type
        self._data: SortedDict = SortedDict(_make_east_key(key_type))
        self._iteration_lock = 0
        if items is not None:
            for key, value in items.items():
                self._data[key] = value

    def _lock_for_iteration(self) -> None:
        """Lock dict for iteration."""
        self._iteration_lock += 1

    def _unlock_for_iteration(self) -> None:
        """Unlock dict after iteration."""
        self._iteration_lock -= 1

    def _check_not_iterating(self) -> None:
        """Check if dict is being iterated."""
        if self._iteration_lock > 0:
            raise RuntimeError("Cannot modify Dict during iteration")

    def __getitem__(self, key: Any) -> Any:
        """Get value for key."""
        return self._data[key]

    def __setitem__(self, key: Any, value: Any) -> None:
        """Set value for key."""
        self._check_not_iterating()
        self._data[key] = value

    def __delitem__(self, key: Any) -> None:
        """Delete key from dict."""
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
        """Get iterator over values."""
        return iter(self._data.values())

    def items(self) -> Iterator[Any]:
        """Return iterator over (key, value) pairs in sorted order."""
        return iter(self._data.items())

    def get(self, key: Any, default: Any = None) -> Any:
        """Get value for key, returning default if not found."""
        return self._data.get(key, default)

    def pop(self, key: Any, *args: Any) -> Any:
        """Remove and return value for key."""
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


# =============================================================================
# EastStruct - Immutable record type
# =============================================================================


class EastStruct(dict, Generic[T]):
    """Hashable, immutable struct wrapper.

    Wraps a plain dict to make it hashable for use in Sets and Dicts.
    Behaves like a dict but implements __hash__() for hashability.

    Generic type parameter T should be a TypedDict describing the structure.
    """

    def __init__(self, data: dict[str, Any]):
        """Create an immutable struct from a dict."""
        super().__init__(data)
        self._hash: int | None = None

    def __hash__(self) -> int:  # type: ignore[override]
        """Compute hash based on sorted field items."""
        if self._hash is None:
            items = []
            for k in sorted(self.keys()):
                v = self[k]
                try:
                    items.append((k, hash(v)))
                except TypeError:
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


# =============================================================================
# EastVariant - Tagged union type (memory-optimized with __slots__)
# =============================================================================


class EastVariant(Generic[V]):
    """Hashable, immutable variant wrapper.

    Represents a tagged union with "type" and "value" fields.
    Uses __slots__ for memory efficiency.
    Provides dict-like access
    """

    __slots__ = ("_tag", "_value", "_hash")

    def __init__(self, tag: str, value: Any):
        """Create an immutable variant."""
        self._tag = tag
        self._value = value
        self._hash: int | None = None

    @property
    def type(self) -> str:
        """Get the variant's type (case name)."""
        return self._tag

    @property
    def value(self) -> Any:
        """Get the variant's value."""
        return self._value

    def __hash__(self) -> int:
        """Compute hash based on type and value."""
        if self._hash is None:
            try:
                self._hash = hash((self._tag, self._value))
            except TypeError:
                self._hash = hash((self._tag, id(self._value)))
        return self._hash

    def __eq__(self, other: object) -> bool:
        """Check equality with another variant."""
        if isinstance(other, EastVariant):
            return self._tag == other._tag and self._value == other._value
        if isinstance(other, dict):
            return (
                other.get("type") == self._tag
                and other.get("value") == self._value
                and len(other) == 2
            )
        return NotImplemented

    # Dict-like access for backward compatibility
    def __getitem__(self, key: str) -> Any:
        """Get value by key (dict-like access)."""
        if key == "type":
            return self._tag
        if key == "value":
            return self._value
        raise KeyError(key)

    def __contains__(self, key: object) -> bool:
        """Check if key exists."""
        return key in ("type", "value")

    def __len__(self) -> int:
        """Return number of fields (always 2)."""
        return 2

    def __iter__(self) -> Iterator[str]:
        """Iterate over keys."""
        return iter(("type", "value"))

    def keys(self) -> tuple[str, str]:
        """Return keys."""
        return ("type", "value")

    def values(self) -> tuple[Any, Any]:
        """Return values."""
        return (self._tag, self._value)

    def items(self) -> tuple[tuple[str, str], tuple[str, Any]]:
        """Return items as tuples."""
        return (("type", self._tag), ("value", self._value))

    def get(self, key: str, default: Any = None) -> Any:
        """Get value by key with default."""
        if key == "type":
            return self._tag
        if key == "value":
            return self._value
        return default

    def __repr__(self) -> str:
        """Return variant representation."""
        return f"EastVariant(type={self._tag!r}, value={self._value!r})"


# =============================================================================
# EastOption - Option variant (some/none)
# =============================================================================


class EastOption(EastVariant, Generic[OptionT]):
    """Option variant - either some(OptionT) or none.

    A specialized variant for optional values. The type parameter OptionT
    represents the inner type when the option is "some".
    """

    __slots__ = ()  # No additional slots needed

    def __init__(self, tag: str, value: OptionT | None):
        """Create an Option variant."""
        if tag not in ("some", "none"):
            raise ValueError(f"EastOption tag must be 'some' or 'none', got '{tag}'")
        super().__init__(tag, value)

    def __hash__(self) -> int:
        """Inherit hash from EastVariant."""
        return super().__hash__()

    def __repr__(self) -> str:
        """Return option representation."""
        return f"EastOption(type={self._tag!r}, value={self._value!r})"


def EastSome(value: OptionT) -> EastVariant[OptionT]:
    """Create a 'some' variant for optional values.

    Args:
        value: The value to wrap

    Returns:
        EastVariant with type="some"
    """
    return EastVariant("some", value)


# Singleton for 'none' variant - reuse same instance to save memory
_east_none_singleton: EastVariant[None] | None = None


def EastNone() -> EastVariant[None]:
    """Create a 'none' variant for optional values.

    Returns the same singleton instance for memory efficiency.

    Returns:
        EastVariant with type="none" and value=east_null
    """
    global _east_none_singleton
    if _east_none_singleton is None:
        _east_none_singleton = EastVariant("none", east_null)
    return _east_none_singleton


# Pre-create the singleton at module load time
_east_none_singleton = EastVariant("none", east_null)


# =============================================================================
# EastRef - Mutable reference cell
# =============================================================================

# Symbol for nominal typing (brand)
REF_SYMBOL = object()


class EastRef(Generic[T]):
    """Mutable reference cell containing a value.

    EastRef-cells are mutable containers with identity semantics:
    - Two refs are equal only if they're the same object
    - Refs support aliasing in serialization
    - Refs are invariant in the type system
    """

    __slots__ = ("value", "_brand")

    def __init__(self, value: T):
        """Create a new east_ref-cell."""
        self.value: T = value
        self._brand = REF_SYMBOL

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"EastRef({self.value!r})"

    def __str__(self) -> str:
        """Human-readable string."""
        return f"&{self.value}"


def east_ref(value: T) -> EastRef[T]:
    """Create a new mutable reference cell."""
    return EastRef(value)


def is_east_ref(v: Any) -> TypeGuard[EastRef]:
    """Check if a value is a ref-cell."""
    return isinstance(v, EastRef) and hasattr(v, "_brand") and v._brand is REF_SYMBOL


def deref(r: EastRef[T]) -> T:
    """Retrieve the current value from a east_ref-cell."""
    return r.value


def set_ref(r: EastRef[T], value: T) -> None:
    """Update the value stored in a east_ref-cell."""
    r.value = value


# =============================================================================
# DateTime helpers
# =============================================================================


def ensure_utc_datetime(dt: datetime) -> datetime:
    """Ensure datetime is UTC-aware."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    if dt.tzinfo != UTC:
        return dt.astimezone(UTC)
    return dt


# =============================================================================
# EastValue - Union of all East value types
# =============================================================================

# Union of all East value types (for type annotations)
EastValue = (
    EastNull
    | bool
    | int
    | float
    | str
    | EastBlob
    | datetime
    | EastArray
    | EastSet
    | EastDict
    | EastStruct
    | EastVariant
    | EastOption
    | EastRef
)


# =============================================================================
# TypeGuard functions for East value types
# =============================================================================


def is_east_null(v: Any) -> TypeGuard[EastNull]:
    """Check if a value is EastNull."""
    return isinstance(v, EastNull)


def is_east_blob(v: Any) -> TypeGuard[EastBlob]:
    """Check if a value is an EastBlob."""
    return isinstance(v, EastBlob)


def is_east_array(v: Any) -> TypeGuard[EastArray]:
    """Check if a value is an EastArray."""
    return isinstance(v, EastArray)


def is_east_set(v: Any) -> TypeGuard[EastSet]:
    """Check if a value is an EastSet."""
    return isinstance(v, EastSet)


def is_east_dict(v: Any) -> TypeGuard[EastDict]:
    """Check if a value is an EastDict."""
    return isinstance(v, EastDict)


def is_east_struct(v: Any) -> TypeGuard[EastStruct]:
    """Check if a value is an EastStruct (plain dict without 'type' key)."""
    return isinstance(v, dict) and "type" not in v


def is_east_variant(v: Any) -> TypeGuard[EastVariant]:
    """Check if a value is an EastVariant."""
    # Primary check: is it an EastVariant instance?
    if isinstance(v, EastVariant):
        return True
    # Backward compatibility: dict with 'type' and 'value' keys
    return isinstance(v, dict) and "type" in v and "value" in v and len(v) == 2


def is_east_option(v: Any) -> TypeGuard[EastOption]:
    """Check if a value is an EastOption (variant with 'some' or 'none' tag)."""
    return isinstance(v, EastOption)


# =============================================================================
# Type checking and inference
# =============================================================================


def is_value_of(
    value: EastValue,
    typ: EastType,
    type_ctx: list[EastType] | None = None,
    nodes_visited: set[int] | None = None,
) -> bool:
    """Check if a value conforms to an East type.

    Args:
        value: The value to check
        typ: The East type to validate against
        type_ctx: Internal parameter for resolving recursive type references
        nodes_visited: Internal parameter for cycle detection in values

    Returns:
        True if value matches type, False otherwise
    """
    # Initialize type context if needed
    if type_ctx is None:
        type_ctx = []

    # Handle Never type
    if typ["type"] == "Never":
        return False

    # Handle primitive types
    if typ["type"] == "Null":
        return value is None or isinstance(value, EastNull)
    if typ["type"] == "Boolean":
        return isinstance(value, bool)
    if typ["type"] == "Integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if typ["type"] == "Float":
        return isinstance(value, float)
    if typ["type"] == "String":
        return isinstance(value, str)
    if typ["type"] == "DateTime":
        return isinstance(value, datetime)
    if typ["type"] == "Blob":
        return isinstance(value, (bytes, bytearray, EastBlob))

    # Handle EastRef type
    if typ["type"] == "Ref":
        if not isinstance(value, EastRef):
            return False
        # Push current type onto context for recursive references
        type_ctx.append(typ)
        try:
            return is_value_of(value.value, typ["value"], type_ctx, nodes_visited)  # type: ignore[typeddict-item]
        finally:
            type_ctx.pop()

    # Handle Array type
    if typ["type"] == "Array":
        if not isinstance(value, EastArray):
            return False
        # Push current type onto context for recursive references
        type_ctx.append(typ)
        try:
            for elem in value:
                if not is_value_of(elem, typ["value"], type_ctx, nodes_visited):  # type: ignore[typeddict-item]
                    return False
            return True
        finally:
            type_ctx.pop()

    # Handle Set type
    if typ["type"] == "Set":
        if not isinstance(value, EastSet):
            return False
        # Push current type onto context for recursive references
        type_ctx.append(typ)
        try:
            for elem in value:
                if not is_value_of(elem, typ["value"], type_ctx, nodes_visited):  # type: ignore[typeddict-item]
                    return False
            return True
        finally:
            type_ctx.pop()

    # Handle Dict type
    if typ["type"] == "Dict":
        if not isinstance(value, EastDict):
            return False
        dict_type = typ["value"]
        # Push current type onto context for recursive references
        type_ctx.append(typ)
        try:
            for k, v in value.items():
                if not is_value_of(k, dict_type["key"], type_ctx, nodes_visited):
                    return False
                if not is_value_of(v, dict_type["value"], type_ctx, nodes_visited):
                    return False
            return True
        finally:
            type_ctx.pop()

    # Handle Struct type
    if typ["type"] == "Struct":
        if not is_east_struct(value):
            return False
        # Check fields match
        value_fields = list(value.items())
        type_fields = typ["value"]
        if len(value_fields) != len(type_fields):
            return False
        # Push current type onto context for recursive references
        type_ctx.append(typ)
        try:
            for i, field_def in enumerate(type_fields):
                field_name = field_def["name"]
                field_type = field_def["type"]
                if i >= len(value_fields):
                    return False
                val_name, val_value = value_fields[i]
                if val_name != field_name:
                    return False
                if not is_value_of(val_value, field_type, type_ctx, nodes_visited):
                    return False
            return True
        finally:
            type_ctx.pop()

    # Handle Variant type
    if typ["type"] == "Variant":
        if not is_east_variant(value):
            return False
        variant_tag = value.type
        variant_value = value.value
        # Find the case type
        cases = typ["value"]
        # Push current type onto context for recursive references
        type_ctx.append(typ)
        try:
            for case in cases:
                if case["name"] == variant_tag:
                    return is_value_of(variant_value, case["type"], type_ctx, nodes_visited)
            return False  # Case not found
        finally:
            type_ctx.pop()

    # Handle Recursive type
    if typ["type"] == "Recursive":
        scope_id = typ["value"]
        if not isinstance(scope_id, int):
            raise ValueError(f"Recursive type must have integer scope_id, got {type(scope_id)}")

        # Resolve the scope_id to the actual type from the context stack
        stack_index = len(type_ctx) - scope_id
        if stack_index < 0 or stack_index >= len(type_ctx):
            raise ValueError(
                f"Invalid recursive scope_id {scope_id} (type_ctx len={len(type_ctx)}, calculated index={stack_index})"
            )

        resolved_type = type_ctx[stack_index]

        # Check for value cycles to avoid infinite recursion
        value_id = id(value)
        if nodes_visited is None:
            nodes_visited = set()
        if value_id in nodes_visited:
            return True  # Already validated this object
        nodes_visited.add(value_id)

        return is_value_of(value, resolved_type, type_ctx, nodes_visited)

    # Handle Function type
    if typ["type"] == "Function":
        raise TypeError("JavaScript/Python functions cannot be converted to East functions")

    # Unknown type
    raise NotImplementedError(f"is_value_of not implemented for type: {typ}")


def type_of(value: EastValue) -> EastType:
    """Infer the East type of a Python value.

    Args:
        value: Python value

    Returns:
        East type

    Raises:
        TypeError: If value type cannot be inferred
    """
    # Lazy imports to avoid circular dependencies
    from east.types.types import (
        ArrayType,
        BlobType,
        BooleanType,
        DateTimeType,
        DictType,
        FloatType,
        IntegerType,
        NullType,
        RefType,
        SetType,
        StringType,
        StructType,
        VariantType,
    )

    if value is None or isinstance(value, EastNull):
        return NullType
    if isinstance(value, bool):
        return BooleanType
    if isinstance(value, int):
        return IntegerType
    if isinstance(value, float):
        return FloatType
    if isinstance(value, str):
        return StringType
    if isinstance(value, bytes):
        return BlobType
    if isinstance(value, datetime):
        return DateTimeType
    if isinstance(value, EastArray):
        return ArrayType(value.element_type)
    if isinstance(value, EastSet):
        return SetType(value.element_type)
    if isinstance(value, EastDict):
        return DictType(value.key_type, value.value_type)
    if isinstance(value, EastRef):
        # EastRef doesn't store type info at runtime - infer from contained value
        return RefType(type_of(value.value))
    if isinstance(value, dict):
        # Check if it's a variant value
        if "type" in value and "value" in value and len(value) == 2:
            # It's a variant - but we don't know the full variant type
            # Return a generic variant with just this case
            case_value_type = type_of(value["value"])
            return VariantType([(value["type"], case_value_type)])
        # It's a struct value
        field_types_list = []
        for key, val in value.items():
            field_types_list.append((key, type_of(val)))
        return StructType(field_types_list)
    if callable(value):
        # Can't infer function types from Python callables
        raise TypeError(f"Cannot infer type of callable {value}")

    raise TypeError(f"Cannot infer type of {type(value).__name__}")


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # EastValue union type
    "EastValue",
    # EastNull
    "EastNull",
    "east_null",
    # EastBlob
    "EastBlob",
    # Containers
    "EastArray",
    "EastSet",
    "EastDict",
    # Structural
    "EastStruct",
    "EastVariant",
    "EastOption",
    "EastSome",
    "EastNone",
    # EastRef
    "EastRef",
    "REF_SYMBOL",
    "east_ref",
    "is_east_ref",
    "deref",
    "set_ref",
    # DateTime
    "ensure_utc_datetime",
    # TypeGuard functions
    "is_east_null",
    "is_east_blob",
    "is_east_array",
    "is_east_set",
    "is_east_dict",
    "is_east_struct",
    "is_east_variant",
    "is_east_option",
    # Type checking and inference
    "is_value_of",
    "type_of",
]
