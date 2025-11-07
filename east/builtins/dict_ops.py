"""Dict builtin functions."""

from typing import Any

from east.builtins.registry import register_builtin
from east.types.containers import EastArray, EastDict


def dict_size(d: EastDict) -> int:
    """Get size of dict.

    Args:
        d: Dict

    Returns:
        Number of entries in dict
    """
    return len(d)


def dict_has(d: EastDict, key: Any) -> bool:
    """Check if dict has key.

    Args:
        d: Dict
        key: Key to check

    Returns:
        True if key is in dict
    """
    return key in d


def dict_get(d: EastDict, key: Any) -> Any:
    """Get value for key.

    Args:
        d: Dict
        key: Key to look up

    Returns:
        Value for key

    Raises:
        KeyError: If key not in dict
    """
    return d[key]


def dict_set(d: EastDict, key: Any, value: Any) -> None:
    """Set value for key (mutation).

    Args:
        d: Dict
        key: Key to set
        value: Value to set
    """
    d[key] = value


def dict_remove(d: EastDict, key: Any) -> None:
    """Remove key from dict (mutation).

    Args:
        d: Dict
        key: Key to remove

    Raises:
        KeyError: If key not in dict
    """
    del d[key]


def dict_clear(d: EastDict) -> None:
    """Remove all entries from dict (mutation).

    Args:
        d: Dict
    """
    d.clear()


def dict_keys(d: EastDict) -> EastArray:
    """Get array of dict keys.

    Args:
        d: Dict

    Returns:
        Array of keys (sorted)
    """
    return EastArray(d.key_type, list(d.keys()))


def dict_values(d: EastDict) -> EastArray:
    """Get array of dict values.

    Args:
        d: Dict

    Returns:
        Array of values (in key order)
    """
    return EastArray(d.value_type, list(d.values()))


def dict_entries(d: EastDict) -> EastArray:
    """Get array of dict entries as [key, value] arrays.

    Args:
        d: Dict

    Returns:
        Array of [key, value] pairs
    """
    from east.types.type_system import ArrayType

    # Create array of [key, value] pairs
    pair_type = ArrayType(d.key_type)  # Simplified: just use key type
    entries = []
    for k, v in d.items():
        # Each entry is a simple list [k, v]
        entries.append([k, v])
    return EastArray(pair_type, entries)


def dict_merge(a: EastDict, b: EastDict) -> EastDict:
    """Merge two dicts (b overwrites a for duplicate keys).

    Args:
        a: First dict
        b: Second dict

    Returns:
        New dict with merged entries
    """
    merged = dict(a.items())
    merged.update(b.items())
    return EastDict(a.key_type, a.value_type, merged)


def dict_get_or_default(d: EastDict, key: Any, default: Any) -> Any:
    """Get value for key or return default if key not in dict.

    Args:
        d: Dict
        key: Key to look up
        default: Default value if key not in dict

    Returns:
        Value for key, or default if key not in dict
    """
    return d.get(key, default)


def dict_copy(d: EastDict) -> EastDict:
    """Create shallow copy of dict.

    Args:
        d: Dict

    Returns:
        New dict with same entries
    """
    return EastDict(d.key_type, d.value_type, dict(d.items()))


def dict_update(d: EastDict, key: Any, value: Any) -> None:
    """Update value for existing key (mutation).

    Args:
        d: Dict
        key: Key to update
        value: New value

    Raises:
        KeyError: If key not in dict
    """
    if key not in d:
        raise KeyError(f"Key not in dict: {key}")
    d[key] = value


# Register all dict builtins
register_builtin("DictSize", dict_size)
register_builtin("DictHas", dict_has)
register_builtin("DictGet", dict_get)
register_builtin("DictGetOrDefault", dict_get_or_default)
register_builtin("DictInsert", dict_set)  # Renamed from DictSet
register_builtin("DictUpdate", dict_update)
register_builtin("DictDelete", dict_remove)  # Renamed from DictRemove
register_builtin("DictClear", dict_clear)
register_builtin("DictKeys", dict_keys)
# Note: DictValues and DictEntries not in spec but kept for convenience
register_builtin("DictValues", dict_values)
register_builtin("DictEntries", dict_entries)
register_builtin("DictMerge", dict_merge)
register_builtin("DictCopy", dict_copy)


__all__ = [
    "dict_size",
    "dict_has",
    "dict_get",
    "dict_get_or_default",
    "dict_set",
    "dict_update",
    "dict_remove",
    "dict_clear",
    "dict_keys",
    "dict_values",
    "dict_entries",
    "dict_merge",
    "dict_copy",
]
