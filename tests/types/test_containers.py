"""Tests for East container types."""

import pytest

from east.types.containers import EastArray, EastDict, EastSet
from east.types.type_system import IntegerType, StringType


class TestEastArray:
    """Tests for EastArray."""

    def test_create_empty(self):
        """Create empty array."""
        arr = EastArray(IntegerType, [])
        assert len(arr) == 0
        assert arr.element_type == IntegerType

    def test_create_with_items(self):
        """Create array with initial items."""
        arr = EastArray(IntegerType, [1, 2, 3])
        assert len(arr) == 3
        assert arr[0] == 1
        assert arr[1] == 2
        assert arr[2] == 3

    def test_indexing(self):
        """Index into array."""
        arr = EastArray(StringType, ["a", "b", "c"])
        assert arr[0] == "a"
        assert arr[-1] == "c"

    def test_mutation(self):
        """Arrays are mutable."""
        arr = EastArray(IntegerType, [1, 2, 3])
        arr.append(4)
        assert len(arr) == 4
        assert arr[3] == 4

        arr[0] = 10
        assert arr[0] == 10

        arr.pop()
        assert len(arr) == 3

    def test_iteration(self):
        """Iterate over array."""
        arr = EastArray(IntegerType, [1, 2, 3])
        items = list(arr)
        assert items == [1, 2, 3]

    def test_repr(self):
        """Array repr in East format."""
        arr = EastArray(IntegerType, [])
        assert repr(arr) == "[]"

        arr = EastArray(IntegerType, [1, 2, 3])
        assert repr(arr) == "[1, 2, 3]"

    def test_list_methods(self):
        """Array inherits from list."""
        arr = EastArray(IntegerType, [3, 1, 2])
        arr.sort()
        assert arr == [1, 2, 3]

        arr.reverse()
        assert arr == [3, 2, 1]

        arr.extend([4, 5])
        assert arr == [3, 2, 1, 4, 5]


class TestEastSet:
    """Tests for EastSet."""

    def test_create_empty(self):
        """Create empty set."""
        s = EastSet(IntegerType, None)
        assert len(s) == 0
        assert s.element_type == IntegerType

    def test_create_with_items(self):
        """Create set with initial items."""
        s = EastSet(IntegerType, [3, 1, 2, 1])  # Duplicate 1
        assert len(s) == 3  # No duplicates
        # Items are sorted
        assert list(s) == [1, 2, 3]

    def test_add(self):
        """Add items to set."""
        s = EastSet(IntegerType)
        s.add(3)
        s.add(1)
        s.add(2)
        assert len(s) == 3
        assert list(s) == [1, 2, 3]  # Sorted

    def test_add_duplicate(self):
        """Adding duplicate has no effect."""
        s = EastSet(IntegerType, [1, 2, 3])
        s.add(2)
        assert len(s) == 3
        assert list(s) == [1, 2, 3]

    def test_remove(self):
        """Remove item from set."""
        s = EastSet(IntegerType, [1, 2, 3])
        s.remove(2)
        assert len(s) == 2
        assert list(s) == [1, 3]

    def test_remove_missing(self):
        """Removing missing item raises KeyError."""
        s = EastSet(IntegerType, [1, 2, 3])
        with pytest.raises(KeyError):
            s.remove(10)

    def test_discard(self):
        """Discard removes item if present."""
        s = EastSet(IntegerType, [1, 2, 3])
        s.discard(2)
        assert len(s) == 2

        # Discarding missing item doesn't raise
        s.discard(10)
        assert len(s) == 2

    def test_contains(self):
        """Check membership."""
        s = EastSet(IntegerType, [1, 2, 3])
        assert 1 in s
        assert 2 in s
        assert 10 not in s

    def test_clear(self):
        """Clear all items."""
        s = EastSet(IntegerType, [1, 2, 3])
        s.clear()
        assert len(s) == 0

    def test_iteration(self):
        """Iterate in sorted order."""
        s = EastSet(IntegerType, [3, 1, 2])
        items = list(s)
        assert items == [1, 2, 3]

    def test_equality(self):
        """Sets are equal if they have same elements."""
        s1 = EastSet(IntegerType, [1, 2, 3])
        s2 = EastSet(IntegerType, [3, 2, 1])  # Different order
        s3 = EastSet(IntegerType, [1, 2])
        assert s1 == s2
        assert s1 != s3

    def test_repr_empty(self):
        """Empty set repr."""
        s = EastSet(IntegerType)
        assert repr(s) == "{}"

    def test_repr_with_items(self):
        """Set repr in East format."""
        s = EastSet(IntegerType, [3, 1, 2])
        assert repr(s) == "{1, 2, 3}"

    def test_ordering(self):
        """Sets maintain sorted order."""
        s = EastSet(StringType)
        s.add("zebra")
        s.add("apple")
        s.add("banana")
        assert list(s) == ["apple", "banana", "zebra"]


class TestEastDict:
    """Tests for EastDict."""

    def test_create_empty(self):
        """Create empty dict."""
        d = EastDict(StringType, IntegerType, None)
        assert len(d) == 0
        assert d.key_type == StringType
        assert d.value_type == IntegerType

    def test_create_with_items(self):
        """Create dict with initial items."""
        d = EastDict(StringType, IntegerType, {"a": 1, "b": 2})
        assert len(d) == 2
        assert d["a"] == 1
        assert d["b"] == 2

    def test_setitem_getitem(self):
        """Set and get items."""
        d = EastDict(StringType, IntegerType)
        d["x"] = 10
        d["y"] = 20
        assert d["x"] == 10
        assert d["y"] == 20

    def test_getitem_missing(self):
        """Getting missing key raises KeyError."""
        d = EastDict(StringType, IntegerType, {"a": 1})
        with pytest.raises(KeyError):
            _ = d["missing"]

    def test_delitem(self):
        """Delete key."""
        d = EastDict(StringType, IntegerType, {"a": 1, "b": 2})
        del d["a"]
        assert len(d) == 1
        assert "a" not in d
        assert "b" in d

    def test_contains(self):
        """Check key membership."""
        d = EastDict(StringType, IntegerType, {"a": 1, "b": 2})
        assert "a" in d
        assert "b" in d
        assert "c" not in d

    def test_keys(self):
        """Get keys in sorted order."""
        d = EastDict(StringType, IntegerType, {"c": 3, "a": 1, "b": 2})
        keys = list(d.keys())
        assert keys == ["a", "b", "c"]

    def test_values(self):
        """Get values in key sort order."""
        d = EastDict(StringType, IntegerType, {"c": 3, "a": 1, "b": 2})
        values = list(d.values())
        assert values == [1, 2, 3]  # Values for a, b, c

    def test_items(self):
        """Get items in key sort order."""
        d = EastDict(StringType, IntegerType, {"c": 3, "a": 1, "b": 2})
        items = list(d.items())
        assert items == [("a", 1), ("b", 2), ("c", 3)]

    def test_iteration(self):
        """Iterating dict yields keys."""
        d = EastDict(StringType, IntegerType, {"c": 3, "a": 1, "b": 2})
        keys = list(d)
        assert keys == ["a", "b", "c"]

    def test_get(self):
        """Get with default."""
        d = EastDict(StringType, IntegerType, {"a": 1})
        assert d.get("a") == 1
        assert d.get("b") is None
        assert d.get("b", 999) == 999

    def test_pop(self):
        """Pop key."""
        d = EastDict(StringType, IntegerType, {"a": 1, "b": 2})
        value = d.pop("a")
        assert value == 1
        assert "a" not in d
        assert len(d) == 1

    def test_pop_missing(self):
        """Pop missing key with default."""
        d = EastDict(StringType, IntegerType, {"a": 1})
        with pytest.raises(KeyError):
            d.pop("missing")

        value = d.pop("missing", 999)
        assert value == 999

    def test_clear(self):
        """Clear all items."""
        d = EastDict(StringType, IntegerType, {"a": 1, "b": 2})
        d.clear()
        assert len(d) == 0

    def test_equality(self):
        """Dicts are equal if they have same key-value pairs."""
        d1 = EastDict(StringType, IntegerType, {"a": 1, "b": 2})
        d2 = EastDict(StringType, IntegerType, {"b": 2, "a": 1})  # Different order
        d3 = EastDict(StringType, IntegerType, {"a": 1})
        assert d1 == d2
        assert d1 != d3

    def test_repr_empty(self):
        """Empty dict repr."""
        d = EastDict(StringType, IntegerType)
        assert repr(d) == "{:}"

    def test_repr_with_items(self):
        """Dict repr in East format."""
        d = EastDict(StringType, IntegerType, {"c": 3, "a": 1, "b": 2})
        assert repr(d) == "{'a': 1, 'b': 2, 'c': 3}"


class TestContainerOrdering:
    """Tests for container ordering using East total ordering."""

    def test_set_with_mixed_integers(self):
        """Set sorts integers correctly."""
        s = EastSet(IntegerType, [100, 2, 30, 1])
        assert list(s) == [1, 2, 30, 100]

    def test_dict_with_string_keys(self):
        """Dict sorts string keys correctly."""
        d = EastDict(StringType, IntegerType, {"zebra": 1, "apple": 2, "banana": 3})
        assert list(d.keys()) == ["apple", "banana", "zebra"]

    def test_set_updates_maintain_order(self):
        """Adding items maintains sorted order."""
        s = EastSet(IntegerType)
        s.add(5)
        s.add(1)
        s.add(10)
        s.add(3)
        assert list(s) == [1, 3, 5, 10]

    def test_dict_updates_maintain_order(self):
        """Adding keys maintains sorted order."""
        d = EastDict(StringType, IntegerType)
        d["zebra"] = 1
        d["apple"] = 2
        d["banana"] = 3
        assert list(d.keys()) == ["apple", "banana", "zebra"]
