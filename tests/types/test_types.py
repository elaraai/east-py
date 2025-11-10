"""Consolidated type system tests."""

from datetime import UTC, datetime, timedelta, timezone

import pytest

from east.serialization.east_printer import print_identifier, print_type
from east.types.containers import EastArray, EastDict, EastSet
from east.types.primitives import Blob, Null, ensure_utc_datetime, null, validate_east_value
from east.types.structural import Case, EastStruct, EastVariant, make_case
from east.types.type_system import (
    ArrayType,
    BlobType,
    BooleanType,
    DateTimeType,
    DictType,
    EastType,
    EastTypeType,
    FloatType,
    FunctionType,
    IntegerType,
    NeverType,
    NullType,
    OptionType,
    RecursiveTypeMarker,
    SetType,
    SomeType,
    StringType,
    StructType,
    TypeMismatchError,
    VariantType,
    _StructTypeClass,
    _VariantTypeClass,
    east_type_of,
    is_data_type,
    is_immutable_type,
    is_subtype,
    is_type_equal,
    is_value_of,
    recursive_type,
    type_equal,
    type_intersect,
    type_of,
    type_union,
)


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


"""Tests for East primitive types."""


class TestNull:
    """Tests for the Null type."""

    def test_singleton(self):
        """Null is a singleton."""
        n1 = Null()
        n2 = Null()
        assert n1 is n2
        assert n1 is null

    def test_repr(self):
        """Null repr is 'null'."""
        assert repr(null) == "null"
        assert str(null) == "null"

    def test_equality(self):
        """Null equals only itself."""
        assert null == null
        assert null == Null()
        assert null != None  # noqa: E711
        assert null != 0
        assert null != False  # noqa: E712

    def test_hash(self):
        """Null is hashable."""
        assert hash(null) == hash(None)
        # Can be used in sets/dicts
        s = {null}
        assert null in s

    def test_ordering(self):
        """Null is not less than itself."""
        assert not (null < null)
        assert null <= null
        assert not (null > null)
        assert null >= null


class TestBlob:
    """Tests for the Blob type."""

    def test_from_bytes(self):
        """Create Blob from bytes."""
        b = Blob(b"hello")
        assert len(b) == 5
        assert b[0] == ord("h")

    def test_from_list(self):
        """Create Blob from list of integers."""
        b = Blob([0x00, 0xFF, 0xAA])
        assert len(b) == 3
        assert b[0] == 0x00
        assert b[1] == 0xFF
        assert b[2] == 0xAA

    def test_from_blob(self):
        """Create Blob from another Blob."""
        b1 = Blob(b"test")
        b2 = Blob(b1)
        assert b1 == b2
        assert b1 is not b2  # Different objects
        assert b1.data is b2.data  # But share underlying bytes

    def test_empty(self):
        """Empty Blob."""
        b = Blob(b"")
        assert len(b) == 0
        assert repr(b) == "0x"

    def test_indexing(self):
        """Index into Blob."""
        b = Blob(b"abc")
        assert b[0] == ord("a")
        assert b[1] == ord("b")
        assert b[2] == ord("c")
        assert b[-1] == ord("c")

    def test_slicing(self):
        """Slice Blob returns new Blob."""
        b = Blob(b"hello")
        b2 = b[1:4]
        assert isinstance(b2, Blob)
        assert len(b2) == 3
        assert b2[0] == ord("e")

    def test_immutable(self):
        """Blob is immutable."""
        b = Blob(b"test")
        with pytest.raises((TypeError, AttributeError)):
            b[0] = 65  # type: ignore

    def test_equality(self):
        """Blobs with same bytes are equal."""
        b1 = Blob(b"test")
        b2 = Blob(b"test")
        b3 = Blob(b"different")
        assert b1 == b2
        assert b1 != b3
        assert b1 != "test"

    def test_ordering(self):
        """Blobs have lexicographic ordering."""
        b1 = Blob(b"aaa")
        b2 = Blob(b"aab")
        b3 = Blob(b"aba")
        assert b1 < b2 < b3
        assert not (b2 < b1)

    def test_hash(self):
        """Blobs are hashable."""
        b1 = Blob(b"test")
        b2 = Blob(b"test")
        assert hash(b1) == hash(b2)
        # Can be used in sets
        s = {b1, b2}
        assert len(s) == 1

    def test_repr_short(self):
        """Short blob repr in hex."""
        b = Blob(b"\x00\xff\xaa")
        assert repr(b) == "0x00ffaa"

    def test_repr_long(self):
        """Long blob repr is truncated."""
        b = Blob(bytes(range(256)) + b"\xff")
        r = repr(b)
        assert r.startswith("0x")
        assert r.endswith("...")
        assert len(r) < 600  # Truncated


class TestDateTime:
    """Tests for DateTime handling."""

    def test_ensure_utc_naive(self):
        """Naive datetime assumed to be UTC."""
        dt = datetime(2025, 1, 1, 12, 0, 0)
        result = ensure_utc_datetime(dt)
        assert result.tzinfo == UTC
        assert result.year == 2025
        assert result.hour == 12

    def test_ensure_utc_already_utc(self):
        """UTC datetime unchanged."""
        dt = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
        result = ensure_utc_datetime(dt)
        assert result is dt  # Same object

    def test_ensure_utc_conversion(self):
        """Other timezone converted to UTC."""
        # UTC-5 timezone
        tz_minus_5 = timezone(timedelta(hours=-5))
        dt = datetime(2025, 1, 1, 12, 0, 0, tzinfo=tz_minus_5)
        result = ensure_utc_datetime(dt)
        assert result.tzinfo == UTC
        assert result.hour == 17  # 12 + 5 = 17 UTC


class TestValidation:
    """Tests for value validation."""

    def test_validate_null(self):
        """Validate Null type."""
        validate_east_value(null, "Null")
        validate_east_value(Null(), "Null")
        with pytest.raises(TypeError):
            validate_east_value(None, "Null")

    def test_validate_boolean(self):
        """Validate Boolean type."""
        validate_east_value(True, "Boolean")
        validate_east_value(False, "Boolean")
        with pytest.raises(TypeError):
            validate_east_value(1, "Boolean")
        with pytest.raises(TypeError):
            validate_east_value(0, "Boolean")

    def test_validate_integer(self):
        """Validate Integer type."""
        validate_east_value(42, "Integer")
        validate_east_value(-10, "Integer")
        validate_east_value(0, "Integer")
        # Booleans are not integers in East
        with pytest.raises(TypeError):
            validate_east_value(True, "Integer")
        with pytest.raises(TypeError):
            validate_east_value(3.14, "Integer")

    def test_validate_float(self):
        """Validate Float type."""
        validate_east_value(3.14, "Float")
        validate_east_value(0.0, "Float")
        validate_east_value(float("inf"), "Float")
        validate_east_value(float("-inf"), "Float")
        validate_east_value(float("nan"), "Float")
        with pytest.raises(TypeError):
            validate_east_value(42, "Float")

    def test_validate_string(self):
        """Validate String type."""
        validate_east_value("hello", "String")
        validate_east_value("", "String")
        validate_east_value("unicode: \u2603", "String")
        with pytest.raises(TypeError):
            validate_east_value(b"bytes", "String")

    def test_validate_blob(self):
        """Validate Blob type."""
        validate_east_value(Blob(b"test"), "Blob")
        with pytest.raises(TypeError):
            validate_east_value(b"bytes", "Blob")

    def test_validate_datetime(self):
        """Validate DateTime type."""
        validate_east_value(datetime.now(), "DateTime")
        validate_east_value(datetime(2025, 1, 1, tzinfo=UTC), "DateTime")
        with pytest.raises(TypeError):
            validate_east_value("2025-01-01", "DateTime")

    def test_validate_unknown_type(self):
        """Unknown type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown East primitive type"):
            validate_east_value(42, "UnknownType")


class TestPrimitiveTypes:
    """Integration tests for primitive type interactions."""

    def test_bool_is_not_int(self):
        """In East, Boolean is distinct from Integer."""
        # Python's bool is a subclass of int, but East treats them separately
        assert isinstance(True, bool)
        assert isinstance(True, int)  # Python quirk
        # Our validation correctly distinguishes them
        validate_east_value(True, "Boolean")
        with pytest.raises(TypeError):
            validate_east_value(True, "Integer")

    def test_special_floats(self):
        """Special float values work correctly."""
        inf = float("inf")
        neg_inf = float("-inf")
        nan = float("nan")

        # All are valid floats
        validate_east_value(inf, "Float")
        validate_east_value(neg_inf, "Float")
        validate_east_value(nan, "Float")

        # Ordering works
        assert neg_inf < 0.0 < inf
        # NaN is not equal to anything, including itself
        assert nan != nan
        assert not (nan < nan)
        assert not (nan > nan)


"""Tests for East structural types (Struct and Variant)."""


class TestStructType:
    """Tests for StructType."""

    def test_create(self):
        """Create a struct type."""
        st = _StructTypeClass((("name", StringType), ("age", IntegerType)))
        assert len(st.fields) == 2
        assert st.field_names() == ["name", "age"]
        assert st.field_types() == [StringType, IntegerType]

    def test_field_index(self):
        """Get field index by name."""
        st = _StructTypeClass((("name", StringType), ("age", IntegerType)))
        assert st.field_index("name") == 0
        assert st.field_index("age") == 1

    def test_field_index_missing(self):
        """Field index raises KeyError for missing field."""
        st = _StructTypeClass((("name", StringType),))
        with pytest.raises(KeyError):
            st.field_index("missing")

    def test_create_instance(self):
        """Create struct instance."""
        st = _StructTypeClass((("name", StringType), ("age", IntegerType)))
        instance = st.create(name="Alice", age=30)
        assert isinstance(instance, EastStruct)
        assert instance._east_type == st
        assert instance._values == ("Alice", 30)

    def test_create_instance_missing_field(self):
        """Creating instance without all fields raises ValueError."""
        st = _StructTypeClass((("name", StringType), ("age", IntegerType)))
        with pytest.raises(ValueError, match="Expected 2 fields"):
            st.create(name="Alice")

    def test_create_instance_extra_field(self):
        """Creating instance with extra field raises ValueError."""
        st = _StructTypeClass((("name", StringType), ("age", IntegerType)))
        with pytest.raises(ValueError, match="Expected 2 fields"):
            st.create(name="Alice", age=30, extra="value")

    def test_empty_struct_type(self):
        """Create empty struct type."""
        st = _StructTypeClass(())
        assert len(st.fields) == 0
        assert st.field_names() == []


class TestEastStruct:
    """Tests for EastStruct."""

    def test_field_access(self):
        """Access fields by name."""
        st = _StructTypeClass((("name", StringType), ("age", IntegerType)))
        instance = st.create(name="Alice", age=30)
        assert instance.name == "Alice"
        assert instance.age == 30

    def test_field_access_missing(self):
        """Accessing missing field raises AttributeError."""
        st = _StructTypeClass((("name", StringType),))
        instance = st.create(name="Alice")
        with pytest.raises(AttributeError):
            _ = instance.missing

    def test_immutable(self):
        """Structs are immutable."""
        st = _StructTypeClass((("name", StringType), ("age", IntegerType)))
        instance = st.create(name="Alice", age=30)
        with pytest.raises((AttributeError, TypeError)):
            instance.name = "Bob"  # type: ignore

    def test_equality(self):
        """Structural equality."""
        st = _StructTypeClass((("name", StringType), ("age", IntegerType)))
        s1 = st.create(name="Alice", age=30)
        s2 = st.create(name="Alice", age=30)
        s3 = st.create(name="Bob", age=30)
        assert s1 == s2
        assert s1 != s3

    def test_equality_different_types(self):
        """Structs with different types aren't equal."""
        st1 = _StructTypeClass((("name", StringType),))
        st2 = _StructTypeClass((("name", StringType), ("age", IntegerType)))
        s1 = st1.create(name="Alice")
        s2 = st2.create(name="Alice", age=30)
        assert s1 != s2

    def test_ordering(self):
        """Structural ordering."""
        st = _StructTypeClass((("name", StringType), ("age", IntegerType)))
        s1 = st.create(name="Alice", age=25)
        s2 = st.create(name="Alice", age=30)
        s3 = st.create(name="Bob", age=20)
        assert s1 < s2  # Same name, age 25 < 30
        assert s1 < s3  # Alice < Bob

    def test_hash(self):
        """Structs are hashable."""
        st = _StructTypeClass((("name", StringType), ("age", IntegerType)))
        s1 = st.create(name="Alice", age=30)
        s2 = st.create(name="Alice", age=30)
        assert hash(s1) == hash(s2)
        # Can be used in sets
        structs = {s1, s2}
        assert len(structs) == 1

    def test_repr_empty(self):
        """Empty struct repr."""
        st = _StructTypeClass(())
        instance = st.create()
        assert repr(instance) == "()"

    def test_repr(self):
        """Struct repr in East format."""
        st = _StructTypeClass((("name", StringType), ("age", IntegerType)))
        instance = st.create(name="Alice", age=30)
        assert repr(instance) == "(name='Alice', age=30)"


class TestVariantType:
    """Tests for VariantType."""

    def test_create(self):
        """Create a variant type."""
        vt = _VariantTypeClass((("Some", IntegerType), ("None", StringType)))
        assert len(vt.cases) == 2
        assert vt.case_names() == ["Some", "None"]
        assert vt.case_types() == [IntegerType, StringType]

    def test_case_type(self):
        """Get case type by name."""
        vt = _VariantTypeClass((("Some", IntegerType), ("None", StringType)))
        assert vt.case_type("Some") == IntegerType
        assert vt.case_type("None") == StringType

    def test_case_type_missing(self):
        """Case type raises KeyError for missing case."""
        vt = _VariantTypeClass((("Some", IntegerType),))
        with pytest.raises(KeyError):
            vt.case_type("Missing")

    def test_create_instance(self):
        """Create variant instance."""
        vt = _VariantTypeClass((("Some", IntegerType), ("None", StringType)))
        instance = vt.create("Some", 42)
        assert isinstance(instance, EastVariant)
        assert instance._east_type == vt
        assert instance.tag == "Some"
        assert instance.value == 42

    def test_create_instance_with_null(self):
        """Create variant instance with default null value."""
        vt = _VariantTypeClass((("Some", IntegerType), ("None", StringType)))
        instance = vt.create("None")
        assert instance.tag == "None"
        assert instance.value == null

    def test_create_instance_invalid_case(self):
        """Creating instance with invalid case raises KeyError."""
        vt = _VariantTypeClass((("Some", IntegerType),))
        with pytest.raises(KeyError):
            vt.create("Invalid", 42)


class TestCase:
    """Tests for Case."""

    def test_create(self):
        """Create a case."""
        c = Case("Some", 42)
        assert c.tag == "Some"
        assert c.value == 42

    def test_equality(self):
        """Cases equal if tag and value equal."""
        c1 = Case("Some", 42)
        c2 = Case("Some", 42)
        c3 = Case("Some", 43)
        c4 = Case("Other", 42)
        assert c1 == c2
        assert c1 != c3
        assert c1 != c4

    def test_ordering(self):
        """Cases ordered by tag, then value."""
        c1 = Case("A", 2)
        c2 = Case("A", 3)
        c3 = Case("B", 1)
        assert c1 < c2  # Same tag, value 2 < 3
        assert c1 < c3  # Tag A < B
        assert c2 < c3  # Tag A < B

    def test_hash(self):
        """Cases are hashable."""
        c1 = Case("Some", 42)
        c2 = Case("Some", 42)
        assert hash(c1) == hash(c2)

    def test_repr_with_value(self):
        """Case repr with value."""
        c = Case("Some", 42)
        assert repr(c) == ".Some 42"

    def test_repr_null_value(self):
        """Case repr with null value."""
        c = Case("None", null)
        assert repr(c) == ".None"


class TestEastVariant:
    """Tests for EastVariant."""

    def test_tag_value_access(self):
        """Access tag and value."""
        vt = _VariantTypeClass((("Some", IntegerType), ("None", StringType)))
        instance = vt.create("Some", 42)
        assert instance.tag == "Some"
        assert instance.value == 42

    def test_equality(self):
        """Structural equality."""
        vt = _VariantTypeClass((("Some", IntegerType), ("None", StringType)))
        v1 = vt.create("Some", 42)
        v2 = vt.create("Some", 42)
        v3 = vt.create("Some", 43)
        v4 = vt.create("None", "test")
        assert v1 == v2
        assert v1 != v3
        assert v1 != v4

    def test_equality_different_types(self):
        """Variants with different types aren't equal."""
        vt1 = _VariantTypeClass((("Some", IntegerType),))
        vt2 = _VariantTypeClass((("Some", IntegerType), ("None", StringType)))
        v1 = vt1.create("Some", 42)
        v2 = vt2.create("Some", 42)
        assert v1 != v2

    def test_ordering(self):
        """Structural ordering."""
        vt = _VariantTypeClass((("A", IntegerType), ("B", IntegerType)))
        v1 = vt.create("A", 1)
        v2 = vt.create("A", 2)
        v3 = vt.create("B", 0)
        assert v1 < v2  # Same tag, value 1 < 2
        assert v1 < v3  # Tag A < B

    def test_hash(self):
        """Variants are hashable."""
        vt = _VariantTypeClass((("Some", IntegerType),))
        v1 = vt.create("Some", 42)
        v2 = vt.create("Some", 42)
        assert hash(v1) == hash(v2)
        # Can be used in sets
        variants = {v1, v2}
        assert len(variants) == 1

    def test_repr(self):
        """Variant repr in East format."""
        vt = _VariantTypeClass((("Some", IntegerType), ("None", StringType)))
        v1 = vt.create("Some", 42)
        v2 = vt.create("None")
        assert repr(v1) == ".Some 42"
        assert repr(v2) == ".None"


class TestMakeCase:
    """Tests for make_case helper."""

    def test_with_value(self):
        """Make case with value."""
        c = make_case("Some", 42)
        assert c.tag == "Some"
        assert c.value == 42

    def test_without_value(self):
        """Make case defaults to null."""
        c = make_case("None")
        assert c.tag == "None"
        assert c.value == null

    def test_explicit_none(self):
        """Explicit None becomes null."""
        c = make_case("None", None)
        assert c.tag == "None"
        assert c.value == null


class TestOptionPattern:
    """Test the common Option pattern with variants."""

    def test_option_some(self):
        """Option Some case."""
        option_type = _VariantTypeClass((("Some", IntegerType), ("None", IntegerType)))
        some = option_type.create("Some", 42)
        assert some.tag == "Some"
        assert some.value == 42

    def test_option_none(self):
        """Option None case."""
        option_type = _VariantTypeClass((("Some", IntegerType), ("None", IntegerType)))
        none = option_type.create("None")
        assert none.tag == "None"
        assert none.value == null

    def test_pattern_matching_with_tag(self):
        """Pattern match using tag."""
        option_type = _VariantTypeClass((("Some", IntegerType), ("None", IntegerType)))

        def unwrap_or(opt: EastVariant, default: int) -> int:
            if opt.tag == "Some":
                return opt.value
            return default

        some = option_type.create("Some", 42)
        none = option_type.create("None")

        assert unwrap_or(some, 0) == 42
        assert unwrap_or(none, 0) == 0


"""Tests for East type constructors with validation.

Ported from East/src/types.spec.ts - Type constructors test suite
"""


class TestTypeConstructors:
    """Test suite for type constructor validation."""

    def test_array_type_should_create_array_types(self):
        """ArrayType should create array types."""
        typ = ArrayType(IntegerType)
        assert typ.tag == "Array"
        assert typ.value == IntegerType

    def test_array_type_should_throw_for_function_element_types(self):
        """ArrayType should throw for function element types."""
        with pytest.raises(
            TypeError, match=r"Array value type must be a \(non-function\) data type"
        ):
            ArrayType(FunctionType([], NullType, []))

    def test_set_type_should_create_set_types(self):
        """SetType should create set types."""
        typ = SetType(StringType)
        assert typ.tag == "Set"
        assert typ.value == StringType

    def test_set_type_should_throw_for_mutable_key_types(self):
        """SetType should throw for mutable key types."""
        with pytest.raises(TypeError, match=r"Set key type must be an immutable type"):
            SetType(ArrayType(IntegerType))

    def test_dict_type_should_create_dict_types(self):
        """DictType should create dict types."""
        typ = DictType(StringType, IntegerType)
        assert typ.tag == "Dict"
        # DictType creates a struct with key and value fields
        dict_struct = typ.value
        assert dict_struct.key == StringType
        assert dict_struct.value == IntegerType

    def test_dict_type_should_throw_for_mutable_key_types(self):
        """DictType should throw for mutable key types."""
        with pytest.raises(TypeError, match=r"Dict key type must be an immutable type"):
            DictType(ArrayType(IntegerType), StringType)

    def test_dict_type_should_throw_for_function_value_types(self):
        """DictType should throw for function value types."""
        with pytest.raises(
            TypeError, match=r"Dict value type must be a \(non-function\) data type"
        ):
            DictType(StringType, FunctionType([], NullType, []))

    def test_struct_type_should_create_struct_types(self):
        """StructType should create struct types."""
        typ = StructType([("x", IntegerType), ("y", FloatType)])
        assert typ.tag == "Struct"
        fields = typ.value
        assert len(fields) == 2
        assert fields[0].name == "x"
        assert fields[0].type == IntegerType
        assert fields[1].name == "y"
        assert fields[1].type == FloatType

    def test_variant_type_should_create_variant_types_with_sorted_cases(self):
        """VariantType should create variant types with sorted cases."""
        # Pass cases in unsorted order
        typ = VariantType([("b", IntegerType), ("a", StringType)])
        assert typ.tag == "Variant"
        cases = typ.value
        # Cases should be sorted alphabetically
        case_names = [case.name for case in cases]
        assert case_names == ["a", "b"]
        assert cases[0].type == StringType
        assert cases[1].type == IntegerType

    def test_function_type_should_create_function_types(self):
        """FunctionType should create function types."""
        typ = FunctionType([IntegerType, StringType], BooleanType, [])
        assert typ.tag == "Function"
        func = typ.value
        assert func.inputs == [IntegerType, StringType]
        assert func.output == BooleanType
        assert func.platforms == []

    def test_option_type_should_create_option_types(self):
        """OptionType should create option types."""
        typ = OptionType(IntegerType)
        assert typ.tag == "Variant"
        cases = typ.value
        case_names = [case.name for case in cases]
        # Should have none and some cases, sorted
        assert case_names == ["none", "some"]
        assert cases[0].type == NullType
        assert cases[1].type == IntegerType


"""Additional coverage tests for East type operations - edge cases and error paths.

Ported from East/src/types.spec.ts - Additional coverage tests section
"""


class TestTypeEqualEdgeCases:
    """Edge case tests for type_equal function."""

    def test_should_handle_k1_gt_k2_variant_case_mismatch(self):
        """TypeEqual should handle k1 > k2 variant case mismatch."""
        t1 = VariantType([("a", IntegerType), ("c", StringType)])
        t2 = VariantType([("a", IntegerType), ("b", StringType)])
        with pytest.raises(
            TypeMismatchError, match=r"variant case b is not present in both variants"
        ):
            type_equal(t1, t2)

    def test_should_succeed_for_equal_variant_types(self):
        """TypeEqual should succeed for equal variant types."""
        t1 = VariantType([("a", IntegerType), ("b", StringType)])
        t2 = VariantType([("a", IntegerType), ("b", StringType)])
        result = type_equal(t1, t2)
        assert result.tag == "Variant"

    def test_should_succeed_for_equal_function_types(self):
        """TypeEqual should succeed for equal function types."""
        t1 = FunctionType([IntegerType, StringType], FloatType, [])
        t2 = FunctionType([IntegerType, StringType], FloatType, [])
        result = type_equal(t1, t2)
        assert result.tag == "Function"

    def test_should_propagate_errors_from_nested_types(self):
        """TypeEqual should propagate errors from nested types."""
        t1 = ArrayType(IntegerType)
        t2 = ArrayType(FloatType)
        with pytest.raises(TypeMismatchError):
            type_equal(t1, t2)

    def test_should_handle_variant_case_where_k1_lt_k2(self):
        """TypeEqual should handle variant case where k1 < k2."""
        t1 = VariantType([("a", IntegerType), ("b", StringType)])
        t2 = VariantType([("a", IntegerType), ("c", StringType)])
        with pytest.raises(
            TypeMismatchError, match=r"variant case b is not present in both variants"
        ):
            type_equal(t1, t2)

    def test_with_nested_type_mismatch_in_array(self):
        """TypeEqual with nested type mismatch in array."""
        t1 = StructType([("x", ArrayType(IntegerType))])
        t2 = StructType([("x", ArrayType(FloatType))])
        with pytest.raises(TypeMismatchError):
            type_equal(t1, t2)

    def test_should_throw_when_comparing_variant_with_non_variant(self):
        """TypeEqual should throw when comparing Variant with non-Variant."""
        t1 = VariantType([("a", IntegerType)])
        t2 = IntegerType
        with pytest.raises(TypeMismatchError, match=r"is not equal to.*incompatible types"):
            type_equal(t1, t2)

    def test_should_throw_when_comparing_function_with_non_function(self):
        """TypeEqual should throw when comparing Function with non-Function."""
        t1 = FunctionType([IntegerType], NullType, [])
        t2 = IntegerType
        with pytest.raises(TypeMismatchError, match=r"is not equal to.*incompatible types"):
            type_equal(t1, t2)

    def test_should_succeed_for_equal_dict_types(self):
        """TypeEqual should succeed for equal Dict types."""
        t1 = DictType(StringType, IntegerType)
        t2 = DictType(StringType, IntegerType)
        result = type_equal(t1, t2)
        assert result.tag == "Dict"

    def test_should_throw_when_comparing_dict_with_non_dict(self):
        """TypeEqual should throw when comparing Dict with non-Dict."""
        t1 = DictType(StringType, IntegerType)
        t2 = IntegerType
        with pytest.raises(TypeMismatchError, match=r"is not equal to.*incompatible types"):
            type_equal(t1, t2)

    def test_should_succeed_for_equal_struct_types(self):
        """TypeEqual should succeed for equal Struct types."""
        t1 = StructType([("x", IntegerType), ("y", StringType)])
        t2 = StructType([("x", IntegerType), ("y", StringType)])
        result = type_equal(t1, t2)
        assert result.tag == "Struct"

    def test_should_throw_when_comparing_struct_with_non_struct(self):
        """TypeEqual should throw when comparing Struct with non-Struct."""
        t1 = StructType([("x", IntegerType)])
        t2 = IntegerType
        with pytest.raises(TypeMismatchError, match=r"is not equal to.*incompatible types"):
            type_equal(t1, t2)

    def test_should_throw_when_comparing_array_with_non_array(self):
        """TypeEqual should throw when comparing Array with non-Array."""
        t1 = ArrayType(IntegerType)
        t2 = IntegerType
        with pytest.raises(TypeMismatchError, match=r"is not equal to.*incompatible types"):
            type_equal(t1, t2)

    def test_should_succeed_for_equal_set_types(self):
        """TypeEqual should succeed for equal Set types."""
        t1 = SetType(StringType)
        t2 = SetType(StringType)
        result = type_equal(t1, t2)
        assert result.tag == "Set"

    def test_should_throw_when_comparing_set_with_non_set(self):
        """TypeEqual should throw when comparing Set with non-Set."""
        t1 = SetType(StringType)
        t2 = IntegerType
        with pytest.raises(TypeMismatchError, match=r"is not equal to.*incompatible types"):
            type_equal(t1, t2)

    def test_catch_block_with_deeply_nested_error(self):
        """TypeEqual catch block with deeply nested error."""
        t1 = DictType(StringType, ArrayType(IntegerType))
        t2 = DictType(StringType, ArrayType(FloatType))
        with pytest.raises(TypeMismatchError):
            type_equal(t1, t2)


class TestTypeIntersectEdgeCases:
    """Edge case tests for type_intersect function."""

    def test_should_throw_for_functions_with_different_argument_counts(self):
        """TypeIntersect should throw for functions with different argument counts."""
        t1 = FunctionType([IntegerType], NullType, [])
        t2 = FunctionType([IntegerType, StringType], NullType, [])
        with pytest.raises(
            TypeMismatchError, match=r"functions take different number of arguments"
        ):
            type_intersect(t1, t2)

    def test_should_succeed_for_compatible_function_types(self):
        """TypeIntersect should succeed for compatible function types."""
        t1 = FunctionType([IntegerType], FloatType, [])
        t2 = FunctionType([IntegerType], FloatType, [])
        result = type_intersect(t1, t2)
        assert result.tag == "Function"

    def test_should_throw_when_intersecting_function_with_non_function(self):
        """TypeIntersect should throw when intersecting Function with non-Function."""
        t1 = FunctionType([IntegerType], NullType, [])
        t2 = IntegerType
        with pytest.raises(TypeMismatchError, match=r"Cannot intersect.*incompatible types"):
            type_intersect(t1, t2)

    def test_catch_block_with_nested_type_error(self):
        """TypeIntersect catch block with nested type error."""
        t1 = ArrayType(IntegerType)
        t2 = ArrayType(FloatType)
        with pytest.raises(TypeMismatchError):
            type_intersect(t1, t2)

    def test_should_throw_when_intersecting_variant_with_non_variant(self):
        """TypeIntersect should throw when intersecting Variant with non-Variant."""
        t1 = VariantType([("a", IntegerType)])
        t2 = IntegerType
        with pytest.raises(TypeMismatchError, match=r"Cannot intersect.*incompatible types"):
            type_intersect(t1, t2)

    def test_should_succeed_for_compatible_struct_types(self):
        """TypeIntersect should succeed for compatible struct types."""
        t1 = StructType([("x", IntegerType), ("y", StringType)])
        t2 = StructType([("x", IntegerType), ("y", StringType)])
        result = type_intersect(t1, t2)
        assert result.tag == "Struct"

    def test_should_throw_when_intersecting_struct_with_non_struct(self):
        """TypeIntersect should throw when intersecting Struct with non-Struct."""
        t1 = StructType([("x", IntegerType)])
        t2 = IntegerType
        with pytest.raises(TypeMismatchError, match=r"Cannot intersect.*incompatible types"):
            type_intersect(t1, t2)

    def test_should_succeed_for_compatible_dict_types(self):
        """TypeIntersect should succeed for compatible dict types."""
        t1 = DictType(StringType, IntegerType)
        t2 = DictType(StringType, IntegerType)
        result = type_intersect(t1, t2)
        assert result.tag == "Dict"

    def test_should_throw_when_intersecting_dict_with_non_dict(self):
        """TypeIntersect should throw when intersecting Dict with non-Dict."""
        t1 = DictType(StringType, IntegerType)
        t2 = IntegerType
        with pytest.raises(TypeMismatchError, match=r"Cannot intersect.*incompatible types"):
            type_intersect(t1, t2)

    def test_should_succeed_for_compatible_set_types(self):
        """TypeIntersect should succeed for compatible set types."""
        t1 = SetType(StringType)
        t2 = SetType(StringType)
        result = type_intersect(t1, t2)
        assert result.tag == "Set"

    def test_should_throw_when_intersecting_set_with_non_set(self):
        """TypeIntersect should throw when intersecting Set with non-Set."""
        t1 = SetType(StringType)
        t2 = IntegerType
        with pytest.raises(TypeMismatchError, match=r"Cannot intersect.*incompatible types"):
            type_intersect(t1, t2)

    def test_should_succeed_for_compatible_array_types(self):
        """TypeIntersect should succeed for compatible array types."""
        t1 = ArrayType(IntegerType)
        t2 = ArrayType(IntegerType)
        result = type_intersect(t1, t2)
        assert result.tag == "Array"

    def test_should_throw_when_intersecting_array_with_non_array(self):
        """TypeIntersect should throw when intersecting Array with non-Array."""
        t1 = ArrayType(IntegerType)
        t2 = IntegerType
        with pytest.raises(TypeMismatchError, match=r"Cannot intersect.*incompatible types"):
            type_intersect(t1, t2)


class TestTypeUnionEdgeCases:
    """Edge case tests for type_union function."""

    def test_should_throw_for_functions_with_different_argument_counts(self):
        """TypeUnion should throw for functions with different argument counts."""
        t1 = FunctionType([IntegerType], NullType, [])
        t2 = FunctionType([IntegerType, StringType], NullType, [])
        with pytest.raises(
            TypeMismatchError, match=r"functions take different number of arguments"
        ):
            type_union(t1, t2)

    def test_should_throw_when_unioning_function_with_non_function(self):
        """TypeUnion should throw when unioning Function with non-Function."""
        t1 = FunctionType([IntegerType], NullType, [])
        t2 = IntegerType
        with pytest.raises(TypeMismatchError, match=r"Cannot union.*incompatible types"):
            type_union(t1, t2)

    def test_should_throw_when_unioning_dict_with_non_dict(self):
        """TypeUnion should throw when unioning Dict with non-Dict."""
        t1 = DictType(StringType, IntegerType)
        t2 = IntegerType
        with pytest.raises(TypeMismatchError, match=r"Cannot union.*incompatible types"):
            type_union(t1, t2)

    def test_should_throw_when_unioning_struct_with_non_struct(self):
        """TypeUnion should throw when unioning Struct with non-Struct."""
        t1 = StructType([("x", IntegerType)])
        t2 = IntegerType
        with pytest.raises(TypeMismatchError, match=r"Cannot union.*incompatible types"):
            type_union(t1, t2)

    def test_should_throw_when_unioning_variant_with_non_variant(self):
        """TypeUnion should throw when unioning Variant with non-Variant."""
        t1 = VariantType([("a", IntegerType)])
        t2 = IntegerType
        with pytest.raises(TypeMismatchError, match=r"Cannot union.*incompatible types"):
            type_union(t1, t2)

    def test_should_throw_when_unioning_set_with_non_set(self):
        """TypeUnion should throw when unioning Set with non-Set."""
        t1 = SetType(StringType)
        t2 = IntegerType
        with pytest.raises(TypeMismatchError, match=r"Cannot union.*incompatible types"):
            type_union(t1, t2)


class TestOptionAndSomeTypes:
    """Tests for SomeType and OptionType helpers."""

    def test_some_type_should_create_option_variant_with_some_case(self):
        """SomeType should create option variant with some case."""
        typ = SomeType(IntegerType)
        assert typ.tag == "Variant"
        cases = typ.value
        case_dict = {case.name: case.type for case in cases}
        assert "some" in case_dict
        assert case_dict["some"] == IntegerType

    def test_option_type_should_create_variant_with_none_and_some_cases(self):
        """OptionType should create variant with none and some cases."""
        typ = OptionType(IntegerType)
        assert typ.tag == "Variant"
        cases = typ.value
        case_dict = {case.name: case.type for case in cases}
        assert case_dict["none"] == NullType
        assert case_dict["some"] == IntegerType


class TestIsSubtypeEdgeCases:
    """Edge case tests for is_subtype function."""

    def test_should_return_false_for_incompatible_variant_types(self):
        """isSubtype should return false for incompatible variant types."""
        t1 = VariantType([("a", IntegerType)])
        t2 = IntegerType
        assert is_subtype(t1, t2) is False

    def test_should_return_false_for_incompatible_function_types(self):
        """isSubtype should return false for incompatible function types."""
        t1 = FunctionType([IntegerType], NullType, [])
        t2 = IntegerType
        assert is_subtype(t1, t2) is False

    def test_should_return_false_for_set_compared_to_non_set(self):
        """isSubtype should return false for Set compared to non-Set."""
        t1 = SetType(StringType)
        t2 = IntegerType
        assert is_subtype(t1, t2) is False

    def test_should_return_false_for_dict_compared_to_non_dict(self):
        """isSubtype should return false for Dict compared to non-Dict."""
        t1 = DictType(StringType, IntegerType)
        t2 = IntegerType
        assert is_subtype(t1, t2) is False

    def test_should_return_false_for_struct_compared_to_non_struct(self):
        """isSubtype should return false for Struct compared to non-Struct."""
        t1 = StructType([("x", IntegerType)])
        t2 = IntegerType
        assert is_subtype(t1, t2) is False

    def test_should_return_false_for_array_compared_to_non_array(self):
        """isSubtype should return false for Array compared to non-Array."""
        t1 = ArrayType(IntegerType)
        t2 = IntegerType
        assert is_subtype(t1, t2) is False


class TestIsTypeEqualEdgeCases:
    """Edge case tests for is_type_equal function."""

    def test_should_return_false_for_function_compared_to_non_function(self):
        """isTypeEqual should return false for Function compared to non-Function."""
        t1 = FunctionType([IntegerType], NullType, [])
        t2 = IntegerType
        assert is_type_equal(t1, t2) is False


class TestIsValueOfEdgeCases:
    """Edge case tests for is_value_of function."""

    def test_should_return_false_for_non_set_value_with_set_type(self):
        """isValueOf should return false for non-Set value with Set type."""
        assert is_value_of([], SetType(IntegerType)) is False

    def test_should_return_false_for_non_map_value_with_dict_type(self):
        """isValueOf should return false for non-Map value with Dict type."""
        assert is_value_of([], DictType(StringType, IntegerType)) is False


"""Tests for East type operation functions.

Ported from East/src/types.spec.ts - TypeUnion, TypeIntersect, TypeEqual test suites
"""


class TestTypeUnion:
    """Test suite for type_union function."""

    def test_never_is_identity_for_union(self):
        """Never is identity for union."""
        assert type_union(NeverType, IntegerType) == IntegerType
        assert type_union(IntegerType, NeverType) == IntegerType

    def test_should_union_same_primitive_types(self):
        """should union same primitive types."""
        assert type_union(IntegerType, IntegerType) == IntegerType

    def test_should_throw_for_different_primitive_types(self):
        """should throw for different primitive types."""
        with pytest.raises(
            TypeMismatchError, match=r"Cannot union \.Integer with \.Float: incompatible types"
        ):
            type_union(IntegerType, FloatType)

    def test_should_union_array_types_with_same_element_type(self):
        """should union array types with same element type."""
        result = type_union(ArrayType(IntegerType), ArrayType(IntegerType))
        assert result.tag == "Array"

    def test_should_throw_for_array_types_with_different_element_types(self):
        """should throw for array types with different element types."""
        with pytest.raises(
            TypeMismatchError, match=r"\.Integer is not equal to \.Float: incompatible types"
        ):
            type_union(ArrayType(IntegerType), ArrayType(FloatType))

    def test_should_union_variant_types(self):
        """should union variant types."""
        t1 = VariantType([("a", IntegerType), ("b", StringType)])
        t2 = VariantType([("b", StringType), ("c", FloatType)])
        result = type_union(t1, t2)
        assert result.tag == "Variant"
        # Should have all cases: a, b, c
        cases = result.value
        case_dict = {case.name: case.type for case in cases}
        assert "a" in case_dict
        assert "b" in case_dict
        assert "c" in case_dict

    def test_should_union_struct_types(self):
        """should union struct types."""
        t1 = StructType([("x", IntegerType), ("y", FloatType)])
        t2 = StructType([("x", IntegerType), ("y", FloatType)])
        result = type_union(t1, t2)
        assert result.tag == "Struct"

    def test_should_throw_for_structs_with_different_field_count(self):
        """should throw for structs with different field count."""
        t1 = StructType([("x", IntegerType)])
        t2 = StructType([("x", IntegerType), ("y", FloatType)])
        with pytest.raises(TypeMismatchError, match=r"structs contain different number of fields"):
            type_union(t1, t2)

    def test_should_throw_for_structs_with_different_field_names_at_position_0(self):
        """should throw for structs with different field names at position 0."""
        t1 = StructType([("x", IntegerType)])
        t2 = StructType([("y", IntegerType)])
        with pytest.raises(TypeMismatchError, match=r"struct field 0 has mismatched names x and y"):
            type_union(t1, t2)

    def test_should_throw_for_structs_with_mismatched_field_names_in_multi_field_structs(self):
        """should throw for structs with mismatched field names in multi-field structs."""
        t1 = StructType([("a", IntegerType), ("b", StringType), ("c", FloatType)])
        t2 = StructType([("a", IntegerType), ("x", StringType), ("c", FloatType)])
        with pytest.raises(TypeMismatchError, match=r"struct field 1 has mismatched names b and x"):
            type_union(t1, t2)

    def test_should_union_function_types(self):
        """should union function types."""
        t1 = FunctionType([IntegerType], IntegerType, [])
        t2 = FunctionType([IntegerType], FloatType, [])
        with pytest.raises(
            TypeMismatchError, match=r"Cannot union \.Integer with \.Float: incompatible types"
        ):
            type_union(t1, t2)


class TestTypeIntersect:
    """Test suite for type_intersect function."""

    def test_never_is_absorbing_for_intersection(self):
        """Never is absorbing for intersection."""
        assert type_intersect(NeverType, IntegerType) == NeverType
        assert type_intersect(IntegerType, NeverType) == NeverType

    def test_should_intersect_same_primitive_types(self):
        """should intersect same primitive types."""
        assert type_intersect(IntegerType, IntegerType) == IntegerType

    def test_should_throw_for_different_primitive_types(self):
        """should throw for different primitive types."""
        with pytest.raises(
            TypeMismatchError, match=r"Cannot intersect \.Integer with \.Float: incompatible types"
        ):
            type_intersect(IntegerType, FloatType)

    def test_should_intersect_variant_types(self):
        """should intersect variant types."""
        t1 = VariantType([("a", IntegerType), ("b", StringType), ("c", FloatType)])
        t2 = VariantType([("b", StringType), ("c", FloatType), ("d", BooleanType)])
        result = type_intersect(t1, t2)
        assert result.tag == "Variant"
        # TypeIntersect for variants keeps cases in t1 that are also in t2
        cases = result.value
        case_dict = {case.name: case.type for case in cases}
        assert case_dict == {"b": StringType, "c": FloatType}

    def test_should_throw_for_variants_with_no_overlapping_cases(self):
        """should throw for variants with no overlapping cases."""
        t1 = VariantType([("a", IntegerType)])
        t2 = VariantType([("b", StringType)])
        with pytest.raises(TypeMismatchError, match=r"variants have no overlapping cases"):
            type_intersect(t1, t2)


class TestTypeEqual:
    """Test suite for type_equal function."""

    def test_should_accept_equal_primitive_types(self):
        """should accept equal primitive types."""
        assert type_equal(IntegerType, IntegerType) == IntegerType

    def test_should_throw_for_unequal_primitive_types(self):
        """should throw for unequal primitive types."""
        with pytest.raises(
            TypeMismatchError, match=r"\.Integer is not equal to \.Float: incompatible types"
        ):
            type_equal(IntegerType, FloatType)

    def test_should_accept_equal_array_types(self):
        """should accept equal array types."""
        result = type_equal(ArrayType(IntegerType), ArrayType(IntegerType))
        assert result.tag == "Array"

    def test_should_throw_for_unequal_variant_case_names(self):
        """should throw for unequal variant case names."""
        t1 = VariantType([("a", IntegerType), ("c", StringType)])
        t2 = VariantType([("a", IntegerType), ("b", StringType)])
        with pytest.raises(
            TypeMismatchError,
            match=r"\.Variant.*is not equal to.*variant case .* is not present in both variants",
        ):
            type_equal(t1, t2)

    def test_should_throw_for_variants_with_different_case_count(self):
        """should throw for variants with different case count."""
        t1 = VariantType([("a", IntegerType)])
        t2 = VariantType([("a", IntegerType), ("b", StringType)])
        with pytest.raises(
            TypeMismatchError,
            match=r"\.Variant.*is not equal to.*variants contain different number of cases",
        ):
            type_equal(t1, t2)

    def test_should_throw_for_functions_with_different_argument_count(self):
        """should throw for functions with different argument count."""
        t1 = FunctionType([IntegerType], NullType, [])
        t2 = FunctionType([IntegerType, StringType], NullType, [])
        with pytest.raises(
            TypeMismatchError,
            match=r"\.Function.*is not equal to.*functions take different number of arguments",
        ):
            type_equal(t1, t2)


"""Tests for East type predicate functions.

Ported from East/src/types.spec.ts - isDataType, isImmutableType, isValueOf test suites
"""


class TestIsDataType:
    """Test suite for is_data_type predicate."""

    def test_should_return_true_for_primitive_data_types(self):
        """should return true for primitive data types."""
        assert is_data_type(NeverType) is True
        assert is_data_type(NullType) is True
        assert is_data_type(BooleanType) is True
        assert is_data_type(IntegerType) is True
        assert is_data_type(FloatType) is True
        assert is_data_type(StringType) is True
        assert is_data_type(DateTimeType) is True
        assert is_data_type(BlobType) is True

    def test_should_return_true_for_collection_data_types(self):
        """should return true for collection data types."""
        assert is_data_type(ArrayType(IntegerType)) is True
        assert is_data_type(SetType(StringType)) is True
        assert is_data_type(DictType(StringType, IntegerType)) is True

    def test_should_return_true_for_struct_with_data_fields(self):
        """should return true for struct with data fields."""
        typ = StructType([("x", IntegerType), ("y", FloatType)])
        assert is_data_type(typ) is True

    def test_should_throw_error_for_struct_with_function_field(self):
        """should throw error for struct with function field."""
        with pytest.raises(TypeError, match=r"Struct field f must be a \(non-function\) data type"):
            StructType([("x", IntegerType), ("f", FunctionType([], NullType, []))])

    def test_should_return_true_for_variant_with_data_cases(self):
        """should return true for variant with data cases."""
        typ = VariantType([("none", NullType), ("some", IntegerType)])
        assert is_data_type(typ) is True

    def test_should_throw_error_for_variant_with_function_case(self):
        """should throw error for variant with function case."""
        with pytest.raises(
            TypeError, match=r"Variant case func must be a \(non-function\) data type"
        ):
            VariantType([("data", IntegerType), ("func", FunctionType([], NullType, []))])

    def test_should_return_false_for_function_types(self):
        """should return false for function types."""
        assert is_data_type(FunctionType([], NullType, [])) is False


class TestIsImmutableType:
    """Test suite for is_immutable_type predicate."""

    def test_should_return_true_for_primitive_immutable_types(self):
        """should return true for primitive immutable types."""
        assert is_immutable_type(NeverType) is True
        assert is_immutable_type(NullType) is True
        assert is_immutable_type(BooleanType) is True
        assert is_immutable_type(IntegerType) is True
        assert is_immutable_type(FloatType) is True
        assert is_immutable_type(StringType) is True
        assert is_immutable_type(DateTimeType) is True
        assert is_immutable_type(BlobType) is True

    def test_should_return_false_for_mutable_collection_types(self):
        """should return false for mutable collection types."""
        assert is_immutable_type(ArrayType(IntegerType)) is False
        assert is_immutable_type(SetType(StringType)) is False
        assert is_immutable_type(DictType(StringType, IntegerType)) is False

    def test_should_return_true_for_struct_with_immutable_fields(self):
        """should return true for struct with immutable fields."""
        typ = StructType([("x", IntegerType), ("y", StringType)])
        assert is_immutable_type(typ) is True

    def test_should_return_false_for_struct_with_mutable_field(self):
        """should return false for struct with mutable field."""
        typ = StructType([("x", IntegerType), ("arr", ArrayType(IntegerType))])
        assert is_immutable_type(typ) is False

    def test_should_return_true_for_variant_with_immutable_cases(self):
        """should return true for variant with immutable cases."""
        typ = VariantType([("none", NullType), ("some", IntegerType)])
        assert is_immutable_type(typ) is True

    def test_should_return_false_for_variant_with_mutable_case(self):
        """should return false for variant with mutable case."""
        typ = VariantType([("data", IntegerType), ("list", ArrayType(IntegerType))])
        assert is_immutable_type(typ) is False

    def test_should_return_false_for_function_types(self):
        """should return false for function types."""
        assert is_immutable_type(FunctionType([], NullType, [])) is False


class TestIsValueOf:
    """Test suite for is_value_of predicate."""

    def test_should_validate_primitive_values(self):
        """should validate primitive values."""
        assert is_value_of(None, NullType) is True
        assert is_value_of(True, BooleanType) is True
        assert is_value_of(False, BooleanType) is True
        assert is_value_of(42, IntegerType) is True
        assert is_value_of(3.14, FloatType) is True
        assert is_value_of("hello", StringType) is True
        assert is_value_of(b"bytes", BlobType) is True
        assert is_value_of(Blob(b"bytes"), BlobType) is True

    def test_should_reject_wrong_primitive_values(self):
        """should reject wrong primitive values."""
        assert is_value_of(42, StringType) is False
        assert is_value_of("hello", IntegerType) is False
        assert is_value_of(True, IntegerType) is False  # bool is not int
        assert is_value_of(3.14, IntegerType) is False

    def test_should_return_false_for_never_type(self):
        """should return false for Never type."""
        assert is_value_of(None, NeverType) is False
        assert is_value_of(42, NeverType) is False
        assert is_value_of("anything", NeverType) is False

    def test_should_validate_array_values(self):
        """should validate array values."""
        arr = EastArray(IntegerType, [1, 2, 3])
        assert is_value_of(arr, ArrayType(IntegerType)) is True
        # Lists also work
        assert is_value_of([1, 2, 3], ArrayType(IntegerType)) is True
        # Wrong element type
        assert is_value_of([1, "two", 3], ArrayType(IntegerType)) is False

    def test_should_validate_set_values(self):
        """should validate set values."""
        s = EastSet(IntegerType, {1, 2, 3})
        assert is_value_of(s, SetType(IntegerType)) is True
        # Python sets also work
        assert is_value_of({1, 2, 3}, SetType(IntegerType)) is True

    def test_should_validate_dict_values(self):
        """should validate dict values."""
        d = EastDict(StringType, IntegerType, {"a": 1, "b": 2})
        assert is_value_of(d, DictType(StringType, IntegerType)) is True
        # Python dicts also work
        assert is_value_of({"a": 1, "b": 2}, DictType(StringType, IntegerType)) is True

    def test_should_validate_struct_values(self):
        """should validate struct values."""
        struct_type = _StructTypeClass((("x", IntegerType), ("y", FloatType)))
        struct_val = struct_type.create(x=42, y=3.14)
        assert is_value_of(struct_val, StructType([("x", IntegerType), ("y", FloatType)])) is True

    def test_should_validate_variant_values(self):
        """should validate variant values."""
        variant_type = _VariantTypeClass((("none", NullType), ("some", IntegerType)))
        none_val = variant_type.create("none", None)
        some_val = variant_type.create("some", 42)

        variant_type_from_cases = VariantType([("none", NullType), ("some", IntegerType)])
        assert is_value_of(none_val, variant_type_from_cases) is True
        assert is_value_of(some_val, variant_type_from_cases) is True

    def test_should_throw_for_function_type(self):
        """should throw for Function type."""
        with pytest.raises(
            TypeError, match=r"JavaScript/Python functions cannot be converted to East functions"
        ):
            is_value_of(lambda x: x, FunctionType([], NullType, []))


class TestIsTypeEqual:
    """Test suite for is_type_equal predicate."""

    def test_should_compare_primitive_types(self):
        """should compare primitive types."""
        assert is_type_equal(NullType, NullType) is True
        assert is_type_equal(IntegerType, IntegerType) is True
        assert is_type_equal(IntegerType, FloatType) is False

    def test_should_compare_array_types(self):
        """should compare array types."""
        assert is_type_equal(ArrayType(IntegerType), ArrayType(IntegerType)) is True
        assert is_type_equal(ArrayType(IntegerType), ArrayType(FloatType)) is False
        assert is_type_equal(ArrayType(IntegerType), IntegerType) is False

    def test_should_compare_set_types(self):
        """should compare set types."""
        assert is_type_equal(SetType(StringType), SetType(StringType)) is True
        assert is_type_equal(SetType(StringType), SetType(IntegerType)) is False

    def test_should_compare_dict_types(self):
        """should compare dict types."""
        assert (
            is_type_equal(DictType(StringType, IntegerType), DictType(StringType, IntegerType))
            is True
        )
        assert (
            is_type_equal(DictType(StringType, IntegerType), DictType(IntegerType, StringType))
            is False
        )

    def test_should_compare_struct_types(self):
        """should compare struct types."""
        t1 = StructType([("x", IntegerType), ("y", FloatType)])
        t2 = StructType([("x", IntegerType), ("y", FloatType)])
        t3 = StructType([("x", IntegerType), ("y", StringType)])
        assert is_type_equal(t1, t2) is True
        assert is_type_equal(t1, t3) is False

    def test_should_compare_variant_types(self):
        """should compare variant types."""
        t1 = VariantType([("none", NullType), ("some", IntegerType)])
        t2 = VariantType([("none", NullType), ("some", IntegerType)])
        t3 = VariantType([("none", NullType), ("some", FloatType)])
        assert is_type_equal(t1, t2) is True
        assert is_type_equal(t1, t3) is False

    def test_should_compare_function_types(self):
        """should compare function types."""
        t1 = FunctionType([IntegerType], StringType, [])
        t2 = FunctionType([IntegerType], StringType, [])
        t3 = FunctionType([FloatType], StringType, [])
        assert is_type_equal(t1, t2) is True
        assert is_type_equal(t1, t3) is False


class TestIsSubtype:
    """Test suite for is_subtype predicate."""

    def test_never_is_subtype_of_everything(self):
        """Never is subtype of everything."""
        assert is_subtype(NeverType, NullType) is True
        assert is_subtype(NeverType, IntegerType) is True
        assert is_subtype(NeverType, FunctionType([], NullType, [])) is True

    def test_primitive_types_are_only_subtypes_of_themselves(self):
        """primitive types are only subtypes of themselves."""
        assert is_subtype(IntegerType, IntegerType) is True
        assert is_subtype(IntegerType, FloatType) is False

    def test_variant_subtyping_fewer_cases_is_subtype(self):
        """variant subtyping - fewer cases is subtype."""
        t1 = VariantType([("a", IntegerType), ("b", StringType), ("c", FloatType)])
        t2 = VariantType([("a", IntegerType), ("b", StringType)])
        assert is_subtype(t1, t2) is False
        assert is_subtype(t2, t1) is True

    def test_struct_subtyping_is_structural(self):
        """struct subtyping is structural."""
        t1 = StructType([("x", IntegerType), ("y", FloatType)])
        t2 = StructType([("x", IntegerType), ("y", FloatType)])
        assert is_subtype(t1, t2) is True

    def test_function_subtyping_contravariant_inputs_covariant_output(self):
        """function subtyping - contravariant inputs, covariant output."""
        t1 = FunctionType([IntegerType], NeverType, [])
        t2 = FunctionType([IntegerType], IntegerType, [])
        # t1 has output Never which is subtype of Integer, so t1 <: t2
        assert is_subtype(t1, t2) is True


"""Tests for East type printing functions.

Ported from East/src/types.spec.ts - printType and printIdentifier test suites
"""


class TestPrintType:
    """Test suite for print_type function."""

    def test_should_print_primitive_types(self):
        """should print primitive types."""
        assert print_type(NeverType) == ".Never"
        assert print_type(NullType) == ".Null"
        assert print_type(BooleanType) == ".Boolean"
        assert print_type(IntegerType) == ".Integer"
        assert print_type(FloatType) == ".Float"
        assert print_type(StringType) == ".String"
        assert print_type(DateTimeType) == ".DateTime"
        assert print_type(BlobType) == ".Blob"

    def test_should_print_collection_types(self):
        """should print collection types."""
        assert print_type(ArrayType(IntegerType)) == ".Array .Integer"
        assert print_type(SetType(StringType)) == ".Set .String"
        assert (
            print_type(DictType(StringType, IntegerType)) == ".Dict (key=.String, value=.Integer)"
        )

    def test_should_print_struct_types(self):
        """should print struct types."""
        assert print_type(StructType([("x", IntegerType)])) == '.Struct [(name="x", type=.Integer)]'
        assert (
            print_type(StructType([("x", IntegerType), ("y", FloatType)]))
            == '.Struct [(name="x", type=.Integer), (name="y", type=.Float)]'
        )

    def test_should_print_variant_types(self):
        """should print variant types."""
        assert (
            print_type(VariantType([("none", NullType)])) == '.Variant [(name="none", type=.Null)]'
        )
        assert (
            print_type(VariantType([("none", NullType), ("some", IntegerType)]))
            == '.Variant [(name="none", type=.Null), (name="some", type=.Integer)]'
        )

    def test_should_print_function_types(self):
        """should print function types."""
        assert (
            print_type(FunctionType([], NullType, []))
            == ".Function (inputs=[], output=.Null, platforms=[])"
        )
        assert (
            print_type(FunctionType([IntegerType, StringType], BooleanType, []))
            == ".Function (inputs=[.Integer, .String], output=.Boolean, platforms=[])"
        )


class TestPrintIdentifier:
    """Test suite for print_identifier function."""

    def test_should_print_valid_identifiers_as_is(self):
        """should print valid identifiers as-is."""
        assert print_identifier("foo") == "foo"
        assert print_identifier("_bar") == "_bar"
        assert print_identifier("foo123") == "foo123"

    def test_should_escape_invalid_identifiers(self):
        """should escape invalid identifiers."""
        assert print_identifier("foo bar") == "`foo bar`"
        assert print_identifier("123") == "`123`"
        assert print_identifier("foo-bar") == "`foo-bar`"

    def test_should_escape_special_characters_in_identifiers(self):
        """should escape special characters in identifiers."""
        assert print_identifier("foo`bar") == "`foo\\`bar`"
        assert print_identifier("foo\\bar") == "`foo\\\\bar`"


"""Tests for East type system."""


class TestPrimitiveTypeConstructors:
    """Tests for primitive type constructors."""

    def test_null_type(self):
        """NullType is an EastType."""
        assert isinstance(NullType, EastType)
        assert NullType.tag == "Null"
        assert NullType.value == null

    def test_boolean_type(self):
        """BooleanType is an EastType."""
        assert isinstance(BooleanType, EastType)
        assert BooleanType.tag == "Boolean"
        assert BooleanType.value == null

    def test_integer_type(self):
        """IntegerType is an EastType."""
        assert isinstance(IntegerType, EastType)
        assert IntegerType.tag == "Integer"
        assert IntegerType.value == null

    def test_float_type(self):
        """FloatType is an EastType."""
        assert isinstance(FloatType, EastType)
        assert FloatType.tag == "Float"
        assert FloatType.value == null

    def test_string_type(self):
        """StringType is an EastType."""
        assert isinstance(StringType, EastType)
        assert StringType.tag == "String"
        assert StringType.value == null

    def test_blob_type(self):
        """BlobType is an EastType."""
        assert isinstance(BlobType, EastType)
        assert BlobType.tag == "Blob"
        assert BlobType.value == null

    def test_datetime_type(self):
        """DateTimeType is an EastType."""
        assert isinstance(DateTimeType, EastType)
        assert DateTimeType.tag == "DateTime"
        assert DateTimeType.value == null

    def test_never_type(self):
        """NeverType is an EastType."""
        assert isinstance(NeverType, EastType)
        assert NeverType.tag == "Never"
        assert NeverType.value == null


class TestContainerTypeConstructors:
    """Tests for container type constructors."""

    def test_array_type(self):
        """ArrayType creates array types."""
        arr_type = ArrayType(IntegerType)
        assert isinstance(arr_type, EastType)
        assert arr_type.tag == "Array"
        assert arr_type.value == IntegerType

    def test_array_of_arrays(self):
        """Nested arrays."""
        arr_type = ArrayType(ArrayType(IntegerType))
        assert arr_type.tag == "Array"
        assert arr_type.value.tag == "Array"
        assert arr_type.value.value == IntegerType

    def test_set_type(self):
        """SetType creates set types."""
        set_type = SetType(StringType)
        assert isinstance(set_type, EastType)
        assert set_type.tag == "Set"
        assert set_type.value == StringType

    def test_dict_type(self):
        """DictType creates dict types."""
        dict_type = DictType(StringType, IntegerType)
        assert isinstance(dict_type, EastType)
        assert dict_type.tag == "Dict"
        # Dict value is a struct with key and value fields
        dict_struct = dict_type.value
        assert dict_struct.key == StringType
        assert dict_struct.value == IntegerType


class TestStructuralTypeConstructors:
    """Tests for structural type constructors."""

    def test_struct_type_from_fields(self):
        """StructType creates struct types."""
        fields = [("name", StringType), ("age", IntegerType)]
        struct_type = StructType(fields)
        assert isinstance(struct_type, EastType)
        assert struct_type.tag == "Struct"
        # Value is a list of field structs
        field_structs = struct_type.value
        assert len(field_structs) == 2
        assert field_structs[0].name == "name"
        assert field_structs[0].type == StringType
        assert field_structs[1].name == "age"
        assert field_structs[1].type == IntegerType

    def test_empty_struct_type(self):
        """Empty struct type."""
        struct_type = StructType([])
        assert struct_type.tag == "Struct"
        assert struct_type.value == []

    def test_variant_type_from_cases(self):
        """VariantType creates variant types."""
        cases = [("Some", IntegerType), ("None", NullType)]
        variant_type = VariantType(cases)
        assert isinstance(variant_type, EastType)
        assert variant_type.tag == "Variant"
        # Value is a list of case structs
        case_structs = variant_type.value
        assert len(case_structs) == 2
        # Cases are sorted by name
        assert case_structs[0].name == "None"
        assert case_structs[0].type == NullType
        assert case_structs[1].name == "Some"
        assert case_structs[1].type == IntegerType

    def test_variant_sorting(self):
        """Variant cases are sorted by name."""
        cases = [("Zebra", IntegerType), ("Apple", StringType), ("Banana", BlobType)]
        variant_type = VariantType(cases)
        case_structs = variant_type.value
        assert case_structs[0].name == "Apple"
        assert case_structs[1].name == "Banana"
        assert case_structs[2].name == "Zebra"


class TestFunctionType:
    """Tests for function types."""

    def test_function_type(self):
        """FunctionType creates function types."""
        func_type = FunctionType([IntegerType, StringType], BooleanType, ["platform1"])
        assert isinstance(func_type, EastType)
        assert func_type.tag == "Function"
        # Value is a struct with inputs, output, platforms
        func_struct = func_type.value
        assert func_struct.inputs == [IntegerType, StringType]
        assert func_struct.output == BooleanType
        assert func_struct.platforms == ["platform1"]

    def test_function_no_inputs(self):
        """Function with no inputs."""
        func_type = FunctionType([], IntegerType, [])
        func_struct = func_type.value
        assert func_struct.inputs == []
        assert func_struct.output == IntegerType


class TestTypeEquality:
    """Tests for type equality."""

    def test_primitive_types_equal(self):
        """Same primitive types are equal."""
        assert IntegerType == IntegerType
        assert StringType == StringType
        assert BooleanType != IntegerType

    def test_array_types_equal(self):
        """Array types equal if element types equal."""
        arr1 = ArrayType(IntegerType)
        arr2 = ArrayType(IntegerType)
        arr3 = ArrayType(StringType)
        assert arr1 == arr2
        assert arr1 != arr3

    def test_dict_types_equal(self):
        """Dict types equal if key and value types equal."""
        dict1 = DictType(StringType, IntegerType)
        dict2 = DictType(StringType, IntegerType)
        dict3 = DictType(IntegerType, StringType)
        assert dict1 == dict2
        assert dict1 != dict3

    def test_struct_types_equal(self):
        """Struct types equal if fields equal."""
        struct1 = StructType([("name", StringType), ("age", IntegerType)])
        struct2 = StructType([("name", StringType), ("age", IntegerType)])
        struct3 = StructType([("name", StringType)])
        assert struct1 == struct2
        assert struct1 != struct3

    def test_struct_field_order_matters(self):
        """Field order matters for struct equality."""
        struct1 = StructType([("name", StringType), ("age", IntegerType)])
        struct2 = StructType([("age", IntegerType), ("name", StringType)])
        assert struct1 != struct2

    def test_variant_types_equal(self):
        """Variant types equal if cases equal (after sorting)."""
        variant1 = VariantType([("Some", IntegerType), ("None", NullType)])
        variant2 = VariantType([("None", NullType), ("Some", IntegerType)])  # Different order
        variant3 = VariantType([("Some", IntegerType)])
        assert variant1 == variant2  # Order doesn't matter (sorted)
        assert variant1 != variant3


class TestTypeHashing:
    """Tests for type hashing."""

    def test_primitive_types_hashable(self):
        """Primitive types are hashable."""
        types_set = {IntegerType, StringType, BooleanType, IntegerType}
        assert len(types_set) == 3  # IntegerType appears twice

    def test_container_types_hashable(self):
        """Container types are hashable."""
        arr1 = ArrayType(IntegerType)
        arr2 = ArrayType(IntegerType)
        arr3 = ArrayType(StringType)
        types_set = {arr1, arr2, arr3}
        assert len(types_set) == 2  # arr1 and arr2 are equal

    def test_struct_types_hashable(self):
        """Struct types are hashable."""
        struct1 = StructType([("name", StringType)])
        struct2 = StructType([("name", StringType)])
        types_set = {struct1, struct2}
        assert len(types_set) == 1


class TestRecursiveTypes:
    """Tests for recursive types."""

    def test_recursive_type_builder(self):
        """recursive_type builds recursive types."""
        # Create a linked list type: List = Variant { Cons(Integer, List), Nil }
        list_type = recursive_type(
            lambda self: VariantType(
                [
                    ("Cons", StructType([("value", IntegerType), ("next", self)])),
                    ("Nil", NullType),
                ]
            )
        )
        assert isinstance(list_type, EastType)
        assert list_type.tag == "Variant"
        cases = list_type.value
        assert len(cases) == 2

        # Check Cons case has recursive reference
        cons_case = cases[0]  # Cons (alphabetically first)
        assert cons_case.name == "Cons"
        cons_struct_type = cons_case.type
        cons_fields = cons_struct_type.value
        assert len(cons_fields) == 2
        assert cons_fields[0].name == "value"
        assert cons_fields[0].type == IntegerType
        assert cons_fields[1].name == "next"
        # The recursive reference
        next_type = cons_fields[1].type
        assert next_type.tag == "Recursive"
        # The value is a RecursiveTypeMarker (new marker-based approach)
        assert isinstance(next_type.value, RecursiveTypeMarker)

    def test_recursive_array(self):
        """Recursive array type."""
        # Array of Array of ... Integer
        nested_arr = recursive_type(lambda self: ArrayType(self))
        assert nested_arr.tag == "Array"
        inner = nested_arr.value
        assert inner.tag == "Recursive"
        # The value is a RecursiveTypeMarker (new marker-based approach)
        assert isinstance(inner.value, RecursiveTypeMarker)

    def test_nested_recursive(self):
        """Nested recursive types."""
        # Tree: Variant { Node(Integer, Array<Tree>), Leaf }
        tree_type = recursive_type(
            lambda self: VariantType(
                [
                    (
                        "Node",
                        StructType([("value", IntegerType), ("children", ArrayType(self))]),
                    ),
                    ("Leaf", NullType),
                ]
            )
        )
        assert tree_type.tag == "Variant"
        cases = tree_type.value

        # Check Node case
        # cases[0] is Leaf (alphabetically first)
        node_case = cases[1]
        assert node_case.name == "Node"
        node_struct = node_case.type
        node_fields = node_struct.value
        children_field = node_fields[1]
        assert children_field.name == "children"
        # children is Array<Recursive(0)>
        children_type = children_field.type
        assert children_type.tag == "Array"
        assert children_type.value.tag == "Recursive"


class TestEastTypeType:
    """Tests for EastTypeType - the type of types."""

    def test_east_type_type_exists(self):
        """EastTypeType is defined."""
        assert EastTypeType is not None
        assert isinstance(EastTypeType, EastType)

    def test_east_type_type_is_variant(self):
        """EastTypeType is a variant."""
        assert EastTypeType.tag == "Variant"

    def test_east_type_has_all_cases(self):
        """EastTypeType has all type cases."""
        cases = EastTypeType.value
        case_names = [case.name for case in cases]
        expected_cases = [
            "Array",
            "Blob",
            "Boolean",
            "DateTime",
            "Dict",
            "Float",
            "Function",
            "Integer",
            "Never",
            "Null",
            "Recursive",
            "Ref",
            "Set",
            "String",
            "Struct",
            "Variant",
        ]
        assert sorted(case_names) == sorted(expected_cases)

    def test_integer_type_has_east_type(self):
        """IntegerType has _east_type of EastTypeType."""
        assert IntegerType._east_type == EastTypeType

    def test_array_type_has_east_type(self):
        """ArrayType instances have _east_type of EastTypeType."""
        arr = ArrayType(IntegerType)
        assert arr._east_type == EastTypeType


class TestComplexTypes:
    """Tests for complex type combinations."""

    def test_array_of_structs(self):
        """Array of structs."""
        person_type = StructType([("name", StringType), ("age", IntegerType)])
        people_type = ArrayType(person_type)
        assert people_type.tag == "Array"
        assert people_type.value == person_type

    def test_dict_of_variants(self):
        """Dict with variant values."""
        option_type = VariantType([("Some", IntegerType), ("None", NullType)])
        dict_type = DictType(StringType, option_type)
        assert dict_type.tag == "Dict"
        dict_struct = dict_type.value
        assert dict_struct.value == option_type

    def test_nested_containers(self):
        """Nested containers."""
        # Dict<String, Array<Set<Integer>>>
        inner = SetType(IntegerType)
        middle = ArrayType(inner)
        outer = DictType(StringType, middle)
        assert outer.tag == "Dict"
        outer_struct = outer.value
        assert outer_struct.value.tag == "Array"
        assert outer_struct.value.value.tag == "Set"
        assert outer_struct.value.value.value == IntegerType

    def test_function_with_struct_params(self):
        """Function with struct parameters."""
        person_type = StructType([("name", StringType), ("age", IntegerType)])
        func_type = FunctionType([person_type], StringType, ["greet"])
        assert func_type.tag == "Function"
        func_struct = func_type.value
        assert func_struct.inputs[0] == person_type
        assert func_struct.output == StringType


class TestRuntimeTypeCreation:
    """Tests for creating runtime type instances."""

    def test_create_struct_from_type(self):
        """Create struct instances from _StructTypeClass."""
        # First create the EastType representing a struct type
        _person_east_type = StructType([("name", StringType), ("age", IntegerType)])

        # Then create a runtime _StructTypeClass from the field specs
        person_runtime_type = _StructTypeClass((("name", StringType), ("age", IntegerType)))

        # Create an instance
        person = person_runtime_type.create(name="Alice", age=30)
        assert person.name == "Alice"
        assert person.age == 30

    def test_create_variant_from_type(self):
        """Create variant instances from _VariantTypeClass."""
        # Create the EastType representing an option type
        _option_east_type = VariantType([("Some", IntegerType), ("None", NullType)])

        # Create a runtime _VariantTypeClass
        option_runtime_type = _VariantTypeClass((("Some", IntegerType), ("None", NullType)))

        # Create instances
        some = option_runtime_type.create("Some", 42)
        none = option_runtime_type.create("None")
        assert some.tag == "Some"
        assert some.value == 42
        assert none.tag == "None"


class TestTypeRepr:
    """Tests for type representation."""

    def test_primitive_type_repr(self):
        """Primitive types have readable repr."""
        assert repr(IntegerType) == ".Integer"
        assert repr(StringType) == ".String"

    def test_array_type_repr(self):
        """Array type repr."""
        arr = ArrayType(IntegerType)
        assert repr(arr) == ".Array .Integer"

    def test_nested_type_repr(self):
        """Nested type repr."""
        arr = ArrayType(ArrayType(IntegerType))
        assert repr(arr) == ".Array .Array .Integer"


class TestTypeOf:
    """Tests for type_of function."""

    def test_null(self):
        """type_of null."""
        assert type_of(null) == NullType
        assert type_of(None) == NullType

    def test_boolean(self):
        """type_of boolean."""
        assert type_of(True) == BooleanType
        assert type_of(False) == BooleanType

    def test_integer(self):
        """type_of integer."""
        assert type_of(42) == IntegerType
        assert type_of(0) == IntegerType
        assert type_of(-100) == IntegerType

    def test_float(self):
        """type_of float."""
        assert type_of(3.14) == FloatType
        assert type_of(0.0) == FloatType
        assert type_of(float("nan")) == FloatType

    def test_string(self):
        """type_of string."""
        assert type_of("hello") == StringType
        assert type_of("") == StringType

    def test_blob(self):
        """type_of blob."""
        b = Blob(b"test")
        assert type_of(b) == BlobType

    def test_datetime(self):
        """type_of datetime."""
        dt = datetime.now(UTC)
        assert type_of(dt) == DateTimeType

    def test_array(self):
        """type_of array."""
        from east.types.containers import EastArray

        arr = EastArray(IntegerType, [1, 2, 3])
        arr_type = type_of(arr)
        assert arr_type.tag == "Array"
        assert arr_type.value == IntegerType

    def test_set(self):
        """type_of set."""
        from east.types.containers import EastSet

        s = EastSet(StringType, ["a", "b"])
        set_type = type_of(s)
        assert set_type.tag == "Set"
        assert set_type.value == StringType

    def test_dict(self):
        """type_of dict."""
        from east.types.containers import EastDict

        d = EastDict(StringType, IntegerType, {"a": 1})
        dict_type = type_of(d)
        assert dict_type.tag == "Dict"
        assert dict_type.value.key == StringType
        assert dict_type.value.value == IntegerType

    def test_struct(self):
        """type_of struct."""
        person_type = _StructTypeClass((("name", StringType), ("age", IntegerType)))
        person = person_type.create(name="Alice", age=30)
        assert type_of(person) == person_type

    def test_variant(self):
        """type_of variant."""
        option_type = _VariantTypeClass((("Some", IntegerType), ("None", NullType)))
        some = option_type.create("Some", 42)
        assert type_of(some) == option_type

    def test_east_type(self):
        """type_of EastType."""
        assert type_of(IntegerType) == EastTypeType
        assert type_of(ArrayType(StringType)) == EastTypeType
        assert type_of(EastTypeType) == EastTypeType

    def test_unknown_type(self):
        """type_of raises TypeError for unknown types."""
        import pytest

        with pytest.raises(TypeError, match="Unknown East type"):
            type_of(object())


class TestEastTypeOf:
    """Tests for east_type_of function that infers types from raw Python values."""

    def test_should_infer_primitive_types(self):
        """should infer primitive types."""
        from east.types.type_system import east_type_of

        assert east_type_of(None) == NullType
        assert east_type_of(True) == BooleanType
        assert east_type_of(42) == IntegerType
        assert east_type_of(3.14) == FloatType
        assert east_type_of("hello") == StringType

    def test_should_infer_date_type(self):
        """should infer Date type."""

        dt = datetime.now(UTC)
        assert east_type_of(dt) == DateTimeType

    def test_should_infer_blob_type(self):
        """should infer Blob type."""

        assert east_type_of(b"bytes") == BlobType
        assert east_type_of(Blob(b"bytes")) == BlobType

    def test_should_infer_array_types(self):
        """should infer array types."""

        typ = east_type_of([1, 2, 3])
        assert typ.tag == "Array"
        assert typ.value == IntegerType

    def test_should_infer_struct_types(self):
        """should infer struct types."""

        typ = east_type_of({"x": 42, "y": "hello"})
        assert typ.tag == "Struct"
        fields = typ.value
        field_dict = {f.name: f.type for f in fields}
        assert field_dict["x"] == IntegerType
        assert field_dict["y"] == StringType

    def test_should_throw_for_functions(self):
        """should throw for functions."""
        import pytest

        with pytest.raises(
            TypeError, match="JavaScript/Python functions cannot be converted to East functions"
        ):
            east_type_of(lambda: None)

    def test_should_throw_for_unknown_values(self):
        """should throw for unknown values."""
        import pytest

        # Symbols in JS, custom objects in Python
        with pytest.raises(TypeError, match="Cannot determine East type for value"):
            east_type_of(object())
