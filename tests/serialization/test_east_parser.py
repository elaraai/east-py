"""Tests for East text format parser."""

import math
from datetime import datetime

import pytest

from east.serialization.east_parser import ParseError, parse_east
from east.types.primitives import Blob, null
from east.types.type_system import (
    ArrayType,
    BlobType,
    BooleanType,
    DateTimeType,
    DictType,
    FloatType,
    IntegerType,
    NullType,
    SetType,
    StringType,
    StructTypeFromFields,
    VariantTypeFromCases,
)


class TestPrimitives:
    """Tests for parsing primitive values."""

    def test_null(self):
        """Parse null."""
        result = parse_east(NullType, "null")
        assert result == null

    def test_true(self):
        """Parse true."""
        result = parse_east(BooleanType, "true")
        assert result is True

    def test_false(self):
        """Parse false."""
        result = parse_east(BooleanType, "false")
        assert result is False

    def test_integer(self):
        """Parse integer."""
        result = parse_east(IntegerType, "42")
        assert result == 42

    def test_negative_integer(self):
        """Parse negative integer."""
        result = parse_east(IntegerType, "-123")
        assert result == -123

    def test_float(self):
        """Parse float."""
        result = parse_east(FloatType, "3.14")
        assert result == 3.14

    def test_nan(self):
        """Parse NaN."""
        result = parse_east(FloatType, "NaN")
        assert math.isnan(result)

    def test_infinity(self):
        """Parse Infinity."""
        result = parse_east(FloatType, "Infinity")
        assert result == float("inf")

    def test_string(self):
        """Parse string."""
        result = parse_east(StringType, '"hello"')
        assert result == "hello"

    def test_string_with_escapes(self):
        """Parse string with escapes."""
        result = parse_east(StringType, r'"line1\nline2"')
        assert result == "line1\nline2"

    def test_blob(self):
        """Parse blob."""
        result = parse_east(BlobType, "0xabcd")
        assert isinstance(result, Blob)
        assert result == Blob(b"\xab\xcd")

    def test_empty_blob(self):
        """Parse empty blob."""
        result = parse_east(BlobType, "0x")
        assert result == Blob(b"")

    def test_datetime(self):
        """Parse datetime."""
        result = parse_east(DateTimeType, "2024-01-15T10:30:00Z")
        assert isinstance(result, datetime)
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15


class TestArrays:
    """Tests for parsing arrays."""

    def test_empty_array(self):
        """Parse empty array."""
        result = parse_east(ArrayType(IntegerType), "[]")
        assert len(result) == 0
        assert result.element_type == IntegerType

    def test_integer_array(self):
        """Parse integer array."""
        result = parse_east(ArrayType(IntegerType), "[1, 2, 3]")
        assert list(result) == [1, 2, 3]
        assert result.element_type == IntegerType

    def test_string_array(self):
        """Parse string array."""
        result = parse_east(ArrayType(StringType), '["a", "b", "c"]')
        assert list(result) == ["a", "b", "c"]

    def test_nested_array(self):
        """Parse nested array."""
        result = parse_east(ArrayType(ArrayType(IntegerType)), "[[1, 2], [3, 4]]")
        assert len(result) == 2
        assert list(result[0]) == [1, 2]
        assert list(result[1]) == [3, 4]

    def test_array_trailing_comma(self):
        """Parse array with trailing comma."""
        result = parse_east(ArrayType(IntegerType), "[1, 2, 3,]")
        assert list(result) == [1, 2, 3]


class TestSets:
    """Tests for parsing sets."""

    def test_empty_set(self):
        """Parse empty set."""
        result = parse_east(SetType(IntegerType), "{}")
        assert len(result) == 0
        assert result.element_type == IntegerType

    def test_integer_set(self):
        """Parse integer set."""
        result = parse_east(SetType(IntegerType), "{3, 1, 2}")
        # Sets are sorted
        assert list(result) == [1, 2, 3]

    def test_string_set(self):
        """Parse string set."""
        result = parse_east(SetType(StringType), '{"c", "a", "b"}')
        assert list(result) == ["a", "b", "c"]

    def test_set_deduplication(self):
        """Parse set with duplicates."""
        result = parse_east(SetType(IntegerType), "{1, 2, 1, 3, 2}")
        assert list(result) == [1, 2, 3]


class TestDicts:
    """Tests for parsing dicts."""

    def test_empty_dict(self):
        """Parse empty dict."""
        result = parse_east(DictType(StringType, IntegerType), "{:}")
        assert len(result) == 0
        assert result.key_type == StringType
        assert result.value_type == IntegerType

    def test_string_int_dict(self):
        """Parse string to int dict."""
        result = parse_east(DictType(StringType, IntegerType), '{"a": 1, "b": 2}')
        assert result["a"] == 1
        assert result["b"] == 2

    def test_dict_sorted_keys(self):
        """Parse dict with unsorted keys."""
        result = parse_east(DictType(StringType, IntegerType), '{"c": 3, "a": 1, "b": 2}')
        # Keys should be sorted
        assert list(result.keys()) == ["a", "b", "c"]


class TestStructs:
    """Tests for parsing structs."""

    def test_empty_struct(self):
        """Parse empty struct."""
        struct_type = StructTypeFromFields([])
        result = parse_east(struct_type, "()")
        assert result._values == ()

    def test_simple_struct(self):
        """Parse simple struct."""
        struct_type = StructTypeFromFields([("name", StringType), ("age", IntegerType)])
        result = parse_east(struct_type, '(name="Alice", age=30)')
        assert result.name == "Alice"
        assert result.age == 30

    def test_struct_field_order(self):
        """Parse struct with different field order."""
        struct_type = StructTypeFromFields([("name", StringType), ("age", IntegerType)])
        result = parse_east(struct_type, '(age=30, name="Alice")')
        assert result.name == "Alice"
        assert result.age == 30

    def test_nested_struct(self):
        """Parse nested struct."""
        inner_type = StructTypeFromFields([("x", IntegerType), ("y", IntegerType)])
        outer_type = StructTypeFromFields([("point", inner_type)])
        result = parse_east(outer_type, "(point=(x=10, y=20))")
        assert result.point.x == 10
        assert result.point.y == 20


class TestVariants:
    """Tests for parsing variants."""

    def test_variant_with_null_value(self):
        """Parse variant with null value."""
        variant_type = VariantTypeFromCases([("Some", IntegerType), ("None", NullType)])
        result = parse_east(variant_type, ".None")
        assert result.tag == "None"
        assert result.value == null

    def test_variant_with_value(self):
        """Parse variant with value."""
        variant_type = VariantTypeFromCases([("Some", IntegerType), ("None", NullType)])
        result = parse_east(variant_type, ".Some 42")
        assert result.tag == "Some"
        assert result.value == 42

    def test_variant_with_struct(self):
        """Parse variant with struct value."""
        struct_type = StructTypeFromFields([("x", IntegerType), ("y", IntegerType)])
        variant_type = VariantTypeFromCases([("Point", struct_type), ("None", NullType)])
        result = parse_east(variant_type, ".Point (x=10, y=20)")
        assert result.tag == "Point"
        assert result.value.x == 10
        assert result.value.y == 20


class TestComplexTypes:
    """Tests for complex nested types."""

    def test_array_of_structs(self):
        """Parse array of structs."""
        person_type = StructTypeFromFields([("name", StringType), ("age", IntegerType)])
        array_type = ArrayType(person_type)
        result = parse_east(array_type, '[(name="Alice", age=30), (name="Bob", age=25)]')
        assert len(result) == 2
        assert result[0].name == "Alice"
        assert result[1].name == "Bob"

    def test_dict_of_arrays(self):
        """Parse dict with array values."""
        dict_type = DictType(StringType, ArrayType(IntegerType))
        result = parse_east(dict_type, '{"a": [1, 2], "b": [3, 4]}')
        assert list(result["a"]) == [1, 2]
        assert list(result["b"]) == [3, 4]

    def test_variant_in_array(self):
        """Parse array of variants."""
        option_type = VariantTypeFromCases([("Some", IntegerType), ("None", NullType)])
        array_type = ArrayType(option_type)
        result = parse_east(array_type, "[.Some 1, .None, .Some 3]")
        assert len(result) == 3
        assert result[0].tag == "Some"
        assert result[1].tag == "None"
        assert result[2].tag == "Some"


class TestWhitespace:
    """Tests for whitespace handling."""

    def test_whitespace_ignored(self):
        """Parse with extra whitespace."""
        result = parse_east(IntegerType, "  42  ")
        assert result == 42

    def test_newlines_ignored(self):
        """Parse with newlines."""
        result = parse_east(ArrayType(IntegerType), "[\n  1,\n  2,\n  3\n]")
        assert list(result) == [1, 2, 3]

    def test_comments_ignored(self):
        """Parse with comments."""
        result = parse_east(
            ArrayType(IntegerType),
            """[
            1,  # first item
            2,  # second item
            3   # third item
        ]""",
        )
        assert list(result) == [1, 2, 3]


class TestErrors:
    """Tests for error handling."""

    def test_wrong_type(self):
        """Parse wrong type raises error."""
        with pytest.raises(ParseError):
            parse_east(IntegerType, '"string"')

    def test_unclosed_array(self):
        """Unclosed array raises error."""
        with pytest.raises(ParseError):
            parse_east(ArrayType(IntegerType), "[1, 2, 3")

    def test_unknown_field(self):
        """Unknown struct field raises error."""
        struct_type = StructTypeFromFields([("name", StringType)])
        with pytest.raises(ParseError, match="Unknown field"):
            parse_east(struct_type, '(name="Alice", age=30)')

    def test_unknown_variant_case(self):
        """Unknown variant case raises error."""
        variant_type = VariantTypeFromCases([("Some", IntegerType)])
        with pytest.raises(ParseError, match="Unknown variant case"):
            parse_east(variant_type, ".None")

    def test_extra_tokens(self):
        """Extra tokens raise error."""
        with pytest.raises(ParseError, match="Unexpected token"):
            parse_east(IntegerType, "42 43")


class TestRoundTrip:
    """Tests for parsing printed values."""

    def test_integer_roundtrip(self):
        """Parse and print integer."""
        result = parse_east(IntegerType, "42")
        assert result == 42

    def test_array_roundtrip(self):
        """Parse and print array."""
        result = parse_east(ArrayType(IntegerType), "[1, 2, 3]")
        assert list(result) == [1, 2, 3]

    def test_struct_roundtrip(self):
        """Parse and print struct."""
        struct_type = StructTypeFromFields([("name", StringType), ("age", IntegerType)])
        result = parse_east(struct_type, '(name="Alice", age=30)')
        assert result.name == "Alice"
        assert result.age == 30
