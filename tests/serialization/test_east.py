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
    StructType,
    VariantType,
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
        """Parse string with escapes - only \\ and \\\" are supported."""
        # East text format does not support \n - it should error
        with pytest.raises(ValueError, match=r"[Uu]nsupported escape"):
            parse_east(StringType, r'"line1\nline2"')

        # But backslash and quote escapes work
        result = parse_east(StringType, r'"path\\to\\file"')
        assert result == r"path\to\file"

        result = parse_east(StringType, r'"say \"hello\""')
        assert result == 'say "hello"'

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
        """Parse array with trailing comma - should error."""
        from east.serialization.east_parser import ParseError

        with pytest.raises(ParseError, match=r"[Tt]railing comma"):
            parse_east(ArrayType(IntegerType), "[1, 2, 3,]")


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
        struct_type = StructType([])
        result = parse_east(struct_type, "()")
        assert result._values == ()

    def test_simple_struct(self):
        """Parse simple struct."""
        struct_type = StructType([("name", StringType), ("age", IntegerType)])
        result = parse_east(struct_type, '(name="Alice", age=30)')
        assert result.name == "Alice"
        assert result.age == 30

    def test_struct_field_order(self):
        """Parse struct with different field order."""
        struct_type = StructType([("name", StringType), ("age", IntegerType)])
        result = parse_east(struct_type, '(age=30, name="Alice")')
        assert result.name == "Alice"
        assert result.age == 30

    def test_nested_struct(self):
        """Parse nested struct."""
        inner_type = StructType([("x", IntegerType), ("y", IntegerType)])
        outer_type = StructType([("point", inner_type)])
        result = parse_east(outer_type, "(point=(x=10, y=20))")
        assert result.point.x == 10
        assert result.point.y == 20


class TestVariants:
    """Tests for parsing variants."""

    def test_variant_with_null_value(self):
        """Parse variant with null value."""
        variant_type = VariantType([("Some", IntegerType), ("None", NullType)])
        result = parse_east(variant_type, ".None")
        assert result.tag == "None"
        assert result.value == null

    def test_variant_with_value(self):
        """Parse variant with value."""
        variant_type = VariantType([("Some", IntegerType), ("None", NullType)])
        result = parse_east(variant_type, ".Some 42")
        assert result.tag == "Some"
        assert result.value == 42

    def test_variant_with_struct(self):
        """Parse variant with struct value."""
        struct_type = StructType([("x", IntegerType), ("y", IntegerType)])
        variant_type = VariantType([("Point", struct_type), ("None", NullType)])
        result = parse_east(variant_type, ".Point (x=10, y=20)")
        assert result.tag == "Point"
        assert result.value.x == 10
        assert result.value.y == 20


class TestComplexTypes:
    """Tests for complex nested types."""

    def test_array_of_structs(self):
        """Parse array of structs."""
        person_type = StructType([("name", StringType), ("age", IntegerType)])
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
        option_type = VariantType([("Some", IntegerType), ("None", NullType)])
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
        struct_type = StructType([("name", StringType)])
        with pytest.raises(ParseError, match="unknown field"):
            parse_east(struct_type, '(name="Alice", age=30)')

    def test_unknown_variant_case(self):
        """Unknown variant case raises error."""
        variant_type = VariantType([("Some", IntegerType)])
        with pytest.raises(ParseError, match="unknown variant case"):
            parse_east(variant_type, ".None")

    def test_extra_tokens(self):
        """Extra tokens raise error."""
        with pytest.raises(ParseError, match="unexpected token"):
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
        struct_type = StructType([("name", StringType), ("age", IntegerType)])
        result = parse_east(struct_type, '(name="Alice", age=30)')
        assert result.name == "Alice"
        assert result.age == 30


"""Additional tests for East text format edge cases and error handling.

Covers test cases from east.spec.ts that aren't yet in test_east_parser.py or test_east_printer.py
"""


from east.serialization.east_printer import print_east
from east.types.containers import EastArray, EastDict, EastSet

# =============================================================================
# Test Suite: parseFor - Collection Trailing Commas
# =============================================================================


class TestCollectionTrailingCommas:
    """Tests for trailing comma error handling."""

    def test_should_error_on_array_with_trailing_comma(self):
        """Should error on array with trailing comma."""
        array_type = ArrayType(StringType)

        with pytest.raises(ParseError):
            parse_east(array_type, '["a", "b",]')

    def test_should_error_on_set_with_trailing_comma(self):
        """Should error on set with trailing comma."""
        set_type = SetType(StringType)

        with pytest.raises(ParseError):
            parse_east(set_type, '{"a", "b",}')

    def test_should_error_on_dict_with_trailing_comma(self):
        """Should error on dict with trailing comma."""
        dict_type = DictType(StringType, IntegerType)

        with pytest.raises(ParseError):
            parse_east(dict_type, '{"a": 1, "b": 2,}')


# =============================================================================
# Test Suite: parseFor - Struct Error Handling
# =============================================================================


class TestStructErrorHandling:
    """Tests for struct parsing error cases."""

    def test_should_error_on_missing_required_field(self):
        """Should error on missing required field."""
        struct_type = StructType(
            [
                ("name", StringType),
                ("age", IntegerType),
            ]
        )

        with pytest.raises(ParseError, match=r"Expected 2 fields"):
            parse_east(struct_type, '(name="Alice")')

    def test_should_error_on_unknown_field(self):
        """Should error on unknown field."""
        struct_type = StructType(
            [
                ("name", StringType),
            ]
        )

        with pytest.raises(ParseError, match=r"unknown field|unexpected"):
            parse_east(struct_type, '(name="Alice", age=30)')

    def test_should_parse_struct_with_quoted_field_names(self):
        """Should parse struct with quoted field names (backticks)."""
        struct_type = StructType(
            [
                ("field-with-dash", StringType),
                ("123numeric", IntegerType),
            ]
        )

        result = parse_east(struct_type, '(`field-with-dash`="value", `123numeric`=42)')
        assert getattr(result, "field-with-dash") == "value"
        assert getattr(result, "123numeric") == 42


# =============================================================================
# Test Suite: parseFor - Variant Error Handling
# =============================================================================


class TestVariantErrorHandling:
    """Tests for variant parsing error cases."""

    def test_should_parse_nullary_variant_with_null_provided(self):
        """Should parse nullary variant with null provided."""
        from east.types.primitives import null
        from east.types.type_system import NullType

        variant_type = VariantType(
            [
                ("success", NullType),
                ("error", StringType),
            ]
        )

        result = parse_east(variant_type, ".success null")
        assert result.tag == "success"
        assert result.value == null

    def test_should_error_on_unknown_variant_case(self):
        """Should error on unknown variant case."""
        variant_type = VariantType(
            [
                ("none", IntegerType),
                ("some", IntegerType),
            ]
        )

        with pytest.raises(ParseError, match=r"unknown variant"):
            parse_east(variant_type, ".other 42")

    def test_should_error_on_variant_case_with_incorrect_payload(self):
        """Should error when variant payload doesn't match expected type."""
        variant_type = VariantType(
            [
                ("int_case", IntegerType),
            ]
        )

        with pytest.raises(ParseError):
            parse_east(variant_type, 'int_case "string"')  # Should be integer

    def test_should_error_when_data_provided_for_nullary_case(self):
        """Should error when data is provided for nullary variant case."""
        from east.types.type_system import NullType

        variant_type = VariantType(
            [
                ("success", NullType),
                ("error", StringType),
            ]
        )

        # Providing "unexpected" data for nullary .success case
        with pytest.raises(ParseError, match=r"expected null|expected end|unexpected"):
            parse_east(variant_type, '.success "unexpected"')

    def test_should_error_when_no_data_provided_for_data_case(self):
        """Should error when no data is provided for variant case that requires data."""
        from east.types.type_system import NullType

        variant_type = VariantType(
            [
                ("success", NullType),
                ("error", StringType),
            ]
        )

        # Not providing required string data for .error case
        with pytest.raises(ParseError, match=r"expected|end of input|unexpected end"):
            parse_east(variant_type, ".error")


# =============================================================================
# Test Suite: parseFor - Complex Nested Structures
# =============================================================================


class TestComplexNestedStructures:
    """Tests for complex nested structure parsing."""

    def test_should_parse_complex_nested_structure(self):
        """Should parse complex nested structure."""
        user_type = StructType(
            [
                ("name", StringType),
                ("age", IntegerType),
            ]
        )

        struct_type = StructType(
            [
                ("users", ArrayType(user_type)),
                ("count", IntegerType),
            ]
        )

        text = '(users=[(name="Alice", age=30), (name="Bob", age=25)], count=2)'
        result = parse_east(struct_type, text)

        assert result.count == 2
        assert len(result.users) == 2
        assert result.users[0].name == "Alice"
        assert result.users[1].age == 25

    def test_should_parse_deeply_nested_structure_with_multiple_collection_types(self):
        """Should parse deeply nested structure with multiple collection types."""
        inner_struct = StructType(
            [
                ("values", SetType(IntegerType)),
            ]
        )

        outer_type = DictType(StringType, ArrayType(inner_struct))

        text = '{"key": [(values={1, 2, 3}), (values={4, 5})]}'
        result = parse_east(outer_type, text)

        assert "key" in result
        assert len(result["key"]) == 2
        assert 1 in result["key"][0].values


# =============================================================================
# Test Suite: printFor and Round-trip - Complex Structures
# =============================================================================


class TestComplexRoundTrip:
    """Tests for complex structure round-tripping."""

    def test_nested_arrays_should_round_trip(self):
        """Nested arrays should round-trip."""
        type_val = ArrayType(ArrayType(IntegerType))
        value = EastArray(
            ArrayType(IntegerType),
            [
                EastArray(IntegerType, [1, 2]),
                EastArray(IntegerType, [3, 4, 5]),
            ],
        )

        printed = print_east(value, type_val)
        parsed = parse_east(type_val, printed)

        assert len(parsed) == 2
        assert list(parsed[0]) == [1, 2]
        assert list(parsed[1]) == [3, 4, 5]

    def test_struct_with_array_field_should_round_trip(self):
        """Struct with array field should round-trip."""
        struct_type = StructType(
            [
                ("name", StringType),
                ("scores", ArrayType(IntegerType)),
            ]
        )

        value = {
            "name": "test",
            "scores": EastArray(IntegerType, [100, 95, 87]),
        }

        printed = print_east(value, struct_type)
        parsed = parse_east(struct_type, printed)

        assert parsed.name == "test"
        assert list(parsed.scores) == [100, 95, 87]

    def test_deeply_nested_structure_should_round_trip(self):
        """Deeply nested structure should round-trip."""
        inner_type = StructType(
            [
                ("x", IntegerType),
                ("y", IntegerType),
            ]
        )

        middle_type = StructType(
            [
                ("point", inner_type),
                ("label", StringType),
            ]
        )

        outer_type = ArrayType(middle_type)

        value = EastArray(
            middle_type,
            [
                {"point": {"x": 1, "y": 2}, "label": "a"},
                {"point": {"x": 3, "y": 4}, "label": "b"},
            ],
        )

        printed = print_east(value, outer_type)
        parsed = parse_east(outer_type, printed)

        assert len(parsed) == 2
        assert parsed[0].point.x == 1
        assert parsed[1].label == "b"


# =============================================================================
# Test Suite: printFor - Edge Cases
# =============================================================================


class TestPrintEdgeCases:
    """Tests for printing edge cases."""

    def test_float_formatting_special_values(self):
        """Float formatting should handle special values correctly."""
        # Infinity
        printed = print_east(float("inf"), FloatType)
        assert printed == "Infinity"

        # Negative infinity
        printed = print_east(float("-inf"), FloatType)
        assert printed == "-Infinity"

        # NaN
        printed = print_east(float("nan"), FloatType)
        assert printed == "NaN"

    def test_float_formatting_preserves_precision(self):
        """Float formatting should preserve necessary precision."""
        # Decimal values with many digits
        printed = print_east(0.123456789, FloatType)
        parsed = parse_east(FloatType, printed)
        assert abs(parsed - 0.123456789) < 1e-15

        # Large numbers
        printed = print_east(1e10, FloatType)
        parsed = parse_east(FloatType, printed)
        assert parsed == 1e10

        # Precise decimal values
        printed = print_east(3.141592653589793, FloatType)
        parsed = parse_east(FloatType, printed)
        assert abs(parsed - 3.141592653589793) < 1e-15

        # Small positive numbers
        printed = print_east(0.00001, FloatType)
        parsed = parse_east(FloatType, printed)
        assert abs(parsed - 0.00001) < 1e-15

    def test_string_escaping(self):
        """String should properly escape special characters."""
        # Backslash
        value = "path\\to\\file"
        printed = print_east(value, StringType)
        assert "\\\\" in printed  # Should be escaped

        # Quotes
        value = 'say "hello"'
        printed = print_east(value, StringType)
        assert '\\"' in printed  # Should be escaped

        # Newlines and tabs are NOT escaped - they appear literally
        value = "line1\nline2"
        printed = print_east(value, StringType)
        assert "\n" in printed  # Actual newline, not \\n
        assert "\\n" not in printed  # Not escaped

    def test_empty_collections_formatting(self):
        """Empty collections should format correctly."""
        # Empty array
        value = EastArray(IntegerType, [])
        printed = print_east(value, ArrayType(IntegerType))
        assert printed == "[]"

        # Empty set
        value = EastSet(StringType, set())
        printed = print_east(value, SetType(StringType))
        assert printed == "{}"

        # Empty dict - uses {:} to distinguish from empty set
        value = EastDict(StringType, IntegerType, {})
        printed = print_east(value, DictType(StringType, IntegerType))
        assert printed == "{:}"


# =============================================================================
# Test Suite: parseFor - Error Messages
# =============================================================================


class TestErrorMessages:
    """Tests for proper error messages."""

    def test_should_return_error_for_type_mismatch(self):
        """Should return error for type mismatch."""
        with pytest.raises(ParseError):
            parse_east(IntegerType, '"not an integer"')

    def test_should_return_error_for_malformed_input(self):
        """Should return error for malformed input."""
        with pytest.raises((ParseError, ValueError)):
            parse_east(StringType, '"unterminated string')

    def test_should_return_error_for_extra_tokens(self):
        """Should return error for extra tokens after value."""
        with pytest.raises(ParseError, match=r"unexpected token"):
            parse_east(IntegerType, "42 extra")


# =============================================================================
# Test Suite: String Parsing Edge Cases
# =============================================================================


class TestStringParsingEdgeCases:
    """Tests for string parsing edge cases."""

    def test_should_parse_strings_with_basic_content(self):
        """Should parse strings with various content."""
        # Alphanumeric
        result = parse_east(StringType, '"Hello123"')
        assert result == "Hello123"

        # With spaces
        result = parse_east(StringType, '"Hello World"')
        assert result == "Hello World"

        # Empty string
        result = parse_east(StringType, '""')
        assert result == ""

    def test_should_error_on_unsupported_escape_sequence_newline(self):
        """Should error on unsupported escape sequence \\n."""
        # East text format doesn't support \n - must use actual newlines
        with pytest.raises(ValueError, match=r"[Uu]nsupported escape"):
            parse_east(StringType, '"line1\\nline2"')

    def test_should_error_on_unsupported_escape_sequence_tab(self):
        """Should error on unsupported escape sequence \\t."""
        # East text format doesn't support \t - must use actual tabs
        with pytest.raises(ValueError, match=r"[Uu]nsupported escape"):
            parse_east(StringType, '"tab\\there"')


"""Advanced tests for East text format - Never, Function, Aliases, Recursive types.

These tests cover the remaining test cases from east.spec.ts that weren't
yet ported to Python.
"""


from east.types.type_system import (
    FunctionType,
    NeverType,
    _StructTypeClass,
    _VariantTypeClass,
    recursive_type,
)

# =============================================================================
# Test Suite: Never Type Handling
# =============================================================================


class TestNeverType:
    """Tests for Never type handling."""

    def test_should_throw_when_printing_never_type(self):
        """Should throw when printing Never type."""
        with pytest.raises(
            (ValueError, TypeError), match=r"[Aa]ttempted to print|Cannot print|Never"
        ):
            print_east(None, NeverType)

    def test_should_throw_when_parsing_never_type(self):
        """Should throw when parsing Never type."""
        with pytest.raises(
            (ParseError, ValueError), match=r"[Aa]ttempted to parse|Cannot parse|Never"
        ):
            parse_east(NeverType, "")


# =============================================================================
# Test Suite: Function Type Handling
# =============================================================================


class TestFunctionType:
    """Tests for Function type handling."""

    def test_should_print_function_type_as_lambda(self):
        """Should print Function type as λ."""
        func_type = FunctionType([], IntegerType, [])

        # Create a simple function
        def example_func():
            return 42

        result = print_east(example_func, func_type)
        assert result == "λ"

    def test_should_throw_when_parsing_function_type(self):
        """Should throw when creating parser for Function type."""
        func_type = FunctionType([], IntegerType, [])

        with pytest.raises(
            (ValueError, NotImplementedError, ParseError), match=r"Cannot parse|Function"
        ):
            parse_east(func_type, "λ")


# =============================================================================
# Test Suite: Alias Detection
# =============================================================================


class TestAliasDetection:
    """Tests for alias detection in printing.

    These tests verify that shared references are detected and printed using
    relative reference syntax (e.g., "1#.a").
    """

    def test_should_detect_array_aliases_in_struct(self):
        """Should detect array aliases in struct."""
        struct_type = StructType([("a", ArrayType(IntegerType)), ("b", ArrayType(IntegerType))])

        # Create struct with shared array reference
        shared_array = EastArray(IntegerType, [1, 2, 3])
        value = {"a": shared_array, "b": shared_array}

        result = print_east(value, struct_type)
        # Should print with alias reference
        assert result == "(a=[1, 2, 3], b=1#.a)"

    def test_should_detect_set_aliases_in_struct(self):
        """Should detect set aliases in struct."""
        struct_type = StructType([("a", SetType(IntegerType)), ("b", SetType(IntegerType))])

        # Create struct with shared set reference
        shared_set = EastSet(IntegerType, [1, 2, 3])
        value = {"a": shared_set, "b": shared_set}

        result = print_east(value, struct_type)
        # Should print with alias reference
        assert result == "(a={1, 2, 3}, b=1#.a)"

    def test_should_detect_dict_aliases_in_struct(self):
        """Should detect dict aliases in struct."""
        struct_type = StructType(
            [("a", DictType(IntegerType, StringType)), ("b", DictType(IntegerType, StringType))]
        )

        # Create struct with shared dict reference
        shared_dict = EastDict(IntegerType, StringType, {1: "x", 2: "y"})
        value = {"a": shared_dict, "b": shared_dict}

        result = print_east(value, struct_type)
        # Should print with alias reference
        assert result == '(a={1: "x", 2: "y"}, b=1#.a)'

    def test_should_detect_nested_array_aliases(self):
        """Should detect nested array aliases."""
        array_type = ArrayType(ArrayType(IntegerType))

        # Create array with shared inner array
        inner = EastArray(IntegerType, [1, 2])
        value = EastArray(ArrayType(IntegerType), [inner, inner, inner])

        result = print_east(value, array_type)
        # Should print with alias references
        assert result == "[[1, 2], 1#[0], 1#[0]]"


# =============================================================================
# Test Suite: Reference Parsing
# =============================================================================


class TestReferenceParsing:
    """Tests for parsing reference syntax.

    These tests verify that references (e.g., "1#.a") can be parsed correctly
    and resolve to the correct shared values.
    """

    def test_should_parse_array_reference_in_struct(self):
        """Should parse array reference in struct."""
        struct_type = StructType([("a", ArrayType(IntegerType)), ("b", ArrayType(IntegerType))])

        # Parse struct with reference
        text = "(a=[1, 2, 3], b=1#.a)"
        result = parse_east(struct_type, text)

        # Both fields should reference the same array object
        assert result.a is result.b
        assert list(result.a) == [1, 2, 3]

    def test_should_parse_set_reference_in_struct(self):
        """Should parse set reference in struct."""
        struct_type = StructType([("a", SetType(IntegerType)), ("b", SetType(IntegerType))])

        # Parse struct with reference
        text = "(a={1, 2, 3}, b=1#.a)"
        result = parse_east(struct_type, text)

        # Both fields should reference the same set object
        assert result.a is result.b
        assert set(result.a) == {1, 2, 3}

    def test_should_parse_dict_reference_in_struct(self):
        """Should parse dict reference in struct."""
        struct_type = StructType(
            [("a", DictType(IntegerType, StringType)), ("b", DictType(IntegerType, StringType))]
        )

        # Parse struct with reference
        text = '(a={1: "x", 2: "y"}, b=1#.a)'
        result = parse_east(struct_type, text)

        # Both fields should reference the same dict object
        assert result.a is result.b
        assert dict(result.a._data) == {1: "x", 2: "y"}

    def test_should_parse_nested_array_references(self):
        """Should parse nested array references."""
        array_type = ArrayType(ArrayType(IntegerType))

        # Parse array with references to first element
        text = "[[1, 2], 1#[0], 1#[0]]"
        result = parse_east(array_type, text)

        # All three elements should reference the same inner array
        assert result[0] is result[1]
        assert result[0] is result[2]
        assert list(result[0]) == [1, 2]

    def test_should_roundtrip_with_shared_references(self):
        """Should round-trip values with shared references."""
        from east.types.type_system import _StructTypeClass

        struct_type = StructType([("a", ArrayType(IntegerType)), ("b", ArrayType(IntegerType))])

        # Create struct with shared array reference
        shared_array = EastArray(IntegerType, [1, 2, 3])
        runtime_type = _StructTypeClass(
            (("a", ArrayType(IntegerType)), ("b", ArrayType(IntegerType)))
        )
        original = runtime_type.create(a=shared_array, b=shared_array)

        # Print and parse back
        printed = print_east(original, struct_type)
        parsed = parse_east(struct_type, printed)

        # Should maintain shared reference
        assert parsed.a is parsed.b
        assert list(parsed.a) == [1, 2, 3]


# =============================================================================
# Test Suite: Recursive Type Printing
# =============================================================================


class TestRecursiveTypePrinting:
    """Tests for printing recursive types without cycles."""

    def test_should_print_tree_without_cycles(self):
        """Should print tree without cycles."""
        # Define Tree type: variant { leaf: null, node: { value, left, right } }
        tree_type = recursive_type(
            lambda self: VariantType(
                [
                    ("leaf", NullType),
                    ("node", StructType([("value", IntegerType), ("left", self), ("right", self)])),
                ]
            )
        )

        # Build the actual tree type
        inner_struct = StructType(
            [("value", IntegerType), ("left", tree_type), ("right", tree_type)]
        )
        actual_variant_type = _VariantTypeClass((("leaf", NullType), ("node", inner_struct)))

        # Create simple tree: node(1, leaf, leaf)
        leaf = actual_variant_type.create("leaf")
        tree = actual_variant_type.create("node", {"value": 1, "left": leaf, "right": leaf})

        result = print_east(tree, tree_type)
        assert result == ".node (value=1, left=.leaf, right=.leaf)"

    def test_should_print_larger_tree_without_cycles(self):
        """Should print larger tree without cycles."""
        # Define Tree type
        tree_type = recursive_type(
            lambda self: VariantType(
                [
                    ("leaf", NullType),
                    ("node", StructType([("value", IntegerType), ("left", self), ("right", self)])),
                ]
            )
        )

        inner_struct = StructType(
            [("value", IntegerType), ("left", tree_type), ("right", tree_type)]
        )
        actual_variant_type = _VariantTypeClass((("leaf", NullType), ("node", inner_struct)))

        # Create tree: node(2, node(1, leaf, leaf), node(3, leaf, leaf))
        leaf = actual_variant_type.create("leaf")
        left_subtree = actual_variant_type.create("node", {"value": 1, "left": leaf, "right": leaf})
        right_subtree = actual_variant_type.create(
            "node", {"value": 3, "left": leaf, "right": leaf}
        )
        tree = actual_variant_type.create(
            "node", {"value": 2, "left": left_subtree, "right": right_subtree}
        )

        result = print_east(tree, tree_type)
        expected = ".node (value=2, left=.node (value=1, left=.leaf, right=.leaf), right=.node (value=3, left=.leaf, right=.leaf))"
        assert result == expected

    def test_should_print_linked_list_without_cycles(self):
        """Should print linked list without cycles."""
        # Define LinkedList type: variant { nil: null, cons: { head, tail } }
        list_type = recursive_type(
            lambda self: VariantType(
                [("nil", NullType), ("cons", StructType([("head", IntegerType), ("tail", self)]))]
            )
        )

        inner_struct = StructType([("head", IntegerType), ("tail", list_type)])
        actual_variant_type = _VariantTypeClass((("nil", NullType), ("cons", inner_struct)))

        # Create list: cons(1, cons(2, cons(3, nil)))
        nil = actual_variant_type.create("nil")
        list3 = actual_variant_type.create("cons", {"head": 3, "tail": nil})
        list2 = actual_variant_type.create("cons", {"head": 2, "tail": list3})
        list1 = actual_variant_type.create("cons", {"head": 1, "tail": list2})

        result = print_east(list1, list_type)
        expected = ".cons (head=1, tail=.cons (head=2, tail=.cons (head=3, tail=.nil)))"
        assert result == expected


"""East text format error message tests with EXACT string checks.

These tests verify that error messages match the TypeScript implementation exactly,
similar to the JSON error tests.
"""


class TestBasicErrorMessagesExact:
    """Test basic error messages match TypeScript exactly."""

    def test_type_mismatch(self):
        """Should return error for type mismatch."""
        with pytest.raises(ParseError) as exc_info:
            parse_east(IntegerType, '"not a number"')
        assert (
            str(exc_info.value)
            == "Error occurred because expected integer, got 'not a number' (line 1, col 1) while parsing value of type \".Integer\""
        )

    def test_malformed_input(self):
        """Should return error for malformed input."""
        with pytest.raises(ParseError) as exc_info:
            parse_east(ArrayType(StringType), "[unclosed array")
        assert (
            str(exc_info.value)
            == "Error occurred because expected string, got 'unclosed' at [0] (line 1, col 2) while parsing value of type \".Array .String\""
        )

    def test_extra_tokens(self):
        """Should return error for extra tokens."""
        with pytest.raises(ParseError) as exc_info:
            parse_east(StringType, '"hello" extra')
        assert (
            str(exc_info.value)
            == 'Error occurred because unexpected token IDENTIFIER (line 1, col 9) while parsing value of type ".String"'
        )


class TestTrailingCommaErrorsExact:
    """Test trailing comma errors with exact strings."""

    def test_array_trailing_comma(self):
        """Should error on array with trailing comma."""
        with pytest.raises(ParseError) as exc_info:
            parse_east(ArrayType(IntegerType), "[1, 2,]")
        assert (
            str(exc_info.value)
            == 'Error occurred because trailing comma not allowed (line 1, col 7) while parsing value of type ".Array .Integer"'
        )

    def test_set_trailing_comma(self):
        """Should error on set with trailing comma."""
        with pytest.raises(ParseError) as exc_info:
            parse_east(SetType(IntegerType), "{1, 2,}")
        assert (
            str(exc_info.value)
            == 'Error occurred because trailing comma not allowed (line 1, col 7) while parsing value of type ".Set .Integer"'
        )

    def test_dict_trailing_comma(self):
        """Should error on dict with trailing comma."""
        with pytest.raises(ParseError) as exc_info:
            parse_east(DictType(StringType, IntegerType), '{"a": 1,}')
        assert (
            str(exc_info.value)
            == 'Error occurred because trailing comma not allowed (line 1, col 9) while parsing value of type ".Dict (key=.String, value=.Integer)"'
        )


class TestStructErrorsExact:
    """Test struct parsing errors with exact strings."""

    def test_unknown_field(self):
        """Should error on unknown field."""
        struct_type = StructType([("name", StringType)])
        with pytest.raises(ParseError) as exc_info:
            parse_east(struct_type, '(name="Alice", age=30)')
        assert (
            str(exc_info.value)
            == 'Error occurred because unknown field \'age\' (line 1, col 16) while parsing value of type ".Struct [(name="name", type=.String)]"'
        )

    def test_missing_required_field(self):
        """Should error on missing required field."""
        struct_type = StructType([("name", StringType), ("age", IntegerType)])
        with pytest.raises(ParseError) as exc_info:
            parse_east(struct_type, '(name="Alice")')
        # This will fail validation when creating the struct
        assert "age" in str(exc_info.value).lower() and (
            "missing" in str(exc_info.value).lower() or "expected 2" in str(exc_info.value).lower()
        )


class TestVariantErrorsExact:
    """Test variant parsing errors with exact strings."""

    def test_unknown_variant_case(self):
        """Should error on unknown variant case."""
        variant_type = VariantType([("some", IntegerType)])
        with pytest.raises(ParseError) as exc_info:
            parse_east(variant_type, ".none")
        assert (
            str(exc_info.value)
            == 'Error occurred because unknown variant case \'none\' (line 1, col 1) while parsing value of type ".Variant [(name="some", type=.Integer)]"'
        )


class TestBlobErrorsExact:
    """Test blob parsing errors."""

    def test_blob_odd_hex_digits(self):
        """Should error on blob with odd number of hex digits."""
        with pytest.raises((ParseError, ValueError)) as exc_info:
            parse_east(BlobType, "0x123")
        # Python's bytes.fromhex raises ValueError with specific message
        assert "hex" in str(exc_info.value).lower() or "position" in str(exc_info.value).lower()

    def test_blob_no_0x_prefix(self):
        """Should error on blob not starting with 0x."""
        with pytest.raises(ParseError) as exc_info:
            parse_east(BlobType, "123456")
        assert (
            str(exc_info.value)
            == "Error occurred because expected blob, got '123456' (line 1, col 1) while parsing value of type \".Blob\""
        )


class TestCollectionErrorsExact:
    """Test collection parsing errors with exact/keyword checks."""

    def test_array_without_opening_bracket(self):
        """Should error on array without opening bracket."""
        with pytest.raises(ParseError) as exc_info:
            parse_east(ArrayType(IntegerType), "1, 2, 3]")
        assert "expected '['" in str(exc_info.value).lower() or (
            "[" in str(exc_info.value) and "expected" in str(exc_info.value).lower()
        )

    def test_array_missing_comma_or_bracket(self):
        """Should error on missing comma or closing bracket in array."""
        with pytest.raises(ParseError) as exc_info:
            parse_east(ArrayType(IntegerType), "[1 2]")
        error = str(exc_info.value).lower()
        assert ("," in str(exc_info.value) or "]" in str(exc_info.value)) and "expected" in error

    def test_set_without_opening_brace(self):
        """Should error on set without opening brace."""
        with pytest.raises(ParseError) as exc_info:
            parse_east(SetType(StringType), '"a", "b"}')
        assert "expected '{'" in str(exc_info.value).lower() or (
            "{" in str(exc_info.value) and "expected" in str(exc_info.value).lower()
        )

    def test_set_missing_comma_or_brace(self):
        """Should error on missing comma or closing brace in set."""
        with pytest.raises(ParseError) as exc_info:
            parse_east(SetType(StringType), '{"a" "b"}')
        error = str(exc_info.value).lower()
        assert ("," in str(exc_info.value) or "}" in str(exc_info.value)) and "expected" in error

    def test_dict_without_opening_brace(self):
        """Should error on dict without opening brace."""
        with pytest.raises(ParseError) as exc_info:
            parse_east(DictType(StringType, IntegerType), '"a": 1}')
        assert "expected '{'" in str(exc_info.value).lower() or (
            "{" in str(exc_info.value) and "expected" in str(exc_info.value).lower()
        )

    def test_dict_missing_colon(self):
        """Should error on missing colon in dict entry."""
        with pytest.raises(ParseError) as exc_info:
            parse_east(DictType(StringType, IntegerType), '{"a" 1}')
        error = str(exc_info.value).lower()
        assert ":" in str(exc_info.value) and "expected" in error

    def test_dict_missing_comma_or_brace(self):
        """Should error on missing comma or closing brace in dict."""
        with pytest.raises(ParseError) as exc_info:
            parse_east(DictType(StringType, IntegerType), '{"a": 1 "b": 2}')
        error = str(exc_info.value).lower()
        assert ("," in str(exc_info.value) or "}" in str(exc_info.value)) and "expected" in error


class TestStructDetailedErrorsExact:
    """Test detailed struct parsing errors."""

    def test_struct_without_opening_paren(self):
        """Should error on struct without opening paren."""
        struct_type = StructType([("x", IntegerType)])
        with pytest.raises(ParseError) as exc_info:
            parse_east(struct_type, "x=42)")
        error = str(exc_info.value).lower()
        assert "expected '('" in error or ("(" in str(exc_info.value) and "expected" in error)

    def test_struct_missing_equals(self):
        """Should error on missing equals after struct field name."""
        struct_type = StructType([("x", IntegerType)])
        with pytest.raises(ParseError) as exc_info:
            parse_east(struct_type, "(x 42)")
        error = str(exc_info.value).lower()
        assert "=" in str(exc_info.value) and "expected" in error

    def test_struct_missing_comma_or_paren(self):
        """Should error on missing comma or closing paren in struct."""
        struct_type = StructType([("x", IntegerType), ("y", IntegerType)])
        with pytest.raises(ParseError) as exc_info:
            parse_east(struct_type, "(x=1 y=2)")
        error = str(exc_info.value).lower()
        assert ("," in str(exc_info.value) or ")" in str(exc_info.value)) and "expected" in error


class TestVariantDetailedErrorsExact:
    """Test detailed variant parsing errors."""

    def test_variant_without_dot(self):
        """Should error on variant without dot."""
        variant_type = VariantType([("some", IntegerType)])
        with pytest.raises(ParseError) as exc_info:
            parse_east(variant_type, "some 42")
        error = str(exc_info.value).lower()
        assert "expected" in error and ("." in str(exc_info.value) or "variant" in error)


"""Tests for East text format printer."""

from datetime import UTC


class TestPrintPrimitives:
    """Tests for printing primitive values."""

    def test_null(self):
        """Print null."""
        result = print_east(null, NullType)
        assert result == "null"

    def test_true(self):
        """Print true."""
        result = print_east(True, BooleanType)
        assert result == "true"

    def test_false(self):
        """Print false."""
        result = print_east(False, BooleanType)
        assert result == "false"

    def test_integer(self):
        """Print integer."""
        result = print_east(42, IntegerType)
        assert result == "42"

    def test_negative_integer(self):
        """Print negative integer."""
        result = print_east(-123, IntegerType)
        assert result == "-123"

    def test_float(self):
        """Print float."""
        result = print_east(3.14, FloatType)
        # Float formatting may have extra precision
        assert result.startswith("3.14")
        assert float(result) == 3.14

    def test_nan(self):
        """Print NaN."""
        result = print_east(float("nan"), FloatType)
        assert result == "NaN"

    def test_infinity(self):
        """Print Infinity."""
        result = print_east(float("inf"), FloatType)
        assert result == "Infinity"

    def test_neg_infinity(self):
        """Print -Infinity."""
        result = print_east(float("-inf"), FloatType)
        assert result == "-Infinity"

    def test_string(self):
        """Print string."""
        result = print_east("hello", StringType)
        assert result == '"hello"'

    def test_string_with_quotes(self):
        """Print string with quotes."""
        result = print_east('say "hi"', StringType)
        assert result == r'"say \"hi\""'

    def test_string_with_newline(self):
        """Print string with newline - literal newline, not escaped."""
        result = print_east("line1\nline2", StringType)
        # East text format does not escape \n - newline appears literally
        assert result == '"line1\nline2"'
        assert "\n" in result  # Actual newline character
        assert r"\n" not in result  # Not the escape sequence

    def test_blob(self):
        """Print blob."""
        result = print_east(Blob(b"\xab\xcd"), BlobType)
        assert result == "0xabcd"

    def test_empty_blob(self):
        """Print empty blob."""
        result = print_east(Blob(b""), BlobType)
        assert result == "0x"

    def test_datetime(self):
        """Print datetime."""
        dt = datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC)
        result = print_east(dt, DateTimeType)
        assert "2024-01-15" in result
        assert "10:30:00" in result


class TestPrintArrays:
    """Tests for printing arrays."""

    def test_empty_array(self):
        """Print empty array."""
        from east.types.containers import EastArray

        arr = EastArray(IntegerType, [])
        result = print_east(arr, ArrayType(IntegerType))
        assert result == "[]"

    def test_integer_array(self):
        """Print integer array."""
        from east.types.containers import EastArray

        arr = EastArray(IntegerType, [1, 2, 3])
        result = print_east(arr, ArrayType(IntegerType))
        assert result == "[1, 2, 3]"

    def test_string_array(self):
        """Print string array."""
        from east.types.containers import EastArray

        arr = EastArray(StringType, ["a", "b", "c"])
        result = print_east(arr, ArrayType(StringType))
        assert result == '["a", "b", "c"]'

    def test_nested_array(self):
        """Print nested array."""
        from east.types.containers import EastArray

        inner1 = EastArray(IntegerType, [1, 2])
        inner2 = EastArray(IntegerType, [3, 4])
        arr = EastArray(ArrayType(IntegerType), [inner1, inner2])
        result = print_east(arr, ArrayType(ArrayType(IntegerType)))
        assert result == "[[1, 2], [3, 4]]"


class TestPrintSets:
    """Tests for printing sets."""

    def test_empty_set(self):
        """Print empty set."""
        from east.types.containers import EastSet

        s = EastSet(IntegerType)
        result = print_east(s, SetType(IntegerType))
        assert result == "{}"

    def test_integer_set(self):
        """Print integer set."""
        from east.types.containers import EastSet

        s = EastSet(IntegerType, [1, 2, 3])
        result = print_east(s, SetType(IntegerType))
        # Sets are sorted
        assert result == "{1, 2, 3}"


class TestPrintDicts:
    """Tests for printing dicts."""

    def test_empty_dict(self):
        """Print empty dict."""
        from east.types.containers import EastDict

        d = EastDict(StringType, IntegerType)
        result = print_east(d, DictType(StringType, IntegerType))
        assert result == "{:}"

    def test_string_int_dict(self):
        """Print string to int dict."""
        from east.types.containers import EastDict

        d = EastDict(StringType, IntegerType, {"a": 1, "b": 2})
        result = print_east(d, DictType(StringType, IntegerType))
        # Keys are sorted
        assert result == '{"a": 1, "b": 2}'


class TestPrintStructs:
    """Tests for printing structs."""

    def test_empty_struct(self):
        """Print empty struct."""
        struct_type_def = _StructTypeClass(())
        struct = struct_type_def.create()

        from east.types.type_system import StructType

        struct_east_type = StructType([])
        result = print_east(struct, struct_east_type)
        assert result == "()"

    def test_simple_struct(self):
        """Print simple struct."""
        struct_type = _StructTypeClass((("name", StringType), ("age", IntegerType)))
        struct = struct_type.create(name="Alice", age=30)

        from east.types.type_system import StructType

        struct_east_type = StructType([("name", StringType), ("age", IntegerType)])
        result = print_east(struct, struct_east_type)
        assert result == '(name="Alice", age=30)'

    def test_struct_special_field_name(self):
        """Print struct with field name that needs escaping."""
        struct_type = _StructTypeClass((("my-field", StringType),))
        struct = struct_type.create(**{"my-field": "value"})

        from east.types.type_system import StructType

        struct_east_type = StructType([("my-field", StringType)])
        result = print_east(struct, struct_east_type)
        assert result == '(`my-field`="value")'


class TestPrintVariants:
    """Tests for printing variants."""

    def test_variant_with_null_value(self):
        """Print variant with null value."""
        variant_type = _VariantTypeClass((("Some", IntegerType), ("None", NullType)))
        variant = variant_type.create("None")

        from east.types.type_system import VariantType

        variant_east_type = VariantType([("Some", IntegerType), ("None", NullType)])
        result = print_east(variant, variant_east_type)
        assert result == ".None"

    def test_variant_with_value(self):
        """Print variant with value."""
        variant_type = _VariantTypeClass((("Some", IntegerType), ("None", NullType)))
        variant = variant_type.create("Some", 42)

        from east.types.type_system import VariantType

        variant_east_type = VariantType([("Some", IntegerType), ("None", NullType)])
        result = print_east(variant, variant_east_type)
        assert result == ".Some 42"


class TestPrintComplexTypes:
    """Tests for complex nested types."""

    def test_array_of_structs(self):
        """Print array of structs."""
        from east.types.containers import EastArray
        from east.types.type_system import StructType

        struct_east_type = StructType([("name", StringType), ("age", IntegerType)])
        struct_type = _StructTypeClass((("name", StringType), ("age", IntegerType)))
        s1 = struct_type.create(name="Alice", age=30)
        s2 = struct_type.create(name="Bob", age=25)
        arr = EastArray(struct_east_type, [s1, s2])

        result = print_east(arr, ArrayType(struct_east_type))
        assert result == '[(name="Alice", age=30), (name="Bob", age=25)]'


"""Tests for East tokenizer."""


from east.serialization.east_tokenizer import TokenType, tokenize


class TestKeywords:
    """Tests for keyword tokenization."""

    def test_null(self):
        """Tokenize null."""
        tokens = tokenize("null")
        assert len(tokens) == 2  # null + EOF
        assert tokens[0].type == TokenType.NULL
        assert tokens[0].value is None

    def test_true(self):
        """Tokenize true."""
        tokens = tokenize("true")
        assert tokens[0].type == TokenType.TRUE
        assert tokens[0].value is True

    def test_false(self):
        """Tokenize false."""
        tokens = tokenize("false")
        assert tokens[0].type == TokenType.FALSE
        assert tokens[0].value is False


class TestNumbers:
    """Tests for number tokenization."""

    def test_integer(self):
        """Tokenize integers."""
        tokens = tokenize("42")
        assert tokens[0].type == TokenType.INTEGER
        assert tokens[0].value == 42

    def test_negative_integer(self):
        """Tokenize negative integer."""
        tokens = tokenize("-123")
        assert tokens[0].type == TokenType.INTEGER
        assert tokens[0].value == -123

    def test_zero(self):
        """Tokenize zero."""
        tokens = tokenize("0")
        assert tokens[0].type == TokenType.INTEGER
        assert tokens[0].value == 0

    def test_float(self):
        """Tokenize float."""
        tokens = tokenize("3.14")
        assert tokens[0].type == TokenType.FLOAT
        assert tokens[0].value == 3.14

    def test_negative_float(self):
        """Tokenize negative float."""
        tokens = tokenize("-2.5")
        assert tokens[0].type == TokenType.FLOAT
        assert tokens[0].value == -2.5

    def test_nan(self):
        """Tokenize NaN."""
        tokens = tokenize("NaN")
        assert tokens[0].type == TokenType.FLOAT
        assert math.isnan(tokens[0].value)

    def test_infinity(self):
        """Tokenize Infinity."""
        tokens = tokenize("Infinity")
        assert tokens[0].type == TokenType.FLOAT
        assert tokens[0].value == float("inf")


class TestStrings:
    """Tests for string tokenization."""

    def test_double_quotes(self):
        """Tokenize double-quoted string."""
        tokens = tokenize('"hello"')
        assert tokens[0].type == TokenType.STRING
        assert tokens[0].value == "hello"

    def test_single_quotes(self):
        """Tokenize single-quoted string."""
        tokens = tokenize("'world'")
        assert tokens[0].type == TokenType.STRING
        assert tokens[0].value == "world"

    def test_empty_string(self):
        """Tokenize empty string."""
        tokens = tokenize('""')
        assert tokens[0].type == TokenType.STRING
        assert tokens[0].value == ""

    def test_escaped_quote(self):
        """Tokenize string with escaped quote."""
        tokens = tokenize(r'"say \"hi\""')
        assert tokens[0].value == 'say "hi"'

    def test_escaped_backslash(self):
        """Tokenize string with escaped backslash."""
        tokens = tokenize(r'"path\\to\\file"')
        assert tokens[0].value == r"path\to\file"

    def test_escaped_newline(self):
        """Tokenize string with escaped newline should error."""
        # East text format doesn't support \n - must use actual newlines
        import pytest

        with pytest.raises(ValueError, match=r"[Uu]nsupported escape"):
            tokenize(r'"line1\nline2"')


class TestBlob:
    """Tests for blob tokenization."""

    def test_empty_blob(self):
        """Tokenize empty blob."""
        tokens = tokenize("0x")
        assert tokens[0].type == TokenType.BLOB
        assert tokens[0].value == ""

    def test_blob_lowercase(self):
        """Tokenize blob with lowercase hex."""
        tokens = tokenize("0xabcd")
        assert tokens[0].type == TokenType.BLOB
        assert tokens[0].value == "abcd"

    def test_blob_uppercase(self):
        """Tokenize blob with uppercase hex."""
        tokens = tokenize("0xABCD")
        assert tokens[0].type == TokenType.BLOB
        assert tokens[0].value == "ABCD"

    def test_blob_mixed_case(self):
        """Tokenize blob with mixed case hex."""
        tokens = tokenize("0x12AbCd")
        assert tokens[0].type == TokenType.BLOB
        assert tokens[0].value == "12AbCd"


class TestIdentifiers:
    """Tests for identifier tokenization."""

    def test_simple_identifier(self):
        """Tokenize simple identifier."""
        tokens = tokenize("foo")
        assert tokens[0].type == TokenType.IDENTIFIER
        assert tokens[0].value == "foo"

    def test_identifier_with_underscore(self):
        """Tokenize identifier with underscore."""
        tokens = tokenize("foo_bar")
        assert tokens[0].type == TokenType.IDENTIFIER
        assert tokens[0].value == "foo_bar"

    def test_identifier_with_numbers(self):
        """Tokenize identifier with numbers."""
        tokens = tokenize("var123")
        assert tokens[0].type == TokenType.IDENTIFIER
        assert tokens[0].value == "var123"

    def test_backtick_identifier(self):
        """Tokenize backtick-escaped identifier."""
        tokens = tokenize("`my-special-name`")
        assert tokens[0].type == TokenType.IDENTIFIER
        assert tokens[0].value == "my-special-name"


class TestVariantTags:
    """Tests for variant tag tokenization."""

    def test_simple_tag(self):
        """Tokenize simple variant tag."""
        tokens = tokenize(".Some")
        assert tokens[0].type == TokenType.VARIANT_TAG
        assert tokens[0].value == "Some"

    def test_tag_with_underscore(self):
        """Tokenize tag with underscore."""
        tokens = tokenize(".My_Tag")
        assert tokens[0].type == TokenType.VARIANT_TAG
        assert tokens[0].value == "My_Tag"


class TestDelimiters:
    """Tests for delimiter tokenization."""

    def test_brackets(self):
        """Tokenize brackets."""
        tokens = tokenize("[]")
        assert tokens[0].type == TokenType.LBRACKET
        assert tokens[1].type == TokenType.RBRACKET

    def test_braces(self):
        """Tokenize braces."""
        tokens = tokenize("{}")
        assert tokens[0].type == TokenType.LBRACE
        assert tokens[1].type == TokenType.RBRACE

    def test_parens(self):
        """Tokenize parentheses."""
        tokens = tokenize("()")
        assert tokens[0].type == TokenType.LPAREN
        assert tokens[1].type == TokenType.RPAREN

    def test_comma(self):
        """Tokenize comma."""
        tokens = tokenize(",")
        assert tokens[0].type == TokenType.COMMA

    def test_colon(self):
        """Tokenize colon."""
        tokens = tokenize(":")
        assert tokens[0].type == TokenType.COLON

    def test_equals(self):
        """Tokenize equals."""
        tokens = tokenize("=")
        assert tokens[0].type == TokenType.EQUALS


class TestTokenizeWhitespace:
    """Tests for whitespace handling."""

    def test_spaces(self):
        """Whitespace is skipped."""
        tokens = tokenize("  42  ")
        assert len(tokens) == 2  # number + EOF
        assert tokens[0].type == TokenType.INTEGER

    def test_newlines(self):
        """Newlines are skipped."""
        tokens = tokenize("42\n\n43")
        assert len(tokens) == 3  # num + num + EOF
        assert tokens[0].value == 42
        assert tokens[1].value == 43

    def test_tabs(self):
        """Tabs are skipped."""
        tokens = tokenize("\t42\t")
        assert len(tokens) == 2
        assert tokens[0].value == 42


class TestComments:
    """Tests for comment handling."""

    def test_comment(self):
        """Comments are skipped."""
        tokens = tokenize("# this is a comment\n42")
        assert len(tokens) == 2  # number + EOF
        assert tokens[0].value == 42

    def test_comment_at_end(self):
        """Comment at end of line."""
        tokens = tokenize("42 # comment")
        assert len(tokens) == 2
        assert tokens[0].value == 42


class TestPositionTracking:
    """Tests for line/column position tracking."""

    def test_line_tracking(self):
        """Track line numbers."""
        tokens = tokenize("42\n\n43")
        assert tokens[0].line == 1
        assert tokens[1].line == 3

    def test_column_tracking(self):
        """Track column numbers."""
        tokens = tokenize("  42")
        assert tokens[0].column == 3  # After two spaces


class TestComplexExamples:
    """Tests for complex token sequences."""

    def test_array(self):
        """Tokenize array."""
        tokens = tokenize("[1, 2, 3]")
        assert tokens[0].type == TokenType.LBRACKET
        assert tokens[1].type == TokenType.INTEGER
        assert tokens[2].type == TokenType.COMMA
        assert tokens[3].type == TokenType.INTEGER
        assert tokens[4].type == TokenType.COMMA
        assert tokens[5].type == TokenType.INTEGER
        assert tokens[6].type == TokenType.RBRACKET

    def test_struct(self):
        """Tokenize struct."""
        tokens = tokenize("(name='Alice', age=30)")
        assert tokens[0].type == TokenType.LPAREN
        assert tokens[1].type == TokenType.IDENTIFIER
        assert tokens[1].value == "name"
        assert tokens[2].type == TokenType.EQUALS
        assert tokens[3].type == TokenType.STRING
        assert tokens[3].value == "Alice"

    def test_variant(self):
        """Tokenize variant."""
        tokens = tokenize(".Some 42")
        assert tokens[0].type == TokenType.VARIANT_TAG
        assert tokens[0].value == "Some"
        assert tokens[1].type == TokenType.INTEGER
        assert tokens[1].value == 42


class TestTokenizeErrors:
    """Tests for error handling."""

    def test_unterminated_string(self):
        """Unterminated string raises error."""
        with pytest.raises(ValueError, match="Unterminated string"):
            tokenize('"hello')

    def test_invalid_variant_tag(self):
        """Invalid variant tag raises error."""
        with pytest.raises(ValueError, match="Invalid variant tag"):
            tokenize(". ")

    def test_unexpected_character(self):
        """Unexpected character raises error."""
        with pytest.raises(ValueError, match="Unexpected character"):
            tokenize("@")


class TestIRTypes:
    """Test IR type round-trip through East text format."""

    def test_increment_function_roundtrip(self):
        """Test round-trip of increment function IR: (x: Integer) -> x + 1"""
        from east.ir.builders import ir_builtin, ir_function, ir_value, ir_variable, location
        from east.serialization.east_parser import parse_east
        from east.serialization.east_printer import print_east
        from east.types.type_system import FunctionType, IntegerType, IRType
        from east.utils.ordering import equal_for

        # Build the IR using builders
        loc = location("node:internal/modules/esm/loader", 651, 26)
        param = ir_variable(IntegerType, "_0", loc, mutable=False, captured=False)
        value_1 = ir_value(IntegerType, loc, 1)
        body = ir_builtin(IntegerType, loc, "IntegerAdd", [], [param, value_1])
        func_type = FunctionType([IntegerType], IntegerType, [])
        original_ir = ir_function(func_type, loc, [], [param], body)

        # Print to East text format
        printed = print_east(original_ir, IRType)

        # Parse back from East text format
        parsed_ir = parse_east(IRType, printed)

        # Compare using type-aware equality
        equal_fn = equal_for(IRType)
        assert equal_fn(parsed_ir, original_ir), (
            f"IR round-trip failed:\n"
            f"Original: {original_ir}\n"
            f"Printed: {printed}\n"
            f"Parsed: {parsed_ir}"
        )


"""Fuzz tests for East text format serialization.

These tests generate random types and values, then verify round-trip
print → parse correctness using property-based testing.
"""


from east.testing.fuzz import fuzzer_test
from east.utils.ordering import equal_for


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_should_round_trip_random_types_and_values():
    """Should round-trip random types and values through East text format."""

    def test_east_round_trip(type_val):
        """Create a test function for round-tripping a specific type."""
        equal = equal_for(type_val)

        def test_value(value):
            # Print to East text format
            printed = print_east(value, type_val)

            # Parse back
            parsed = parse_east(type_val, printed)

            # Check equality
            if not equal(parsed, value):
                raise AssertionError(
                    f"Round-trip failed: values not equal\n"
                    f"Original: {value}\n"
                    f"Printed: {printed}\n"
                    f"Parsed: {parsed}"
                )

        return test_value

    # Test 100 random types with 10 samples each
    success = await fuzzer_test(test_east_round_trip, n_types=100, n_samples=10)
    assert success, "Fuzz test failed - see stderr for details"
