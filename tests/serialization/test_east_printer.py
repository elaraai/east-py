"""Tests for East text format printer."""

from datetime import UTC, datetime

from east.serialization.east_printer import print_east
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
        assert result == "3.14"

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
        """Print string with newline."""
        result = print_east("line1\nline2", StringType)
        assert result == r'"line1\nline2"'

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


class TestArrays:
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


class TestSets:
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


class TestDicts:
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


class TestStructs:
    """Tests for printing structs."""

    def test_empty_struct(self):
        """Print empty struct."""
        struct_type_def = StructType(())
        struct = struct_type_def.create()

        from east.types.type_system import StructTypeFromFields

        struct_east_type = StructTypeFromFields([])
        result = print_east(struct, struct_east_type)
        assert result == "()"

    def test_simple_struct(self):
        """Print simple struct."""
        struct_type = StructType((("name", StringType), ("age", IntegerType)))
        struct = struct_type.create(name="Alice", age=30)

        from east.types.type_system import StructTypeFromFields

        struct_east_type = StructTypeFromFields([("name", StringType), ("age", IntegerType)])
        result = print_east(struct, struct_east_type)
        assert result == '(name="Alice", age=30)'

    def test_struct_special_field_name(self):
        """Print struct with field name that needs escaping."""
        struct_type = StructType((("my-field", StringType),))
        struct = struct_type.create(**{"my-field": "value"})

        from east.types.type_system import StructTypeFromFields

        struct_east_type = StructTypeFromFields([("my-field", StringType)])
        result = print_east(struct, struct_east_type)
        assert result == '(`my-field`="value")'


class TestVariants:
    """Tests for printing variants."""

    def test_variant_with_null_value(self):
        """Print variant with null value."""
        variant_type = VariantType((("Some", IntegerType), ("None", NullType)))
        variant = variant_type.create("None")

        from east.types.type_system import VariantTypeFromCases

        variant_east_type = VariantTypeFromCases([("Some", IntegerType), ("None", NullType)])
        result = print_east(variant, variant_east_type)
        assert result == ".None"

    def test_variant_with_value(self):
        """Print variant with value."""
        variant_type = VariantType((("Some", IntegerType), ("None", NullType)))
        variant = variant_type.create("Some", 42)

        from east.types.type_system import VariantTypeFromCases

        variant_east_type = VariantTypeFromCases([("Some", IntegerType), ("None", NullType)])
        result = print_east(variant, variant_east_type)
        assert result == ".Some 42"


class TestComplexTypes:
    """Tests for complex nested types."""

    def test_array_of_structs(self):
        """Print array of structs."""
        from east.types.containers import EastArray
        from east.types.type_system import StructTypeFromFields

        struct_east_type = StructTypeFromFields([("name", StringType), ("age", IntegerType)])
        struct_type = StructType((("name", StringType), ("age", IntegerType)))
        s1 = struct_type.create(name="Alice", age=30)
        s2 = struct_type.create(name="Bob", age=25)
        arr = EastArray(struct_east_type, [s1, s2])

        result = print_east(arr, ArrayType(struct_east_type))
        assert result == '[(name="Alice", age=30), (name="Bob", age=25)]'
