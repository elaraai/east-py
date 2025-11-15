"""Tests for builtin functions."""

import math
from datetime import UTC, datetime

import pytest

from east.builtins import get_builtin, list_builtins
from east.datetime_format import tokenize_datetime_format
from east.types.containers import EastArray, EastDict, EastSet
from east.types.types import IntegerType, StringType


class TestBuiltinRegistry:
    """Test builtin registry."""

    def test_list_builtins_not_empty(self):
        """Test that builtins are registered."""
        builtins = list_builtins()
        assert len(builtins) > 0
        assert "IntegerAdd" in builtins
        assert "StringConcat" in builtins

    def test_get_builtin(self):
        """Test getting a builtin."""
        func = get_builtin("IntegerAdd")
        assert callable(func)
        assert func(2, 3) == 5

    def test_get_unknown_builtin_raises(self):
        """Test that unknown builtin raises KeyError."""
        with pytest.raises(KeyError):
            get_builtin("NonexistentBuiltin")


class TestBooleanBuiltins:
    """Test boolean operations."""

    def test_boolean_and(self):
        """Test BooleanAnd."""
        func = get_builtin("BooleanAnd")
        assert func(True, True) is True
        assert func(True, False) is False
        assert func(False, True) is False
        assert func(False, False) is False

    def test_boolean_or(self):
        """Test BooleanOr."""
        func = get_builtin("BooleanOr")
        assert func(True, True) is True
        assert func(True, False) is True
        assert func(False, True) is True
        assert func(False, False) is False

    def test_boolean_not(self):
        """Test BooleanNot."""
        func = get_builtin("BooleanNot")
        assert func(True) is False
        assert func(False) is True

    def test_boolean_xor(self):
        """Test BooleanXor."""
        func = get_builtin("BooleanXor")
        assert func(True, True) is False
        assert func(True, False) is True
        assert func(False, True) is True
        assert func(False, False) is False


class TestComparisonBuiltins:
    """Test comparison operations."""

    def test_equal(self):
        """Test Equal."""
        func = get_builtin("Equal")
        assert func(42, 42, IntegerType) is True
        assert func(42, 43, IntegerType) is False
        assert func("hello", "hello", StringType) is True
        assert func("hello", "world", StringType) is False

    def test_not_equal(self):
        """Test NotEqual."""
        func = get_builtin("NotEqual")
        assert func(42, 43, IntegerType) is True
        assert func(42, 42, IntegerType) is False

    def test_less(self):
        """Test Less."""
        func = get_builtin("Less")
        assert func(1, 2, IntegerType) is True
        assert func(2, 1, IntegerType) is False
        assert func(2, 2, IntegerType) is False

    def test_greater(self):
        """Test Greater."""
        func = get_builtin("Greater")
        assert func(2, 1, IntegerType) is True
        assert func(1, 2, IntegerType) is False
        assert func(2, 2, IntegerType) is False

    def test_is(self):
        """Test Is (identity comparison)."""
        from east.types.containers import EastArray
        from east.types.types import ArrayType

        func = get_builtin("Is")
        a = EastArray(IntegerType, [])
        b = EastArray(IntegerType, [])
        arr_type = ArrayType(IntegerType)
        assert func(a, a, arr_type) is True
        assert func(a, b, arr_type) is False
        assert func(1, 1, IntegerType) is True
        assert func("hello", "hello", StringType) is True

    def test_less_equal(self):
        """Test LessEqual."""
        func = get_builtin("LessEqual")
        assert func(1, 2, IntegerType) is True
        assert func(2, 2, IntegerType) is True
        assert func(3, 2, IntegerType) is False

    def test_greater_equal(self):
        """Test GreaterEqual."""
        func = get_builtin("GreaterEqual")
        assert func(3, 2, IntegerType) is True
        assert func(2, 2, IntegerType) is True
        assert func(1, 2, IntegerType) is False


class TestIntegerBuiltins:
    """Test integer operations."""

    def test_integer_add(self):
        """Test IntegerAdd."""
        func = get_builtin("IntegerAdd")
        assert func(2, 3) == 5
        assert func(-1, 1) == 0

    def test_integer_subtract(self):
        """Test IntegerSubtract."""
        func = get_builtin("IntegerSubtract")
        assert func(5, 3) == 2
        assert func(1, 1) == 0

    def test_integer_multiply(self):
        """Test IntegerMultiply."""
        func = get_builtin("IntegerMultiply")
        assert func(3, 4) == 12
        assert func(-2, 3) == -6

    def test_integer_divide(self):
        """Test IntegerDivide."""
        func = get_builtin("IntegerDivide")
        assert func(10, 3) == 3
        assert func(10, 2) == 5

    def test_integer_remainder(self):
        """Test IntegerRemainder."""
        func = get_builtin("IntegerRemainder")
        assert func(10, 3) == 1
        assert func(10, 2) == 0

    def test_integer_pow(self):
        """Test IntegerPow."""
        func = get_builtin("IntegerPow")
        assert func(2, 3) == 8
        assert func(5, 0) == 1

    def test_integer_negate(self):
        """Test IntegerNegate."""
        func = get_builtin("IntegerNegate")
        assert func(5) == -5
        assert func(-3) == 3

    def test_integer_abs(self):
        """Test IntegerAbs."""
        func = get_builtin("IntegerAbs")
        assert func(-5) == 5
        assert func(5) == 5

    def test_integer_to_float(self):
        """Test IntegerToFloat."""
        func = get_builtin("IntegerToFloat")
        assert func(42) == 42.0
        assert isinstance(func(42), float)

    def test_integer_sign(self):
        """Test IntegerSign."""
        func = get_builtin("IntegerSign")
        assert func(5) == 1
        assert func(-5) == -1
        assert func(0) == 0

    def test_integer_log(self):
        """Test IntegerLog."""
        func = get_builtin("IntegerLog")
        assert func(8, 2) == 3  # log2(8) = 3
        assert func(100, 10) == 2  # log10(100) = 2
        assert func(27, 3) == 3  # log3(27) = 3
        assert func(7, 2) == 2  # floor(log2(7)) = 2
        # Edge cases return 0 (matches TypeScript behavior)
        assert func(0, 2) == 0  # a == 0
        assert func(10, 1) == 0  # base <= 1


class TestFloatBuiltins:
    """Test float operations."""

    def test_float_add(self):
        """Test FloatAdd."""
        func = get_builtin("FloatAdd")
        assert func(2.5, 3.5) == 6.0

    def test_float_subtract(self):
        """Test FloatSubtract."""
        func = get_builtin("FloatSubtract")
        assert func(5.5, 2.5) == 3.0

    def test_float_multiply(self):
        """Test FloatMultiply."""
        func = get_builtin("FloatMultiply")
        assert func(2.5, 4.0) == 10.0

    def test_float_divide(self):
        """Test FloatDivide."""
        func = get_builtin("FloatDivide")
        assert func(10.0, 2.0) == 5.0

    def test_float_sqrt(self):
        """Test FloatSqrt."""
        func = get_builtin("FloatSqrt")
        assert func(9.0) == 3.0
        assert func(2.0) == pytest.approx(1.414213, rel=1e-5)

    def test_float_remainder(self):
        """Test FloatRemainder."""
        func = get_builtin("FloatRemainder")
        assert func(10.0, 3.0) == pytest.approx(1.0)
        assert func(5.5, 2.0) == pytest.approx(1.5)

    def test_float_pow(self):
        """Test FloatPow."""
        func = get_builtin("FloatPow")
        assert func(2.0, 3.0) == 8.0
        assert func(9.0, 0.5) == pytest.approx(3.0)

    def test_float_negate(self):
        """Test FloatNegate."""
        func = get_builtin("FloatNegate")
        assert func(5.5) == -5.5
        assert func(-3.2) == 3.2

    def test_float_abs(self):
        """Test FloatAbs."""
        func = get_builtin("FloatAbs")
        assert func(-5.5) == 5.5
        assert func(5.5) == 5.5

    def test_float_sign(self):
        """Test FloatSign."""
        func = get_builtin("FloatSign")
        assert func(5.5) == 1.0
        assert func(-5.5) == -1.0
        assert func(0.0) == 0.0
        assert func(float("nan")) == 0.0

    def test_float_log(self):
        """Test FloatLog."""
        func = get_builtin("FloatLog")
        assert func(math.e) == pytest.approx(1.0)
        assert func(1.0) == 0.0

    def test_float_exp(self):
        """Test FloatExp."""
        func = get_builtin("FloatExp")
        assert func(0.0) == 1.0
        assert func(1.0) == pytest.approx(math.e)

    def test_float_sin(self):
        """Test FloatSin."""
        func = get_builtin("FloatSin")
        assert func(0.0) == 0.0
        assert func(math.pi / 2) == pytest.approx(1.0)

    def test_float_cos(self):
        """Test FloatCos."""
        func = get_builtin("FloatCos")
        assert func(0.0) == 1.0
        assert func(math.pi) == pytest.approx(-1.0)

    def test_float_tan(self):
        """Test FloatTan."""
        func = get_builtin("FloatTan")
        assert func(0.0) == 0.0
        assert func(math.pi / 4) == pytest.approx(1.0)

    def test_float_to_integer(self):
        """Test FloatToInteger."""
        func = get_builtin("FloatToInteger")
        assert func(5.9) == 5
        assert func(-5.9) == -5
        assert isinstance(func(5.9), int)


class TestStringBuiltins:
    """Test string operations."""

    def test_string_concat(self):
        """Test StringConcat."""
        func = get_builtin("StringConcat")
        assert func("hello", "world") == "helloworld"
        assert func("", "test") == "test"

    def test_string_length(self):
        """Test StringLength."""
        func = get_builtin("StringLength")
        assert func("hello") == 5
        assert func("") == 0

    def test_string_index_of(self):
        """Test StringIndexOf."""
        func = get_builtin("StringIndexOf")
        assert func("hello world", "world") == 6
        assert func("hello world", "foo") == -1

    def test_string_split(self):
        """Test StringSplit."""
        func = get_builtin("StringSplit")
        result = func("a,b,c", ",")
        assert isinstance(result, EastArray)
        assert list(result) == ["a", "b", "c"]

    def test_string_trim(self):
        """Test StringTrim."""
        func = get_builtin("StringTrim")
        assert func("  hello  ") == "hello"

    def test_string_lower_case(self):
        """Test StringLowerCase."""
        func = get_builtin("StringLowerCase")
        assert func("HELLO") == "hello"

    def test_string_upper_case(self):
        """Test StringUpperCase."""
        func = get_builtin("StringUpperCase")
        assert func("hello") == "HELLO"

    def test_regex_contains(self):
        """Test RegexContains."""
        func = get_builtin("RegexContains")
        # Case-sensitive match
        assert func("Hello World", r"World", "") is True
        assert func("Hello World", r"world", "") is False
        # Case-insensitive match
        assert func("Hello World", r"world", "i") is True
        # No match
        assert func("Hello World", r"xyz", "") is False
        # Pattern matching
        assert func("Hello123", r"\d+", "") is True

    def test_regex_index_of(self):
        """Test RegexIndexOf."""
        func = get_builtin("RegexIndexOf")
        # Found at beginning
        assert func("Hello World", r"\w+", "") == 0
        # Found at position 6
        assert func("Hello World", r"World", "") == 6
        # Not found
        assert func("Hello World", r"xyz", "") == -1
        # Case-insensitive
        assert func("Hello World", r"world", "i") == 6

    def test_regex_replace(self):
        """Test RegexReplace."""
        func = get_builtin("RegexReplace")
        # Simple replacement (params: text, pattern, flags, replacement)
        assert func("Hello World", r"World", "", "Python") == "Hello Python"
        # Case-insensitive replacement
        assert func("Hello World", r"world", "i", "Python") == "Hello Python"
        # ReplaceAll semantics (replaces all matches)
        assert func("foo bar foo", r"foo", "", "baz") == "baz bar baz"
        # Pattern replacement
        assert func("test123", r"\d+", "", "456") == "test456"

    def test_string_repeat(self):
        """Test StringRepeat."""
        func = get_builtin("StringRepeat")
        assert func("ab", 3) == "ababab"
        assert func("x", 0) == ""
        assert func("hello", 1) == "hello"

    def test_string_substring(self):
        """Test StringSubstring."""
        func = get_builtin("StringSubstring")
        # StringSubstring(from, to) - extracts from index 'from' to index 'to' (exclusive)
        assert func("hello world", 0, 5) == "hello"
        assert func("hello world", 6, 11) == "world"  # "world" is at indices 6-10
        assert func("hello", 1, 4) == "ell"  # "ell" is at indices 1-3

    def test_string_trim_start(self):
        """Test StringTrimStart."""
        func = get_builtin("StringTrimStart")
        assert func("  hello  ") == "hello  "
        assert func("hello") == "hello"

    def test_string_trim_end(self):
        """Test StringTrimEnd."""
        func = get_builtin("StringTrimEnd")
        assert func("  hello  ") == "  hello"
        assert func("hello") == "hello"

    def test_string_starts_with(self):
        """Test StringStartsWith."""
        func = get_builtin("StringStartsWith")
        assert func("hello world", "hello") is True
        assert func("hello world", "world") is False
        assert func("hello", "hello") is True

    def test_string_ends_with(self):
        """Test StringEndsWith."""
        func = get_builtin("StringEndsWith")
        assert func("hello world", "world") is True
        assert func("hello world", "hello") is False
        assert func("hello", "hello") is True

    def test_string_contains(self):
        """Test StringContains."""
        func = get_builtin("StringContains")
        assert func("hello world", "world") is True
        assert func("hello world", "lo wo") is True
        assert func("hello world", "xyz") is False

    def test_string_replace(self):
        """Test StringReplace."""
        func = get_builtin("StringReplace")
        assert func("hello world", "world", "python") == "hello python"
        assert func("foo bar foo", "foo", "baz") == "baz bar baz"  # Replaces all
        assert func("hello", "xyz", "abc") == "hello"  # No match

    def test_string_encode_utf8(self):
        """Test StringEncodeUtf8."""
        from east.types.primitives import Blob

        func = get_builtin("StringEncodeUtf8")
        result = func("hello")
        assert isinstance(result, Blob)
        assert result.data == b"hello"

    def test_string_encode_utf16(self):
        """Test StringEncodeUtf16."""
        from east.types.primitives import Blob

        func = get_builtin("StringEncodeUtf16")
        result = func("hello")
        assert isinstance(result, Blob)
        assert result.data == "hello".encode("utf-16")

    def test_string_print_json(self):
        """Test StringPrintJSON."""
        from east.types.types import IntegerType

        func = get_builtin("StringPrintJSON")
        result = func(42, IntegerType)
        assert result == '"42"'

    def test_string_parse_json(self):
        """Test StringParseJSON."""
        from east.types.types import IntegerType

        func = get_builtin("StringParseJSON")
        result = func('"42"', IntegerType)
        assert result == 42

    def test_print(self):
        """Test Print."""
        from east.types.types import IntegerType

        func = get_builtin("Print")
        result = func(42, IntegerType)
        assert isinstance(result, str)
        assert "42" in result

    def test_parse(self):
        """Test Parse."""
        from east.types.types import IntegerType

        func = get_builtin("Parse")
        result = func("42", IntegerType)
        assert result == 42


class TestBlobBuiltins:
    """Test blob operations."""

    def test_blob_size(self):
        """Test BlobSize."""
        from east.types.primitives import Blob

        func = get_builtin("BlobSize")
        assert func(Blob(b"hello")) == 5
        assert func(Blob(b"")) == 0

    def test_blob_get_uint8(self):
        """Test BlobGetUint8."""
        from east.types.primitives import Blob

        func = get_builtin("BlobGetUint8")
        blob = Blob(b"hello")
        assert func(blob, 0) == ord("h")
        assert func(blob, 4) == ord("o")
        with pytest.raises(IndexError):
            func(blob, 10)

    def test_blob_decode_utf8(self):
        """Test BlobDecodeUtf8."""
        from east.types.primitives import Blob

        func = get_builtin("BlobDecodeUtf8")
        assert func(Blob(b"hello")) == "hello"
        assert func(Blob(b"world")) == "world"

    def test_blob_decode_utf16(self):
        """Test BlobDecodeUtf16."""
        from east.types.primitives import Blob

        func = get_builtin("BlobDecodeUtf16")
        blob = Blob("hello".encode("utf-16"))
        assert func(blob) == "hello"

    def test_blob_encode_beast(self):
        """Test BlobEncodeBeast."""
        from east.types.primitives import Blob
        from east.types.types import IntegerType

        func = get_builtin("BlobEncodeBeast")
        result = func(42, IntegerType)
        assert isinstance(result, Blob)

    def test_blob_decode_beast(self):
        """Test BlobDecodeBeast."""
        from east.types.types import IntegerType

        encode_func = get_builtin("BlobEncodeBeast")
        decode_func = get_builtin("BlobDecodeBeast")

        encoded = encode_func(42, IntegerType)
        decoded = decode_func(encoded, IntegerType)
        assert decoded == 42

    def test_blob_encode_beast2(self):
        """Test BlobEncodeBeast2."""
        from east.types.primitives import Blob
        from east.types.types import IntegerType

        func = get_builtin("BlobEncodeBeast2")
        result = func(42, IntegerType)
        assert isinstance(result, Blob)
        assert len(result.data) > 0

    def test_blob_decode_beast2(self):
        """Test BlobDecodeBeast2."""
        from east.types.types import IntegerType

        encode_func = get_builtin("BlobEncodeBeast2")
        decode_func = get_builtin("BlobDecodeBeast2")

        # Test round-trip
        original = 42
        encoded = encode_func(original, IntegerType)
        decoded = decode_func(encoded, IntegerType)
        assert decoded == original


class TestArrayBuiltins:
    """Test array operations."""

    def test_array_size(self):
        """Test ArraySize."""
        func = get_builtin("ArraySize")
        arr = EastArray(IntegerType, [1, 2, 3])
        assert func(arr, IntegerType) == 3

    def test_array_get(self):
        """Test ArrayGet."""
        func = get_builtin("ArrayGet")
        arr = EastArray(IntegerType, [1, 2, 3])
        assert func(arr, 0, IntegerType) == 1
        assert func(arr, 2, IntegerType) == 3

    def test_array_update(self):
        """Test ArrayUpdate."""
        func = get_builtin("ArrayUpdate")
        arr = EastArray(IntegerType, [1, 2, 3])
        func(arr, 1, 42, IntegerType)
        assert arr[1] == 42

    def test_array_push_last(self):
        """Test ArrayPushLast."""
        func = get_builtin("ArrayPushLast")
        arr = EastArray(IntegerType, [1, 2, 3])
        func(arr, 4, IntegerType)
        assert list(arr) == [1, 2, 3, 4]

    def test_array_slice(self):
        """Test ArraySlice."""
        func = get_builtin("ArraySlice")
        arr = EastArray(IntegerType, [1, 2, 3, 4, 5])
        result = func(arr, 1, 4, IntegerType)
        assert list(result) == [2, 3, 4]

    def test_array_concat(self):
        """Test ArrayConcat."""
        func = get_builtin("ArrayConcat")
        a = EastArray(IntegerType, [1, 2])
        b = EastArray(IntegerType, [3, 4])
        result = func(a, b, IntegerType)
        assert list(result) == [1, 2, 3, 4]

    def test_array_push_first(self):
        """Test ArrayPushFirst."""
        func = get_builtin("ArrayPushFirst")
        arr = EastArray(IntegerType, [2, 3])
        func(arr, 1, IntegerType)
        assert list(arr) == [1, 2, 3]

    def test_array_pop_last(self):
        """Test ArrayPopLast."""
        func = get_builtin("ArrayPopLast")
        arr = EastArray(IntegerType, [1, 2, 3])
        result = func(arr, IntegerType)
        assert result == 3
        assert list(arr) == [1, 2]

    def test_array_pop_first(self):
        """Test ArrayPopFirst."""
        func = get_builtin("ArrayPopFirst")
        arr = EastArray(IntegerType, [1, 2, 3])
        result = func(arr, IntegerType)
        assert result == 1
        assert list(arr) == [2, 3]

    def test_array_clear(self):
        """Test ArrayClear."""
        func = get_builtin("ArrayClear")
        arr = EastArray(IntegerType, [1, 2, 3])
        func(arr, IntegerType)
        assert list(arr) == []

    def test_array_copy(self):
        """Test ArrayCopy."""
        func = get_builtin("ArrayCopy")
        arr = EastArray(IntegerType, [1, 2, 3])
        result = func(arr, IntegerType)
        assert list(result) == [1, 2, 3]
        assert result is not arr

    def test_array_range(self):
        """Test ArrayRange."""
        func = get_builtin("ArrayRange")
        result = func(0, 5, 1)
        assert list(result) == [0, 1, 2, 3, 4]

    def test_array_reverse(self):
        """Test ArrayReverse."""
        func = get_builtin("ArrayReverse")
        arr = EastArray(IntegerType, [1, 2, 3])
        result = func(arr, IntegerType)
        assert list(result) == [3, 2, 1]
        assert list(arr) == [1, 2, 3]  # Original unchanged

    def test_array_reverse_in_place(self):
        """Test ArrayReverseInPlace."""
        func = get_builtin("ArrayReverseInPlace")
        arr = EastArray(IntegerType, [1, 2, 3])
        func(arr, IntegerType)
        assert list(arr) == [3, 2, 1]

    def test_array_sort(self):
        """Test ArraySort."""
        func = get_builtin("ArraySort")
        arr = EastArray(IntegerType, [3, 1, 2])
        result = func(arr, lambda x: x, IntegerType, IntegerType)  # Sort by identity
        assert list(result) == [1, 2, 3]
        assert list(arr) == [3, 1, 2]  # Original unchanged

    def test_array_sort_in_place(self):
        """Test ArraySortInPlace."""
        func = get_builtin("ArraySortInPlace")
        arr = EastArray(IntegerType, [3, 1, 2])
        func(arr, lambda x: x, IntegerType, IntegerType)  # Sort by identity
        assert list(arr) == [1, 2, 3]

    def test_array_map(self):
        """Test ArrayMap."""
        func = get_builtin("ArrayMap")
        arr = EastArray(IntegerType, [1, 2, 3])
        result = func(arr, lambda x, i: x * 2, IntegerType, IntegerType)
        assert list(result) == [2, 4, 6]

    def test_array_filter(self):
        """Test ArrayFilter."""
        func = get_builtin("ArrayFilter")
        arr = EastArray(IntegerType, [1, 2, 3, 4])
        result = func(arr, lambda x, i: x % 2 == 0, IntegerType)
        assert list(result) == [2, 4]

    def test_array_fold(self):
        """Test ArrayFold."""
        func = get_builtin("ArrayFold")
        arr = EastArray(IntegerType, [1, 2, 3, 4])
        result = func(arr, 0, lambda acc, x, index: acc + x, IntegerType, IntegerType)
        assert result == 10

    def test_array_get_or_default(self):
        """Test ArrayGetOrDefault."""
        func = get_builtin("ArrayGetOrDefault")
        arr = EastArray(IntegerType, [1, 2, 3])
        assert func(arr, 1, lambda index: 99, IntegerType) == 2
        assert func(arr, 10, lambda index: 99, IntegerType) == 99

    def test_array_generate(self):
        """Test ArrayGenerate."""
        func = get_builtin("ArrayGenerate")
        result = func(5, lambda i: i * 2, IntegerType)
        assert list(result) == [0, 2, 4, 6, 8]

    def test_array_linspace(self):
        """Test ArrayLinspace."""
        func = get_builtin("ArrayLinspace")
        result = func(0.0, 10.0, 5)
        assert len(result) == 5
        assert result[0] == 0.0
        assert result[4] == 10.0

    def test_array_has(self):
        """Test ArrayHas."""
        func = get_builtin("ArrayHas")
        arr = EastArray(IntegerType, [1, 2, 3])
        assert func(arr, 1, IntegerType) is True
        assert func(arr, 10, IntegerType) is False

    def test_array_try_get(self):
        """Test ArrayTryGet."""
        func = get_builtin("ArrayTryGet")
        arr = EastArray(IntegerType, [1, 2, 3])
        result = func(arr, 1, IntegerType)
        assert result["type"] == "some"
        assert result["value"] == 2
        result = func(arr, 10, IntegerType)
        assert result["type"] == "none"

    def test_array_merge(self):
        """Test ArrayMerge."""
        func = get_builtin("ArrayMerge")
        arr = EastArray(IntegerType, [1, 2, 3])
        func(arr, 1, 10, lambda old, new, idx: old + new, IntegerType, IntegerType)
        assert list(arr) == [1, 12, 3]

    def test_array_append(self):
        """Test ArrayAppend."""
        func = get_builtin("ArrayAppend")
        arr1 = EastArray(IntegerType, [1, 2, 3])
        arr2 = EastArray(IntegerType, [4, 5])
        func(arr1, arr2, IntegerType)
        assert list(arr1) == [1, 2, 3, 4, 5]

    def test_array_prepend(self):
        """Test ArrayPrepend."""
        func = get_builtin("ArrayPrepend")
        arr1 = EastArray(IntegerType, [1, 2, 3])
        arr2 = EastArray(IntegerType, [4, 5])
        func(arr1, arr2, IntegerType)
        assert list(arr1) == [4, 5, 1, 2, 3]

    def test_array_merge_all(self):
        """Test ArrayMergeAll."""
        func = get_builtin("ArrayMergeAll")
        arr1 = EastArray(IntegerType, [1, 2, 3])
        arr2 = EastArray(IntegerType, [4, 5, 6])
        func(arr1, arr2, lambda a, b, i: a + b, IntegerType, IntegerType)
        assert list(arr1) == [5, 7, 9]

    def test_array_is_sorted(self):
        """Test ArrayIsSorted."""
        func = get_builtin("ArrayIsSorted")
        arr1 = EastArray(IntegerType, [1, 2, 3])
        arr2 = EastArray(IntegerType, [3, 1, 2])
        assert func(arr1, lambda x: x, IntegerType, IntegerType) is True
        assert func(arr2, lambda x: x, IntegerType, IntegerType) is False

    def test_array_find_sorted_first(self):
        """Test ArrayFindSortedFirst."""
        func = get_builtin("ArrayFindSortedFirst")
        arr = EastArray(IntegerType, [1, 2, 2, 2, 3])
        assert func(arr, 2, lambda x: x, IntegerType, IntegerType) == 1

    def test_array_find_sorted_last(self):
        """Test ArrayFindSortedLast."""
        func = get_builtin("ArrayFindSortedLast")
        arr = EastArray(IntegerType, [1, 2, 2, 2, 3])
        assert func(arr, 2, lambda x: x, IntegerType, IntegerType) == 4

    def test_array_find_sorted_range(self):
        """Test ArrayFindSortedRange."""
        func = get_builtin("ArrayFindSortedRange")
        arr = EastArray(IntegerType, [1, 2, 2, 2, 3])
        result = func(arr, 2, lambda x: x, IntegerType, IntegerType)
        assert result["start"] == 1
        assert result["end"] == 4

    def test_array_find_first(self):
        """Test ArrayFindFirst."""
        func = get_builtin("ArrayFindFirst")
        arr = EastArray(IntegerType, [1, 2, 3, 4])
        result = func(arr, 2, lambda x: x, IntegerType, IntegerType)
        assert result["type"] == "some"
        assert result["value"] == 1

    def test_array_get_keys(self):
        """Test ArrayGetKeys."""
        func = get_builtin("ArrayGetKeys")
        arr = EastArray(IntegerType, [10, 20, 30])
        indices = EastArray(IntegerType, [0, 1, 2])
        result = func(arr, indices, lambda idx: 0, IntegerType)
        assert list(result) == [10, 20, 30]

    def test_array_for_each(self):
        """Test ArrayForEach."""
        func = get_builtin("ArrayForEach")
        arr = EastArray(IntegerType, [1, 2, 3])
        total = [0]
        func(arr, lambda x, i: total.__setitem__(0, total[0] + x), IntegerType, IntegerType)
        assert total[0] == 6

    def test_array_filter_map(self):
        """Test ArrayFilterMap."""
        func = get_builtin("ArrayFilterMap")
        arr = EastArray(IntegerType, [1, 2, 3, 4])
        result = func(
            arr,
            lambda x, i: {"type": "some", "value": x * 2}
            if x % 2 == 0
            else {"type": "none", "value": None},
            IntegerType,
            IntegerType,
        )
        assert list(result) == [4, 8]

    def test_array_first_map(self):
        """Test ArrayFirstMap."""
        func = get_builtin("ArrayFirstMap")
        arr = EastArray(IntegerType, [1, 2, 3, 4])
        result = func(
            arr,
            lambda x, i: {"type": "some", "value": x * 2}
            if x > 2
            else {"type": "none", "value": None},
            IntegerType,
            IntegerType,
        )
        assert result["type"] == "some"
        assert result["value"] == 6

    def test_array_map_reduce(self):
        """Test ArrayMapReduce."""
        func = get_builtin("ArrayMapReduce")
        arr = EastArray(IntegerType, [1, 2, 3])
        result = func(arr, lambda x, i: x * 2, lambda a, b: a + b, IntegerType, IntegerType)
        assert result == 12

    def test_array_string_join(self):
        """Test ArrayStringJoin."""
        from east.types.types import StringType

        func = get_builtin("ArrayStringJoin")
        arr = EastArray(StringType, ["a", "b", "c"])
        result = func(arr, ",")
        assert result == "a,b,c"

    def test_array_to_set(self):
        """Test ArrayToSet."""
        func = get_builtin("ArrayToSet")
        arr = EastArray(IntegerType, [1, 2, 2, 3])
        result = func(arr, lambda x, i: x, IntegerType, IntegerType)
        assert len(result) == 3

    def test_array_to_dict(self):
        """Test ArrayToDict."""
        func = get_builtin("ArrayToDict")
        arr = EastArray(IntegerType, [1, 2, 3])
        result = func(
            arr,
            lambda x, i: x,
            lambda x, i: x * 2,
            lambda old, new, key: old,
            IntegerType,
            IntegerType,
            IntegerType,
        )
        assert len(result) == 3
        assert result[1] == 2

    def test_array_flatten_to_array(self):
        """Test ArrayFlattenToArray."""
        func = get_builtin("ArrayFlattenToArray")
        arr = EastArray(IntegerType, [1, 2, 3])
        result = func(arr, lambda x, i: EastArray(IntegerType, [x, x]), IntegerType, IntegerType)
        assert list(result) == [1, 1, 2, 2, 3, 3]

    def test_array_flatten_to_set(self):
        """Test ArrayFlattenToSet."""
        from east.types.containers import EastSet

        func = get_builtin("ArrayFlattenToSet")
        arr = EastArray(IntegerType, [1, 2, 3])
        result = func(arr, lambda x, i: EastSet(IntegerType, {x, x + 10}), IntegerType, IntegerType)
        assert len(result) == 6

    def test_array_flatten_to_dict(self):
        """Test ArrayFlattenToDict."""
        from east.types.containers import EastDict

        func = get_builtin("ArrayFlattenToDict")
        arr = EastArray(IntegerType, [1, 2])
        result = func(
            arr,
            lambda x, i: EastDict(IntegerType, IntegerType, {x: x * 2}),
            lambda old, new, key: old,
            IntegerType,
            IntegerType,
            IntegerType,
        )
        assert len(result) == 2

    def test_array_group_fold(self):
        """Test ArrayGroupFold."""
        func = get_builtin("ArrayGroupFold")
        arr = EastArray(IntegerType, [1, 2, 3, 4])
        result = func(
            arr,
            lambda x, i: x % 2,
            lambda key: 0,
            lambda acc, x, i: acc + x,
            IntegerType,
            IntegerType,
            IntegerType,
        )
        assert result[0] == 6  # 2 + 4
        assert result[1] == 4  # 1 + 3


class TestSetBuiltins:
    """Test set operations."""

    def test_set_size(self):
        """Test SetSize."""
        func = get_builtin("SetSize")
        s = EastSet(IntegerType, [1, 2, 3])
        assert func(s, IntegerType) == 3

    def test_set_has(self):
        """Test SetHas."""
        func = get_builtin("SetHas")
        s = EastSet(IntegerType, [1, 2, 3])
        assert func(s, 2, IntegerType) is True
        assert func(s, 5, IntegerType) is False

    def test_set_insert(self):
        """Test SetInsert."""
        func = get_builtin("SetInsert")
        s = EastSet(IntegerType, [1, 2, 3])
        func(s, 4, IntegerType)
        assert 4 in s

    def test_set_union(self):
        """Test SetUnion."""
        func = get_builtin("SetUnion")
        a = EastSet(IntegerType, [1, 2])
        b = EastSet(IntegerType, [2, 3])
        result = func(a, b, IntegerType)
        assert list(result) == [1, 2, 3]

    def test_set_intersect(self):
        """Test SetIntersect."""
        func = get_builtin("SetIntersect")
        a = EastSet(IntegerType, [1, 2, 3])
        b = EastSet(IntegerType, [2, 3, 4])
        result = func(a, b, IntegerType)
        assert list(result) == [2, 3]

    def test_set_delete(self):
        """Test SetDelete."""
        func = get_builtin("SetDelete")
        s = EastSet(IntegerType, [1, 2, 3])
        func(s, 2, IntegerType)
        assert list(s) == [1, 3]

    def test_set_clear(self):
        """Test SetClear."""
        func = get_builtin("SetClear")
        s = EastSet(IntegerType, [1, 2, 3])
        func(s, IntegerType)
        assert len(s) == 0

    def test_set_union_in_place(self):
        """Test SetUnionInPlace."""
        func = get_builtin("SetUnionInPlace")
        a = EastSet(IntegerType, [1, 2])
        b = EastSet(IntegerType, [2, 3])
        func(a, b, IntegerType)
        assert list(a) == [1, 2, 3]

    def test_set_diff(self):
        """Test SetDiff."""
        func = get_builtin("SetDiff")
        a = EastSet(IntegerType, [1, 2, 3])
        b = EastSet(IntegerType, [2, 3, 4])
        result = func(a, b, IntegerType)
        assert list(result) == [1]

    def test_set_sym_diff(self):
        """Test SetSymDiff."""
        func = get_builtin("SetSymDiff")
        a = EastSet(IntegerType, [1, 2, 3])
        b = EastSet(IntegerType, [2, 3, 4])
        result = func(a, b, IntegerType)
        assert sorted(result) == [1, 4]

    def test_set_is_subset(self):
        """Test SetIsSubset."""
        func = get_builtin("SetIsSubset")
        a = EastSet(IntegerType, [1, 2])
        b = EastSet(IntegerType, [1, 2, 3])
        assert func(a, b, IntegerType) is True
        assert func(b, a, IntegerType) is False

    def test_set_is_disjoint(self):
        """Test SetIsDisjoint."""
        func = get_builtin("SetIsDisjoint")
        a = EastSet(IntegerType, [1, 2])
        b = EastSet(IntegerType, [3, 4])
        c = EastSet(IntegerType, [2, 3])
        assert func(a, b, IntegerType) is True
        assert func(a, c, IntegerType) is False

    def test_set_copy(self):
        """Test SetCopy."""
        func = get_builtin("SetCopy")
        s = EastSet(IntegerType, [1, 2, 3])
        copy = func(s, IntegerType)
        assert list(copy) == [1, 2, 3]
        assert copy is not s

    def test_set_to_array(self):
        """Test SetToArray."""
        func = get_builtin("SetToArray")
        s = EastSet(IntegerType, [3, 1, 2])
        result = func(s, lambda x: x, IntegerType, IntegerType)  # Identity function
        assert isinstance(result, EastArray)
        assert sorted(result) == [1, 2, 3]

    def test_set_generate(self):
        """Test SetGenerate."""
        func = get_builtin("SetGenerate")
        result = func(5, lambda i: i * 2, lambda x: None, IntegerType)
        assert len(result) == 5
        assert 0 in result
        assert 8 in result

    def test_set_try_insert(self):
        """Test SetTryInsert."""
        func = get_builtin("SetTryInsert")
        s = EastSet(IntegerType, [1, 2])
        assert func(s, 3, IntegerType) is True
        assert func(s, 2, IntegerType) is False

    def test_set_try_delete(self):
        """Test SetTryDelete."""
        func = get_builtin("SetTryDelete")
        s = EastSet(IntegerType, [1, 2, 3])
        assert func(s, 2, IntegerType) is True
        assert func(s, 5, IntegerType) is False

    def test_set_for_each(self):
        """Test SetForEach."""
        func = get_builtin("SetForEach")
        s = EastSet(IntegerType, [1, 2, 3])
        total = [0]
        func(s, lambda x: total.__setitem__(0, total[0] + x), IntegerType, IntegerType)
        assert total[0] == 6

    def test_set_map(self):
        """Test SetMap."""
        func = get_builtin("SetMap")
        s = EastSet(IntegerType, [1, 2, 3])
        result = func(s, lambda x: x * 2, IntegerType, IntegerType)
        assert isinstance(result, EastDict)
        assert result[1] == 2
        assert result[2] == 4

    def test_set_filter(self):
        """Test SetFilter."""
        func = get_builtin("SetFilter")
        s = EastSet(IntegerType, [1, 2, 3, 4])
        result = func(s, lambda x: x % 2 == 0, IntegerType)
        assert len(result) == 2
        assert 2 in result
        assert 4 in result

    def test_set_filter_map(self):
        """Test SetFilterMap."""
        func = get_builtin("SetFilterMap")
        s = EastSet(IntegerType, [1, 2, 3, 4])
        result = func(
            s,
            lambda x: {"type": "some", "value": x * 2}
            if x % 2 == 0
            else {"type": "none", "value": None},
            IntegerType,
            IntegerType,
        )
        assert 2 in result
        assert result[2] == 4

    def test_set_first_map(self):
        """Test SetFirstMap."""
        func = get_builtin("SetFirstMap")
        s = EastSet(IntegerType, [1, 2, 3, 4])
        result = func(
            s,
            lambda x: {"type": "some", "value": x * 2}
            if x > 2
            else {"type": "none", "value": None},
            IntegerType,
            IntegerType,
        )
        assert result["type"] == "some"

    def test_set_map_reduce(self):
        """Test SetMapReduce."""
        func = get_builtin("SetMapReduce")
        s = EastSet(IntegerType, [1, 2, 3])
        result = func(s, lambda x: x * 2, lambda a, b: a + b, IntegerType, IntegerType)
        assert result == 12

    def test_set_reduce(self):
        """Test SetReduce."""
        func = get_builtin("SetReduce")
        s = EastSet(IntegerType, [1, 2, 3])
        result = func(s, lambda acc, x: acc + x, 0, IntegerType, IntegerType)
        assert result == 6

    def test_set_to_set(self):
        """Test SetToSet."""
        func = get_builtin("SetToSet")
        s = EastSet(IntegerType, [1, 2, 3])
        result = func(s, lambda x: x * 2, IntegerType, IntegerType)
        assert len(result) == 3
        assert 2 in result
        assert 6 in result

    def test_set_to_dict(self):
        """Test SetToDict."""
        func = get_builtin("SetToDict")
        s = EastSet(IntegerType, [1, 2, 3])
        result = func(
            s,
            lambda x: x,
            lambda x: x * 2,
            lambda old, new, key: old,
            IntegerType,
            IntegerType,
            IntegerType,
        )
        assert len(result) == 3
        assert result[1] == 2

    def test_set_flatten_to_array(self):
        """Test SetFlattenToArray."""
        func = get_builtin("SetFlattenToArray")
        s = EastSet(IntegerType, [1, 2])
        result = func(s, lambda x: EastArray(IntegerType, [x, x]), IntegerType, IntegerType)
        assert len(result) >= 2

    def test_set_flatten_to_set(self):
        """Test SetFlattenToSet."""
        func = get_builtin("SetFlattenToSet")
        s = EastSet(IntegerType, [1, 2])
        result = func(s, lambda x: EastSet(IntegerType, {x, x + 10}), IntegerType, IntegerType)
        assert len(result) == 4

    def test_set_flatten_to_dict(self):
        """Test SetFlattenToDict."""
        func = get_builtin("SetFlattenToDict")
        s = EastSet(IntegerType, [1, 2])
        result = func(
            s,
            lambda x: EastDict(IntegerType, IntegerType, {x: x * 2}),
            lambda old, new, key: old,
            IntegerType,
            IntegerType,
            IntegerType,
        )
        assert len(result) == 2

    def test_set_group_fold(self):
        """Test SetGroupFold."""
        func = get_builtin("SetGroupFold")
        s = EastSet(IntegerType, [1, 2, 3, 4])
        result = func(
            s,
            lambda x: x % 2,
            lambda key: 0,
            lambda acc, x: acc + x,
            IntegerType,
            IntegerType,
            IntegerType,
        )
        assert result[0] == 6  # 2 + 4
        assert result[1] == 4  # 1 + 3


class TestDictBuiltins:
    """Test dict operations."""

    def test_dict_size(self):
        """Test DictSize."""
        func = get_builtin("DictSize")
        d = EastDict(StringType, IntegerType, {"a": 1, "b": 2})
        assert func(d, StringType, IntegerType) == 2

    def test_dict_has(self):
        """Test DictHas."""
        func = get_builtin("DictHas")
        d = EastDict(StringType, IntegerType, {"a": 1, "b": 2})
        assert func(d, "a", StringType, IntegerType) is True
        assert func(d, "c", StringType, IntegerType) is False

    def test_dict_get(self):
        """Test DictGet."""
        func = get_builtin("DictGet")
        d = EastDict(StringType, IntegerType, {"a": 1, "b": 2})
        assert func(d, "a", StringType, IntegerType) == 1
        assert func(d, "b", StringType, IntegerType) == 2

    def test_dict_insert(self):
        """Test DictInsert."""
        func = get_builtin("DictInsert")
        d = EastDict(StringType, IntegerType, {"a": 1})
        func(d, "b", 2, StringType, IntegerType)
        assert d["b"] == 2

    def test_dict_keys(self):
        """Test DictKeys."""
        func = get_builtin("DictKeys")
        d = EastDict(StringType, IntegerType, {"a": 1, "b": 2})
        result = func(d, StringType, IntegerType)
        assert set(result) == {"a", "b"}

    def test_dict_generate(self):
        """Test DictGenerate."""
        func = get_builtin("DictGenerate")
        result = func(
            5, lambda i: i, lambda i: i * 2, lambda old, new, key: old, IntegerType, IntegerType
        )
        assert len(result) == 5
        assert result[0] == 0
        assert result[4] == 8

    def test_dict_try_get(self):
        """Test DictTryGet."""
        func = get_builtin("DictTryGet")
        d = EastDict(StringType, IntegerType, {"a": 1})
        result = func(d, "a", StringType, IntegerType)
        assert result["type"] == "some"
        assert result["value"] == 1
        result = func(d, "z", StringType, IntegerType)
        assert result["type"] == "none"

    def test_dict_get_or_default(self):
        """Test DictGetOrDefault."""
        func = get_builtin("DictGetOrDefault")
        d = EastDict(StringType, IntegerType, {"a": 1})
        assert func(d, "a", lambda key: 99, StringType, IntegerType) == 1
        assert func(d, "z", lambda key: 99, StringType, IntegerType) == 99

    def test_dict_get_or_insert(self):
        """Test DictGetOrInsert."""
        func = get_builtin("DictGetOrInsert")
        d = EastDict(StringType, IntegerType, {"a": 1})
        assert func(d, "a", lambda key: 99, StringType, IntegerType) == 1
        assert func(d, "b", lambda key: 2, StringType, IntegerType) == 2
        assert d["b"] == 2

    def test_dict_insert_or_update(self):
        """Test DictInsertOrUpdate."""
        func = get_builtin("DictInsertOrUpdate")
        d = EastDict(StringType, IntegerType, {"a": 1})
        func(d, "a", 5, lambda old, new, key: old + new, StringType, IntegerType)
        assert d["a"] == 6
        func(d, "b", 2, lambda old, new, key: old + new, StringType, IntegerType)
        assert d["b"] == 2

    def test_dict_update(self):
        """Test DictUpdate."""
        func = get_builtin("DictUpdate")
        d = EastDict(StringType, IntegerType, {"a": 1})
        func(d, "a", 10, StringType, IntegerType)
        assert d["a"] == 10

    def test_dict_swap(self):
        """Test DictSwap."""
        func = get_builtin("DictSwap")
        d = EastDict(StringType, IntegerType, {"a": 1})
        old = func(d, "a", 10, StringType, IntegerType)
        assert old == 1
        assert d["a"] == 10

    def test_dict_merge(self):
        """Test DictMerge."""
        func = get_builtin("DictMerge")
        d = EastDict(StringType, IntegerType, {"a": 1})
        func(
            d,
            "b",
            2,
            lambda old, new, key: new,
            lambda key: 0,
            StringType,
            IntegerType,
            IntegerType,
        )
        assert len(d) == 2
        assert d["b"] == 2

    def test_dict_delete(self):
        """Test DictDelete."""
        func = get_builtin("DictDelete")
        d = EastDict(StringType, IntegerType, {"a": 1, "b": 2})
        func(d, "a", StringType, IntegerType)
        assert "a" not in d

    def test_dict_try_delete(self):
        """Test DictTryDelete."""
        func = get_builtin("DictTryDelete")
        d = EastDict(StringType, IntegerType, {"a": 1, "b": 2})
        assert func(d, "a", StringType, IntegerType) is True
        assert func(d, "z", StringType, IntegerType) is False

    def test_dict_pop(self):
        """Test DictPop."""
        func = get_builtin("DictPop")
        d = EastDict(StringType, IntegerType, {"a": 1, "b": 2})
        value = func(d, "a", StringType, IntegerType)
        assert value == 1
        assert "a" not in d

    def test_dict_clear(self):
        """Test DictClear."""
        func = get_builtin("DictClear")
        d = EastDict(StringType, IntegerType, {"a": 1, "b": 2})
        func(d, StringType, IntegerType)
        assert len(d) == 0

    def test_dict_union_in_place(self):
        """Test DictUnionInPlace."""
        func = get_builtin("DictUnionInPlace")
        d1 = EastDict(StringType, IntegerType, {"a": 1})
        d2 = EastDict(StringType, IntegerType, {"b": 2})
        func(d1, d2, lambda old, new, key: old, StringType, IntegerType)
        assert len(d1) == 2

    def test_dict_merge_all(self):
        """Test DictMergeAll."""
        func = get_builtin("DictMergeAll")
        d1 = EastDict(StringType, IntegerType, {"a": 1})
        d2 = EastDict(StringType, IntegerType, {"b": 2})
        func(d1, d2, lambda old, new, key: old, lambda key: 0, StringType, IntegerType, IntegerType)
        assert len(d1) == 2

    def test_dict_get_keys(self):
        """Test DictGetKeys."""
        from east.types.containers import EastSet

        func = get_builtin("DictGetKeys")
        d = EastDict(StringType, IntegerType, {"a": 1, "b": 2, "c": 3})
        keys = EastSet(StringType, {"a", "c", "z"})
        result = func(d, keys, lambda key: 99, StringType, IntegerType)
        assert result["a"] == 1
        assert result["c"] == 3
        assert result["z"] == 99

    def test_dict_for_each(self):
        """Test DictForEach."""
        func = get_builtin("DictForEach")
        d = EastDict(StringType, IntegerType, {"a": 1, "b": 2})
        total = [0]
        func(
            d, lambda v, k: total.__setitem__(0, total[0] + v), StringType, IntegerType, IntegerType
        )
        assert total[0] == 3

    def test_dict_copy(self):
        """Test DictCopy."""
        func = get_builtin("DictCopy")
        d = EastDict(StringType, IntegerType, {"a": 1, "b": 2})
        copy = func(d, StringType, IntegerType)
        assert len(copy) == 2
        assert copy is not d

    def test_dict_map(self):
        """Test DictMap."""
        func = get_builtin("DictMap")
        d = EastDict(StringType, IntegerType, {"a": 1, "b": 2})
        result = func(d, lambda v, k: v * 2, StringType, IntegerType, IntegerType)
        assert result["a"] == 2
        assert result["b"] == 4

    def test_dict_filter(self):
        """Test DictFilter."""
        func = get_builtin("DictFilter")
        d = EastDict(StringType, IntegerType, {"a": 1, "b": 2, "c": 3})
        result = func(d, lambda v, k: v % 2 == 0, StringType, IntegerType)
        assert len(result) == 1
        assert result["b"] == 2

    def test_dict_filter_map(self):
        """Test DictFilterMap."""
        func = get_builtin("DictFilterMap")
        d = EastDict(StringType, IntegerType, {"a": 1, "b": 2, "c": 3})
        result = func(
            d,
            lambda v, k: {"type": "some", "value": v * 2}
            if v % 2 == 0
            else {"type": "none", "value": None},
            StringType,
            IntegerType,
            IntegerType,
        )
        assert len(result) == 1
        assert result["b"] == 4

    def test_dict_first_map(self):
        """Test DictFirstMap."""
        func = get_builtin("DictFirstMap")
        d = EastDict(StringType, IntegerType, {"a": 1, "b": 2})
        result = func(
            d,
            lambda v, k: {"type": "some", "value": v * 2}
            if v > 1
            else {"type": "none", "value": None},
            StringType,
            IntegerType,
            IntegerType,
        )
        assert result["type"] == "some"
        assert result["value"] == 4

    def test_dict_map_reduce(self):
        """Test DictMapReduce."""
        func = get_builtin("DictMapReduce")
        d = EastDict(StringType, IntegerType, {"a": 1, "b": 2, "c": 3})
        result = func(
            d, lambda v, k: v * 2, lambda a, b: a + b, StringType, IntegerType, IntegerType
        )
        assert result == 12

    def test_dict_reduce(self):
        """Test DictReduce."""
        func = get_builtin("DictReduce")
        d = EastDict(StringType, IntegerType, {"a": 1, "b": 2})
        result = func(d, lambda acc, v, k: acc + v, 0, StringType, IntegerType, IntegerType)
        assert result == 3

    def test_dict_to_array(self):
        """Test DictToArray."""
        func = get_builtin("DictToArray")
        d = EastDict(StringType, IntegerType, {"a": 1, "b": 2})
        result = func(d, lambda v, k: v, StringType, IntegerType, IntegerType)
        assert len(result) == 2

    def test_dict_to_set(self):
        """Test DictToSet."""
        func = get_builtin("DictToSet")
        d = EastDict(StringType, IntegerType, {"a": 1, "b": 2})
        result = func(d, lambda v, k: v, StringType, IntegerType, IntegerType)
        assert len(result) == 2

    def test_dict_to_dict(self):
        """Test DictToDict."""
        func = get_builtin("DictToDict")
        d = EastDict(StringType, IntegerType, {"a": 1, "b": 2})
        result = func(
            d,
            lambda v, k: k.upper(),
            lambda v, k: v * 2,
            lambda old, new, key: old,
            StringType,
            IntegerType,
            StringType,
            IntegerType,
        )
        assert result["A"] == 2
        assert result["B"] == 4

    def test_dict_flatten_to_array(self):
        """Test DictFlattenToArray."""
        func = get_builtin("DictFlattenToArray")
        d = EastDict(StringType, IntegerType, {"a": 1, "b": 2})
        result = func(
            d, lambda v, k: EastArray(IntegerType, [v, v]), StringType, IntegerType, IntegerType
        )
        assert len(result) == 4

    def test_dict_flatten_to_set(self):
        """Test DictFlattenToSet."""
        from east.types.containers import EastSet

        func = get_builtin("DictFlattenToSet")
        d = EastDict(StringType, IntegerType, {"a": 1, "b": 2})
        result = func(
            d, lambda v, k: EastSet(IntegerType, {v, v + 10}), StringType, IntegerType, IntegerType
        )
        assert len(result) == 4

    def test_dict_flatten_to_dict(self):
        """Test DictFlattenToDict."""
        func = get_builtin("DictFlattenToDict")
        d = EastDict(StringType, IntegerType, {"a": 1})
        result = func(
            d,
            lambda v, k: EastDict(IntegerType, IntegerType, {v: v * 2}),
            lambda old, new, key: old,
            StringType,
            IntegerType,
            IntegerType,
            IntegerType,
        )
        assert len(result) == 1

    def test_dict_group_fold(self):
        """Test DictGroupFold."""
        func = get_builtin("DictGroupFold")
        d = EastDict(StringType, IntegerType, {"a": 1, "b": 2, "c": 3, "d": 4})
        result = func(
            d,
            lambda v, k: v % 2,
            lambda key: 0,
            lambda acc, v, k: acc + v,
            StringType,
            IntegerType,
            IntegerType,
            IntegerType,
        )
        assert result[0] == 6  # 2 + 4
        assert result[1] == 4  # 1 + 3


class TestDateTimeFormatBuiltins:
    """Tests for DateTimePrintFormat and DateTimeParseFormat builtins."""

    def test_datetime_print_format_basic_iso_8601(self):
        """Test DateTimePrintFormat with ISO 8601 format."""
        func = get_builtin("DateTimePrintFormat")
        dt = datetime(2025, 1, 15, 14, 30, 45, 123000, tzinfo=UTC)
        tokens = tokenize_datetime_format("YYYY-MM-DD HH:mm:ss")
        result = func(dt, tokens)
        assert result == "2025-01-15 14:30:45"

    def test_datetime_print_format_with_milliseconds(self):
        """Test DateTimePrintFormat with milliseconds."""
        func = get_builtin("DateTimePrintFormat")
        dt = datetime(2025, 1, 15, 14, 30, 45, 123000, tzinfo=UTC)
        tokens = tokenize_datetime_format("YYYY-MM-DD HH:mm:ss.SSS")
        result = func(dt, tokens)
        assert result == "2025-01-15 14:30:45.123"

    def test_datetime_print_format_12_hour(self):
        """Test DateTimePrintFormat with 12-hour format."""
        func = get_builtin("DateTimePrintFormat")
        dt = datetime(2025, 1, 15, 14, 30, 45, tzinfo=UTC)
        tokens = tokenize_datetime_format("h:mm A")
        result = func(dt, tokens)
        assert result == "2:30 PM"

    def test_datetime_print_format_month_names(self):
        """Test DateTimePrintFormat with month names."""
        func = get_builtin("DateTimePrintFormat")
        dt = datetime(2025, 1, 15, tzinfo=UTC)
        tokens = tokenize_datetime_format("MMMM D, YYYY")
        result = func(dt, tokens)
        assert result == "January 15, 2025"

    def test_datetime_parse_format_basic_iso_8601(self):
        """Test DateTimeParseFormat with ISO 8601 format."""
        func = get_builtin("DateTimeParseFormat")
        tokens = tokenize_datetime_format("YYYY-MM-DD HH:mm:ss")
        result = func("2025-01-15 14:30:45", tokens)

        assert result.year == 2025
        assert result.month == 1
        assert result.day == 15
        assert result.hour == 14
        assert result.minute == 30
        assert result.second == 45

    def test_datetime_parse_format_with_milliseconds(self):
        """Test DateTimeParseFormat with milliseconds."""
        func = get_builtin("DateTimeParseFormat")
        tokens = tokenize_datetime_format("YYYY-MM-DD HH:mm:ss.SSS")
        result = func("2025-01-15 14:30:45.123", tokens)

        assert result.microsecond == 123000

    def test_datetime_parse_format_12_hour(self):
        """Test DateTimeParseFormat with 12-hour format."""
        func = get_builtin("DateTimeParseFormat")
        tokens = tokenize_datetime_format("h:mm A")
        result = func("2:30 PM", tokens)

        assert result.hour == 14
        assert result.minute == 30

    def test_datetime_parse_format_month_names(self):
        """Test DateTimeParseFormat with month names."""
        func = get_builtin("DateTimeParseFormat")
        tokens = tokenize_datetime_format("MMMM D, YYYY")
        result = func("January 15, 2025", tokens)

        assert result.year == 2025
        assert result.month == 1
        assert result.day == 15

    def test_datetime_parse_format_error(self):
        """Test DateTimeParseFormat raises ValueError on parse error."""
        func = get_builtin("DateTimeParseFormat")
        tokens = tokenize_datetime_format("YYYY-MM-DD")

        with pytest.raises(ValueError) as exc_info:
            func("invalid", tokens)

        assert "Failed to parse datetime" in str(exc_info.value)
        assert "position" in str(exc_info.value)

    def test_datetime_format_round_trip_iso_8601(self):
        """Test round trip formatting and parsing with ISO 8601."""
        print_func = get_builtin("DateTimePrintFormat")
        parse_func = get_builtin("DateTimeParseFormat")

        format_str = "YYYY-MM-DD HH:mm:ss.SSS"
        tokens = tokenize_datetime_format(format_str)

        original = datetime(2025, 1, 15, 14, 30, 45, 123000, tzinfo=UTC)
        formatted = print_func(original, tokens)
        parsed = parse_func(formatted, tokens)

        assert parsed.replace(tzinfo=UTC).timestamp() == original.timestamp()

    def test_datetime_format_round_trip_12_hour(self):
        """Test round trip formatting and parsing with 12-hour format."""
        print_func = get_builtin("DateTimePrintFormat")
        parse_func = get_builtin("DateTimeParseFormat")

        format_str = "MMMM D, YYYY h:mm A"
        tokens = tokenize_datetime_format(format_str)

        original = datetime(2025, 1, 15, 14, 30, 0, tzinfo=UTC)
        formatted = print_func(original, tokens)
        parsed = parse_func(formatted, tokens)

        # Hour and minute should match (seconds/milliseconds not in format)
        assert parsed.year == original.year
        assert parsed.month == original.month
        assert parsed.day == original.day
        assert parsed.hour == original.hour
        assert parsed.minute == original.minute


class TestDateTimeBuiltins:
    """Tests for DateTime component and manipulation builtins."""

    def test_datetime_get_year(self):
        """Test DateTimeGetYear."""
        func = get_builtin("DateTimeGetYear")
        dt = datetime(2025, 1, 15, 14, 30, 45, tzinfo=UTC)
        assert func(dt) == 2025

    def test_datetime_get_month(self):
        """Test DateTimeGetMonth."""
        func = get_builtin("DateTimeGetMonth")
        dt = datetime(2025, 1, 15, 14, 30, 45, tzinfo=UTC)
        assert func(dt) == 1

    def test_datetime_get_day_of_month(self):
        """Test DateTimeGetDayOfMonth."""
        func = get_builtin("DateTimeGetDayOfMonth")
        dt = datetime(2025, 1, 15, 14, 30, 45, tzinfo=UTC)
        assert func(dt) == 15

    def test_datetime_get_hour(self):
        """Test DateTimeGetHour."""
        func = get_builtin("DateTimeGetHour")
        dt = datetime(2025, 1, 15, 14, 30, 45, tzinfo=UTC)
        assert func(dt) == 14

    def test_datetime_get_minute(self):
        """Test DateTimeGetMinute."""
        func = get_builtin("DateTimeGetMinute")
        dt = datetime(2025, 1, 15, 14, 30, 45, tzinfo=UTC)
        assert func(dt) == 30

    def test_datetime_get_second(self):
        """Test DateTimeGetSecond."""
        func = get_builtin("DateTimeGetSecond")
        dt = datetime(2025, 1, 15, 14, 30, 45, tzinfo=UTC)
        assert func(dt) == 45

    def test_datetime_get_millisecond(self):
        """Test DateTimeGetMillisecond."""
        func = get_builtin("DateTimeGetMillisecond")
        dt = datetime(2025, 1, 15, 14, 30, 45, 123000, tzinfo=UTC)
        assert func(dt) == 123

    def test_datetime_get_day_of_week(self):
        """Test DateTimeGetDayOfWeek."""
        func = get_builtin("DateTimeGetDayOfWeek")
        # 2025-01-15 is a Wednesday (ISO 8601: 3 = Wed, where 1=Mon, 7=Sun)
        dt = datetime(2025, 1, 15, 14, 30, 45, tzinfo=UTC)
        assert func(dt) == 3

    def test_datetime_from_components(self):
        """Test DateTimeFromComponents."""
        func = get_builtin("DateTimeFromComponents")
        dt = func(2025, 1, 15, 14, 30, 45, 123)
        assert dt.year == 2025
        assert dt.month == 1
        assert dt.day == 15
        assert dt.hour == 14
        assert dt.minute == 30
        assert dt.second == 45
        assert dt.microsecond == 123000

    def test_datetime_to_epoch_milliseconds(self):
        """Test DateTimeToEpochMilliseconds."""
        func = get_builtin("DateTimeToEpochMilliseconds")
        dt = datetime(2025, 1, 15, 14, 30, 45, 123000, tzinfo=UTC)
        epoch_ms = func(dt)
        assert isinstance(epoch_ms, int)
        # Should be approximately 1736951445123
        assert epoch_ms > 1700000000000  # After 2023
        assert epoch_ms < 1800000000000  # Before 2027

    def test_datetime_from_epoch_milliseconds(self):
        """Test DateTimeFromEpochMilliseconds."""
        func = get_builtin("DateTimeFromEpochMilliseconds")
        epoch_ms = 1736951445123
        dt = func(epoch_ms)
        assert dt.year == 2025
        assert dt.month == 1
        assert dt.day == 15

    def test_datetime_add_milliseconds(self):
        """Test DateTimeAddMilliseconds."""
        func = get_builtin("DateTimeAddMilliseconds")
        dt = datetime(2025, 1, 15, 14, 30, 45, 0, tzinfo=UTC)
        result = func(dt, 1000)
        assert result.second == 46

    def test_datetime_duration_milliseconds(self):
        """Test DateTimeDurationMilliseconds."""
        func = get_builtin("DateTimeDurationMilliseconds")
        dt1 = datetime(2025, 1, 15, 14, 30, 45, 0, tzinfo=UTC)
        dt2 = datetime(2025, 1, 15, 14, 30, 46, 0, tzinfo=UTC)
        duration = func(dt1, dt2)
        assert duration == -1000  # dt1 is 1 second before dt2


class TestRefBuiltins:
    """Test ref operations."""

    def test_ref_get(self):
        """Test Ref.Get."""
        from east.types.ref import ref

        func = get_builtin("RefGet")
        r = ref(42)
        assert func(r, IntegerType) == 42

        r_str = ref("hello")
        assert func(r_str, StringType) == "hello"

    def test_ref_update(self):
        """Test Ref.Update."""
        from east.types.ref import deref, ref

        func = get_builtin("RefUpdate")
        r = ref(0)
        result = func(r, 100, IntegerType)
        assert result is None
        assert deref(r) == 100

        # Update again
        func(r, 200, IntegerType)
        assert deref(r) == 200

    def test_ref_merge(self):
        """Test Ref.Merge."""
        from east.types.ref import deref, ref

        func = get_builtin("RefMerge")

        # Test with integer addition
        r = ref(10)
        result = func(r, 5, lambda cur, delta: cur + delta, IntegerType, IntegerType)
        assert result is None
        assert deref(r) == 15

        # Merge again
        func(r, 20, lambda cur, delta: cur + delta, IntegerType, IntegerType)
        assert deref(r) == 35

        # Test with string concatenation
        r_str = ref("hello")
        func(r_str, " world", lambda cur, new: cur + new, StringType, StringType)
        assert deref(r_str) == "hello world"
