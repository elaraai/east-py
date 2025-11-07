"""Tests for builtin functions."""

import math

import pytest

from east.builtins import get_builtin, list_builtins
from east.types.containers import EastArray, EastDict, EastSet
from east.types.type_system import IntegerType, StringType


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


class TestComparisonBuiltins:
    """Test comparison operations."""

    def test_equal(self):
        """Test Equal."""
        func = get_builtin("Equal")
        assert func(42, 42) is True
        assert func(42, 43) is False
        assert func("hello", "hello") is True
        assert func("hello", "world") is False

    def test_not_equal(self):
        """Test NotEqual."""
        func = get_builtin("NotEqual")
        assert func(42, 43) is True
        assert func(42, 42) is False

    def test_less(self):
        """Test Less."""
        func = get_builtin("Less")
        assert func(1, 2) is True
        assert func(2, 1) is False
        assert func(2, 2) is False

    def test_greater(self):
        """Test Greater."""
        func = get_builtin("Greater")
        assert func(2, 1) is True
        assert func(1, 2) is False
        assert func(2, 2) is False


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

    def test_float_is_nan(self):
        """Test FloatIsNaN."""
        func = get_builtin("FloatIsNaN")
        assert func(float("nan")) is True
        assert func(1.0) is False
        assert func(float("inf")) is False

    def test_float_is_infinite(self):
        """Test FloatIsInfinite."""
        func = get_builtin("FloatIsInfinite")
        assert func(float("inf")) is True
        assert func(float("-inf")) is True
        assert func(1.0) is False

    def test_float_is_finite(self):
        """Test FloatIsFinite."""
        func = get_builtin("FloatIsFinite")
        assert func(1.0) is True
        assert func(float("inf")) is False
        assert func(float("nan")) is False

    def test_float_to_string(self):
        """Test FloatToString."""
        func = get_builtin("FloatToString")
        assert func(3.14) == "3.14"
        assert func(float("nan")) == "NaN"
        assert func(float("inf")) == "Infinity"
        assert func(float("-inf")) == "-Infinity"


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

    def test_string_get(self):
        """Test StringGet."""
        func = get_builtin("StringGet")
        assert func("hello", 0) == "h"
        assert func("hello", 4) == "o"

    def test_string_slice(self):
        """Test StringSlice."""
        func = get_builtin("StringSlice")
        assert func("hello", 1, 4) == "ell"
        assert func("hello", 0, 5) == "hello"

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

    def test_string_to_integer(self):
        """Test StringToInteger."""
        func = get_builtin("StringToInteger")
        assert func("42") == 42
        assert func("-100") == -100

    def test_string_to_float(self):
        """Test StringToFloat."""
        func = get_builtin("StringToFloat")
        assert func("3.14") == 3.14
        assert func("NaN") != func("NaN")  # NaN != NaN
        assert math.isnan(func("NaN"))
        assert func("Infinity") == float("inf")


class TestArrayBuiltins:
    """Test array operations."""

    def test_array_size(self):
        """Test ArraySize."""
        func = get_builtin("ArraySize")
        arr = EastArray(IntegerType, [1, 2, 3])
        assert func(arr) == 3

    def test_array_get(self):
        """Test ArrayGet."""
        func = get_builtin("ArrayGet")
        arr = EastArray(IntegerType, [1, 2, 3])
        assert func(arr, 0) == 1
        assert func(arr, 2) == 3

    def test_array_update(self):
        """Test ArrayUpdate."""
        func = get_builtin("ArrayUpdate")
        arr = EastArray(IntegerType, [1, 2, 3])
        func(arr, 1, 42)
        assert arr[1] == 42

    def test_array_push_last(self):
        """Test ArrayPushLast."""
        func = get_builtin("ArrayPushLast")
        arr = EastArray(IntegerType, [1, 2, 3])
        func(arr, 4)
        assert list(arr) == [1, 2, 3, 4]

    def test_array_slice(self):
        """Test ArraySlice."""
        func = get_builtin("ArraySlice")
        arr = EastArray(IntegerType, [1, 2, 3, 4, 5])
        result = func(arr, 1, 4)
        assert list(result) == [2, 3, 4]

    def test_array_concat(self):
        """Test ArrayConcat."""
        func = get_builtin("ArrayConcat")
        a = EastArray(IntegerType, [1, 2])
        b = EastArray(IntegerType, [3, 4])
        result = func(a, b)
        assert list(result) == [1, 2, 3, 4]

    def test_array_contains(self):
        """Test ArrayContains."""
        func = get_builtin("ArrayContains")
        arr = EastArray(IntegerType, [1, 2, 3])
        assert func(arr, 2) is True
        assert func(arr, 5) is False


class TestSetBuiltins:
    """Test set operations."""

    def test_set_size(self):
        """Test SetSize."""
        func = get_builtin("SetSize")
        s = EastSet(IntegerType, [1, 2, 3])
        assert func(s) == 3

    def test_set_has(self):
        """Test SetHas."""
        func = get_builtin("SetHas")
        s = EastSet(IntegerType, [1, 2, 3])
        assert func(s, 2) is True
        assert func(s, 5) is False

    def test_set_insert(self):
        """Test SetInsert."""
        func = get_builtin("SetInsert")
        s = EastSet(IntegerType, [1, 2, 3])
        func(s, 4)
        assert 4 in s

    def test_set_union(self):
        """Test SetUnion."""
        func = get_builtin("SetUnion")
        a = EastSet(IntegerType, [1, 2])
        b = EastSet(IntegerType, [2, 3])
        result = func(a, b)
        assert list(result) == [1, 2, 3]

    def test_set_intersect(self):
        """Test SetIntersect."""
        func = get_builtin("SetIntersect")
        a = EastSet(IntegerType, [1, 2, 3])
        b = EastSet(IntegerType, [2, 3, 4])
        result = func(a, b)
        assert list(result) == [2, 3]


class TestDictBuiltins:
    """Test dict operations."""

    def test_dict_size(self):
        """Test DictSize."""
        func = get_builtin("DictSize")
        d = EastDict(StringType, IntegerType, {"a": 1, "b": 2})
        assert func(d) == 2

    def test_dict_has(self):
        """Test DictHas."""
        func = get_builtin("DictHas")
        d = EastDict(StringType, IntegerType, {"a": 1, "b": 2})
        assert func(d, "a") is True
        assert func(d, "c") is False

    def test_dict_get(self):
        """Test DictGet."""
        func = get_builtin("DictGet")
        d = EastDict(StringType, IntegerType, {"a": 1, "b": 2})
        assert func(d, "a") == 1
        assert func(d, "b") == 2

    def test_dict_insert(self):
        """Test DictInsert."""
        func = get_builtin("DictInsert")
        d = EastDict(StringType, IntegerType, {"a": 1})
        func(d, "b", 2)
        assert d["b"] == 2

    def test_dict_keys(self):
        """Test DictKeys."""
        func = get_builtin("DictKeys")
        d = EastDict(StringType, IntegerType, {"a": 1, "b": 2})
        result = func(d)
        assert list(result) == ["a", "b"]

    def test_dict_values(self):
        """Test DictValues."""
        func = get_builtin("DictValues")
        d = EastDict(StringType, IntegerType, {"a": 1, "b": 2})
        result = func(d)
        assert list(result) == [1, 2]
