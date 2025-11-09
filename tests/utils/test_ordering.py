"""Tests for East comparison/ordering functions.

Ported from East/src/comparison.spec.ts
"""

import math
from datetime import UTC, datetime
from typing import Any

import pytest

from east.types.primitives import Blob
from east.types.type_system import (
    ArrayType,
    BlobType,
    BooleanType,
    DateTimeType,
    DictType,
    FloatType,
    FunctionType,
    IntegerType,
    NeverType,
    NullType,
    SetType,
    StringType,
    StructType,
    VariantType,
    recursive_type,
)
from east.utils.ordering import (
    compare_for,
    equal_for,
    greater_equal_for,
    greater_for,
    is_for,
    less_equal_for,
    less_for,
    not_equal_for,
)


def variant(type_tag: str, value: Any) -> dict[str, Any]:
    """Create a variant value with given type tag and value."""
    return {"type": type_tag, "value": value}


def run(type_val: Any, values: list[Any]) -> None:
    """Helper function to test ordering consistency.

    For a list of values in sorted order, verify that:
    - equal(x, y) is true iff i == j
    - less(x, y) is true iff i < j
    - compare(x, y) returns -1 if i < j, 0 if i == j, 1 if i > j
    """
    equal = equal_for(type_val)
    less = less_for(type_val)
    compare = compare_for(type_val)

    for i in range(len(values)):
        x = values[i]
        for j in range(len(values)):
            y = values[j]
            assert equal(x, y) == (i == j), f"equal({x}, {y}) should be {i == j}"
            assert less(x, y) == (i < j), f"less({x}, {y}) should be {i < j}"
            expected_cmp = -1 if i < j else (0 if i == j else 1)
            assert compare(x, y) == expected_cmp, f"compare({x}, {y}) should be {expected_cmp}"


class TestPrimitiveComparisons:
    """Test comparisons of primitive types."""

    def test_should_compare_nulls(self):
        """Should compare nulls."""
        type_val = NullType
        values = [None]
        run(type_val, values)

    def test_should_compare_booleans(self):
        """Should compare booleans."""
        type_val = BooleanType
        values = [False, True]
        run(type_val, values)

    def test_should_compare_integers(self):
        """Should compare integers."""
        type_val = IntegerType
        values = [
            -9223372036854775808,
            -1,
            0,
            42,
            90071992547409919,
            9223372036854775807,
        ]
        run(type_val, values)

    def test_should_compare_floats(self):
        """Should compare floats."""
        type_val = FloatType
        values = [
            -math.inf,
            -1e8,
            -3.14,
            -1.0,
            -1e-8,
            -1e-16,
            -0.0,
            0.0,
            1e-16,
            1e-8,
            1.0,
            3.14,
            1e8,
            math.inf,
            math.nan,
        ]
        run(type_val, values)

    def test_should_compare_dates(self):
        """Should compare dates."""
        type_val = DateTimeType
        values = [
            datetime.fromtimestamp(0, tz=UTC),
            datetime.fromisoformat("2022-06-29T13:43:00.123Z"),
        ]
        run(type_val, values)

    def test_should_compare_strings(self):
        """Should compare strings."""
        type_val = StringType
        values = [
            "",
            "a",
            "ab",
            "abc",
            "abd",
            "def",
            "いろはにほへとちりぬるを",
        ]
        run(type_val, values)

    def test_should_compare_blobs(self):
        """Should compare blobs."""
        type_val = BlobType
        values = [
            Blob(b""),
            Blob(bytes([0, 0])),
            Blob(bytes([1])),
            Blob(bytes([2])),
        ]
        run(type_val, values)


class TestContainerComparisons:
    """Test comparisons of container types."""

    def test_should_compare_arrays(self):
        """Should compare arrays."""
        type_val = ArrayType(IntegerType)
        values = [
            [],
            [0],
            [0, 1],
            [0, 2, 3],
            [0, 2, 4],
            [1],
        ]
        run(type_val, values)

    def test_should_compare_sets(self):
        """Should compare sets."""
        type_val = SetType(StringType)
        values = [
            set(),
            {"abc"},
            {"abc", "def"},
            {"def"},
        ]
        run(type_val, values)

    def test_should_compare_dicts(self):
        """Should compare dicts."""
        type_val = DictType(StringType, IntegerType)
        values = [
            {},
            {"abc": 0},
            {"abc": 0, "def": 1},
            {"abc": 1},
            {"def": 1},
        ]
        run(type_val, values)

    def test_should_compare_structs(self):
        """Should compare structs."""
        type_val = StructType([("boolean", BooleanType), ("string", StringType)])
        values = [
            {"boolean": False, "string": "good"},
            {"boolean": True, "string": "bad"},
            {"boolean": True, "string": "ok"},
        ]
        run(type_val, values)

    def test_should_compare_variants(self):
        """Should compare variants."""
        type_val = VariantType([("none", NullType), ("some", IntegerType)])
        values = [
            variant("none", None),
            variant("some", 0),
            variant("some", 1),
        ]
        run(type_val, values)


class TestNeverAndFunctionTypes:
    """Test error handling for Never and Function types."""

    def test_should_handle_never_type_comparisons(self):
        """Should handle Never type comparisons."""
        type_val = NeverType
        is_compare = is_for(type_val)
        equal_compare = equal_for(type_val)
        less_compare = less_for(type_val)

        with pytest.raises(RuntimeError, match=r"Attempted to compare values of type \.Never"):
            is_compare(None, None)
        with pytest.raises(RuntimeError, match=r"Attempted to compare values of type \.Never"):
            equal_compare(None, None)
        with pytest.raises(RuntimeError, match=r"Attempted to compare values of type \.Never"):
            less_compare(None, None)

    def test_should_handle_function_type_comparisons(self):
        """Should handle Function type comparisons."""
        type_val = FunctionType([], NullType, [])

        with pytest.raises(RuntimeError, match=r"Attempted to compare values of type \.Function"):
            is_for(type_val)
        with pytest.raises(RuntimeError, match=r"Attempted to compare values of type \.Function"):
            equal_for(type_val)
        with pytest.raises(RuntimeError, match=r"Attempted to compare values of type \.Function"):
            not_equal_for(type_val)
        with pytest.raises(RuntimeError, match=r"Attempted to compare values of type \.Function"):
            less_for(type_val)
        with pytest.raises(RuntimeError, match=r"Attempted to compare values of type \.Function"):
            less_equal_for(type_val)
        with pytest.raises(RuntimeError, match=r"Attempted to compare values of type \.Function"):
            greater_for(type_val)
        with pytest.raises(RuntimeError, match=r"Attempted to compare values of type \.Function"):
            greater_equal_for(type_val)


class TestFloatEdgeCases:
    """Test edge cases for float comparisons (NaN, -0/+0)."""

    def test_should_handle_float_nan_edge_cases(self):
        """Should handle Float NaN edge cases."""
        type_val = FloatType
        is_compare = is_for(type_val)
        equal_compare = equal_for(type_val)
        not_equal_compare = not_equal_for(type_val)
        less_compare = less_for(type_val)

        # NaN == NaN for isFor
        assert is_compare(math.nan, math.nan) is True
        assert is_compare(math.nan, 0.0) is False
        assert is_compare(0.0, math.nan) is False

        # NaN == NaN for equalFor
        assert equal_compare(math.nan, math.nan) is True
        assert equal_compare(math.nan, 0.0) is False

        # notEqual with NaN
        assert not_equal_compare(math.nan, math.nan) is False
        assert not_equal_compare(math.nan, 0.0) is True
        assert not_equal_compare(0.0, math.nan) is True

        # NaN ordering (NaN is greatest)
        assert less_compare(0.0, math.nan) is True
        assert less_compare(math.nan, 0.0) is False
        assert less_compare(math.nan, math.nan) is False

        # -0 vs +0 edge case (they are different)
        assert less_compare(-0.0, 0.0) is True
        assert less_compare(0.0, -0.0) is False
        assert equal_compare(-0.0, 0.0) is False


class TestBlobEdgeCases:
    """Test edge cases for blob comparisons."""

    def test_should_handle_blob_different_lengths(self):
        """Should handle Blob different lengths."""
        type_val = BlobType
        equal_compare = equal_for(type_val)
        not_equal_compare = not_equal_for(type_val)
        less_compare = less_for(type_val)

        blob1 = Blob(bytes([1, 2, 3]))
        blob2 = Blob(bytes([1, 2]))
        blob3 = Blob(bytes([1, 2, 4]))

        # Different lengths
        assert equal_compare(blob1, blob2) is False
        assert not_equal_compare(blob1, blob2) is True
        assert less_compare(blob2, blob1) is True  # shorter is less
        assert less_compare(blob1, blob2) is False

        # Same length, different values
        assert equal_compare(blob1, blob3) is False
        assert less_compare(blob1, blob3) is True
        assert less_compare(blob3, blob1) is False

    def test_should_handle_blob_lexical_comparison_loops(self):
        """Should handle Blob lexical comparison loops."""
        type_val = BlobType
        less_compare = less_for(type_val)
        less_equal_compare = less_equal_for(type_val)
        greater_compare = greater_for(type_val)
        greater_equal_compare = greater_equal_for(type_val)
        not_equal_compare = not_equal_for(type_val)

        # Different bytes in middle - tests loop bodies
        blob1 = Blob(bytes([1, 2, 3, 4]))
        blob2 = Blob(bytes([1, 2, 5, 4]))  # different at index 2

        assert less_compare(blob1, blob2) is True
        assert less_equal_compare(blob1, blob2) is True
        assert greater_compare(blob2, blob1) is True
        assert greater_equal_compare(blob2, blob1) is True
        assert not_equal_compare(blob1, blob2) is True

        # Equal blobs
        blob3 = Blob(bytes([1, 2, 3, 4]))
        assert less_compare(blob1, blob3) is False
        assert less_equal_compare(blob1, blob3) is True
        assert greater_compare(blob1, blob3) is False
        assert greater_equal_compare(blob1, blob3) is True
        assert not_equal_compare(blob1, blob3) is False

    def test_should_handle_blob_is_for_loop_body_for_value_comparison(self):
        """Should handle Blob isFor loop body for value comparison."""
        type_val = BlobType
        is_compare = is_for(type_val)

        # Test the loop body that compares byte-by-byte
        blob1 = Blob(bytes([1, 2, 3, 4, 5]))
        blob2 = Blob(bytes([1, 2, 3, 4, 5]))  # same values
        blob3 = Blob(bytes([1, 2, 3, 9, 5]))  # different at index 3

        assert is_compare(blob1, blob2) is True
        assert is_compare(blob1, blob3) is False


class TestArrayEdgeCases:
    """Test edge cases for array comparisons."""

    def test_should_handle_array_comparisons(self):
        """Should handle Array comparisons."""
        type_val = ArrayType(IntegerType)
        equal_compare = equal_for(type_val)
        not_equal_compare = not_equal_for(type_val)
        is_compare = is_for(type_val)

        arr1 = [1, 2, 3]
        arr2 = [1, 2, 3]  # different array, same values
        arr3 = arr1  # same reference

        # isFor compares by identity for mutable types
        assert is_compare(arr1, arr2) is False  # different objects
        assert is_compare(arr1, arr3) is True  # same object

        # equalFor compares by deep value equality for arrays
        assert equal_compare(arr1, arr2) is True  # same values
        assert equal_compare(arr1, arr3) is True  # same object
        assert not_equal_compare(arr1, arr2) is False  # same values

    def test_should_handle_array_length_comparisons(self):
        """Should handle Array length comparisons."""
        type_val = ArrayType(IntegerType)
        less_compare = less_for(type_val)
        less_equal_compare = less_equal_for(type_val)
        greater_compare = greater_for(type_val)
        greater_equal_compare = greater_equal_for(type_val)

        arr1 = [1, 2]
        arr2 = [1, 2, 3]

        # Prefix comparison
        assert less_compare(arr1, arr2) is True
        assert less_equal_compare(arr1, arr2) is True
        assert greater_compare(arr2, arr1) is True
        assert greater_equal_compare(arr2, arr1) is True


class TestSetEdgeCases:
    """Test edge cases for set comparisons."""

    def test_should_handle_set_value_comparisons(self):
        """Should handle Set value comparisons."""
        type_val = SetType(StringType)
        equal_compare = equal_for(type_val)
        not_equal_compare = not_equal_for(type_val)

        set1 = {"a", "b", "c"}
        set2 = {"a", "b", "c"}
        set3 = {"a", "b", "d"}
        set4 = {"a", "b"}

        # Same values
        assert equal_compare(set1, set2) is True
        assert not_equal_compare(set1, set2) is False

        # Different values
        assert equal_compare(set1, set3) is False
        assert not_equal_compare(set1, set3) is True

        # Different sizes
        assert equal_compare(set1, set4) is False
        assert not_equal_compare(set1, set4) is True

    def test_should_handle_set_prefix_comparisons(self):
        """Should handle Set prefix comparisons."""
        type_val = SetType(IntegerType)
        less_compare = less_for(type_val)
        less_equal_compare = less_equal_for(type_val)
        greater_compare = greater_for(type_val)
        greater_equal_compare = greater_equal_for(type_val)
        compare_compare = compare_for(type_val)

        # SortedSet maintains order
        set1 = {1, 2}
        set2 = {1, 2, 3}
        set3 = {1, 2}

        # Prefix: set1 < set2
        assert less_compare(set1, set2) is True
        assert less_equal_compare(set1, set2) is True
        assert greater_compare(set2, set1) is True
        assert greater_equal_compare(set2, set1) is True
        assert compare_compare(set1, set2) == -1
        assert compare_compare(set2, set1) == 1

        # Equal
        assert less_equal_compare(set1, set3) is True
        assert greater_equal_compare(set1, set3) is True
        assert compare_compare(set1, set3) == 0

    def test_should_handle_set_and_dict_identity_with_is_for(self):
        """Should handle Set and Dict identity with isFor."""
        # isFor uses identity for Set and Dict
        set_type = SetType(IntegerType)
        dict_type = DictType(StringType, IntegerType)

        set1 = {1, 2}
        set2 = {1, 2}
        set3 = set1

        dict1 = {"a": 1}
        dict2 = {"a": 1}
        dict3 = dict1

        set_is_compare = is_for(set_type)
        dict_is_compare = is_for(dict_type)

        # Different objects with same values
        assert set_is_compare(set1, set2) is False
        assert dict_is_compare(dict1, dict2) is False

        # Same reference
        assert set_is_compare(set1, set3) is True
        assert dict_is_compare(dict1, dict3) is True

    def test_should_handle_set_prefix_where_x_size_gt_y_size(self):
        """Should handle Set prefix where x.size > y.size."""
        type_val = SetType(IntegerType)
        greater_compare = greater_for(type_val)

        # x is longer and y is prefix
        set1 = {1, 2, 3, 4}
        set2 = {1, 2, 3}

        assert greater_compare(set1, set2) is True

    def test_should_handle_set_greater_for_when_all_elements_match(self):
        """Should handle Set greaterFor when all elements match."""
        type_val = SetType(IntegerType)
        greater_compare = greater_for(type_val)

        # Same size and all elements equal
        set1 = {1, 2, 3}
        set2 = {1, 2, 3}

        # Should return x.size > y.size which is false
        assert greater_compare(set1, set2) is False


class TestDictEdgeCases:
    """Test edge cases for dict comparisons."""

    def test_should_handle_dict_value_comparisons(self):
        """Should handle Dict value comparisons."""
        type_val = DictType(StringType, IntegerType)
        equal_compare = equal_for(type_val)
        not_equal_compare = not_equal_for(type_val)

        dict1 = {"a": 1, "b": 2}
        dict2 = {"a": 1, "b": 2}
        dict3 = {"a": 1, "b": 3}
        dict4 = {"a": 1}
        dict5 = {"a": 1, "c": 2}

        # Same values
        assert equal_compare(dict1, dict2) is True
        assert not_equal_compare(dict1, dict2) is False

        # Different values (same keys)
        assert equal_compare(dict1, dict3) is False
        assert not_equal_compare(dict1, dict3) is True

        # Different size
        assert equal_compare(dict1, dict4) is False

        # Missing key
        assert equal_compare(dict1, dict5) is False
        assert not_equal_compare(dict1, dict5) is True

    def test_should_handle_dict_prefix_comparisons(self):
        """Should handle Dict prefix comparisons."""
        type_val = DictType(StringType, IntegerType)
        less_compare = less_for(type_val)
        less_equal_compare = less_equal_for(type_val)
        greater_compare = greater_for(type_val)
        greater_equal_compare = greater_equal_for(type_val)
        compare_compare = compare_for(type_val)

        dict1 = {"a": 1}
        dict2 = {"a": 1, "b": 2}
        dict3 = {"a": 1}

        # Prefix: dict1 < dict2
        assert less_compare(dict1, dict2) is True
        assert less_equal_compare(dict1, dict2) is True
        assert greater_compare(dict2, dict1) is True
        assert greater_equal_compare(dict2, dict1) is True
        assert compare_compare(dict1, dict2) == -1
        assert compare_compare(dict2, dict1) == 1

        # Equal
        assert less_equal_compare(dict1, dict3) is True
        assert greater_equal_compare(dict1, dict3) is True
        assert compare_compare(dict1, dict3) == 0

    def test_should_handle_dict_prefix_where_x_size_gt_y_size(self):
        """Should handle Dict prefix where x.size > y.size."""
        type_val = DictType(StringType, IntegerType)
        greater_compare = greater_for(type_val)

        # x is longer and y is prefix
        dict1 = {"a": 1, "b": 2, "c": 3}
        dict2 = {"a": 1, "b": 2}

        assert greater_compare(dict1, dict2) is True

    def test_should_handle_dict_greater_for_when_all_entries_match(self):
        """Should handle Dict greaterFor when all entries match."""
        type_val = DictType(StringType, IntegerType)
        greater_compare = greater_for(type_val)

        # Same size and all entries equal
        dict1 = {"a": 1, "b": 2}
        dict2 = {"a": 1, "b": 2}

        # Should return x.size > y.size which is false
        assert greater_compare(dict1, dict2) is False


class TestStructEdgeCases:
    """Test edge cases for struct comparisons."""

    def test_should_handle_struct_field_mismatches(self):
        """Should handle Struct field mismatches."""
        type_val = StructType([("a", IntegerType), ("b", StringType)])
        equal_compare = equal_for(type_val)
        not_equal_compare = not_equal_for(type_val)
        less_compare = less_for(type_val)

        struct1 = {"a": 1, "b": "hello"}
        struct2 = {"a": 1, "b": "hello"}
        struct3 = {"a": 2, "b": "hello"}
        struct4 = {"a": 1, "b": "world"}

        # Same values
        assert equal_compare(struct1, struct2) is True
        assert not_equal_compare(struct1, struct2) is False

        # First field differs
        assert equal_compare(struct1, struct3) is False
        assert not_equal_compare(struct1, struct3) is True
        assert less_compare(struct1, struct3) is True
        assert less_compare(struct3, struct1) is False

        # Second field differs
        assert equal_compare(struct1, struct4) is False
        assert less_compare(struct1, struct4) is True

    def test_should_handle_struct_field_by_field_comparison(self):
        """Should handle Struct field-by-field comparison."""
        type_val = StructType([("a", IntegerType), ("b", IntegerType)])
        less_equal_compare = less_equal_for(type_val)
        greater_equal_compare = greater_equal_for(type_val)

        struct1 = {"a": 1, "b": 2}
        struct2 = {"a": 1, "b": 3}
        struct3 = {"a": 1, "b": 2}

        # struct1 <= struct2
        assert less_equal_compare(struct1, struct2) is True
        assert greater_equal_compare(struct2, struct1) is True

        # struct1 == struct3
        assert less_equal_compare(struct1, struct3) is True
        assert greater_equal_compare(struct1, struct3) is True

    def test_should_handle_struct_field_mismatch_in_is_for(self):
        """Should handle Struct field mismatch in isFor."""
        type_val = StructType([("x", IntegerType), ("y", IntegerType)])
        is_compare = is_for(type_val)

        struct1 = {"x": 1, "y": 2}
        struct2 = {"x": 1, "y": 3}  # different y

        # isFor checks all fields, returns false on first mismatch
        assert is_compare(struct1, struct2) is False

    def test_should_handle_struct_greater_for_with_all_fields_equal(self):
        """Should handle Struct greaterFor with all fields equal."""
        type_val = StructType([("x", IntegerType), ("y", IntegerType)])
        greater_compare = greater_for(type_val)

        struct1 = {"x": 1, "y": 2}
        struct2 = {"x": 1, "y": 2}

        # All fields equal means not greater
        assert greater_compare(struct1, struct2) is False


class TestVariantEdgeCases:
    """Test edge cases for variant comparisons."""

    def test_should_handle_variant_type_mismatches(self):
        """Should handle Variant type mismatches."""
        type_val = VariantType([("none", NullType), ("some", IntegerType)])
        equal_compare = equal_for(type_val)
        not_equal_compare = not_equal_for(type_val)
        less_compare = less_for(type_val)
        compare_compare = compare_for(type_val)

        v1 = variant("none", None)
        v2 = variant("none", None)
        v3 = variant("some", 5)
        v4 = variant("some", 10)

        # Same type and value
        assert equal_compare(v1, v2) is True
        assert not_equal_compare(v1, v2) is False

        # Different type
        assert equal_compare(v1, v3) is False
        assert not_equal_compare(v1, v3) is True
        assert less_compare(v1, v3) is True  # 'none' < 'some'
        assert less_compare(v3, v1) is False
        assert compare_compare(v1, v3) == -1
        assert compare_compare(v3, v1) == 1

        # Same type, different value
        assert equal_compare(v3, v4) is False
        assert less_compare(v3, v4) is True

    def test_should_handle_variant_less_equal_and_greater_equal(self):
        """Should handle Variant lessEqual and greaterEqual."""
        type_val = VariantType([("a", IntegerType), ("b", IntegerType)])
        less_equal_compare = less_equal_for(type_val)
        greater_equal_compare = greater_equal_for(type_val)

        v1 = variant("a", 5)
        v2 = variant("b", 3)
        v3 = variant("b", 5)
        v4 = variant("b", 3)

        # v1 < v2 (by type)
        assert less_equal_compare(v1, v2) is True
        assert greater_equal_compare(v2, v1) is True

        # v2 < v3 (same type, by value)
        assert less_equal_compare(v2, v3) is True
        assert greater_equal_compare(v3, v2) is True

        # v2 == v4
        assert less_equal_compare(v2, v4) is True
        assert greater_equal_compare(v2, v4) is True


class TestNullTypeComparisons:
    """Test null type comparisons."""

    def test_should_handle_null_type_comparisons(self):
        """Should handle Null type comparisons."""
        type_val = NullType
        is_compare = is_for(type_val)
        equal_compare = equal_for(type_val)
        less_compare = less_for(type_val)
        less_equal_compare = less_equal_for(type_val)
        greater_compare = greater_for(type_val)
        greater_equal_compare = greater_equal_for(type_val)

        # All null values are equal
        assert is_compare(None, None) is True
        assert equal_compare(None, None) is True
        assert less_compare(None, None) is False
        assert less_equal_compare(None, None) is True
        assert greater_compare(None, None) is False
        assert greater_equal_compare(None, None) is True


class TestNeverTypeEdgeCases:
    """Test Never type edge cases for all comparison functions."""

    def test_should_handle_never_type_in_not_equal_for(self):
        """Should handle Never type in notEqualFor."""
        type_val = NeverType
        not_equal_compare = not_equal_for(type_val)

        with pytest.raises(RuntimeError, match=r"Attempted to compare values of type \.Never"):
            not_equal_compare(None, None)

    def test_should_handle_never_type_in_less_equal_for(self):
        """Should handle Never type in lessEqualFor."""
        type_val = NeverType
        less_equal_compare = less_equal_for(type_val)

        with pytest.raises(RuntimeError, match=r"Attempted to compare values of type \.Never"):
            less_equal_compare(None, None)

    def test_should_handle_never_type_in_greater_equal_for(self):
        """Should handle Never type in greaterEqualFor."""
        type_val = NeverType
        greater_equal_compare = greater_equal_for(type_val)

        with pytest.raises(RuntimeError, match=r"Attempted to compare values of type \.Never"):
            greater_equal_compare(None, None)

    def test_should_handle_never_type_in_greater_for(self):
        """Should handle Never type in greaterFor."""
        type_val = NeverType
        greater_compare = greater_for(type_val)

        with pytest.raises(RuntimeError, match=r"Attempted to compare values of type \.Never"):
            greater_compare(None, None)


class TestRecursiveTypes:
    """Test comparisons with recursive types (trees, lists, DAGs, cycles)."""

    def test_should_compare_tree_shaped_recursive_data_binary_tree(self):
        """Should compare tree-shaped recursive data (binary tree)."""

        # Binary tree type: { value: Integer, left: Tree | null, right: Tree | null }
        def TreeType(self):
            return VariantType(
                [
                    ("leaf", NullType),
                    (
                        "node",
                        StructType(
                            [
                                ("value", IntegerType),
                                ("left", self),
                                ("right", self),
                            ]
                        ),
                    ),
                ]
            )

        tree_type = recursive_type(TreeType)

        equal_compare = equal_for(tree_type)
        not_equal_compare = not_equal_for(tree_type)
        compare = compare_for(tree_type)

        # Leaf nodes
        leaf = variant("leaf", None)
        assert equal_compare(leaf, leaf) is True
        assert not_equal_compare(leaf, leaf) is False
        assert compare(leaf, leaf) == 0

        # Simple nodes
        node1 = variant("node", {"value": 1, "left": leaf, "right": leaf})
        node2 = variant("node", {"value": 1, "left": leaf, "right": leaf})
        node3 = variant("node", {"value": 2, "left": leaf, "right": leaf})

        assert equal_compare(node1, node2) is True
        assert equal_compare(node1, node3) is False
        assert not_equal_compare(node1, node3) is True
        assert compare(node1, node2) == 0
        assert compare(node1, node3) == -1
        assert compare(node3, node1) == 1

        # Nested tree structures
        left_tree = variant("node", {"value": 10, "left": leaf, "right": leaf})
        right_tree = variant("node", {"value": 20, "left": leaf, "right": leaf})
        tree1 = variant("node", {"value": 5, "left": left_tree, "right": right_tree})
        tree2 = variant("node", {"value": 5, "left": left_tree, "right": right_tree})
        tree3 = variant("node", {"value": 5, "left": right_tree, "right": left_tree})  # swapped

        assert equal_compare(tree1, tree2) is True
        assert equal_compare(tree1, tree3) is False
        assert not_equal_compare(tree1, tree3) is True
        assert compare(tree1, tree2) == 0
        assert compare(tree1, tree3) == -1  # left_tree < right_tree
        assert compare(tree3, tree1) == 1

        # Test lessFor, greaterFor, lessEqualFor, greaterEqualFor
        less = less_for(tree_type)
        greater = greater_for(tree_type)
        less_equal = less_equal_for(tree_type)
        greater_equal = greater_equal_for(tree_type)

        # Leaf nodes
        assert less(leaf, leaf) is False
        assert greater(leaf, leaf) is False
        assert less_equal(leaf, leaf) is True
        assert greater_equal(leaf, leaf) is True

        # Simple nodes
        assert less(node1, node2) is False  # equal
        assert less(node1, node3) is True
        assert less(node3, node1) is False
        assert greater(node3, node1) is True
        assert greater(node1, node3) is False
        assert less_equal(node1, node2) is True
        assert less_equal(node1, node3) is True
        assert greater_equal(node3, node1) is True

        # Nested structures
        assert less(tree1, tree3) is True
        assert less(tree3, tree1) is False
        assert greater(tree3, tree1) is True
        assert less_equal(tree1, tree2) is True  # equal
        assert greater_equal(tree1, tree2) is True  # equal

        # Test isFor (RecursiveType is immutable, so isFor uses structural equality)
        is_compare = is_for(tree_type)

        # Same object
        assert is_compare(leaf, leaf) is True
        assert is_compare(tree1, tree1) is True

        # Different objects, structurally equal (immutable type, so should be true)
        assert is_compare(node1, node2) is True
        assert is_compare(tree1, tree2) is True

        # Different values
        assert is_compare(node1, node3) is False
        assert is_compare(tree1, tree3) is False
        assert is_compare(leaf, node1) is False

    def test_should_compare_tree_shaped_recursive_data_linked_list(self):
        """Should compare tree-shaped recursive data (linked list)."""

        # Linked list type: { value: Integer, next: List | null }
        def ListType(self):
            return VariantType(
                [
                    ("nil", NullType),
                    (
                        "cons",
                        StructType(
                            [
                                ("value", IntegerType),
                                ("next", self),
                            ]
                        ),
                    ),
                ]
            )

        list_type = recursive_type(ListType)

        equal_compare = equal_for(list_type)
        compare = compare_for(list_type)

        # Empty lists
        nil = variant("nil", None)
        assert equal_compare(nil, nil) is True
        assert compare(nil, nil) == 0

        # Single-element lists
        list1 = variant("cons", {"value": 1, "next": nil})
        list2 = variant("cons", {"value": 1, "next": nil})
        list3 = variant("cons", {"value": 2, "next": nil})

        assert equal_compare(list1, list2) is True
        assert equal_compare(list1, list3) is False
        assert compare(list1, list2) == 0
        assert compare(list1, list3) == -1
        assert compare(list3, list1) == 1

        # Multi-element lists
        list4 = variant(
            "cons",
            {
                "value": 1,
                "next": variant(
                    "cons", {"value": 2, "next": variant("cons", {"value": 3, "next": nil})}
                ),
            },
        )
        list5 = variant(
            "cons",
            {
                "value": 1,
                "next": variant(
                    "cons", {"value": 2, "next": variant("cons", {"value": 3, "next": nil})}
                ),
            },
        )
        list6 = variant(
            "cons",
            {
                "value": 1,
                "next": variant(
                    "cons", {"value": 2, "next": variant("cons", {"value": 4, "next": nil})}
                ),
            },
        )

        assert equal_compare(list4, list5) is True
        assert equal_compare(list4, list6) is False
        assert compare(list4, list5) == 0
        assert compare(list4, list6) == -1  # 3 < 4
        assert compare(list6, list4) == 1

        # Test lessFor, greaterFor, lessEqualFor, greaterEqualFor
        less = less_for(list_type)
        greater = greater_for(list_type)
        less_equal = less_equal_for(list_type)
        greater_equal = greater_equal_for(list_type)

        # Empty lists
        assert less(nil, nil) is False
        assert greater(nil, nil) is False
        assert less_equal(nil, nil) is True
        assert greater_equal(nil, nil) is True

        # Single-element lists
        assert less(list1, list2) is False  # equal
        assert less(list1, list3) is True
        assert less(list3, list1) is False
        assert greater(list3, list1) is True
        assert less_equal(list1, list2) is True
        assert greater_equal(list3, list1) is True

        # Multi-element lists
        assert less(list4, list5) is False  # equal
        assert less(list4, list6) is True
        assert greater(list6, list4) is True
        assert less_equal(list4, list5) is True
        assert greater_equal(list4, list5) is True

        # Test isFor (RecursiveType is immutable, so isFor uses structural equality)
        is_compare = is_for(list_type)

        # Same object
        assert is_compare(nil, nil) is True
        assert is_compare(list1, list1) is True

        # Different objects, structurally equal
        assert is_compare(list1, list2) is True
        assert is_compare(list4, list5) is True

        # Different values
        assert is_compare(list1, list3) is False
        assert is_compare(list4, list6) is False
        assert is_compare(nil, list1) is False

    def test_should_compare_dag_shaped_recursive_data_shared_subtrees(self):
        """Should compare DAG-shaped recursive data (shared subtrees)."""

        # Binary tree with shared subtrees (DAG, not a tree)
        def TreeType(self):
            return VariantType(
                [
                    ("leaf", NullType),
                    (
                        "node",
                        StructType(
                            [
                                ("value", IntegerType),
                                ("left", self),
                                ("right", self),
                            ]
                        ),
                    ),
                ]
            )

        tree_type = recursive_type(TreeType)

        equal_compare = equal_for(tree_type)
        compare = compare_for(tree_type)

        leaf = variant("leaf", None)
        shared_subtree = variant("node", {"value": 10, "left": leaf, "right": leaf})

        # Two trees sharing the same subtree object
        dag1 = variant("node", {"value": 5, "left": shared_subtree, "right": shared_subtree})
        dag2 = variant("node", {"value": 5, "left": shared_subtree, "right": shared_subtree})

        # Fast path: same object reference
        assert equal_compare(dag1, dag1) is True
        assert compare(dag1, dag1) == 0

        # Different objects but structurally equal (including shared structure)
        assert equal_compare(dag1, dag2) is True
        assert compare(dag1, dag2) == 0

        # Different tree with same values but different structure
        different_subtree = variant("node", {"value": 10, "left": leaf, "right": leaf})
        dag3 = variant("node", {"value": 5, "left": different_subtree, "right": different_subtree})
        assert equal_compare(dag1, dag3) is True  # Structurally equal
        assert compare(dag1, dag3) == 0

        # Different values
        dag4 = variant("node", {"value": 5, "left": shared_subtree, "right": leaf})
        assert equal_compare(dag1, dag4) is False
        assert compare(dag1, dag4) == 1  # dag1 has node on right, dag4 has leaf
        assert compare(dag4, dag1) == -1

        # Test lessFor, greaterFor, lessEqualFor, greaterEqualFor
        less = less_for(tree_type)
        greater = greater_for(tree_type)
        less_equal = less_equal_for(tree_type)
        greater_equal = greater_equal_for(tree_type)

        assert less(dag1, dag1) is False  # same object
        assert less_equal(dag1, dag1) is True
        assert greater_equal(dag1, dag1) is True
        assert less(dag1, dag2) is False  # equal
        assert less_equal(dag1, dag2) is True
        assert less(dag4, dag1) is True  # leaf < node
        assert greater(dag1, dag4) is True
        assert greater_equal(dag1, dag4) is True

        # Test isFor (RecursiveType is immutable, so isFor uses structural equality)
        is_compare = is_for(tree_type)

        # Same object
        assert is_compare(dag1, dag1) is True
        assert is_compare(shared_subtree, shared_subtree) is True

        # Different objects, structurally equal (even with shared subtrees)
        assert is_compare(dag1, dag2) is True
        assert is_compare(dag1, dag3) is True  # Different subtree objects but same structure

        # Different values
        assert is_compare(dag1, dag4) is False
        assert is_compare(leaf, dag1) is False

    def test_should_compare_circular_recursive_data_self_loop(self):
        """Should compare circular recursive data (self-loop)."""

        # Linked list with cycles
        def ListType(self):
            return VariantType(
                [
                    ("nil", NullType),
                    (
                        "cons",
                        StructType(
                            [
                                ("value", IntegerType),
                                ("next", ArrayType(self)),  # note: should have exactly one value
                            ]
                        ),
                    ),
                ]
            )

        list_type = recursive_type(ListType)

        equal_compare = equal_for(list_type)
        compare = compare_for(list_type)

        # Create a circular list: A -> A
        self_loop = variant("cons", {"value": 1, "next": []})
        self_loop["value"]["next"].append(self_loop)

        # Compare with itself
        assert equal_compare(self_loop, self_loop) is True
        assert compare(self_loop, self_loop) == 0

        # Create another identical circular list
        self_loop2 = variant("cons", {"value": 1, "next": []})
        self_loop2["value"]["next"].append(self_loop2)

        # Two different circular lists with same structure and values
        assert equal_compare(self_loop, self_loop2) is True
        assert compare(self_loop, self_loop2) == 0  # Both cycle at same depth

        # Different value in cycle
        self_loop3 = variant("cons", {"value": 2, "next": []})
        self_loop3["value"]["next"].append(self_loop3)

        assert equal_compare(self_loop, self_loop3) is False
        assert compare(self_loop, self_loop3) == -1  # 1 < 2
        assert compare(self_loop3, self_loop) == 1

        # Non-circular list vs circular list
        nil = variant("nil", None)
        non_circular = variant("cons", {"value": 1, "next": [nil]})
        assert compare(non_circular, self_loop) == 1  # "nil" > "cons"
        assert compare(self_loop, non_circular) == -1

        # Test lessFor, greaterFor, lessEqualFor, greaterEqualFor
        less = less_for(list_type)
        greater = greater_for(list_type)
        less_equal = less_equal_for(list_type)
        greater_equal = greater_equal_for(list_type)

        # Same circular list
        assert less(self_loop, self_loop) is False
        assert greater(self_loop, self_loop) is False
        assert less_equal(self_loop, self_loop) is True
        assert greater_equal(self_loop, self_loop) is True

        # Two identical circular lists
        assert less(self_loop, self_loop2) is False  # equal
        assert less_equal(self_loop, self_loop2) is True
        assert greater_equal(self_loop, self_loop2) is True

        # Different values in cycles
        assert less(self_loop, self_loop3) is True  # 1 < 2
        assert less(self_loop3, self_loop) is False
        assert greater(self_loop3, self_loop) is True
        assert less_equal(self_loop, self_loop3) is True
        assert greater_equal(self_loop3, self_loop) is True

        # Non-circular vs circular
        assert less(non_circular, self_loop) is False  # "nil" > "cons"
        assert greater(self_loop, non_circular) is False
        assert less_equal(non_circular, self_loop) is False
        assert greater_equal(self_loop, non_circular) is False

        # Test isFor (RecursiveType is immutable, so isFor uses structural equality with cycle detection)
        is_compare = is_for(list_type)

        # Same object
        assert is_compare(self_loop, self_loop) is True
        assert is_compare(nil, nil) is True

        # Different objects, structurally equal (different array identity)
        assert is_compare(self_loop, self_loop2) is False

        # Different values in cycle
        assert is_compare(self_loop, self_loop3) is False

        # Non-circular vs circular
        assert is_compare(non_circular, self_loop) is False
        assert is_compare(nil, self_loop) is False

    def test_should_compare_circular_recursive_data_cycle_in_chain(self):
        """Should compare circular recursive data (cycle in chain)."""

        # Linked list type
        def ListType(self):
            return VariantType(
                [
                    ("nil", NullType),
                    (
                        "cons",
                        StructType(
                            [
                                ("value", IntegerType),
                                ("next", ArrayType(self)),
                            ]
                        ),
                    ),
                ]
            )

        list_type = recursive_type(ListType)

        equal_compare = equal_for(list_type)
        compare = compare_for(list_type)

        # Create a cycle: A(1) -> B(2) -> C(3) -> B(2) (cycle at depth 2)
        nil = variant("nil", None)
        node_b = variant("cons", {"value": 2, "next": [nil]})
        node_c = variant("cons", {"value": 3, "next": [node_b]})
        node_b["value"]["next"] = [node_c]  # Create the cycle
        list_a = variant("cons", {"value": 1, "next": [node_b]})

        # Compare with itself
        assert equal_compare(list_a, list_a) is True
        assert compare(list_a, list_a) == 0

        # Create another identical cyclic structure
        node_b2 = variant("cons", {"value": 2, "next": [nil]})
        node_c2 = variant("cons", {"value": 3, "next": [node_b2]})
        node_b2["value"]["next"] = [node_c2]
        list_a2 = variant("cons", {"value": 1, "next": [node_b2]})

        # Two different cyclic lists with same structure
        assert equal_compare(list_a, list_a2) is True
        assert compare(list_a, list_a2) == 0  # Both cycle at same depth

        # Different cycle structure (cycle at different depth) Y(1) -> X(2) -> X(2)
        node_x = variant("cons", {"value": 2, "next": [nil]})
        node_x["value"]["next"] = [node_x]  # Self-loop instead
        list_y = variant("cons", {"value": 1, "next": [node_x]})

        # Different cycle structure should not be equal
        assert equal_compare(list_a, list_y) is False
        # list_a has B->C->B cycle, list_y has X->X cycle
        # When comparing, X cycles earlier (at depth 1) while C is still acyclic
        # So list_y is "more infinite", making list_y > list_a
        assert compare(list_a, list_y) == 1
        assert compare(list_y, list_a) == -1

        # Test lessFor, greaterFor, lessEqualFor, greaterEqualFor
        less = less_for(list_type)
        greater = greater_for(list_type)
        less_equal = less_equal_for(list_type)
        greater_equal = greater_equal_for(list_type)

        # Same list
        assert less(list_a, list_a) is False
        assert greater(list_a, list_a) is False
        assert less_equal(list_a, list_a) is True
        assert greater_equal(list_a, list_a) is True

        # Two identical cyclic structures
        assert less(list_a, list_a2) is False  # equal
        assert less_equal(list_a, list_a2) is True
        assert greater_equal(list_a, list_a2) is True

        # Different cycle structures
        assert less(list_a, list_y) is False  # list_y < list_a (2 < 3)
        assert less(list_y, list_a) is True
        assert greater(list_a, list_y) is True
        assert greater(list_y, list_a) is False
        assert less_equal(list_a, list_y) is False
        assert less_equal(list_y, list_a) is True
        assert greater_equal(list_a, list_y) is True
        assert greater_equal(list_y, list_a) is False

        # Test isFor (RecursiveType is immutable, so isFor uses structural equality with cycle detection)
        is_compare = is_for(list_type)

        # Same object
        assert is_compare(list_a, list_a) is True

        # Different objects, structurally equal (different array identity)
        assert is_compare(list_a, list_a2) is False

        # Different cycle structures
        assert is_compare(list_a, list_y) is False

    def test_should_compare_circular_recursive_data_binary_tree_with_cycle(self):
        """Should compare circular recursive data (binary tree with cycle)."""

        # Binary tree type
        def TreeType(self):
            return VariantType(
                [
                    ("leaf", NullType),
                    (
                        "node",
                        StructType(
                            [
                                ("value", IntegerType),
                                ("left", ArrayType(self)),
                                ("right", ArrayType(self)),
                            ]
                        ),
                    ),
                ]
            )

        tree_type = recursive_type(TreeType)

        equal_compare = equal_for(tree_type)
        compare = compare_for(tree_type)

        leaf = variant("leaf", None)

        # Create a tree with a cycle: root -> left -> root
        root = variant("node", {"value": 1, "left": [leaf], "right": [leaf]})
        left_child = variant("node", {"value": 2, "left": [root], "right": [leaf]})
        root["value"]["left"] = [left_child]

        # Compare with itself
        assert equal_compare(root, root) is True
        assert compare(root, root) == 0

        # Create another identical cyclic tree
        root2 = variant("node", {"value": 1, "left": [leaf], "right": [leaf]})
        left_child2 = variant("node", {"value": 2, "left": [root2], "right": [leaf]})
        root2["value"]["left"] = [left_child2]

        # Two different cyclic trees with same structure
        assert equal_compare(root, root2) is True
        assert compare(root, root2) == 0

        # Different structure (no cycle)
        root3 = variant(
            "node",
            {
                "value": 1,
                "left": [variant("node", {"value": 2, "left": [leaf], "right": [leaf]})],
                "right": [leaf],
            },
        )
        assert equal_compare(root, root3) is False
        assert compare(root, root3) == 1  # cyclic > acyclic ("infinite" > finite)
        assert compare(root3, root) == -1

        # Test lessFor, greaterFor, lessEqualFor, greaterEqualFor
        less = less_for(tree_type)
        greater = greater_for(tree_type)
        less_equal = less_equal_for(tree_type)
        greater_equal = greater_equal_for(tree_type)

        # Same tree
        assert less(root, root) is False
        assert greater(root, root) is False
        assert less_equal(root, root) is True
        assert greater_equal(root, root) is True

        # Two identical cyclic trees
        assert less(root, root2) is False  # equal
        assert less_equal(root, root2) is True
        assert greater_equal(root, root2) is True

        # Cyclic vs acyclic
        assert less(root3, root) is True  # acyclic < cyclic
        assert less(root, root3) is False
        assert greater(root, root3) is True
        assert less_equal(root3, root) is True
        assert greater_equal(root, root3) is True

        # Test isFor (RecursiveType is immutable, so isFor uses structural equality with cycle detection)
        is_compare = is_for(tree_type)

        # Same object
        assert is_compare(root, root) is True
        assert is_compare(leaf, leaf) is True

        # Different objects, structurally equal (different array identity)
        assert is_compare(root, root2) is False

        # Cyclic vs acyclic
        assert is_compare(root, root3) is False

    def test_should_compare_nested_recursive_types_tree_of_lists(self):
        """Should compare nested recursive types (tree of lists)."""

        # Linked list type
        def ListType(self):
            return VariantType(
                [
                    ("nil", NullType),
                    (
                        "cons",
                        StructType(
                            [
                                ("value", IntegerType),
                                ("next", self),
                            ]
                        ),
                    ),
                ]
            )

        list_type = recursive_type(ListType)

        # Binary tree type containing lists at each node
        def TreeOfListsType(self):
            return VariantType(
                [
                    ("leaf", NullType),
                    (
                        "node",
                        StructType(
                            [
                                ("list", list_type),
                                ("left", self),
                                ("right", self),
                            ]
                        ),
                    ),
                ]
            )

        tree_of_lists_type = recursive_type(TreeOfListsType)

        equal_compare = equal_for(tree_of_lists_type)
        compare = compare_for(tree_of_lists_type)

        # Create some lists
        nil = variant("nil", None)
        list1 = variant("cons", {"value": 1, "next": variant("cons", {"value": 2, "next": nil})})
        list2 = variant("cons", {"value": 3, "next": nil})

        # Create trees containing these lists
        leaf = variant("leaf", None)
        tree1 = variant(
            "node",
            {
                "list": list1,
                "left": variant("node", {"list": list2, "left": leaf, "right": leaf}),
                "right": leaf,
            },
        )

        tree2 = variant(
            "node",
            {
                "list": list1,
                "left": variant("node", {"list": list2, "left": leaf, "right": leaf}),
                "right": leaf,
            },
        )

        # Same structure
        assert equal_compare(tree1, tree2) is True
        assert compare(tree1, tree2) == 0

        # Different list content
        tree3 = variant(
            "node",
            {
                "list": list2,  # Different list
                "left": variant("node", {"list": list2, "left": leaf, "right": leaf}),
                "right": leaf,
            },
        )

        assert equal_compare(tree1, tree3) is False
        assert compare(tree1, tree3) == -1  # list1 < list2 (starts with 1 vs 3)
        assert compare(tree3, tree1) == 1

        # Test lessFor, greaterFor, lessEqualFor, greaterEqualFor
        less = less_for(tree_of_lists_type)
        greater = greater_for(tree_of_lists_type)
        less_equal = less_equal_for(tree_of_lists_type)
        greater_equal = greater_equal_for(tree_of_lists_type)

        # Same tree
        assert less(tree1, tree1) is False
        assert greater(tree1, tree1) is False
        assert less_equal(tree1, tree1) is True
        assert greater_equal(tree1, tree1) is True

        # Equal trees
        assert less(tree1, tree2) is False
        assert less_equal(tree1, tree2) is True
        assert greater_equal(tree1, tree2) is True

        # Different list content
        assert less(tree1, tree3) is True  # list1 < list2
        assert less(tree3, tree1) is False
        assert greater(tree3, tree1) is True
        assert less_equal(tree1, tree3) is True
        assert greater_equal(tree3, tree1) is True

        # Test isFor (RecursiveType is immutable, so isFor uses structural equality)
        is_compare = is_for(tree_of_lists_type)

        # Same object
        assert is_compare(tree1, tree1) is True
        assert is_compare(leaf, leaf) is True

        # Different objects, structurally equal
        assert is_compare(tree1, tree2) is True

        # Different list content
        assert is_compare(tree1, tree3) is False


class TestInvalidTypes:
    """Test error handling for invalid types."""

    def test_should_throw_for_invalid_type_in_is_for(self):
        """Should throw for invalid type in isFor."""
        # Force execution of unreachable error path using invalid type
        invalid_type = type("InvalidType", (), {"tag": "InvalidType"})()
        with pytest.raises(
            RuntimeError, match=r"Unknown type encountered during type printing: InvalidType"
        ):
            is_for(invalid_type)

    def test_should_throw_for_invalid_type_in_less_equal_for(self):
        """Should throw for invalid type in lessEqualFor."""
        # Force execution of unreachable error path using invalid type
        invalid_type = type("InvalidType", (), {"tag": "InvalidType"})()
        with pytest.raises(
            RuntimeError, match=r"Unknown type encountered during type printing: InvalidType"
        ):
            less_equal_for(invalid_type)

    def test_should_throw_for_invalid_type_in_greater_for(self):
        """Should throw for invalid type in greaterFor."""
        # Force execution of unreachable error path using invalid type
        invalid_type = type("InvalidType", (), {"tag": "InvalidType"})()
        with pytest.raises(
            RuntimeError, match=r"Unknown type encountered during type printing: InvalidType"
        ):
            greater_for(invalid_type)
