"""Tests for default value generation.

Ported from East/src/default.spec.ts
"""

from datetime import datetime as DateTime

import pytest

from east.types.containers import EastArray, EastDict, EastSet
from east.types.primitives import Blob
from east.types.types import (
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
)
from east.utils.default import default_value, minimal_value


class TestDefaultValue:
    """Tests for default_value function."""

    def test_should_throw_for_never_type(self):
        """Should throw for Never type."""
        with pytest.raises(RuntimeError, match=r"Cannot create a default value of type \.Never"):
            default_value(NeverType)

    def test_should_return_none_for_null_type(self):
        """Should return None for Null type."""
        result = default_value(NullType)
        assert result is None

    def test_should_return_false_for_boolean_type(self):
        """Should return False for Boolean type."""
        result = default_value(BooleanType)
        assert result is False

    def test_should_return_0_for_integer_type(self):
        """Should return 0 for Integer type."""
        result = default_value(IntegerType)
        assert result == 0

    def test_should_return_0_0_for_float_type(self):
        """Should return 0.0 for Float type."""
        result = default_value(FloatType)
        assert result == 0.0

    def test_should_return_empty_string_for_string_type(self):
        """Should return empty string for String type."""
        result = default_value(StringType)
        assert result == ""

    def test_should_return_epoch_date_for_datetime_type(self):
        """Should return epoch date for DateTime type."""
        result = default_value(DateTimeType)
        assert isinstance(result, DateTime)
        assert result.timestamp() == 0

    def test_should_return_empty_blob_for_blob_type(self):
        """Should return empty Blob for Blob type."""
        result = default_value(BlobType)
        assert isinstance(result, Blob)
        assert len(result.data) == 0

    def test_should_return_empty_array_for_array_type(self):
        """Should return empty array for Array type."""
        result = default_value(ArrayType(IntegerType))
        assert isinstance(result, EastArray)
        assert len(result) == 0

    def test_should_return_empty_set_for_set_type(self):
        """Should return empty EastSet for Set type."""
        result = default_value(SetType(IntegerType))
        assert isinstance(result, EastSet)
        assert len(result) == 0

    def test_should_return_empty_dict_for_dict_type(self):
        """Should return empty EastDict for Dict type."""
        result = default_value(DictType(StringType, IntegerType))
        assert isinstance(result, EastDict)
        assert len(result) == 0

    def test_should_return_struct_with_default_field_values(self):
        """Should return struct with default field values for Struct type."""
        type_val = StructType(
            [
                ("name", StringType),
                ("age", IntegerType),
                ("active", BooleanType),
            ]
        )

        result = default_value(type_val)

        assert result == {
            "active": False,
            "age": 0,
            "name": "",
        }

    def test_should_return_nested_struct_with_default_values(self):
        """Should return nested struct with default values."""
        user_type = StructType(
            [
                ("name", StringType),
                ("age", IntegerType),
            ]
        )
        type_val = StructType(
            [
                ("user", user_type),
                ("score", FloatType),
            ]
        )

        result = default_value(type_val)

        assert result == {
            "score": 0.0,
            "user": {"age": 0, "name": ""},
        }

    def test_should_return_first_variant_case_with_default_value(self):
        """Should return first variant case with default value for Variant type."""
        type_val = VariantType(
            [
                ("none", NullType),
                ("some", IntegerType),
            ]
        )

        result = default_value(type_val)

        # Should return the first case (none) with its default value (null)
        assert result["type"] == "none"
        assert result["value"] is None

    def test_should_throw_for_empty_variant_type(self):
        """Should throw for empty Variant type."""
        type_val = VariantType([])
        with pytest.raises(RuntimeError, match=r"Cannot create a value of an empty variant"):
            default_value(type_val)

    def test_should_throw_for_function_type(self):
        """Should throw for Function type."""
        type_val = FunctionType([], NullType, [])
        with pytest.raises(RuntimeError, match=r"Cannot create a default value of type \.Function"):
            default_value(type_val)


class TestMinimalValue:
    """Tests for minimal_value function."""

    def test_should_throw_for_never_type(self):
        """Should throw for Never type."""
        with pytest.raises(RuntimeError, match=r"Cannot create a default value of type \.Never"):
            minimal_value(NeverType)

    def test_should_return_none_for_null_type(self):
        """Should return None for Null type."""
        result = minimal_value(NullType)
        assert result is None

    def test_should_return_false_for_boolean_type(self):
        """Should return False for Boolean type."""
        result = minimal_value(BooleanType)
        assert result is False

    def test_should_return_0_for_integer_type(self):
        """Should return 0 for Integer type."""
        result = minimal_value(IntegerType)
        assert result == 0

    def test_should_return_0_0_for_float_type(self):
        """Should return 0.0 for Float type."""
        result = minimal_value(FloatType)
        assert result == 0.0

    def test_should_return_empty_string_for_string_type(self):
        """Should return empty string for String type."""
        result = minimal_value(StringType)
        assert result == ""

    def test_should_return_epoch_date_for_datetime_type(self):
        """Should return epoch date for DateTime type."""
        result = minimal_value(DateTimeType)
        assert isinstance(result, DateTime)
        assert result.timestamp() == 0

    def test_should_return_empty_blob_for_blob_type(self):
        """Should return empty Blob for Blob type."""
        result = minimal_value(BlobType)
        assert isinstance(result, Blob)
        assert len(result.data) == 0

    def test_should_return_empty_array_for_array_type(self):
        """Should return empty array for Array type."""
        result = minimal_value(ArrayType(IntegerType))
        assert isinstance(result, EastArray)
        assert len(result) == 0

    def test_should_return_empty_set_for_set_type(self):
        """Should return empty EastSet for Set type."""
        result = minimal_value(SetType(StringType))
        assert isinstance(result, EastSet)
        assert len(result) == 0

    def test_should_return_empty_dict_for_dict_type(self):
        """Should return empty EastDict for Dict type."""
        result = minimal_value(DictType(IntegerType, StringType))
        assert isinstance(result, EastDict)
        assert len(result) == 0

    def test_should_return_struct_with_minimal_field_values(self):
        """Should return struct with minimal field values for Struct type."""
        type_val = StructType(
            [
                ("x", IntegerType),
                ("y", FloatType),
            ]
        )

        result = minimal_value(type_val)

        assert result == {
            "x": 0,
            "y": 0.0,
        }

    def test_should_return_first_variant_case_with_minimal_value(self):
        """Should return first variant case with minimal value for Variant type."""
        type_val = VariantType(
            [
                ("a", IntegerType),
                ("b", StringType),
            ]
        )

        result = minimal_value(type_val)

        # Should return the first case (a) with its default value (0)
        assert result["type"] == "a"
        assert result["value"] == 0

    def test_should_throw_for_empty_variant_type(self):
        """Should throw for empty Variant type."""
        type_val = VariantType([])
        with pytest.raises(RuntimeError, match=r"Cannot create a value of an empty variant"):
            minimal_value(type_val)

    def test_should_throw_for_function_type(self):
        """Should throw for Function type."""
        type_val = FunctionType([IntegerType], StringType, [])
        with pytest.raises(RuntimeError, match=r"Cannot create a default value of type \.Function"):
            minimal_value(type_val)
