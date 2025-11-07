"""Tests for East type predicate functions.

Ported from East/src/types.spec.ts - isDataType, isImmutableType, isValueOf test suites
"""

import pytest

from east.types.containers import EastArray, EastDict, EastSet
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
    StructTypeFromFields,
    VariantType,
    VariantTypeFromCases,
    is_data_type,
    is_immutable_type,
    is_subtype,
    is_type_equal,
    is_value_of,
)


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
        typ = StructTypeFromFields([("x", IntegerType), ("y", FloatType)])
        assert is_data_type(typ) is True

    def test_should_throw_error_for_struct_with_function_field(self):
        """should throw error for struct with function field."""
        with pytest.raises(TypeError, match=r"Struct field f must be a \(non-function\) data type"):
            StructTypeFromFields([("x", IntegerType), ("f", FunctionType([], NullType, []))])

    def test_should_return_true_for_variant_with_data_cases(self):
        """should return true for variant with data cases."""
        typ = VariantTypeFromCases([("none", NullType), ("some", IntegerType)])
        assert is_data_type(typ) is True

    def test_should_throw_error_for_variant_with_function_case(self):
        """should throw error for variant with function case."""
        with pytest.raises(
            TypeError, match=r"Variant case func must be a \(non-function\) data type"
        ):
            VariantTypeFromCases([("data", IntegerType), ("func", FunctionType([], NullType, []))])

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
        typ = StructTypeFromFields([("x", IntegerType), ("y", StringType)])
        assert is_immutable_type(typ) is True

    def test_should_return_false_for_struct_with_mutable_field(self):
        """should return false for struct with mutable field."""
        typ = StructTypeFromFields([("x", IntegerType), ("arr", ArrayType(IntegerType))])
        assert is_immutable_type(typ) is False

    def test_should_return_true_for_variant_with_immutable_cases(self):
        """should return true for variant with immutable cases."""
        typ = VariantTypeFromCases([("none", NullType), ("some", IntegerType)])
        assert is_immutable_type(typ) is True

    def test_should_return_false_for_variant_with_mutable_case(self):
        """should return false for variant with mutable case."""
        typ = VariantTypeFromCases([("data", IntegerType), ("list", ArrayType(IntegerType))])
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
        struct_type = StructType((("x", IntegerType), ("y", FloatType)))
        struct_val = struct_type.create(x=42, y=3.14)
        assert (
            is_value_of(struct_val, StructTypeFromFields([("x", IntegerType), ("y", FloatType)]))
            is True
        )

    def test_should_validate_variant_values(self):
        """should validate variant values."""
        variant_type = VariantType((("none", NullType), ("some", IntegerType)))
        none_val = variant_type.create("none", None)
        some_val = variant_type.create("some", 42)

        variant_type_from_cases = VariantTypeFromCases([("none", NullType), ("some", IntegerType)])
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
        t1 = StructTypeFromFields([("x", IntegerType), ("y", FloatType)])
        t2 = StructTypeFromFields([("x", IntegerType), ("y", FloatType)])
        t3 = StructTypeFromFields([("x", IntegerType), ("y", StringType)])
        assert is_type_equal(t1, t2) is True
        assert is_type_equal(t1, t3) is False

    def test_should_compare_variant_types(self):
        """should compare variant types."""
        t1 = VariantTypeFromCases([("none", NullType), ("some", IntegerType)])
        t2 = VariantTypeFromCases([("none", NullType), ("some", IntegerType)])
        t3 = VariantTypeFromCases([("none", NullType), ("some", FloatType)])
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
        t1 = VariantTypeFromCases([("a", IntegerType), ("b", StringType), ("c", FloatType)])
        t2 = VariantTypeFromCases([("a", IntegerType), ("b", StringType)])
        assert is_subtype(t1, t2) is False
        assert is_subtype(t2, t1) is True

    def test_struct_subtyping_is_structural(self):
        """struct subtyping is structural."""
        t1 = StructTypeFromFields([("x", IntegerType), ("y", FloatType)])
        t2 = StructTypeFromFields([("x", IntegerType), ("y", FloatType)])
        assert is_subtype(t1, t2) is True

    def test_function_subtyping_contravariant_inputs_covariant_output(self):
        """function subtyping - contravariant inputs, covariant output."""
        t1 = FunctionType([IntegerType], NeverType, [])
        t2 = FunctionType([IntegerType], IntegerType, [])
        # t1 has output Never which is subtype of Integer, so t1 <: t2
        assert is_subtype(t1, t2) is True
