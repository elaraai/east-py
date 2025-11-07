"""Tests for East type operation functions.

Ported from East/src/types.spec.ts - TypeUnion, TypeIntersect, TypeEqual test suites
"""

import pytest

from east.types.type_system import (
    ArrayType,
    BooleanType,
    FloatType,
    FunctionType,
    IntegerType,
    NeverType,
    StringType,
    StructTypeFromFields,
    TypeMismatchError,
    VariantTypeFromCases,
    type_intersect,
    type_union,
)


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
        t1 = VariantTypeFromCases([("a", IntegerType), ("b", StringType)])
        t2 = VariantTypeFromCases([("b", StringType), ("c", FloatType)])
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
        t1 = StructTypeFromFields([("x", IntegerType), ("y", FloatType)])
        t2 = StructTypeFromFields([("x", IntegerType), ("y", FloatType)])
        result = type_union(t1, t2)
        assert result.tag == "Struct"

    def test_should_throw_for_structs_with_different_field_count(self):
        """should throw for structs with different field count."""
        t1 = StructTypeFromFields([("x", IntegerType)])
        t2 = StructTypeFromFields([("x", IntegerType), ("y", FloatType)])
        with pytest.raises(TypeMismatchError, match=r"structs contain different number of fields"):
            type_union(t1, t2)

    def test_should_throw_for_structs_with_different_field_names_at_position_0(self):
        """should throw for structs with different field names at position 0."""
        t1 = StructTypeFromFields([("x", IntegerType)])
        t2 = StructTypeFromFields([("y", IntegerType)])
        with pytest.raises(TypeMismatchError, match=r"struct field 0 has mismatched names x and y"):
            type_union(t1, t2)

    def test_should_throw_for_structs_with_mismatched_field_names_in_multi_field_structs(self):
        """should throw for structs with mismatched field names in multi-field structs."""
        t1 = StructTypeFromFields([("a", IntegerType), ("b", StringType), ("c", FloatType)])
        t2 = StructTypeFromFields([("a", IntegerType), ("x", StringType), ("c", FloatType)])
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
        t1 = VariantTypeFromCases([("a", IntegerType), ("b", StringType), ("c", FloatType)])
        t2 = VariantTypeFromCases([("b", StringType), ("c", FloatType), ("d", BooleanType)])
        result = type_intersect(t1, t2)
        assert result.tag == "Variant"
        # TypeIntersect for variants keeps cases in t1 that are also in t2
        cases = result.value
        case_dict = {case.name: case.type for case in cases}
        assert case_dict == {"b": StringType, "c": FloatType}

    def test_should_throw_for_variants_with_no_overlapping_cases(self):
        """should throw for variants with no overlapping cases."""
        t1 = VariantTypeFromCases([("a", IntegerType)])
        t2 = VariantTypeFromCases([("b", StringType)])
        with pytest.raises(TypeMismatchError, match=r"variants have no overlapping cases"):
            type_intersect(t1, t2)
