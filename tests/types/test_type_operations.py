"""Tests for East type operation functions.

Ported from East/src/types.spec.ts - TypeUnion, TypeIntersect, TypeEqual test suites
"""

import pytest

from east.types.type_system import (
    ArrayType,
    FloatType,
    FunctionType,
    IntegerType,
    NeverType,
    StringType,
    StructTypeFromFields,
    TypeMismatchError,
    VariantTypeFromCases,
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
