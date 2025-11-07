"""Additional coverage tests for East type operations - edge cases and error paths.

Ported from East/src/types.spec.ts - Additional coverage tests section
"""

import pytest

from east.types.type_system import (
    ArrayType,
    DictType,
    FloatType,
    FunctionType,
    IntegerType,
    NullType,
    OptionType,
    SetType,
    SomeType,
    StringType,
    StructTypeFromFields,
    TypeMismatchError,
    VariantTypeFromCases,
    is_subtype,
    is_type_equal,
    is_value_of,
    type_equal,
    type_intersect,
    type_union,
)


class TestTypeEqualEdgeCases:
    """Edge case tests for type_equal function."""

    def test_should_handle_k1_gt_k2_variant_case_mismatch(self):
        """TypeEqual should handle k1 > k2 variant case mismatch."""
        t1 = VariantTypeFromCases([("a", IntegerType), ("c", StringType)])
        t2 = VariantTypeFromCases([("a", IntegerType), ("b", StringType)])
        with pytest.raises(
            TypeMismatchError, match=r"variant case b is not present in both variants"
        ):
            type_equal(t1, t2)

    def test_should_succeed_for_equal_variant_types(self):
        """TypeEqual should succeed for equal variant types."""
        t1 = VariantTypeFromCases([("a", IntegerType), ("b", StringType)])
        t2 = VariantTypeFromCases([("a", IntegerType), ("b", StringType)])
        result = type_equal(t1, t2)
        assert result.tag == "Variant"

    def test_should_succeed_for_equal_function_types(self):
        """TypeEqual should succeed for equal function types."""
        t1 = FunctionType([IntegerType, StringType], FloatType, [])
        t2 = FunctionType([IntegerType, StringType], FloatType, [])
        result = type_equal(t1, t2)
        assert result.tag == "Function"

    def test_should_propagate_errors_from_nested_types(self):
        """TypeEqual should propagate errors from nested types."""
        t1 = ArrayType(IntegerType)
        t2 = ArrayType(FloatType)
        with pytest.raises(TypeMismatchError):
            type_equal(t1, t2)

    def test_should_handle_variant_case_where_k1_lt_k2(self):
        """TypeEqual should handle variant case where k1 < k2."""
        t1 = VariantTypeFromCases([("a", IntegerType), ("b", StringType)])
        t2 = VariantTypeFromCases([("a", IntegerType), ("c", StringType)])
        with pytest.raises(
            TypeMismatchError, match=r"variant case b is not present in both variants"
        ):
            type_equal(t1, t2)

    def test_with_nested_type_mismatch_in_array(self):
        """TypeEqual with nested type mismatch in array."""
        t1 = StructTypeFromFields([("x", ArrayType(IntegerType))])
        t2 = StructTypeFromFields([("x", ArrayType(FloatType))])
        with pytest.raises(TypeMismatchError):
            type_equal(t1, t2)

    def test_should_throw_when_comparing_variant_with_non_variant(self):
        """TypeEqual should throw when comparing Variant with non-Variant."""
        t1 = VariantTypeFromCases([("a", IntegerType)])
        t2 = IntegerType
        with pytest.raises(TypeMismatchError, match=r"is not equal to.*incompatible types"):
            type_equal(t1, t2)

    def test_should_throw_when_comparing_function_with_non_function(self):
        """TypeEqual should throw when comparing Function with non-Function."""
        t1 = FunctionType([IntegerType], NullType, [])
        t2 = IntegerType
        with pytest.raises(TypeMismatchError, match=r"is not equal to.*incompatible types"):
            type_equal(t1, t2)

    def test_should_succeed_for_equal_dict_types(self):
        """TypeEqual should succeed for equal Dict types."""
        t1 = DictType(StringType, IntegerType)
        t2 = DictType(StringType, IntegerType)
        result = type_equal(t1, t2)
        assert result.tag == "Dict"

    def test_should_throw_when_comparing_dict_with_non_dict(self):
        """TypeEqual should throw when comparing Dict with non-Dict."""
        t1 = DictType(StringType, IntegerType)
        t2 = IntegerType
        with pytest.raises(TypeMismatchError, match=r"is not equal to.*incompatible types"):
            type_equal(t1, t2)

    def test_should_succeed_for_equal_struct_types(self):
        """TypeEqual should succeed for equal Struct types."""
        t1 = StructTypeFromFields([("x", IntegerType), ("y", StringType)])
        t2 = StructTypeFromFields([("x", IntegerType), ("y", StringType)])
        result = type_equal(t1, t2)
        assert result.tag == "Struct"

    def test_should_throw_when_comparing_struct_with_non_struct(self):
        """TypeEqual should throw when comparing Struct with non-Struct."""
        t1 = StructTypeFromFields([("x", IntegerType)])
        t2 = IntegerType
        with pytest.raises(TypeMismatchError, match=r"is not equal to.*incompatible types"):
            type_equal(t1, t2)

    def test_should_throw_when_comparing_array_with_non_array(self):
        """TypeEqual should throw when comparing Array with non-Array."""
        t1 = ArrayType(IntegerType)
        t2 = IntegerType
        with pytest.raises(TypeMismatchError, match=r"is not equal to.*incompatible types"):
            type_equal(t1, t2)

    def test_should_succeed_for_equal_set_types(self):
        """TypeEqual should succeed for equal Set types."""
        t1 = SetType(StringType)
        t2 = SetType(StringType)
        result = type_equal(t1, t2)
        assert result.tag == "Set"

    def test_should_throw_when_comparing_set_with_non_set(self):
        """TypeEqual should throw when comparing Set with non-Set."""
        t1 = SetType(StringType)
        t2 = IntegerType
        with pytest.raises(TypeMismatchError, match=r"is not equal to.*incompatible types"):
            type_equal(t1, t2)

    def test_catch_block_with_deeply_nested_error(self):
        """TypeEqual catch block with deeply nested error."""
        t1 = DictType(StringType, ArrayType(IntegerType))
        t2 = DictType(StringType, ArrayType(FloatType))
        with pytest.raises(TypeMismatchError):
            type_equal(t1, t2)


class TestTypeIntersectEdgeCases:
    """Edge case tests for type_intersect function."""

    def test_should_throw_for_functions_with_different_argument_counts(self):
        """TypeIntersect should throw for functions with different argument counts."""
        t1 = FunctionType([IntegerType], NullType, [])
        t2 = FunctionType([IntegerType, StringType], NullType, [])
        with pytest.raises(
            TypeMismatchError, match=r"functions take different number of arguments"
        ):
            type_intersect(t1, t2)

    def test_should_succeed_for_compatible_function_types(self):
        """TypeIntersect should succeed for compatible function types."""
        t1 = FunctionType([IntegerType], FloatType, [])
        t2 = FunctionType([IntegerType], FloatType, [])
        result = type_intersect(t1, t2)
        assert result.tag == "Function"

    def test_should_throw_when_intersecting_function_with_non_function(self):
        """TypeIntersect should throw when intersecting Function with non-Function."""
        t1 = FunctionType([IntegerType], NullType, [])
        t2 = IntegerType
        with pytest.raises(TypeMismatchError, match=r"Cannot intersect.*incompatible types"):
            type_intersect(t1, t2)

    def test_catch_block_with_nested_type_error(self):
        """TypeIntersect catch block with nested type error."""
        t1 = ArrayType(IntegerType)
        t2 = ArrayType(FloatType)
        with pytest.raises(TypeMismatchError):
            type_intersect(t1, t2)

    def test_should_throw_when_intersecting_variant_with_non_variant(self):
        """TypeIntersect should throw when intersecting Variant with non-Variant."""
        t1 = VariantTypeFromCases([("a", IntegerType)])
        t2 = IntegerType
        with pytest.raises(TypeMismatchError, match=r"Cannot intersect.*incompatible types"):
            type_intersect(t1, t2)

    def test_should_succeed_for_compatible_struct_types(self):
        """TypeIntersect should succeed for compatible struct types."""
        t1 = StructTypeFromFields([("x", IntegerType), ("y", StringType)])
        t2 = StructTypeFromFields([("x", IntegerType), ("y", StringType)])
        result = type_intersect(t1, t2)
        assert result.tag == "Struct"

    def test_should_throw_when_intersecting_struct_with_non_struct(self):
        """TypeIntersect should throw when intersecting Struct with non-Struct."""
        t1 = StructTypeFromFields([("x", IntegerType)])
        t2 = IntegerType
        with pytest.raises(TypeMismatchError, match=r"Cannot intersect.*incompatible types"):
            type_intersect(t1, t2)

    def test_should_succeed_for_compatible_dict_types(self):
        """TypeIntersect should succeed for compatible dict types."""
        t1 = DictType(StringType, IntegerType)
        t2 = DictType(StringType, IntegerType)
        result = type_intersect(t1, t2)
        assert result.tag == "Dict"

    def test_should_throw_when_intersecting_dict_with_non_dict(self):
        """TypeIntersect should throw when intersecting Dict with non-Dict."""
        t1 = DictType(StringType, IntegerType)
        t2 = IntegerType
        with pytest.raises(TypeMismatchError, match=r"Cannot intersect.*incompatible types"):
            type_intersect(t1, t2)

    def test_should_succeed_for_compatible_set_types(self):
        """TypeIntersect should succeed for compatible set types."""
        t1 = SetType(StringType)
        t2 = SetType(StringType)
        result = type_intersect(t1, t2)
        assert result.tag == "Set"

    def test_should_throw_when_intersecting_set_with_non_set(self):
        """TypeIntersect should throw when intersecting Set with non-Set."""
        t1 = SetType(StringType)
        t2 = IntegerType
        with pytest.raises(TypeMismatchError, match=r"Cannot intersect.*incompatible types"):
            type_intersect(t1, t2)

    def test_should_succeed_for_compatible_array_types(self):
        """TypeIntersect should succeed for compatible array types."""
        t1 = ArrayType(IntegerType)
        t2 = ArrayType(IntegerType)
        result = type_intersect(t1, t2)
        assert result.tag == "Array"

    def test_should_throw_when_intersecting_array_with_non_array(self):
        """TypeIntersect should throw when intersecting Array with non-Array."""
        t1 = ArrayType(IntegerType)
        t2 = IntegerType
        with pytest.raises(TypeMismatchError, match=r"Cannot intersect.*incompatible types"):
            type_intersect(t1, t2)


class TestTypeUnionEdgeCases:
    """Edge case tests for type_union function."""

    def test_should_throw_for_functions_with_different_argument_counts(self):
        """TypeUnion should throw for functions with different argument counts."""
        t1 = FunctionType([IntegerType], NullType, [])
        t2 = FunctionType([IntegerType, StringType], NullType, [])
        with pytest.raises(
            TypeMismatchError, match=r"functions take different number of arguments"
        ):
            type_union(t1, t2)

    def test_should_throw_when_unioning_function_with_non_function(self):
        """TypeUnion should throw when unioning Function with non-Function."""
        t1 = FunctionType([IntegerType], NullType, [])
        t2 = IntegerType
        with pytest.raises(TypeMismatchError, match=r"Cannot union.*incompatible types"):
            type_union(t1, t2)

    def test_should_throw_when_unioning_dict_with_non_dict(self):
        """TypeUnion should throw when unioning Dict with non-Dict."""
        t1 = DictType(StringType, IntegerType)
        t2 = IntegerType
        with pytest.raises(TypeMismatchError, match=r"Cannot union.*incompatible types"):
            type_union(t1, t2)

    def test_should_throw_when_unioning_struct_with_non_struct(self):
        """TypeUnion should throw when unioning Struct with non-Struct."""
        t1 = StructTypeFromFields([("x", IntegerType)])
        t2 = IntegerType
        with pytest.raises(TypeMismatchError, match=r"Cannot union.*incompatible types"):
            type_union(t1, t2)

    def test_should_throw_when_unioning_variant_with_non_variant(self):
        """TypeUnion should throw when unioning Variant with non-Variant."""
        t1 = VariantTypeFromCases([("a", IntegerType)])
        t2 = IntegerType
        with pytest.raises(TypeMismatchError, match=r"Cannot union.*incompatible types"):
            type_union(t1, t2)

    def test_should_throw_when_unioning_set_with_non_set(self):
        """TypeUnion should throw when unioning Set with non-Set."""
        t1 = SetType(StringType)
        t2 = IntegerType
        with pytest.raises(TypeMismatchError, match=r"Cannot union.*incompatible types"):
            type_union(t1, t2)


class TestOptionAndSomeTypes:
    """Tests for SomeType and OptionType helpers."""

    def test_some_type_should_create_option_variant_with_some_case(self):
        """SomeType should create option variant with some case."""
        typ = SomeType(IntegerType)
        assert typ.tag == "Variant"
        cases = typ.value
        case_dict = {case.name: case.type for case in cases}
        assert "some" in case_dict
        assert case_dict["some"] == IntegerType

    def test_option_type_should_create_variant_with_none_and_some_cases(self):
        """OptionType should create variant with none and some cases."""
        typ = OptionType(IntegerType)
        assert typ.tag == "Variant"
        cases = typ.value
        case_dict = {case.name: case.type for case in cases}
        assert case_dict["none"] == NullType
        assert case_dict["some"] == IntegerType


class TestIsSubtypeEdgeCases:
    """Edge case tests for is_subtype function."""

    def test_should_return_false_for_incompatible_variant_types(self):
        """isSubtype should return false for incompatible variant types."""
        t1 = VariantTypeFromCases([("a", IntegerType)])
        t2 = IntegerType
        assert is_subtype(t1, t2) is False

    def test_should_return_false_for_incompatible_function_types(self):
        """isSubtype should return false for incompatible function types."""
        t1 = FunctionType([IntegerType], NullType, [])
        t2 = IntegerType
        assert is_subtype(t1, t2) is False

    def test_should_return_false_for_set_compared_to_non_set(self):
        """isSubtype should return false for Set compared to non-Set."""
        t1 = SetType(StringType)
        t2 = IntegerType
        assert is_subtype(t1, t2) is False

    def test_should_return_false_for_dict_compared_to_non_dict(self):
        """isSubtype should return false for Dict compared to non-Dict."""
        t1 = DictType(StringType, IntegerType)
        t2 = IntegerType
        assert is_subtype(t1, t2) is False

    def test_should_return_false_for_struct_compared_to_non_struct(self):
        """isSubtype should return false for Struct compared to non-Struct."""
        t1 = StructTypeFromFields([("x", IntegerType)])
        t2 = IntegerType
        assert is_subtype(t1, t2) is False

    def test_should_return_false_for_array_compared_to_non_array(self):
        """isSubtype should return false for Array compared to non-Array."""
        t1 = ArrayType(IntegerType)
        t2 = IntegerType
        assert is_subtype(t1, t2) is False


class TestIsTypeEqualEdgeCases:
    """Edge case tests for is_type_equal function."""

    def test_should_return_false_for_function_compared_to_non_function(self):
        """isTypeEqual should return false for Function compared to non-Function."""
        t1 = FunctionType([IntegerType], NullType, [])
        t2 = IntegerType
        assert is_type_equal(t1, t2) is False


class TestIsValueOfEdgeCases:
    """Edge case tests for is_value_of function."""

    def test_should_return_false_for_non_set_value_with_set_type(self):
        """isValueOf should return false for non-Set value with Set type."""
        assert is_value_of([], SetType(IntegerType)) is False

    def test_should_return_false_for_non_map_value_with_dict_type(self):
        """isValueOf should return false for non-Map value with Dict type."""
        assert is_value_of([], DictType(StringType, IntegerType)) is False
