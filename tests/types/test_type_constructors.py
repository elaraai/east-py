"""Tests for East type constructors with validation.

Ported from East/src/types.spec.ts - Type constructors test suite
"""

import pytest

from east.types.type_system import (
    ArrayType,
    BooleanType,
    DictType,
    FloatType,
    FunctionType,
    IntegerType,
    NullType,
    OptionType,
    SetType,
    StringType,
    StructTypeFromFields,
    VariantTypeFromCases,
)


class TestTypeConstructors:
    """Test suite for type constructor validation."""

    def test_array_type_should_create_array_types(self):
        """ArrayType should create array types."""
        typ = ArrayType(IntegerType)
        assert typ.tag == "Array"
        assert typ.value == IntegerType

    def test_array_type_should_throw_for_function_element_types(self):
        """ArrayType should throw for function element types."""
        with pytest.raises(
            TypeError, match=r"Array value type must be a \(non-function\) data type"
        ):
            ArrayType(FunctionType([], NullType, []))

    def test_set_type_should_create_set_types(self):
        """SetType should create set types."""
        typ = SetType(StringType)
        assert typ.tag == "Set"
        assert typ.value == StringType

    def test_set_type_should_throw_for_mutable_key_types(self):
        """SetType should throw for mutable key types."""
        with pytest.raises(TypeError, match=r"Set key type must be an immutable type"):
            SetType(ArrayType(IntegerType))

    def test_dict_type_should_create_dict_types(self):
        """DictType should create dict types."""
        typ = DictType(StringType, IntegerType)
        assert typ.tag == "Dict"
        # DictType creates a struct with key and value fields
        dict_struct = typ.value
        assert dict_struct.key == StringType
        assert dict_struct.value == IntegerType

    def test_dict_type_should_throw_for_mutable_key_types(self):
        """DictType should throw for mutable key types."""
        with pytest.raises(TypeError, match=r"Dict key type must be an immutable type"):
            DictType(ArrayType(IntegerType), StringType)

    def test_dict_type_should_throw_for_function_value_types(self):
        """DictType should throw for function value types."""
        with pytest.raises(
            TypeError, match=r"Dict value type must be a \(non-function\) data type"
        ):
            DictType(StringType, FunctionType([], NullType, []))

    def test_struct_type_should_create_struct_types(self):
        """StructType should create struct types."""
        typ = StructTypeFromFields([("x", IntegerType), ("y", FloatType)])
        assert typ.tag == "Struct"
        fields = typ.value
        assert len(fields) == 2
        assert fields[0].name == "x"
        assert fields[0].type == IntegerType
        assert fields[1].name == "y"
        assert fields[1].type == FloatType

    def test_variant_type_should_create_variant_types_with_sorted_cases(self):
        """VariantType should create variant types with sorted cases."""
        # Pass cases in unsorted order
        typ = VariantTypeFromCases([("b", IntegerType), ("a", StringType)])
        assert typ.tag == "Variant"
        cases = typ.value
        # Cases should be sorted alphabetically
        case_names = [case.name for case in cases]
        assert case_names == ["a", "b"]
        assert cases[0].type == StringType
        assert cases[1].type == IntegerType

    def test_function_type_should_create_function_types(self):
        """FunctionType should create function types."""
        typ = FunctionType([IntegerType, StringType], BooleanType, [])
        assert typ.tag == "Function"
        func = typ.value
        assert func.inputs == [IntegerType, StringType]
        assert func.output == BooleanType
        assert func.platforms == []

    def test_option_type_should_create_option_types(self):
        """OptionType should create option types."""
        typ = OptionType(IntegerType)
        assert typ.tag == "Variant"
        cases = typ.value
        case_names = [case.name for case in cases]
        # Should have none and some cases, sorted
        assert case_names == ["none", "some"]
        assert cases[0].type == NullType
        assert cases[1].type == IntegerType
