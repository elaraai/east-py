"""Tests for East type system."""

from datetime import UTC, datetime

from east.types.primitives import Blob, null
from east.types.type_system import (
    ArrayType,
    BlobType,
    BooleanType,
    DateTimeType,
    DictType,
    EastType,
    EastTypeType,
    FloatType,
    FunctionType,
    IntegerType,
    NeverType,
    NullType,
    RecursiveTypeRef,
    SetType,
    StringType,
    StructType,
    StructTypeFromFields,
    VariantType,
    VariantTypeFromCases,
    recursive_type,
    type_of,
)


class TestPrimitiveTypeConstructors:
    """Tests for primitive type constructors."""

    def test_null_type(self):
        """NullType is an EastType."""
        assert isinstance(NullType, EastType)
        assert NullType.tag == "Null"
        assert NullType.value == null

    def test_boolean_type(self):
        """BooleanType is an EastType."""
        assert isinstance(BooleanType, EastType)
        assert BooleanType.tag == "Boolean"
        assert BooleanType.value == null

    def test_integer_type(self):
        """IntegerType is an EastType."""
        assert isinstance(IntegerType, EastType)
        assert IntegerType.tag == "Integer"
        assert IntegerType.value == null

    def test_float_type(self):
        """FloatType is an EastType."""
        assert isinstance(FloatType, EastType)
        assert FloatType.tag == "Float"
        assert FloatType.value == null

    def test_string_type(self):
        """StringType is an EastType."""
        assert isinstance(StringType, EastType)
        assert StringType.tag == "String"
        assert StringType.value == null

    def test_blob_type(self):
        """BlobType is an EastType."""
        assert isinstance(BlobType, EastType)
        assert BlobType.tag == "Blob"
        assert BlobType.value == null

    def test_datetime_type(self):
        """DateTimeType is an EastType."""
        assert isinstance(DateTimeType, EastType)
        assert DateTimeType.tag == "DateTime"
        assert DateTimeType.value == null

    def test_never_type(self):
        """NeverType is an EastType."""
        assert isinstance(NeverType, EastType)
        assert NeverType.tag == "Never"
        assert NeverType.value == null


class TestContainerTypeConstructors:
    """Tests for container type constructors."""

    def test_array_type(self):
        """ArrayType creates array types."""
        arr_type = ArrayType(IntegerType)
        assert isinstance(arr_type, EastType)
        assert arr_type.tag == "Array"
        assert arr_type.value == IntegerType

    def test_array_of_arrays(self):
        """Nested arrays."""
        arr_type = ArrayType(ArrayType(IntegerType))
        assert arr_type.tag == "Array"
        assert arr_type.value.tag == "Array"
        assert arr_type.value.value == IntegerType

    def test_set_type(self):
        """SetType creates set types."""
        set_type = SetType(StringType)
        assert isinstance(set_type, EastType)
        assert set_type.tag == "Set"
        assert set_type.value == StringType

    def test_dict_type(self):
        """DictType creates dict types."""
        dict_type = DictType(StringType, IntegerType)
        assert isinstance(dict_type, EastType)
        assert dict_type.tag == "Dict"
        # Dict value is a struct with key and value fields
        dict_struct = dict_type.value
        assert dict_struct.key == StringType
        assert dict_struct.value == IntegerType


class TestStructuralTypeConstructors:
    """Tests for structural type constructors."""

    def test_struct_type_from_fields(self):
        """StructTypeFromFields creates struct types."""
        fields = [("name", StringType), ("age", IntegerType)]
        struct_type = StructTypeFromFields(fields)
        assert isinstance(struct_type, EastType)
        assert struct_type.tag == "Struct"
        # Value is a list of field structs
        field_structs = struct_type.value
        assert len(field_structs) == 2
        assert field_structs[0].name == "name"
        assert field_structs[0].type == StringType
        assert field_structs[1].name == "age"
        assert field_structs[1].type == IntegerType

    def test_empty_struct_type(self):
        """Empty struct type."""
        struct_type = StructTypeFromFields([])
        assert struct_type.tag == "Struct"
        assert struct_type.value == []

    def test_variant_type_from_cases(self):
        """VariantTypeFromCases creates variant types."""
        cases = [("Some", IntegerType), ("None", NullType)]
        variant_type = VariantTypeFromCases(cases)
        assert isinstance(variant_type, EastType)
        assert variant_type.tag == "Variant"
        # Value is a list of case structs
        case_structs = variant_type.value
        assert len(case_structs) == 2
        # Cases are sorted by name
        assert case_structs[0].name == "None"
        assert case_structs[0].type == NullType
        assert case_structs[1].name == "Some"
        assert case_structs[1].type == IntegerType

    def test_variant_sorting(self):
        """Variant cases are sorted by name."""
        cases = [("Zebra", IntegerType), ("Apple", StringType), ("Banana", BlobType)]
        variant_type = VariantTypeFromCases(cases)
        case_structs = variant_type.value
        assert case_structs[0].name == "Apple"
        assert case_structs[1].name == "Banana"
        assert case_structs[2].name == "Zebra"


class TestFunctionType:
    """Tests for function types."""

    def test_function_type(self):
        """FunctionType creates function types."""
        func_type = FunctionType([IntegerType, StringType], BooleanType, ["platform1"])
        assert isinstance(func_type, EastType)
        assert func_type.tag == "Function"
        # Value is a struct with inputs, output, platforms
        func_struct = func_type.value
        assert func_struct.inputs == [IntegerType, StringType]
        assert func_struct.output == BooleanType
        assert func_struct.platforms == ["platform1"]

    def test_function_no_inputs(self):
        """Function with no inputs."""
        func_type = FunctionType([], IntegerType, [])
        func_struct = func_type.value
        assert func_struct.inputs == []
        assert func_struct.output == IntegerType


class TestTypeEquality:
    """Tests for type equality."""

    def test_primitive_types_equal(self):
        """Same primitive types are equal."""
        assert IntegerType == IntegerType
        assert StringType == StringType
        assert BooleanType != IntegerType

    def test_array_types_equal(self):
        """Array types equal if element types equal."""
        arr1 = ArrayType(IntegerType)
        arr2 = ArrayType(IntegerType)
        arr3 = ArrayType(StringType)
        assert arr1 == arr2
        assert arr1 != arr3

    def test_dict_types_equal(self):
        """Dict types equal if key and value types equal."""
        dict1 = DictType(StringType, IntegerType)
        dict2 = DictType(StringType, IntegerType)
        dict3 = DictType(IntegerType, StringType)
        assert dict1 == dict2
        assert dict1 != dict3

    def test_struct_types_equal(self):
        """Struct types equal if fields equal."""
        struct1 = StructTypeFromFields([("name", StringType), ("age", IntegerType)])
        struct2 = StructTypeFromFields([("name", StringType), ("age", IntegerType)])
        struct3 = StructTypeFromFields([("name", StringType)])
        assert struct1 == struct2
        assert struct1 != struct3

    def test_struct_field_order_matters(self):
        """Field order matters for struct equality."""
        struct1 = StructTypeFromFields([("name", StringType), ("age", IntegerType)])
        struct2 = StructTypeFromFields([("age", IntegerType), ("name", StringType)])
        assert struct1 != struct2

    def test_variant_types_equal(self):
        """Variant types equal if cases equal (after sorting)."""
        variant1 = VariantTypeFromCases([("Some", IntegerType), ("None", NullType)])
        variant2 = VariantTypeFromCases(
            [("None", NullType), ("Some", IntegerType)]
        )  # Different order
        variant3 = VariantTypeFromCases([("Some", IntegerType)])
        assert variant1 == variant2  # Order doesn't matter (sorted)
        assert variant1 != variant3


class TestTypeHashing:
    """Tests for type hashing."""

    def test_primitive_types_hashable(self):
        """Primitive types are hashable."""
        types_set = {IntegerType, StringType, BooleanType, IntegerType}
        assert len(types_set) == 3  # IntegerType appears twice

    def test_container_types_hashable(self):
        """Container types are hashable."""
        arr1 = ArrayType(IntegerType)
        arr2 = ArrayType(IntegerType)
        arr3 = ArrayType(StringType)
        types_set = {arr1, arr2, arr3}
        assert len(types_set) == 2  # arr1 and arr2 are equal

    def test_struct_types_hashable(self):
        """Struct types are hashable."""
        struct1 = StructTypeFromFields([("name", StringType)])
        struct2 = StructTypeFromFields([("name", StringType)])
        types_set = {struct1, struct2}
        assert len(types_set) == 1


class TestRecursiveTypes:
    """Tests for recursive types."""

    def test_recursive_type_ref(self):
        """RecursiveTypeRef creates recursive references."""
        ref = RecursiveTypeRef(0)
        assert isinstance(ref, EastType)
        assert ref.tag == "Recursive"
        assert ref.value == 0

    def test_recursive_type_builder(self):
        """recursive_type builds recursive types."""
        # Create a linked list type: List = Variant { Cons(Integer, List), Nil }
        list_type = recursive_type(
            lambda self: VariantTypeFromCases(
                [
                    ("Cons", StructTypeFromFields([("value", IntegerType), ("next", self)])),
                    ("Nil", NullType),
                ]
            )
        )
        assert isinstance(list_type, EastType)
        assert list_type.tag == "Variant"
        cases = list_type.value
        assert len(cases) == 2

        # Check Cons case has recursive reference
        cons_case = cases[0]  # Cons (alphabetically first)
        assert cons_case.name == "Cons"
        cons_struct_type = cons_case.type
        cons_fields = cons_struct_type.value
        assert len(cons_fields) == 2
        assert cons_fields[0].name == "value"
        assert cons_fields[0].type == IntegerType
        assert cons_fields[1].name == "next"
        # The recursive reference
        next_type = cons_fields[1].type
        assert next_type.tag == "Recursive"
        assert next_type.value == 0  # Refers to depth 0 (the list type itself)

    def test_recursive_array(self):
        """Recursive array type."""
        # Array of Array of ... Integer
        nested_arr = recursive_type(lambda self: ArrayType(self))
        assert nested_arr.tag == "Array"
        inner = nested_arr.value
        assert inner.tag == "Recursive"
        assert inner.value == 0

    def test_nested_recursive(self):
        """Nested recursive types."""
        # Tree: Variant { Node(Integer, Array<Tree>), Leaf }
        tree_type = recursive_type(
            lambda self: VariantTypeFromCases(
                [
                    (
                        "Node",
                        StructTypeFromFields(
                            [("value", IntegerType), ("children", ArrayType(self))]
                        ),
                    ),
                    ("Leaf", NullType),
                ]
            )
        )
        assert tree_type.tag == "Variant"
        cases = tree_type.value

        # Check Node case
        # cases[0] is Leaf (alphabetically first)
        node_case = cases[1]
        assert node_case.name == "Node"
        node_struct = node_case.type
        node_fields = node_struct.value
        children_field = node_fields[1]
        assert children_field.name == "children"
        # children is Array<Recursive(0)>
        children_type = children_field.type
        assert children_type.tag == "Array"
        assert children_type.value.tag == "Recursive"


class TestEastTypeType:
    """Tests for EastTypeType - the type of types."""

    def test_east_type_type_exists(self):
        """EastTypeType is defined."""
        assert EastTypeType is not None
        assert isinstance(EastTypeType, EastType)

    def test_east_type_type_is_variant(self):
        """EastTypeType is a variant."""
        assert EastTypeType.tag == "Variant"

    def test_east_type_has_all_cases(self):
        """EastTypeType has all type cases."""
        cases = EastTypeType.value
        case_names = [case.name for case in cases]
        expected_cases = [
            "Array",
            "Blob",
            "Boolean",
            "DateTime",
            "Dict",
            "Float",
            "Function",
            "Integer",
            "Never",
            "Null",
            "Recursive",
            "Set",
            "String",
            "Struct",
            "Variant",
        ]
        assert sorted(case_names) == sorted(expected_cases)

    def test_integer_type_has_east_type(self):
        """IntegerType has _east_type of EastTypeType."""
        assert IntegerType._east_type == EastTypeType

    def test_array_type_has_east_type(self):
        """ArrayType instances have _east_type of EastTypeType."""
        arr = ArrayType(IntegerType)
        assert arr._east_type == EastTypeType


class TestComplexTypes:
    """Tests for complex type combinations."""

    def test_array_of_structs(self):
        """Array of structs."""
        person_type = StructTypeFromFields([("name", StringType), ("age", IntegerType)])
        people_type = ArrayType(person_type)
        assert people_type.tag == "Array"
        assert people_type.value == person_type

    def test_dict_of_variants(self):
        """Dict with variant values."""
        option_type = VariantTypeFromCases([("Some", IntegerType), ("None", NullType)])
        dict_type = DictType(StringType, option_type)
        assert dict_type.tag == "Dict"
        dict_struct = dict_type.value
        assert dict_struct.value == option_type

    def test_nested_containers(self):
        """Nested containers."""
        # Dict<String, Array<Set<Integer>>>
        inner = SetType(IntegerType)
        middle = ArrayType(inner)
        outer = DictType(StringType, middle)
        assert outer.tag == "Dict"
        outer_struct = outer.value
        assert outer_struct.value.tag == "Array"
        assert outer_struct.value.value.tag == "Set"
        assert outer_struct.value.value.value == IntegerType

    def test_function_with_struct_params(self):
        """Function with struct parameters."""
        person_type = StructTypeFromFields([("name", StringType), ("age", IntegerType)])
        func_type = FunctionType([person_type], StringType, ["greet"])
        assert func_type.tag == "Function"
        func_struct = func_type.value
        assert func_struct.inputs[0] == person_type
        assert func_struct.output == StringType


class TestRuntimeTypeCreation:
    """Tests for creating runtime type instances."""

    def test_create_struct_from_type(self):
        """Create struct instances from StructType."""
        # First create the EastType representing a struct type
        _person_east_type = StructTypeFromFields([("name", StringType), ("age", IntegerType)])

        # Then create a runtime StructType from the field specs
        person_runtime_type = StructType((("name", StringType), ("age", IntegerType)))

        # Create an instance
        person = person_runtime_type.create(name="Alice", age=30)
        assert person.name == "Alice"
        assert person.age == 30

    def test_create_variant_from_type(self):
        """Create variant instances from VariantType."""
        # Create the EastType representing an option type
        _option_east_type = VariantTypeFromCases([("Some", IntegerType), ("None", NullType)])

        # Create a runtime VariantType
        option_runtime_type = VariantType((("Some", IntegerType), ("None", NullType)))

        # Create instances
        some = option_runtime_type.create("Some", 42)
        none = option_runtime_type.create("None")
        assert some.tag == "Some"
        assert some.value == 42
        assert none.tag == "None"


class TestTypeRepr:
    """Tests for type representation."""

    def test_primitive_type_repr(self):
        """Primitive types have readable repr."""
        assert repr(IntegerType) == ".Integer"
        assert repr(StringType) == ".String"

    def test_array_type_repr(self):
        """Array type repr."""
        arr = ArrayType(IntegerType)
        assert repr(arr) == ".Array .Integer"

    def test_nested_type_repr(self):
        """Nested type repr."""
        arr = ArrayType(ArrayType(IntegerType))
        assert repr(arr) == ".Array .Array .Integer"


class TestTypeOf:
    """Tests for type_of function."""

    def test_null(self):
        """type_of null."""
        assert type_of(null) == NullType
        assert type_of(None) == NullType

    def test_boolean(self):
        """type_of boolean."""
        assert type_of(True) == BooleanType
        assert type_of(False) == BooleanType

    def test_integer(self):
        """type_of integer."""
        assert type_of(42) == IntegerType
        assert type_of(0) == IntegerType
        assert type_of(-100) == IntegerType

    def test_float(self):
        """type_of float."""
        assert type_of(3.14) == FloatType
        assert type_of(0.0) == FloatType
        assert type_of(float("nan")) == FloatType

    def test_string(self):
        """type_of string."""
        assert type_of("hello") == StringType
        assert type_of("") == StringType

    def test_blob(self):
        """type_of blob."""
        b = Blob(b"test")
        assert type_of(b) == BlobType

    def test_datetime(self):
        """type_of datetime."""
        dt = datetime.now(UTC)
        assert type_of(dt) == DateTimeType

    def test_array(self):
        """type_of array."""
        from east.types.containers import EastArray

        arr = EastArray(IntegerType, [1, 2, 3])
        arr_type = type_of(arr)
        assert arr_type.tag == "Array"
        assert arr_type.value == IntegerType

    def test_set(self):
        """type_of set."""
        from east.types.containers import EastSet

        s = EastSet(StringType, ["a", "b"])
        set_type = type_of(s)
        assert set_type.tag == "Set"
        assert set_type.value == StringType

    def test_dict(self):
        """type_of dict."""
        from east.types.containers import EastDict

        d = EastDict(StringType, IntegerType, {"a": 1})
        dict_type = type_of(d)
        assert dict_type.tag == "Dict"
        assert dict_type.value.key == StringType
        assert dict_type.value.value == IntegerType

    def test_struct(self):
        """type_of struct."""
        person_type = StructType((("name", StringType), ("age", IntegerType)))
        person = person_type.create(name="Alice", age=30)
        assert type_of(person) == person_type

    def test_variant(self):
        """type_of variant."""
        option_type = VariantType((("Some", IntegerType), ("None", NullType)))
        some = option_type.create("Some", 42)
        assert type_of(some) == option_type

    def test_east_type(self):
        """type_of EastType."""
        assert type_of(IntegerType) == EastTypeType
        assert type_of(ArrayType(StringType)) == EastTypeType
        assert type_of(EastTypeType) == EastTypeType

    def test_unknown_type(self):
        """type_of raises TypeError for unknown types."""
        import pytest

        with pytest.raises(TypeError, match="Unknown East type"):
            type_of(object())
