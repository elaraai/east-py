"""East type system.

EastType is a recursive variant representing all East types.
It's homoiconic - the type of types is itself an East value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from east.types.primitives import null
from east.types.structural import Case, EastStruct, EastVariant, make_case


@dataclass(frozen=True, eq=True)
class StructType:
    """Runtime representation of a struct type.

    Tracks field names and their types.
    """

    fields: tuple[tuple[str, EastType], ...]

    def field_names(self) -> list[str]:
        """Get list of field names."""
        return [name for name, _ in self.fields]

    def field_types(self) -> list[EastType]:
        """Get list of field types."""
        return [typ for _, typ in self.fields]

    def field_index(self, name: str) -> int:
        """Get index of field by name.

        Args:
            name: Field name

        Returns:
            Index of field

        Raises:
            KeyError: If field not found
        """
        for i, (field_name, _) in enumerate(self.fields):
            if field_name == name:
                return i
        raise KeyError(f"No field named '{name}'")

    def create(self, **kwargs: Any) -> EastStruct:
        """Create a struct instance with these field values.

        Args:
            **kwargs: Field name-value pairs

        Returns:
            EastStruct instance

        Raises:
            ValueError: If wrong number of fields or missing/extra fields
        """
        if len(kwargs) != len(self.fields):
            raise ValueError(f"Expected {len(self.fields)} fields, got {len(kwargs)}")

        values = []
        for name, _ in self.fields:
            if name not in kwargs:
                raise ValueError(f"Missing field '{name}'")
            values.append(kwargs[name])

        return EastStruct(self, tuple(values))


@dataclass(frozen=True, eq=True)
class VariantType:
    """Runtime representation of a variant type.

    Tracks case names and their types.
    """

    cases: tuple[tuple[str, EastType], ...]

    def case_names(self) -> list[str]:
        """Get list of case names."""
        return [name for name, _ in self.cases]

    def case_types(self) -> list[EastType]:
        """Get list of case types."""
        return [typ for _, typ in self.cases]

    def case_type(self, name: str) -> EastType:
        """Get type of a case by name.

        Args:
            name: Case name

        Returns:
            Type of the case

        Raises:
            KeyError: If case not found
        """
        for case_name, case_type in self.cases:
            if case_name == name:
                return case_type
        raise KeyError(f"No case named '{name}'")

    def create(self, tag: str, value: Any = None) -> EastVariant:
        """Create a variant instance with this case.

        Args:
            tag: Case tag
            value: Case value (defaults to null)

        Returns:
            EastVariant instance

        Raises:
            KeyError: If case not found
        """
        # Validate case exists
        _ = self.case_type(tag)

        if value is None:
            value = null

        return EastVariant(self, Case(tag, value))


# Forward reference type for EastType during definition
class EastType(EastVariant):
    """East type representation.

    This is a recursive variant with cases for all East types.
    """

    _east_type_value: EastType | None = None  # Will be set after bootstrap

    def __init__(self, case: Case):
        """Create an EastType from a case.

        Args:
            case: The case representing this type
        """
        # Don't call super().__init__() to avoid needing _east_type yet
        object.__setattr__(self, "_case", case)

    @property
    def _east_type(self) -> EastType:
        """Get the type of EastType (homoiconic!)."""
        if EastType._east_type_value is None:
            raise RuntimeError("EastType not fully bootstrapped")
        return EastType._east_type_value

    @property
    def tag(self) -> str:
        """Get the type tag.

        Returns type tag like "Array", "Integer", etc.
        """
        return self._case.tag

    @property
    def value(self) -> Any:
        """Get the associated value for this type."""
        return self._case.value

    def __eq__(self, other: object) -> bool:
        """Type equality compares cases, not _east_type.

        This breaks infinite recursion since we don't compare _east_type.
        """
        if not isinstance(other, EastType):
            return NotImplemented
        return self._case == other._case

    def __hash__(self) -> int:
        """Hash based on case only."""
        return hash(self._case)


# Type constructors for primitive types
NullType = EastType(make_case("Null"))
BooleanType = EastType(make_case("Boolean"))
IntegerType = EastType(make_case("Integer"))
FloatType = EastType(make_case("Float"))
StringType = EastType(make_case("String"))
BlobType = EastType(make_case("Blob"))
DateTimeType = EastType(make_case("DateTime"))
NeverType = EastType(make_case("Never"))


def ArrayType(element_type: EastType) -> EastType:
    """Create an array type.

    Args:
        element_type: Type of array elements

    Returns:
        Array type
    """
    return EastType(make_case("Array", element_type))


def SetType(element_type: EastType) -> EastType:
    """Create a set type.

    Args:
        element_type: Type of set elements

    Returns:
        Set type
    """
    return EastType(make_case("Set", element_type))


def DictType(key_type: EastType, value_type: EastType) -> EastType:
    """Create a dict type.

    Args:
        key_type: Type of dict keys
        value_type: Type of dict values

    Returns:
        Dict type
    """
    # Dict type contains a struct with key and value fields
    dict_struct_type = StructType((("key", key_type), ("value", value_type)))
    dict_struct = dict_struct_type.create(key=key_type, value=value_type)
    return EastType(make_case("Dict", dict_struct))


def StructTypeFromFields(fields: list[tuple[str, EastType]]) -> EastType:
    """Create a struct type from field specifications.

    Args:
        fields: List of (name, type) pairs

    Returns:
        Struct type (as EastType)
    """
    # Each field is represented as a struct with name and type
    field_struct_type = StructType((("name", StringType), ("type", EastTypeType)))
    field_structs = [field_struct_type.create(name=name, type=typ) for name, typ in fields]
    return EastType(make_case("Struct", field_structs))


def VariantTypeFromCases(cases: list[tuple[str, EastType]]) -> EastType:
    """Create a variant type from case specifications.

    Args:
        cases: List of (name, type) pairs

    Returns:
        Variant type (as EastType)
    """
    # Sort cases by name (East requires this)
    sorted_cases = sorted(cases, key=lambda x: x[0])

    # Each case is represented as a struct with name and type
    case_struct_type = StructType((("name", StringType), ("type", EastTypeType)))
    case_structs = [case_struct_type.create(name=name, type=typ) for name, typ in sorted_cases]
    return EastType(make_case("Variant", case_structs))


def FunctionType(inputs: list[EastType], output: EastType, platforms: list[str]) -> EastType:
    """Create a function type.

    Args:
        inputs: List of input types
        output: Output type
        platforms: List of platform names

    Returns:
        Function type
    """
    func_struct_type = StructType(
        (
            ("inputs", ArrayType(EastTypeType)),
            ("output", EastTypeType),
            ("platforms", ArrayType(StringType)),
        )
    )
    func_struct = func_struct_type.create(inputs=inputs, output=output, platforms=platforms)
    return EastType(make_case("Function", func_struct))


def RecursiveTypeRef(depth: int) -> EastType:
    """Create a recursive type reference.

    Args:
        depth: Number of levels up to reference

    Returns:
        Recursive type reference
    """
    return EastType(make_case("Recursive", depth))


# The type of EastType itself (homoiconic!)
# This is a recursive variant, so we build it carefully
EastTypeType: EastType = None  # type: ignore  # Will be set below


def recursive_type(builder) -> EastType:  # type: ignore[no-untyped-def]
    """Build a recursive type.

    Args:
        builder: Function that takes a self-reference and returns a type

    Returns:
        The recursive type
    """
    # Create a placeholder reference
    placeholder = RecursiveTypeRef(-1)

    # Build the type using the placeholder
    result = builder(placeholder)

    # Apply recursive depth to resolve the placeholder
    return _apply_recursive_depth(result, 0)


def _apply_recursive_depth(typ: EastType, depth: int) -> EastType:
    """Replace recursive placeholders with proper depth references.

    Args:
        typ: Type to process
        depth: Current depth

    Returns:
        Type with resolved recursive references
    """
    # Handle raw StructType and VariantType objects
    if isinstance(typ, StructType):
        new_fields = [(name, _apply_recursive_depth(field_type, depth)) for name, field_type in typ.fields]
        return StructTypeFromFields(new_fields)

    if isinstance(typ, VariantType):
        new_cases = [(name, _apply_recursive_depth(case_type, depth)) for name, case_type in typ.cases]
        return VariantTypeFromCases(new_cases)

    tag = typ.tag

    if tag == "Recursive":
        n = typ.value
        if n == -1:
            # Placeholder - replace with current depth
            return RecursiveTypeRef(depth)
        if n >= depth:
            raise ValueError("Malformed recursive type")
        return typ

    if tag == "Array":
        element_type = _apply_recursive_depth(typ.value, depth)
        return ArrayType(element_type)

    if tag == "Set":
        element_type = _apply_recursive_depth(typ.value, depth)
        return SetType(element_type)

    if tag == "Dict":
        dict_struct = typ.value
        key_type = _apply_recursive_depth(dict_struct.key, depth)
        value_type = _apply_recursive_depth(dict_struct.value, depth)
        return DictType(key_type, value_type)

    if tag == "Struct":
        fields = typ.value
        new_fields = [(field.name, _apply_recursive_depth(field.type, depth)) for field in fields]
        return StructTypeFromFields(new_fields)

    if tag == "Variant":
        cases = typ.value
        new_cases = [(case.name, _apply_recursive_depth(case.type, depth)) for case in cases]
        return VariantTypeFromCases(new_cases)

    if tag == "Function":
        func = typ.value
        new_inputs = [_apply_recursive_depth(inp, depth) for inp in func.inputs]
        new_output = _apply_recursive_depth(func.output, depth)
        return FunctionType(new_inputs, new_output, func.platforms)

    # Primitive types don't contain recursive references
    return typ


# Bootstrap EastTypeType - the type of all types
# This is the actual definition from the Julia code, translated to Python
EastTypeType = recursive_type(
    lambda self: VariantTypeFromCases(
        [
            ("Array", self),
            ("Blob", NullType),
            ("Boolean", NullType),
            ("DateTime", NullType),
            ("Dict", StructTypeFromFields([("key", self), ("value", self)])),
            ("Float", NullType),
            (
                "Function",
                StructTypeFromFields(
                    [
                        ("inputs", ArrayType(self)),
                        ("output", self),
                        ("platforms", ArrayType(StringType)),
                    ]
                ),
            ),
            ("Integer", NullType),
            ("Never", NullType),
            ("Null", NullType),
            ("Recursive", IntegerType),
            ("Set", self),
            ("String", NullType),
            ("Struct", ArrayType(StructTypeFromFields([("name", StringType), ("type", self)]))),
            ("Variant", ArrayType(StructTypeFromFields([("name", StringType), ("type", self)]))),
        ]
    )
)

# Now set the _east_type for EastType properly
EastType._east_type_value = EastTypeType


def type_of(value: Any) -> EastType:
    """Get the EastType of a value.

    Args:
        value: The value to get the type of

    Returns:
        The EastType representing the type of the value

    Raises:
        TypeError: If the value is not a valid East value
    """
    from datetime import datetime

    from east.types.containers import EastArray, EastDict, EastSet
    from east.types.primitives import Blob, Null
    from east.types.structural import EastStruct, EastVariant

    # Check for None first (convert to Null)
    if value is None or isinstance(value, Null):
        return NullType

    # Boolean (must come before int since bool is subclass of int)
    if isinstance(value, bool):
        return BooleanType

    # Integer
    if isinstance(value, int):
        return IntegerType

    # Float
    if isinstance(value, float):
        return FloatType

    # String
    if isinstance(value, str):
        return StringType

    # Blob
    if isinstance(value, Blob):
        return BlobType

    # DateTime
    if isinstance(value, datetime):
        return DateTimeType

    # EastType itself
    if isinstance(value, EastType):
        return EastTypeType

    # Containers
    if isinstance(value, EastArray):
        return ArrayType(value.element_type)

    if isinstance(value, EastSet):
        return SetType(value.element_type)

    if isinstance(value, EastDict):
        return DictType(value.key_type, value.value_type)

    # Structural types have _east_type
    if isinstance(value, EastStruct | EastVariant):
        return value._east_type  # type: ignore[return-value]

    # Unknown type
    raise TypeError(f"Unknown East type for value: {type(value).__name__}")


__all__ = [
    "StructType",
    "VariantType",
    "EastType",
    "NullType",
    "BooleanType",
    "IntegerType",
    "FloatType",
    "StringType",
    "BlobType",
    "DateTimeType",
    "NeverType",
    "ArrayType",
    "SetType",
    "DictType",
    "StructTypeFromFields",
    "VariantTypeFromCases",
    "FunctionType",
    "RecursiveTypeRef",
    "EastTypeType",
    "recursive_type",
    "type_of",
]
