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


# Type predicate functions (defined early for use in type constructors)
def is_data_type(typ: EastType, recursive_type: EastType | None = None) -> bool:
    """Check if a type is a data type (non-function).

    Data types exclude functions but include all other types.
    Used to validate type parameters that must be serializable.

    Args:
        typ: Type to check
        recursive_type: Internal parameter for cycle detection

    Returns:
        True if the type is a data type, False otherwise
    """
    # Avoid infinite loops in recursive types
    if recursive_type is not None and typ == recursive_type:
        return True

    tag = typ.tag

    if tag == "Function":
        return False
    if tag == "Array":
        # Array constructors check their value type are data types
        return True
    if tag == "Set":
        # Set constructors check their key type, which must be immutable
        return True
    if tag == "Dict":
        # Dict constructors check their value type are data types
        return True
    if tag == "Struct":
        fields = typ.value
        return all(is_data_type(field.type, recursive_type) for field in fields)
    if tag == "Variant":
        cases = typ.value
        return all(is_data_type(case.type, recursive_type) for case in cases)
    if tag == "Recursive":
        # Recursive references are always valid for data type check
        return True
    # Primitive types are data types
    return True


def is_immutable_type(typ: EastType, recursive_type: EastType | None = None) -> bool:
    """Check if a type is immutable.

    Immutable types exclude mutable collections (Array, Set, Dict) and functions.
    Used to validate key types for Set and Dict.

    Args:
        typ: Type to check
        recursive_type: Internal parameter for cycle detection

    Returns:
        True if the type is immutable, False otherwise
    """
    # Avoid infinite loops in recursive types
    if recursive_type is not None and typ == recursive_type:
        return True

    tag = typ.tag

    if tag in ("Array", "Set", "Dict", "Function"):
        return False
    if tag == "Struct":
        fields = typ.value
        return all(is_immutable_type(field.type, recursive_type) for field in fields)
    if tag == "Variant":
        cases = typ.value
        return all(is_immutable_type(case.type, recursive_type) for case in cases)
    if tag == "Recursive":
        # Recursive references are always valid for immutable check
        return True
    # Primitive types are immutable
    return True


def ArrayType(element_type: EastType) -> EastType:
    """Create an array type.

    Args:
        element_type: Type of array elements

    Returns:
        Array type

    Raises:
        TypeError: If element_type is not a data type
    """
    if not is_data_type(element_type):
        from east.serialization.east_printer import print_type

        raise TypeError(
            f"Array value type must be a (non-function) data type, got {print_type(element_type)}"
        )
    return EastType(make_case("Array", element_type))


def SetType(element_type: EastType) -> EastType:
    """Create a set type.

    Args:
        element_type: Type of set elements (must be immutable)

    Returns:
        Set type

    Raises:
        TypeError: If element_type is not immutable
    """
    if not is_immutable_type(element_type):
        from east.serialization.east_printer import print_type

        raise TypeError(f"Set key type must be an immutable type, got {print_type(element_type)}")
    return EastType(make_case("Set", element_type))


def DictType(key_type: EastType, value_type: EastType) -> EastType:
    """Create a dict type.

    Args:
        key_type: Type of dict keys (must be immutable)
        value_type: Type of dict values (must be data type)

    Returns:
        Dict type

    Raises:
        TypeError: If key_type is not immutable or value_type is not a data type
    """
    if not is_immutable_type(key_type):
        from east.serialization.east_printer import print_type

        raise TypeError(f"Dict key type must be an immutable type, got {print_type(key_type)}")
    if not is_data_type(value_type):
        from east.serialization.east_printer import print_type

        raise TypeError(
            f"Dict value type must be a (non-function) data type, got {print_type(value_type)}"
        )
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

    Raises:
        TypeError: If any field type is not a data type
    """
    # Validate all field types are data types
    for name, typ in fields:
        if not is_data_type(typ):
            from east.serialization.east_printer import print_type

            raise TypeError(
                f"Struct field {name} must be a (non-function) data type, got {print_type(typ)}"
            )

    # Each field is represented as a struct with name and type
    field_struct_type = StructType((("name", StringType), ("type", EastTypeType)))
    field_structs = [field_struct_type.create(name=name, type=typ) for name, typ in fields]
    return EastType(make_case("Struct", field_structs))


def VariantTypeFromCases(cases: list[tuple[str, EastType]]) -> EastType:
    """Create a variant type from case specifications.

    Args:
        cases: List of (name, type) pairs (will be sorted alphabetically)

    Returns:
        Variant type (as EastType)

    Raises:
        TypeError: If any case type is not a data type
    """
    # Validate all case types are data types
    for name, typ in cases:
        if not is_data_type(typ):
            from east.serialization.east_printer import print_type

            raise TypeError(
                f"Variant case {name} must be a (non-function) data type, got {print_type(typ)}"
            )

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
        new_fields = [
            (name, _apply_recursive_depth(field_type, depth)) for name, field_type in typ.fields
        ]
        return StructTypeFromFields(new_fields)

    if isinstance(typ, VariantType):
        new_cases = [
            (name, _apply_recursive_depth(case_type, depth)) for name, case_type in typ.cases
        ]
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


# Helper functions for Option pattern
def SomeType(value_type: EastType) -> EastType:
    """Create a Some type wrapping a value type.

    Args:
        value_type: Type of the wrapped value

    Returns:
        Variant type with 'some' case
    """
    return VariantTypeFromCases([("some", value_type)])


def OptionType(value_type: EastType) -> EastType:
    """Create an Option type for optional values.

    Args:
        value_type: Type of the value when present

    Returns:
        Variant type with 'none' and 'some' cases
    """
    return VariantTypeFromCases([("none", NullType), ("some", value_type)])


def is_type_equal(
    t1: EastType, t2: EastType, r1: EastType | None = None, r2: EastType | None = None
) -> bool:
    """Check if two East types are structurally equal.

    Args:
        t1: First type to compare
        t2: Second type to compare
        r1: Internal parameter for tracking the first recursive type
        r2: Internal parameter for tracking the second recursive type

    Returns:
        True if the types are equal, False otherwise
    """
    # Initialize recursive type tracking on first call
    if r1 is None:
        r1 = t1
    if r2 is None:
        r2 = t2

    # Handle recursive type references
    if t1.tag == "Recursive":
        if t2.tag == "Recursive":
            if t1.value == r1.value if hasattr(r1, "value") and hasattr(t1, "value") else t1 == r1:
                return (
                    t2.value == r2.value
                    if hasattr(r2, "value") and hasattr(t2, "value")
                    else t2 == r2
                )
            if t2.value == r2.value if hasattr(r2, "value") and hasattr(t2, "value") else t2 == r2:
                return False
            # Both are new recursive types, compare their depths
            return t1.value == t2.value
        # Recursive type is transparent
        return t1.value == t2 if isinstance(t1.value, int) else is_type_equal(t1, t2, r1, r2)
    if t2.tag == "Recursive":
        # Recursive type is transparent
        return t2.value == t1 if isinstance(t2.value, int) else is_type_equal(t1, t2, r1, r2)

    # Primitive types
    if t1.tag != t2.tag:
        return False

    tag = t1.tag

    if tag in ("Never", "Null", "Boolean", "Integer", "Float", "String", "DateTime", "Blob"):
        return True

    # Container types
    if tag == "Array":
        return is_type_equal(t1.value, t2.value, r1, r2)
    if tag == "Set":
        return is_type_equal(t1.value, t2.value, r1, r2)
    if tag == "Dict":
        dict1 = t1.value
        dict2 = t2.value
        return is_type_equal(dict1.key, dict2.key, r1, r2) and is_type_equal(
            dict1.value, dict2.value, r1, r2
        )

    # Structural types
    if tag == "Struct":
        fields1 = t1.value
        fields2 = t2.value
        if len(fields1) != len(fields2):
            return False
        for field1, field2 in zip(fields1, fields2, strict=False):
            if field1.name != field2.name:
                return False
            if not is_type_equal(field1.type, field2.type, r1, r2):
                return False
        return True

    if tag == "Variant":
        cases1 = t1.value
        cases2 = t2.value
        if len(cases1) != len(cases2):
            return False
        for case1, case2 in zip(cases1, cases2, strict=False):
            if case1.name != case2.name:
                return False
            if not is_type_equal(case1.type, case2.type, r1, r2):
                return False
        return True

    # Function types
    if tag == "Function":
        func1 = t1.value
        func2 = t2.value
        # Check input count
        if len(func1.inputs) != len(func2.inputs):
            return False
        # Check each input type
        for inp1, inp2 in zip(func1.inputs, func2.inputs, strict=False):
            if not is_type_equal(inp1, inp2, r1, r2):
                return False
        # Check output type
        if not is_type_equal(func1.output, func2.output, r1, r2):
            return False
        # Check platforms
        if func1.platforms is None and func2.platforms is None:
            return True
        if func1.platforms is None or func2.platforms is None:
            return False
        # Compare platform lists (assume already sorted)
        return list(func1.platforms) == list(func2.platforms)

    # Unknown type
    raise TypeError(f"Unknown type encountered during type equality check: {tag}")


def is_value_of(
    value: Any,
    typ: EastType,
    node_type: EastType | None = None,
    nodes_visited: set[int] | None = None,
) -> bool:
    """Check if a JavaScript/Python value conforms to an East type.

    Args:
        value: The value to check
        typ: The East type to validate against
        node_type: Internal parameter for tracking the current recursive type node
        nodes_visited: Internal parameter for tracking visited nodes to detect cycles

    Returns:
        True if the value matches the type, False otherwise
    """
    from datetime import datetime

    from east.types.containers import EastArray, EastDict, EastSet
    from east.types.primitives import Blob, Null
    from east.types.structural import EastStruct, EastVariant

    tag = typ.tag

    if tag == "Never":
        return False
    if tag == "Null":
        return value is None or isinstance(value, Null)
    if tag == "Boolean":
        return isinstance(value, bool)
    if tag == "Integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if tag == "Float":
        return isinstance(value, float)
    if tag == "String":
        return isinstance(value, str)
    if tag == "DateTime":
        return isinstance(value, datetime)
    if tag == "Blob":
        return isinstance(value, (bytes, Blob))

    if tag == "Array":
        if not isinstance(value, (list, EastArray)):
            return False
        element_type = typ.value
        for x in value:
            if not is_value_of(x, element_type, node_type, nodes_visited):
                return False
        return True

    if tag == "Set":
        if not isinstance(value, (set, EastSet)):
            return False
        element_type = typ.value
        for x in value:
            if not is_value_of(x, element_type, node_type, nodes_visited):
                return False
        return True

    if tag == "Dict":
        if not isinstance(value, (dict, EastDict)):
            return False
        dict_struct = typ.value
        key_type = dict_struct.key
        value_type = dict_struct.value
        items = value.items() if isinstance(value, (dict, EastDict)) else []
        for k, v in items:
            if not is_value_of(k, key_type, node_type, nodes_visited):
                return False
            if not is_value_of(v, value_type, node_type, nodes_visited):
                return False
        return True

    if tag == "Struct":
        if not isinstance(value, EastStruct):
            return False
        fields = typ.value
        if len(value._values) != len(fields):
            return False
        for i, field in enumerate(fields):
            if not is_value_of(value._values[i], field.type, node_type, nodes_visited):
                return False
        return True

    if tag == "Variant":
        if not isinstance(value, EastVariant):
            return False
        cases = typ.value
        # Find the matching case
        for case in cases:
            if case.name == value.tag:
                return is_value_of(value.value, case.type, node_type, nodes_visited)
        return False

    if tag == "Recursive":
        # Handle recursive type references with cycle detection
        depth = typ.value
        if node_type == depth if isinstance(depth, int) else False:
            # We're checking against the same recursive type
            if nodes_visited is None:
                nodes_visited = set()
            value_id = id(value)
            if value_id in nodes_visited:
                return True  # Already seen this object
            nodes_visited.add(value_id)
            # Continue checking with the node type
            # Note: this is simplified - full recursive type checking would need the actual node
            return True
        # New recursive type - simplified handling
        return True

    if tag == "Function":
        raise TypeError("JavaScript/Python functions cannot be converted to East functions")

    raise TypeError(f"Unknown type encountered during value type check: {tag}")


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
    "is_data_type",
    "is_immutable_type",
    "is_type_equal",
    "is_value_of",
    "SomeType",
    "OptionType",
]
