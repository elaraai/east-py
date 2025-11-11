"""East type system.

EastType is a recursive variant representing all East types.
It's homoiconic - the type of types is itself an East value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from east.types.primitives import null
from east.types.structural import Case, EastStruct, EastVariant, make_case


class TypeMismatchError(TypeError):
    """Exception raised when types cannot be unified or intersected."""

    pass


@dataclass(frozen=True, eq=True)
class _StructTypeClass:
    """Internal runtime representation of a struct type.

    Tracks field names and their types.
    This is an internal class - use StructType() function to create struct types.
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
class _VariantTypeClass:
    """Internal runtime representation of a variant type.

    Tracks case names and their types.
    This is an internal class - use VariantType() function to create variant types.
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
    if tag == "Ref":
        # Refs are data types (serializable)
        # Constructor already validates inner type is data type
        return True
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

    if tag in ("Array", "Set", "Dict", "Ref", "Function"):
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
    dict_struct_type = _StructTypeClass((("key", key_type), ("value", value_type)))
    dict_struct = dict_struct_type.create(key=key_type, value=value_type)
    return EastType(make_case("Dict", dict_struct))


def RefType(value_type: EastType) -> EastType:
    """Create a ref type.

    Args:
        value_type: Type of the referenced value (must be data type)

    Returns:
        Ref type

    Raises:
        TypeError: If value_type is not a data type

    Examples:
        >>> RefType(IntegerType)  # ref<Integer>
        >>> RefType(ArrayType(StringType))  # ref<Array<String>>
    """
    if not is_data_type(value_type):
        from east.serialization.east_printer import print_type

        raise TypeError(
            f"Ref value type must be a (non-function) data type, got {print_type(value_type)}"
        )
    return EastType(make_case("Ref", value_type))


def StructType(fields: list[tuple[str, EastType]]) -> EastType:
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
    field_struct_type = _StructTypeClass((("name", StringType), ("type", EastTypeType)))
    field_structs = [field_struct_type.create(name=name, type=typ) for name, typ in fields]
    return EastType(make_case("Struct", field_structs))


def VariantType(cases: list[tuple[str, EastType]]) -> EastType:
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
    case_struct_type = _StructTypeClass((("name", StringType), ("type", EastTypeType)))
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
    func_struct_type = _StructTypeClass(
        (
            ("inputs", ArrayType(EastTypeType)),
            ("output", EastTypeType),
            ("platforms", ArrayType(StringType)),
        )
    )
    func_struct = func_struct_type.create(inputs=inputs, output=output, platforms=platforms)
    return EastType(make_case("Function", func_struct))


def RecursiveTypeRef(marker: RecursiveTypeMarker) -> EastType:
    """Create a recursive type reference.

    Args:
        marker: The RecursiveTypeMarker object for this recursive scope

    Returns:
        Recursive type reference
    """
    return EastType(make_case("Recursive", marker))


# The type of EastType itself (homoiconic!)
# This is a recursive variant, so we build it carefully
EastTypeType: EastType = None  # type: ignore  # Will be set below

# Counter for generating unique recursive type IDs (kept for backwards compatibility with existing code)
_recursive_type_counter = 0


class RecursiveTypeMarker:
    """Marker object for recursive type references.

    This matches TypeScript's RecursiveTypeMarker approach where the marker
    is a unique object identity that can be used as a dictionary key.
    """

    def __init__(self):
        self.node = None  # Will be set to the resolved type

    def __repr__(self):
        return f"<RecursiveMarker at {hex(id(self))}>"


def recursive_type(builder) -> EastType:  # type: ignore[no-untyped-def]
    """Build a recursive type using a marker/node approach (matches TypeScript).

    Args:
        builder: Function that takes a marker and returns a type

    Returns:
        The recursive type
    """
    # Create a unique marker for this recursive scope
    marker = RecursiveTypeMarker()

    # Create a placeholder reference that points to the marker
    placeholder = RecursiveTypeRef(marker)

    # Build the type using the placeholder
    result = builder(placeholder)

    # Store the result in the marker's node field
    marker.node = result

    return result


def _apply_recursive_scope_id(typ: EastType, scope_id: int) -> EastType:
    """DEPRECATED: This function is no longer used with the marker-based approach.

    Kept for backwards compatibility but should not be called.

    Args:
        typ: Type to process
        scope_id: The unique scope ID for this recursive type

    Returns:
        Type with resolved recursive references
    """
    # Handle raw _StructTypeClass and _VariantTypeClass objects
    if isinstance(typ, _StructTypeClass):
        new_fields = [
            (name, _apply_recursive_scope_id(field_type, scope_id))
            for name, field_type in typ.fields
        ]
        return StructType(new_fields)

    if isinstance(typ, _VariantTypeClass):
        new_cases = [
            (name, _apply_recursive_scope_id(case_type, scope_id)) for name, case_type in typ.cases
        ]
        return VariantType(new_cases)

    tag = typ.tag

    if tag == "Recursive":
        n = typ.value
        if n == -1:
            # Placeholder - replace with scope ID
            return RecursiveTypeRef(scope_id)
        # Already resolved (from a nested independent recursive type) - leave as-is
        return typ

    if tag == "Array":
        element_type = _apply_recursive_scope_id(typ.value, scope_id)
        return ArrayType(element_type)

    if tag == "Set":
        element_type = _apply_recursive_scope_id(typ.value, scope_id)
        return SetType(element_type)

    if tag == "Dict":
        dict_struct = typ.value
        key_type = _apply_recursive_scope_id(dict_struct.key, scope_id)
        value_type = _apply_recursive_scope_id(dict_struct.value, scope_id)
        return DictType(key_type, value_type)

    if tag == "Struct":
        fields = typ.value
        new_fields = [
            (field.name, _apply_recursive_scope_id(field.type, scope_id)) for field in fields
        ]
        return StructType(new_fields)

    if tag == "Variant":
        cases = typ.value
        new_cases = [(case.name, _apply_recursive_scope_id(case.type, scope_id)) for case in cases]
        return VariantType(new_cases)

    if tag == "Function":
        func = typ.value
        new_inputs = [_apply_recursive_scope_id(inp, scope_id) for inp in func.inputs]
        new_output = _apply_recursive_scope_id(func.output, scope_id)
        return FunctionType(new_inputs, new_output, func.platforms)

    # Primitive types don't contain recursive references
    return typ


# Bootstrap EastTypeType - the type of all types
# This is the actual definition from the Julia code, translated to Python
EastTypeType = recursive_type(
    lambda self: VariantType(
        [
            ("Array", self),
            ("Blob", NullType),
            ("Boolean", NullType),
            ("DateTime", NullType),
            ("Dict", StructType([("key", self), ("value", self)])),
            ("Float", NullType),
            (
                "Function",
                StructType(
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
            ("Ref", self),
            ("Set", self),
            ("String", NullType),
            ("Struct", ArrayType(StructType([("name", StringType), ("type", self)]))),
            ("Variant", ArrayType(StructType([("name", StringType), ("type", self)]))),
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
    return VariantType([("some", value_type)])


def OptionType(value_type: EastType) -> EastType:
    """Create an Option type for optional values.

    Args:
        value_type: Type of the value when present

    Returns:
        Variant type with 'none' and 'some' cases
    """
    return VariantType([("none", NullType), ("some", value_type)])


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


def is_subtype(t1: EastType, t2: EastType) -> bool:
    """Check if one East type is a subtype of another.

    Implements East's subtyping rules:
    - Never is a subtype of all types
    - Mutable collections (Array, Set, Dict) are invariant
    - Variant supports width subtyping (more cases → fewer cases)
    - Function uses contravariant inputs and covariant outputs
    - Struct is covariant in all fields
    - Recursive types are invariant

    Args:
        t1: The potential subtype
        t2: The potential supertype

    Returns:
        True if t1 is a subtype of t2, False otherwise
    """
    # Handle recursive types
    if t1.tag == "Recursive":
        if t2.tag == "Recursive":
            # Recursive types are invariant - must have exact same layout
            return is_type_equal(t1, t2)
        # Recursive type wrapper is transparent but invariant
        return is_type_equal(t1, t2)

    if t2.tag == "Recursive":
        # Head covariance: unfold recursive type once
        # Note: simplified - full implementation would track the node
        return is_subtype(t1, t2)

    # Never is subtype of everything
    if t1.tag == "Never":
        return True

    # Primitives are only subtypes of themselves
    if t1.tag in ("Null", "Boolean", "Integer", "Float", "String", "DateTime", "Blob"):
        return t1.tag == t2.tag

    # Ref type - invariant (mutable)
    if t1.tag == "Ref":
        if t2.tag == "Ref":
            return is_type_equal(t1.value, t2.value)
        return False

    # Mutable collections are invariant (must be exactly equal)
    if t1.tag == "Array":
        if t2.tag == "Array":
            return is_type_equal(t1.value, t2.value)
        return False

    if t1.tag == "Set":
        if t2.tag == "Set":
            return is_type_equal(t1.value, t2.value)
        return False

    if t1.tag == "Dict":
        if t2.tag == "Dict":
            dict1 = t1.value
            dict2 = t2.value
            return is_type_equal(dict1.key, dict2.key) and is_type_equal(dict1.value, dict2.value)
        return False

    # Struct is covariant in fields
    if t1.tag == "Struct":
        if t2.tag == "Struct":
            fields1 = t1.value
            fields2 = t2.value
            if len(fields1) != len(fields2):
                return False
            for field1, field2 in zip(fields1, fields2, strict=False):
                if field1.name != field2.name:
                    return False
                if not is_subtype(field1.type, field2.type):
                    return False
            return True
        return False

    # Variant supports width subtyping (t1 can have more cases)
    if t1.tag == "Variant":
        if t2.tag == "Variant":
            cases1 = t1.value
            cases2 = t2.value
            # Build dict of t2 cases for lookup
            cases2_dict = {case.name: case.type for case in cases2}
            # All cases in t1 must be subtypes of corresponding cases in t2
            for case1 in cases1:
                case2_type = cases2_dict.get(case1.name, NeverType)
                if not is_subtype(case1.type, case2_type):
                    return False
            return True
        return False

    # Function is contravariant in inputs, covariant in output
    if t1.tag == "Function":
        if t2.tag == "Function":
            func1 = t1.value
            func2 = t2.value
            if len(func1.inputs) != len(func2.inputs):
                return False
            # Contravariant inputs: t2's inputs must be subtypes of t1's inputs
            for inp1, inp2 in zip(func1.inputs, func2.inputs, strict=False):
                if not is_subtype(inp2, inp1):
                    return False
            # Covariant output
            return is_subtype(func1.output, func2.output)
        return False

    raise TypeError(f"Unknown type encountered during subtype check: {t1.tag}")


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
        return isinstance(value, bytes | Blob)

    if tag == "Ref":
        from east.types.ref import Ref

        if not isinstance(value, Ref):
            return False
        value_type = typ.value
        return is_value_of(value.value, value_type, node_type, nodes_visited)

    if tag == "Array":
        if not isinstance(value, list | EastArray):
            return False
        element_type = typ.value
        return all(is_value_of(x, element_type, node_type, nodes_visited) for x in value)

    if tag == "Set":
        if not isinstance(value, set | EastSet):
            return False
        element_type = typ.value
        return all(is_value_of(x, element_type, node_type, nodes_visited) for x in value)

    if tag == "Dict":
        if not isinstance(value, dict | EastDict):
            return False
        dict_struct = typ.value
        key_type = dict_struct.key
        value_type = dict_struct.value
        items = value.items() if isinstance(value, dict | EastDict) else []
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


def type_union(t1: EastType, t2: EastType) -> EastType:
    """Compute the union of two East types at runtime.

    Args:
        t1: First type
        t2: Second type

    Returns:
        The union type

    Raises:
        TypeMismatchError: When the types cannot be unified

    Remarks:
        - Never is the identity for union
        - Same primitives union to themselves
        - Array/Set/Dict require matching inner types (invariant)
        - Struct requires same field count and names, unions field types
        - Variant merges cases (union all cases from both)
        - Function requires matching signatures, unions platforms
    """
    from east.serialization.east_printer import print_type

    try:
        if t1.tag == "Never":
            return t2
        if t2.tag == "Never":
            return t1
        if t1.tag == "Recursive":
            if t2.tag == "Recursive":
                # Both recursive - require exact match (heap invariance)
                return type_equal(t1, t2)
            # Rec(A) ∪ NonRec: If NonRec <: A, union is Rec(A)
            # Note: simplified - TypeScript has TODO about recursive type handling
            if is_subtype(t2, t1):
                return t1
            raise TypeMismatchError(
                f"Cannot union {print_type(t1)} with {print_type(t2)}: incompatible types"
            )
        if t2.tag == "Recursive":
            # NonRec ∪ Rec(B): If NonRec <: B, union is Rec(B)
            if is_subtype(t1, t2):
                return t2
            raise TypeMismatchError(
                f"Cannot union {print_type(t1)} with {print_type(t2)}: incompatible types"
            )
        if t1.tag == "Ref":
            if t2.tag == "Ref":
                return RefType(type_equal(t1.value, t2.value))
            raise TypeMismatchError(
                f"Cannot union {print_type(t1)} with {print_type(t2)}: incompatible types"
            )
        if t1.tag == "Array":
            if t2.tag == "Array":
                return ArrayType(type_equal(t1.value, t2.value))
            raise TypeMismatchError(
                f"Cannot union {print_type(t1)} with {print_type(t2)}: incompatible types"
            )
        if t1.tag == "Set":
            if t2.tag == "Set":
                return SetType(type_equal(t1.value, t2.value))
            raise TypeMismatchError(
                f"Cannot union {print_type(t1)} with {print_type(t2)}: incompatible types"
            )
        if t1.tag == "Dict":
            if t2.tag == "Dict":
                dict1 = t1.value
                dict2 = t2.value
                return DictType(
                    type_equal(dict1.key, dict2.key), type_equal(dict1.value, dict2.value)
                )
            raise TypeMismatchError(
                f"Cannot union {print_type(t1)} with {print_type(t2)}: incompatible types"
            )
        if t1.tag == "Struct":
            if t2.tag == "Struct":
                fields1 = t1.value
                fields2 = t2.value
                if len(fields1) != len(fields2):
                    raise TypeMismatchError(
                        f"Cannot union {print_type(t1)} with {print_type(t2)}: structs contain different number of fields"
                    )
                new_fields = []
                for i, (field1, field2) in enumerate(zip(fields1, fields2, strict=False)):
                    if field1.name != field2.name:
                        raise TypeMismatchError(
                            f"Cannot union {print_type(t1)} with {print_type(t2)}: struct field {i} has mismatched names {field1.name} and {field2.name}"
                        )
                    new_fields.append((field1.name, type_union(field1.type, field2.type)))
                return StructType(new_fields)
            raise TypeMismatchError(
                f"Cannot union {print_type(t1)} with {print_type(t2)}: incompatible types"
            )
        if t1.tag == "Variant":
            if t2.tag == "Variant":
                cases1 = t1.value
                cases2 = t2.value
                # Build dict of all cases
                cases_dict: dict[str, EastType] = {}
                # Add all cases from t1
                for case1 in cases1:
                    cases_dict[case1.name] = case1.type
                # Merge/add cases from t2
                for case2 in cases2:
                    if case2.name in cases_dict:
                        # Union the types
                        cases_dict[case2.name] = type_union(cases_dict[case2.name], case2.type)
                    else:
                        cases_dict[case2.name] = case2.type
                # Convert back to list
                return VariantType(list(cases_dict.items()))
            raise TypeMismatchError(
                f"Cannot union {print_type(t1)} with {print_type(t2)}: incompatible types"
            )
        if t1.tag == "Function":
            if t2.tag == "Function":
                func1 = t1.value
                func2 = t2.value
                if len(func1.inputs) != len(func2.inputs):
                    raise TypeMismatchError(
                        f"Cannot union {print_type(t1)} with {print_type(t2)}: functions take different number of arguments"
                    )
                # Union platforms
                platforms: list[str] | None
                if func1.platforms is None or func2.platforms is None:
                    platforms = None
                else:
                    # Union and sort platforms
                    platforms = sorted(set(func1.platforms) | set(func2.platforms))
                # Contravariant inputs (intersect), covariant output (union)
                new_inputs = [
                    type_intersect(inp1, inp2)
                    for inp1, inp2 in zip(func1.inputs, func2.inputs, strict=False)
                ]
                new_output = type_union(func1.output, func2.output)
                return FunctionType(new_inputs, new_output, platforms or [])
            raise TypeMismatchError(
                f"Cannot union {print_type(t1)} with {print_type(t2)}: incompatible types"
            )
        # Primitives
        if t1.tag == t2.tag:
            return t1
        raise TypeMismatchError(
            f"Cannot union {print_type(t1)} with {print_type(t2)}: incompatible types"
        )
    except TypeMismatchError:
        raise  # Don't wrap our own errors
    except Exception as cause:
        raise TypeMismatchError(f"Cannot union {print_type(t1)} with {print_type(t2)}") from cause


def type_intersect(t1: EastType, t2: EastType) -> EastType:
    """Compute the intersection of two East types at runtime.

    Args:
        t1: First type
        t2: Second type

    Returns:
        The intersection type

    Raises:
        TypeMismatchError: When the types cannot be intersected

    Remarks:
        - Never is absorbing for intersection
        - Same primitives intersect to themselves
        - Variant keeps only overlapping cases
        - For empty variant intersection, raises error
    """
    from east.serialization.east_printer import print_type

    try:
        if t1.tag == "Never":
            return NeverType
        if t2.tag == "Never":
            return NeverType
        if t1.tag == "Recursive":
            if t2.tag == "Recursive":
                # Both recursive - require exact match (heap invariance)
                # TypeScript creates a new RecursiveType with TypeEqual check
                # Simplified: just ensure they're equal
                type_equal(t1, t2)
                return t1
            # Rec(A) ∩ NonRec: If NonRec <: A, intersection is NonRec
            if is_subtype(t2, t1):
                return t2
            raise TypeMismatchError(
                f"Cannot intersect {print_type(t1)} with {print_type(t2)}: incompatible types"
            )
        if t2.tag == "Recursive":
            # NonRec ∩ Rec(B): If NonRec <: B, intersection is NonRec
            if is_subtype(t1, t2):
                return t1
            raise TypeMismatchError(
                f"Cannot intersect {print_type(t1)} with {print_type(t2)}: incompatible types"
            )
        if t1.tag == "Ref":
            if t2.tag == "Ref":
                return RefType(type_equal(t1.value, t2.value))
            raise TypeMismatchError(
                f"Cannot intersect {print_type(t1)} with {print_type(t2)}: incompatible types"
            )
        if t1.tag == "Array":
            if t2.tag == "Array":
                return ArrayType(type_equal(t1.value, t2.value))
            raise TypeMismatchError(
                f"Cannot intersect {print_type(t1)} with {print_type(t2)}: incompatible types"
            )
        if t1.tag == "Set":
            if t2.tag == "Set":
                return SetType(type_equal(t1.value, t2.value))
            raise TypeMismatchError(
                f"Cannot intersect {print_type(t1)} with {print_type(t2)}: incompatible types"
            )
        if t1.tag == "Dict":
            if t2.tag == "Dict":
                dict1 = t1.value
                dict2 = t2.value
                return DictType(
                    type_equal(dict1.key, dict2.key), type_equal(dict1.value, dict2.value)
                )
            raise TypeMismatchError(
                f"Cannot intersect {print_type(t1)} with {print_type(t2)}: incompatible types"
            )
        if t1.tag == "Struct":
            if t2.tag == "Struct":
                fields1 = t1.value
                fields2 = t2.value
                if len(fields1) != len(fields2):
                    raise TypeMismatchError(
                        f"Cannot intersect {print_type(t1)} with {print_type(t2)}: structs contain different number of fields"
                    )
                new_fields = []
                for i, (field1, field2) in enumerate(zip(fields1, fields2, strict=False)):
                    if field1.name != field2.name:
                        raise TypeMismatchError(
                            f"Cannot intersect {print_type(t1)} with {print_type(t2)}: struct field {i} has mismatched names {field1.name} and {field2.name}"
                        )
                    new_fields.append((field1.name, type_intersect(field1.type, field2.type)))
                return StructType(new_fields)
            raise TypeMismatchError(
                f"Cannot intersect {print_type(t1)} with {print_type(t2)}: incompatible types"
            )
        if t1.tag == "Variant":
            if t2.tag == "Variant":
                cases1 = t1.value
                cases2 = t2.value
                # Build dict of t2 cases for lookup
                cases2_dict = {case.name: case.type for case in cases2}
                # Keep only overlapping cases, intersect their types
                new_cases = []
                for case1 in cases1:
                    if case1.name in cases2_dict:
                        new_cases.append(
                            (case1.name, type_intersect(case1.type, cases2_dict[case1.name]))
                        )
                if not new_cases:
                    raise TypeMismatchError(
                        f"Cannot intersect {print_type(t1)} with {print_type(t2)}: variants have no overlapping cases"
                    )
                return VariantType(new_cases)
            raise TypeMismatchError(
                f"Cannot intersect {print_type(t1)} with {print_type(t2)}: incompatible types"
            )
        if t1.tag == "Function":
            if t2.tag == "Function":
                func1 = t1.value
                func2 = t2.value
                if len(func1.inputs) != len(func2.inputs):
                    raise TypeMismatchError(
                        f"Cannot intersect {print_type(t1)} with {print_type(t2)}: functions take different number of arguments"
                    )
                # Intersect platforms
                platforms: list[str] | None
                if func1.platforms is None:
                    platforms = func2.platforms
                elif func2.platforms is None:
                    platforms = func1.platforms
                else:
                    # Intersect platforms
                    platforms = sorted(set(func1.platforms) & set(func2.platforms))
                # Contravariant inputs (union), covariant output (intersect)
                new_inputs = [
                    type_union(inp1, inp2)
                    for inp1, inp2 in zip(func1.inputs, func2.inputs, strict=False)
                ]
                new_output = type_intersect(func1.output, func2.output)
                return FunctionType(new_inputs, new_output, platforms or [])
            raise TypeMismatchError(
                f"Cannot intersect {print_type(t1)} with {print_type(t2)}: incompatible types"
            )
        # Primitives
        if t1.tag == t2.tag:
            return t1
        raise TypeMismatchError(
            f"Cannot intersect {print_type(t1)} with {print_type(t2)}: incompatible types"
        )
    except TypeMismatchError:
        raise  # Don't wrap our own errors
    except Exception as cause:
        raise TypeMismatchError(
            f"Cannot intersect {print_type(t1)} with {print_type(t2)}"
        ) from cause


def type_equal(
    t1: EastType, t2: EastType, r1: EastType | None = None, r2: EastType | None = None
) -> EastType:
    """Assert that two East types are equal, returning the first type.

    Unlike is_type_equal which returns bool, this throws TypeMismatchError on inequality.

    Args:
        t1: First type
        t2: Second type
        r1: Internal parameter for tracking the first recursive type
        r2: Internal parameter for tracking the second recursive type

    Returns:
        The first type if types are equal

    Raises:
        TypeMismatchError: When the types are not equal

    Remarks:
        Used to enforce type constraints in compound type constructors.
    """
    from east.serialization.east_printer import print_type

    if r1 is None:
        r1 = t1
    if r2 is None:
        r2 = t2

    try:
        if t1.tag == "Array":
            if t2.tag == "Array":
                return ArrayType(type_equal(t1.value, t2.value, r1, r2))
            raise TypeMismatchError(
                f"{print_type(t1)} is not equal to {print_type(t2)}: incompatible types"
            )
        if t1.tag == "Set":
            if t2.tag == "Set":
                return SetType(type_equal(t1.value, t2.value, r1, r2))
            raise TypeMismatchError(
                f"{print_type(t1)} is not equal to {print_type(t2)}: incompatible types"
            )
        if t1.tag == "Dict":
            if t2.tag == "Dict":
                dict1 = t1.value
                dict2 = t2.value
                return DictType(
                    type_equal(dict1.key, dict2.key, r1, r2),
                    type_equal(dict1.value, dict2.value, r1, r2),
                )
            raise TypeMismatchError(
                f"{print_type(t1)} is not equal to {print_type(t2)}: incompatible types"
            )
        if t1.tag == "Ref":
            if t2.tag == "Ref":
                return RefType(type_equal(t1.value, t2.value, r1, r2))
            raise TypeMismatchError(
                f"{print_type(t1)} is not equal to {print_type(t2)}: incompatible types"
            )
        if t1.tag == "Struct":
            if t2.tag == "Struct":
                fields1 = t1.value
                fields2 = t2.value
                if len(fields1) != len(fields2):
                    raise TypeMismatchError(
                        f"{print_type(t1)} is not equal to {print_type(t2)}: structs contain different number of fields"
                    )
                new_fields = []
                for i, (field1, field2) in enumerate(zip(fields1, fields2, strict=False)):
                    if field1.name != field2.name:
                        raise TypeMismatchError(
                            f"{print_type(t1)} is not equal to {print_type(t2)}: struct field {i} has mismatched names {field1.name} and {field2.name}"
                        )
                    new_fields.append((field1.name, type_equal(field1.type, field2.type, r1, r2)))
                return StructType(new_fields)
            raise TypeMismatchError(
                f"{print_type(t1)} is not equal to {print_type(t2)}: incompatible types"
            )
        if t1.tag == "Variant":
            if t2.tag == "Variant":
                cases1 = t1.value
                cases2 = t2.value
                if len(cases1) != len(cases2):
                    raise TypeMismatchError(
                        f"{print_type(t1)} is not equal to {print_type(t2)}: variants contain different number of cases"
                    )
                new_cases = []
                for case1, case2 in zip(cases1, cases2, strict=False):
                    if case1.name != case2.name:
                        # More specific error based on ordering
                        if case1.name < case2.name:
                            raise TypeMismatchError(
                                f"{print_type(t1)} is not equal to {print_type(t2)}: variant case {case1.name} is not present in both variants"
                            )
                        raise TypeMismatchError(
                            f"{print_type(t1)} is not equal to {print_type(t2)}: variant case {case2.name} is not present in both variants"
                        )
                    new_cases.append((case1.name, type_equal(case1.type, case2.type, r1, r2)))
                return VariantType(new_cases)
            raise TypeMismatchError(
                f"{print_type(t1)} is not equal to {print_type(t2)}: incompatible types"
            )
        if t1.tag == "Recursive":
            if t2.tag == "Recursive":
                # Check if both are references to the same recursive type
                # Simplified - TypeScript checks t1.node === r1
                if t1.value == getattr(r1, "value", None):
                    if t2.value == getattr(r2, "value", None):
                        return t1  # Both are references to the same recursive type
                    raise TypeMismatchError(
                        f"{print_type(t1)} is not equal to {print_type(t2)}: recursive types do not match"
                    )
                if t2.value == getattr(r2, "value", None):
                    raise TypeMismatchError(
                        f"{print_type(t1)} is not equal to {print_type(t2)}: recursive types do not match"
                    )
                # This is the root of a new recursive type - assert the depths are equal
                if t1.value != t2.value:
                    raise TypeMismatchError(
                        f"{print_type(t1)} is not equal to {print_type(t2)}: recursive types do not match"
                    )
                return t1
            raise TypeMismatchError(
                f"{print_type(t1)} is not equal to {print_type(t2)}: incompatible types"
            )
        if t1.tag == "Function":
            if t2.tag == "Function":
                func1 = t1.value
                func2 = t2.value
                if len(func1.inputs) != len(func2.inputs):
                    raise TypeMismatchError(
                        f"{print_type(t1)} is not equal to {print_type(t2)}: functions take different number of arguments"
                    )
                # Check platforms
                if func1.platforms is None:
                    if func2.platforms is not None:
                        raise TypeMismatchError(
                            f"{print_type(t1)} is not equal to {print_type(t2)}: functions have different platform effects"
                        )
                elif (
                    func2.platforms is None
                    or len(func1.platforms) != len(func2.platforms)
                    or not all(
                        p1 == p2 for p1, p2 in zip(func1.platforms, func2.platforms, strict=False)
                    )
                ):
                    raise TypeMismatchError(
                        f"{print_type(t1)} is not equal to {print_type(t2)}: functions have different platform effects"
                    )
                # Recursively check inputs and output
                new_inputs = [
                    type_equal(inp1, inp2, r1, r2)
                    for inp1, inp2 in zip(func1.inputs, func2.inputs, strict=False)
                ]
                new_output = type_equal(func1.output, func2.output, r1, r2)
                return FunctionType(new_inputs, new_output, func1.platforms or [])
            raise TypeMismatchError(
                f"{print_type(t1)} is not equal to {print_type(t2)}: incompatible types"
            )
        # Primitives
        if t1.tag == t2.tag:
            return t1
        raise TypeMismatchError(
            f"{print_type(t1)} is not equal to {print_type(t2)}: incompatible types"
        )
    except TypeMismatchError:
        raise  # Don't wrap our own errors
    except Exception as cause:
        raise TypeMismatchError(f"{print_type(t1)} is not equal to {print_type(t2)}") from cause


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
    from east.types.ref import Ref
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

    # Ref
    if isinstance(value, Ref):
        return RefType(type_of(value.value))

    # Structural types have _east_type
    if isinstance(value, EastStruct | EastVariant):
        return value._east_type  # type: ignore[return-value]

    # Unknown type
    raise TypeError(f"Unknown East type for value: {type(value).__name__}")


def east_type_of(value: Any) -> EastType:
    """Infer the EastType from a Python value.

    This function infers East types from raw Python values, including:
    - Primitives: None, bool, int, float, str, datetime, bytes
    - Collections: list (as Array), set (as Set), dict (as Struct)
    - East containers: EastArray, EastSet, EastDict
    - Structural types: EastStruct, EastVariant

    Args:
        value: The Python value to infer the type from

    Returns:
        The inferred EastType

    Raises:
        TypeError: If the value cannot be converted to an East type (e.g., functions)
    """
    from datetime import datetime

    from east.types.containers import EastArray, EastDict, EastSet
    from east.types.primitives import Blob, Null
    from east.types.ref import Ref
    from east.types.structural import EastStruct, EastVariant

    # None
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

    # DateTime
    if isinstance(value, datetime):
        return DateTimeType

    # Blob (bytes or Blob)
    if isinstance(value, bytes | Blob):
        return BlobType

    # Array - Python list or EastArray
    if isinstance(value, list):
        if not value:
            msg = "Cannot infer type of empty list"
            raise TypeError(msg)
        # Infer element type from first element
        return ArrayType(east_type_of(value[0]))

    if isinstance(value, EastArray):
        return ArrayType(value.element_type)

    # Set - Python set or EastSet
    if isinstance(value, set):
        if not value:
            msg = "Cannot infer type of empty set"
            raise TypeError(msg)
        # Infer element type from first element
        return SetType(east_type_of(next(iter(value))))

    if isinstance(value, EastSet):
        return SetType(value.element_type)

    # Dict - EastDict
    if isinstance(value, EastDict):
        return DictType(value.key_type, value.value_type)

    # Ref
    if isinstance(value, Ref):
        return RefType(east_type_of(value.value))

    # Function
    if callable(value):
        msg = "JavaScript/Python functions cannot be converted to East functions"
        raise TypeError(msg)

    # Struct - Python dict (plain object)
    if isinstance(value, dict):
        if not value:
            msg = "Cannot infer type of empty dict"
            raise TypeError(msg)
        # Infer struct type from field types
        fields = [(k, east_type_of(v)) for k, v in value.items()]
        return StructType(fields)

    # Structural types
    if isinstance(value, EastStruct | EastVariant):
        return value._east_type  # type: ignore[return-value]

    # EastType itself
    if isinstance(value, EastType):
        return EastTypeType

    # Unknown type
    msg = f"Cannot determine East type for value {value}"
    raise TypeError(msg)


############################################################################################
# IR Types
#
# These types define the structure of East Intermediate Representation (IR).
# IR nodes are East values themselves, enabling cross-language serialization.
############################################################################################

# Type of primitive literal values that can appear in ValueIR nodes
LiteralValueType = VariantType(
    [
        ("Null", NullType),
        ("Boolean", BooleanType),
        ("Integer", IntegerType),
        ("Float", FloatType),
        ("String", StringType),
        ("DateTime", DateTimeType),
        ("Blob", BlobType),
    ]
)

# Location information for IR nodes (filename, line, column)
LocationType = StructType(
    [
        ("filename", StringType),
        ("line", IntegerType),
        ("column", IntegerType),
    ]
)

# Label for loops (used in While, ForArray, ForSet, ForDict, Break, Continue)
IRLabelType = StructType(
    [
        ("name", StringType),
        ("location", LocationType),
    ]
)


# Helper function to build IR struct types
def _ir_struct_type(fields_builder):
    """Build an IR struct type with recursive IR reference.

    Args:
        fields_builder: Function taking ir type and returning list of (name, type) tuples

    Returns:
        Function that takes ir type and returns StructType
    """
    return lambda ir: StructType(fields_builder(ir))


# Define all IR node struct types as builder functions
_ErrorIR = _ir_struct_type(
    lambda ir: [
        ("type", EastTypeType),
        ("location", LocationType),
        ("message", ir),
    ]
)

_TryCatchIR = _ir_struct_type(
    lambda ir: [
        ("type", EastTypeType),
        ("location", LocationType),
        ("try_body", ir),
        ("catch_body", ir),
        ("message", ir),
        ("stack", ir),
        ("finally_body", ir),
    ]
)

_ValueIR = _ir_struct_type(
    lambda ir: [
        ("type", EastTypeType),
        ("location", LocationType),
        ("value", LiteralValueType),
    ]
)

VariableIR = StructType(
    [
        ("type", EastTypeType),
        ("name", StringType),
        ("location", LocationType),
        ("mutable", BooleanType),
        ("captured", BooleanType),
    ]
)

_LetIR = _ir_struct_type(
    lambda ir: [
        ("type", EastTypeType),
        ("location", LocationType),
        ("variable", ir),
        ("value", ir),
    ]
)

_AssignIR = _ir_struct_type(
    lambda ir: [
        ("type", EastTypeType),
        ("location", LocationType),
        ("variable", ir),
        ("value", ir),
    ]
)

_AsIR = _ir_struct_type(
    lambda ir: [
        ("type", EastTypeType),
        ("value", ir),
        ("location", LocationType),
    ]
)

_FunctionIR = _ir_struct_type(
    lambda ir: [
        ("type", EastTypeType),
        ("location", LocationType),
        ("captures", ArrayType(ir)),
        ("parameters", ArrayType(ir)),
        ("body", ir),
    ]
)

_CallIR = _ir_struct_type(
    lambda ir: [
        ("type", EastTypeType),
        ("location", LocationType),
        ("function", ir),
        ("arguments", ArrayType(ir)),
    ]
)

_NewRefIR = _ir_struct_type(
    lambda ir: [
        ("type", EastTypeType),
        ("location", LocationType),
        ("value", ir),
    ]
)

_NewArrayIR = _ir_struct_type(
    lambda ir: [
        ("type", EastTypeType),
        ("location", LocationType),
        ("values", ArrayType(ir)),
    ]
)

_NewSetIR = _ir_struct_type(
    lambda ir: [
        ("type", EastTypeType),
        ("location", LocationType),
        ("values", ArrayType(ir)),
    ]
)

_NewDictEntry = _ir_struct_type(
    lambda ir: [
        ("key", ir),
        ("value", ir),
    ]
)

_NewDictIR = _ir_struct_type(
    lambda ir: [
        ("type", EastTypeType),
        ("location", LocationType),
        ("values", ArrayType(_NewDictEntry(ir))),
    ]
)

_StructField = _ir_struct_type(
    lambda ir: [
        ("name", StringType),
        ("value", ir),
    ]
)

_StructIR = _ir_struct_type(
    lambda ir: [
        ("type", EastTypeType),
        ("location", LocationType),
        ("fields", ArrayType(_StructField(ir))),
    ]
)

_GetFieldIR = _ir_struct_type(
    lambda ir: [
        ("type", EastTypeType),
        ("location", LocationType),
        ("field", StringType),
        ("struct", ir),
    ]
)

_VariantIR = _ir_struct_type(
    lambda ir: [
        ("type", EastTypeType),
        ("location", LocationType),
        ("case", StringType),
        ("value", ir),
    ]
)

_BlockIR = _ir_struct_type(
    lambda ir: [
        ("type", EastTypeType),
        ("location", LocationType),
        ("statements", ArrayType(ir)),
    ]
)

_IfCase = _ir_struct_type(
    lambda ir: [
        ("predicate", ir),
        ("body", ir),
    ]
)

_IfElseIR = _ir_struct_type(
    lambda ir: [
        ("type", EastTypeType),
        ("location", LocationType),
        ("ifs", ArrayType(_IfCase(ir))),
        ("else_body", ir),
    ]
)

_MatchCase = _ir_struct_type(
    lambda ir: [
        ("case", StringType),
        ("variable", ir),
        ("body", ir),
    ]
)

_MatchIR = _ir_struct_type(
    lambda ir: [
        ("type", EastTypeType),
        ("location", LocationType),
        ("variant", ir),
        ("cases", ArrayType(_MatchCase(ir))),
    ]
)

_UnwrapRecursiveIR = _ir_struct_type(
    lambda ir: [
        ("type", EastTypeType),
        ("location", LocationType),
        ("value", ir),
    ]
)

_WrapRecursiveIR = _ir_struct_type(
    lambda ir: [
        ("type", EastTypeType),
        ("location", LocationType),
        ("value", ir),
    ]
)

_WhileIR = _ir_struct_type(
    lambda ir: [
        ("type", EastTypeType),
        ("location", LocationType),
        ("predicate", ir),
        ("label", IRLabelType),
        ("body", ir),
    ]
)

_ForArrayIR = _ir_struct_type(
    lambda ir: [
        ("type", EastTypeType),
        ("location", LocationType),
        ("array", ir),
        ("label", IRLabelType),
        ("key", ir),
        ("value", ir),
        ("body", ir),
    ]
)

_ForSetIR = _ir_struct_type(
    lambda ir: [
        ("type", EastTypeType),
        ("location", LocationType),
        ("set", ir),
        ("label", IRLabelType),
        ("key", ir),
        ("body", ir),
    ]
)

_ForDictIR = _ir_struct_type(
    lambda ir: [
        ("type", EastTypeType),
        ("location", LocationType),
        ("dict", ir),
        ("label", IRLabelType),
        ("key", ir),
        ("value", ir),
        ("body", ir),
    ]
)

_ReturnIR = _ir_struct_type(
    lambda ir: [
        ("type", EastTypeType),
        ("location", LocationType),
        ("value", ir),
    ]
)

_ContinueIR = _ir_struct_type(
    lambda ir: [
        ("type", EastTypeType),
        ("location", LocationType),
        ("label", IRLabelType),
    ]
)

_BreakIR = _ir_struct_type(
    lambda ir: [
        ("type", EastTypeType),
        ("location", LocationType),
        ("label", IRLabelType),
    ]
)

_BuiltinIR = _ir_struct_type(
    lambda ir: [
        ("type", EastTypeType),
        ("location", LocationType),
        ("builtin", StringType),
        ("type_parameters", ArrayType(EastTypeType)),
        ("arguments", ArrayType(ir)),
    ]
)

_PlatformIR = _ir_struct_type(
    lambda ir: [
        ("type", EastTypeType),
        ("location", LocationType),
        ("name", StringType),
        ("arguments", ArrayType(ir)),
    ]
)

# Recursive IR type - the type of all IR nodes
IRType = recursive_type(
    lambda ir: VariantType(
        [
            ("Error", _ErrorIR(ir)),
            ("TryCatch", _TryCatchIR(ir)),
            ("Value", _ValueIR(ir)),
            ("Variable", VariableIR),
            ("Let", _LetIR(ir)),
            ("Assign", _AssignIR(ir)),
            ("As", _AsIR(ir)),
            ("Function", _FunctionIR(ir)),
            ("Call", _CallIR(ir)),
            ("NewRef", _NewRefIR(ir)),
            ("NewArray", _NewArrayIR(ir)),
            ("NewSet", _NewSetIR(ir)),
            ("NewDict", _NewDictIR(ir)),
            ("Struct", _StructIR(ir)),
            ("GetField", _GetFieldIR(ir)),
            ("Variant", _VariantIR(ir)),
            ("Block", _BlockIR(ir)),
            ("IfElse", _IfElseIR(ir)),
            ("Match", _MatchIR(ir)),
            ("UnwrapRecursive", _UnwrapRecursiveIR(ir)),
            ("WrapRecursive", _WrapRecursiveIR(ir)),
            ("While", _WhileIR(ir)),
            ("ForArray", _ForArrayIR(ir)),
            ("ForSet", _ForSetIR(ir)),
            ("ForDict", _ForDictIR(ir)),
            ("Return", _ReturnIR(ir)),
            ("Continue", _ContinueIR(ir)),
            ("Break", _BreakIR(ir)),
            ("Builtin", _BuiltinIR(ir)),
            ("Platform", _PlatformIR(ir)),
        ]
    )
)

# Export the struct types by evaluating them with IRType
ValueIR = _ValueIR(IRType)
BuiltinIR = _BuiltinIR(IRType)
PlatformIR = _PlatformIR(IRType)
FunctionIR = _FunctionIR(IRType)
BlockIR = _BlockIR(IRType)
IfElseIR = _IfElseIR(IRType)
IfCase = _IfCase(IRType)
WhileIR = _WhileIR(IRType)


__all__ = [
    "StructType",
    "VariantType",
    "EastType",
    "TypeMismatchError",
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
    "RefType",
    "StructType",
    "VariantType",
    "FunctionType",
    "RecursiveTypeRef",
    "EastTypeType",
    "recursive_type",
    "type_of",
    "east_type_of",
    "is_data_type",
    "is_immutable_type",
    "is_type_equal",
    "is_subtype",
    "is_value_of",
    "type_union",
    "type_intersect",
    "type_equal",
    "SomeType",
    "OptionType",
    # IR Types
    "LiteralValueType",
    "LocationType",
    "IRLabelType",
    "VariableIR",
    "ValueIR",
    "BuiltinIR",
    "FunctionIR",
    "BlockIR",
    "IfElseIR",
    "IfCase",
    "WhileIR",
    "IRType",
]
