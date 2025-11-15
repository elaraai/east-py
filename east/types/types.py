"""East type system using TypedDict.

Types are plain dicts at runtime, with TypedDict providing static type hints.
This matches TypeScript's approach exactly while maintaining Python type safety.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from east.types.primitives import Null

# =============================================================================
# TypedDict Definitions - Common Struct Types
# =============================================================================


class Location(TypedDict):
    """Source code location."""

    filename: str
    line: int
    column: int


class IRLabel(TypedDict):
    """IR loop label."""

    name: str
    location: Location


# Forward declare for IR types that reference themselves
IRNode = dict  # Union of all IR variant types
LiteralValueVariant = (
    dict  # Variant with cases: Null, Boolean, Integer, Float, String, Blob, DateTime
)


class ValueIRValue(TypedDict):
    """Value inside ValueIR variant (literal value)."""

    type: EastType
    location: Location
    value: LiteralValueVariant


class VariableIRValue(TypedDict):
    """Value inside VariableIR variant."""

    type: EastType
    name: str
    location: Location
    mutable: bool
    captured: bool


class BuiltinIRValue(TypedDict):
    """Value inside BuiltinIR variant."""

    type: EastType
    location: Location
    builtin: str
    type_parameters: Any  # EastArray[EastType]
    arguments: Any  # EastArray[IRNode]


class PlatformIRValue(TypedDict):
    """Value inside PlatformIR variant."""

    type: EastType
    location: Location
    name: str
    arguments: Any  # EastArray[IRNode]


class LetIRValue(TypedDict):
    """Value inside LetIR variant."""

    type: EastType
    location: Location
    variable: IRNode  # VariableIR
    value: IRNode


class AssignIRValue(TypedDict):
    """Value inside AssignIR variant."""

    type: EastType
    location: Location
    variable: IRNode  # VariableIR
    value: IRNode


class AsIRValue(TypedDict):
    """Value inside AsIR variant (type cast)."""

    type: EastType
    value: IRNode
    location: Location


class FunctionIRValue(TypedDict):
    """Value inside FunctionIR variant."""

    type: EastType
    location: Location
    captures: Any  # List[VariableIR]
    parameters: Any  # List[VariableIR]
    body: IRNode


class CallIRValue(TypedDict):
    """Value inside CallIR variant."""

    type: EastType
    location: Location
    function: IRNode
    arguments: Any  # List[IRNode]


class NewRefIRValue(TypedDict):
    """Value inside NewRefIR variant."""

    type: EastType
    location: Location
    value: IRNode


class NewArrayIRValue(TypedDict):
    """Value inside NewArrayIR variant."""

    type: EastType
    location: Location
    values: Any  # List[IRNode]


class NewSetIRValue(TypedDict):
    """Value inside NewSetIR variant."""

    type: EastType
    location: Location
    values: Any  # List[IRNode]


class DictEntry(TypedDict):
    """Dict entry in NewDictIR."""

    key: IRNode
    value: IRNode


class NewDictIRValue(TypedDict):
    """Value inside NewDictIR variant."""

    type: EastType
    location: Location
    entries: list[DictEntry]


class StructField(TypedDict):
    """Struct field in StructIR."""

    name: str
    value: IRNode


class StructIRValue(TypedDict):
    """Value inside StructIR variant."""

    type: EastType
    location: Location
    fields: list[StructField]


class GetFieldIRValue(TypedDict):
    """Value inside GetFieldIR variant."""

    type: EastType
    location: Location
    field: str
    struct: IRNode


class VariantIRValue(TypedDict):
    """Value inside VariantIR variant (variant constructor)."""

    type: EastType
    location: Location
    case: str
    value: IRNode


class BlockIRValue(TypedDict):
    """Value inside BlockIR variant."""

    type: EastType
    location: Location
    statements: Any  # List[IRNode]


class IfCase(TypedDict):
    """If case in IfElseIR."""

    predicate: IRNode
    body: IRNode


class IfElseIRValue(TypedDict):
    """Value inside IfElseIR variant."""

    type: EastType
    location: Location
    ifs: list[IfCase]
    else_body: IRNode


class MatchCase(TypedDict):
    """Match case in MatchIR."""

    case: str
    variable: IRNode  # VariableIR
    body: IRNode


class MatchIRValue(TypedDict):
    """Value inside MatchIR variant."""

    type: EastType
    location: Location
    variant: IRNode
    cases: list[MatchCase]


class UnwrapRecursiveIRValue(TypedDict):
    """Value inside UnwrapRecursiveIR variant."""

    type: EastType
    location: Location
    value: IRNode


class WrapRecursiveIRValue(TypedDict):
    """Value inside WrapRecursiveIR variant."""

    type: EastType
    location: Location
    value: IRNode


class WhileIRValue(TypedDict):
    """Value inside WhileIR variant."""

    type: EastType
    location: Location
    predicate: IRNode
    label: IRLabel
    body: IRNode


class ForArrayIRValue(TypedDict):
    """Value inside ForArrayIR variant."""

    type: EastType
    location: Location
    array: IRNode
    label: IRLabel
    key: IRNode  # VariableIR
    element: IRNode  # VariableIR
    body: IRNode


class ForSetIRValue(TypedDict):
    """Value inside ForSetIR variant."""

    type: EastType
    location: Location
    set: IRNode
    label: IRLabel
    element: IRNode  # VariableIR
    body: IRNode


class ForDictIRValue(TypedDict):
    """Value inside ForDictIR variant."""

    type: EastType
    location: Location
    dict: IRNode
    label: IRLabel
    key: IRNode  # VariableIR
    value: IRNode  # VariableIR
    body: IRNode


class ReturnIRValue(TypedDict):
    """Value inside ReturnIR variant."""

    type: EastType
    location: Location
    value: IRNode


class ContinueIRValue(TypedDict):
    """Value inside ContinueIR variant."""

    type: EastType
    location: Location
    label: str


class BreakIRValue(TypedDict):
    """Value inside BreakIR variant."""

    type: EastType
    location: Location
    label: str


class ErrorIRValue(TypedDict):
    """Value inside ErrorIR variant."""

    type: EastType
    location: Location
    message: IRNode


class TryCatchIRValue(TypedDict):
    """Value inside TryCatchIR variant."""

    type: EastType
    location: Location
    try_body: IRNode
    catch_body: IRNode
    message: IRNode  # VariableIR
    stack: IRNode  # VariableIR
    finally_body: IRNode  # Always an IRNode; Value wrapping Null when no finally block


# =============================================================================
# TypedDict Definitions for Type System
# =============================================================================


class NullTypeDef(TypedDict):
    """Null type - unit type with single value."""

    type: Literal["Null"]
    value: None  # Always null in JSON


class BooleanTypeDef(TypedDict):
    """Boolean type - true or false."""

    type: Literal["Boolean"]
    value: None  # Always null in JSON


class IntegerTypeDef(TypedDict):
    """Integer type - arbitrary precision integers."""

    type: Literal["Integer"]
    value: None  # Always null in JSON


class FloatTypeDef(TypedDict):
    """Float type - 64-bit floating point."""

    type: Literal["Float"]
    value: None  # Always null in JSON


class StringTypeDef(TypedDict):
    """String type - UTF-8 text."""

    type: Literal["String"]
    value: None  # Always null in JSON


class BlobTypeDef(TypedDict):
    """Blob type - immutable binary data."""

    type: Literal["Blob"]
    value: None  # Always null in JSON


class DateTimeTypeDef(TypedDict):
    """DateTime type - UTC timestamps."""

    type: Literal["DateTime"]
    value: None  # Always null in JSON


class NeverTypeDef(TypedDict):
    """Never type - bottom type (no values)."""

    type: Literal["Never"]
    value: None  # Always null in JSON


class ArrayTypeDef(TypedDict):
    """Array type - mutable ordered collection."""

    type: Literal["Array"]
    value: EastType  # Element type


class SetTypeDef(TypedDict):
    """Set type - mutable unordered collection of unique values."""

    type: Literal["Set"]
    value: EastType  # Element type


class DictValueTypeDef(TypedDict):
    """Dict type value (key and value types)."""

    key: EastType
    value: EastType


class DictTypeDef(TypedDict):
    """Dict type - mutable key-value mapping."""

    type: Literal["Dict"]
    value: DictValueTypeDef


class RefTypeDef(TypedDict):
    """Ref type - mutable reference cell."""

    type: Literal["Ref"]
    value: EastType  # Type of referenced value


class StructFieldDef(TypedDict):
    """Struct field definition."""

    name: str
    type: EastType


class StructTypeDef(TypedDict):
    """Struct type - product type with named fields."""

    type: Literal["Struct"]
    value: list[StructFieldDef]


class VariantCaseDef(TypedDict):
    """Variant case definition."""

    name: str
    type: EastType


class VariantTypeDef(TypedDict):
    """Variant type - sum type with tagged cases."""

    type: Literal["Variant"]
    value: list[VariantCaseDef]


class FunctionTypeDef(TypedDict):
    """Function type."""

    type: Literal["Function"]
    value: FunctionTypeValue


class FunctionTypeValue(TypedDict):
    """Function type value (inputs, output, platforms)."""

    inputs: list[EastType]
    output: EastType
    platforms: list[str]


class RecursiveTypeDef(TypedDict):
    """Recursive type reference."""

    type: Literal["Recursive"]
    value: int  # Marker ID


# Union of all type variants
EastType = (
    NullTypeDef
    | BooleanTypeDef
    | IntegerTypeDef
    | FloatTypeDef
    | StringTypeDef
    | BlobTypeDef
    | DateTimeTypeDef
    | NeverTypeDef
    | ArrayTypeDef
    | SetTypeDef
    | DictTypeDef
    | RefTypeDef
    | StructTypeDef
    | VariantTypeDef
    | FunctionTypeDef
    | RecursiveTypeDef
)


# =============================================================================
# Type Constructors (return plain dicts)
# =============================================================================

# Primitive types (singletons)
NullType: NullTypeDef = {"type": "Null", "value": None}
BooleanType: BooleanTypeDef = {"type": "Boolean", "value": None}
IntegerType: IntegerTypeDef = {"type": "Integer", "value": None}
FloatType: FloatTypeDef = {"type": "Float", "value": None}
StringType: StringTypeDef = {"type": "String", "value": None}
BlobType: BlobTypeDef = {"type": "Blob", "value": None}
DateTimeType: DateTimeTypeDef = {"type": "DateTime", "value": None}
NeverType: NeverTypeDef = {"type": "Never", "value": None}


def ArrayType(element_type: EastType) -> ArrayTypeDef:
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
    return {"type": "Array", "value": element_type}


def SetType(element_type: EastType) -> SetTypeDef:
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
    return {"type": "Set", "value": element_type}


def DictType(key_type: EastType, value_type: EastType) -> DictTypeDef:
    """Create a dictionary type.

    Args:
        key_type: Type of dictionary keys (must be immutable)
        value_type: Type of dictionary values

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
    return {"type": "Dict", "value": {"key": key_type, "value": value_type}}


def RefType(value_type: EastType) -> RefTypeDef:
    """Create a reference type.

    Args:
        value_type: Type of referenced value

    Returns:
        Ref type

    Raises:
        TypeError: If value_type is not a data type
    """
    if not is_data_type(value_type):
        from east.serialization.east_printer import print_type

        raise TypeError(
            f"Ref value type must be a (non-function) data type, got {print_type(value_type)}"
        )
    return {"type": "Ref", "value": value_type}


def StructType(fields: list[tuple[str, EastType]]) -> StructTypeDef:
    """Create a struct type.

    Args:
        fields: List of (field_name, field_type) tuples

    Returns:
        Struct type

    Raises:
        TypeError: If any field type is not a data type
    """
    field_defs: list[StructFieldDef] = []
    for name, field_type in fields:
        if not is_data_type(field_type):
            from east.serialization.east_printer import print_type

            raise TypeError(
                f"Struct field '{name}' type must be a (non-function) data type, got {print_type(field_type)}"
            )
        field_defs.append({"name": name, "type": field_type})

    return {"type": "Struct", "value": field_defs}


def VariantType(cases: list[tuple[str, EastType]]) -> VariantTypeDef:
    """Create a variant type.

    Args:
        cases: List of (case_name, case_type) tuples

    Returns:
        Variant type

    Raises:
        TypeError: If any case type is not a data type
        ValueError: If case names are not unique
    """
    # Check for duplicate case names
    case_names = [name for name, _ in cases]
    if len(case_names) != len(set(case_names)):
        raise ValueError(f"Variant case names must be unique, got {case_names}")

    case_defs: list[VariantCaseDef] = []
    for name, case_type in cases:
        if not is_data_type(case_type):
            from east.serialization.east_printer import print_type

            raise TypeError(
                f"Variant case '{name}' type must be a (non-function) data type, got {print_type(case_type)}"
            )
        case_defs.append({"name": name, "type": case_type})

    # Sort cases alphabetically by name
    case_defs.sort(key=lambda c: c["name"])

    return {"type": "Variant", "value": case_defs}


def FunctionType(inputs: list[EastType], output: EastType, platforms: list[str]) -> FunctionTypeDef:
    """Create a function type.

    Args:
        inputs: List of input parameter types
        output: Output return type
        platforms: List of required platform functions

    Returns:
        Function type
    """
    return {
        "type": "Function",
        "value": {"inputs": inputs, "output": output, "platforms": platforms},
    }


def RecursiveTypeRef(marker: int) -> RecursiveTypeDef:
    """Create a recursive type reference.

    Args:
        marker: Recursive type marker ID

    Returns:
        Recursive type reference
    """
    return {"type": "Recursive", "value": marker}


# =============================================================================
# Helper Functions for Working with Types
# =============================================================================


def field_names(struct_type: StructTypeDef) -> list[str]:
    """Get field names from a Struct type.

    Args:
        struct_type: Struct type

    Returns:
        List of field names
    """
    return [field["name"] for field in struct_type["value"]]


def field_types(struct_type: StructTypeDef) -> list[EastType]:
    """Get field types from a Struct type.

    Args:
        struct_type: Struct type

    Returns:
        List of field types
    """
    return [field["type"] for field in struct_type["value"]]


def field_index(struct_type: StructTypeDef, name: str) -> int:
    """Get field index by name for Struct types.

    Args:
        struct_type: Struct type
        name: Field name

    Returns:
        Index of field

    Raises:
        KeyError: If field not found
    """
    for i, field in enumerate(struct_type["value"]):
        if field["name"] == name:
            return i
    raise KeyError(f"No field named '{name}'")


def case_names(variant_type: VariantTypeDef) -> list[str]:
    """Get case names from a Variant type.

    Args:
        variant_type: Variant type

    Returns:
        List of case names
    """
    return [case["name"] for case in variant_type["value"]]


def case_types(variant_type: VariantTypeDef) -> list[EastType]:
    """Get case types from a Variant type.

    Args:
        variant_type: Variant type

    Returns:
        List of case types
    """
    return [case["type"] for case in variant_type["value"]]


def case_type(variant_type: VariantTypeDef, name: str) -> EastType:
    """Get type of a case by name for Variant types.

    Args:
        variant_type: Variant type
        name: Case name

    Returns:
        Type of the case

    Raises:
        KeyError: If case not found
    """
    for case in variant_type["value"]:
        if case["name"] == name:
            return case["type"]
    raise KeyError(f"No case named '{name}'")


# Type checking helpers
def is_struct_type(typ: EastType) -> bool:
    """Check if a type is a Struct type."""
    return typ["type"] == "Struct"


def is_variant_type(typ: EastType) -> bool:
    """Check if a type is a Variant type."""
    return typ["type"] == "Variant"


def is_array_type(typ: EastType) -> bool:
    """Check if a type is an Array type."""
    return typ["type"] == "Array"


def is_function_type(typ: EastType) -> bool:
    """Check if a type is a Function type."""
    return typ["type"] == "Function"


# Value checking helpers
def is_struct_value(value: Any) -> bool:
    """Check if a value is a struct (plain dict without 'type' key)."""
    return isinstance(value, dict) and "type" not in value


def is_variant_value(value: Any) -> bool:
    """Check if a value is a variant (dict with 'type' and 'value' keys)."""
    return isinstance(value, dict) and "type" in value and "value" in value and len(value) == 2


# =============================================================================
# Type Predicates
# =============================================================================


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
    from typing import cast

    # Avoid infinite loops in recursive types
    if recursive_type is not None and typ == recursive_type:
        return True

    tag = typ["type"]

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
        # Type narrowing: we know it's StructTypeDef after the check
        struct_typ = cast(StructTypeDef, typ)
        fields: list[StructFieldDef] = struct_typ["value"]
        return all(is_data_type(cast(EastType, field["type"]), recursive_type) for field in fields)
    if tag == "Variant":
        # Type narrowing: we know it's VariantTypeDef after the check
        variant_typ = cast(VariantTypeDef, typ)
        cases: list[VariantCaseDef] = variant_typ["value"]
        return all(is_data_type(cast(EastType, case["type"]), recursive_type) for case in cases)
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
    from typing import cast

    # Avoid infinite loops in recursive types
    if recursive_type is not None and typ == recursive_type:
        return True

    tag = typ["type"]

    if tag in ("Array", "Set", "Dict", "Ref", "Function"):
        return False
    if tag == "Struct":
        # Type narrowing: we know it's StructTypeDef after the check
        struct_typ = cast(StructTypeDef, typ)
        fields: list[StructFieldDef] = struct_typ["value"]
        return all(
            is_immutable_type(cast(EastType, field["type"]), recursive_type) for field in fields
        )
    if tag == "Variant":
        # Type narrowing: we know it's VariantTypeDef after the check
        variant_typ = cast(VariantTypeDef, typ)
        cases: list[VariantCaseDef] = variant_typ["value"]
        return all(
            is_immutable_type(cast(EastType, case["type"]), recursive_type) for case in cases
        )
    if tag == "Recursive":
        # Recursive references are always valid for immutable check
        return True
    # Primitive types are immutable
    return True


# =============================================================================
# Common Type Constructors
# =============================================================================


def SomeType(value_type: EastType) -> VariantTypeDef:
    """Create an Option.Some variant type (for optional values).

    Args:
        value_type: Type of the wrapped value

    Returns:
        Variant type with 'some' and 'none' cases
    """
    return VariantType([("some", value_type), ("none", NullType)])


def OptionType(value_type: EastType) -> VariantTypeDef:
    """Create an Option type (for optional values).

    Alias for SomeType.

    Args:
        value_type: Type of the wrapped value

    Returns:
        Variant type with 'some' and 'none' cases
    """
    return SomeType(value_type)


# =============================================================================
# Meta Types - Types that describe other types
# =============================================================================

# The type of all East types - defined after recursive_type function below
# Will be initialized at the end of this file
EastTypeType: EastType

# Helper types for IR nodes - will be defined after EastTypeType (see below)
LocationType: EastType
IRLabelType: EastType
VariableType: EastType
LiteralValueType: EastType
# IRType will be defined after recursive_type function (see below)


# =============================================================================
# Exception Types
# =============================================================================


class TypeMismatchError(TypeError):
    """Exception raised when types cannot be unified or intersected."""

    pass


# =============================================================================
# Recursive Type Handling
# =============================================================================


class RecursiveTypeMarker:
    """Temporary marker used during recursive type construction.

    After construction, all marker instances are replaced with integer scope_ids.
    This class should not appear in any final type structures - only integers.
    """

    def __repr__(self) -> str:
        """Return string representation of the marker."""
        return f"<RecursiveMarker at {hex(id(self))}>"


def recursive_type(builder: Any) -> EastType:
    """Build a recursive type with integer scope_ids (matches TypeScript).

    This function creates a recursive type where all self-references use integer
    scope_ids based on their depth in the type structure.

    Args:
        builder: Function that takes a marker and returns the node type

    Returns:
        A type where all self-references use integer scope_ids

    Example:
        ListType = recursive_type(
            lambda self: VariantType([
                ("nil", NullType),
                ("cons", StructType([("head", IntegerType), ("tail", self)]))
            ])
        )
        # The "tail" field will have type .Recursive 2 (integer, not marker)
    """
    # Create a marker for this recursive scope
    marker = RecursiveTypeMarker()

    # Create a placeholder reference that points to the marker
    placeholder: RecursiveTypeDef = {"type": "Recursive", "value": marker}  # type: ignore

    # Build the type using the placeholder
    node = builder(placeholder)

    # Replace all marker instances with integer scope_ids
    def replace_markers(t: Any, stack_depth: int = 0) -> Any:
        """Recursively replace markers with integer scope_ids.

        Args:
            t: Type to process
            stack_depth: Current type_ctx stack depth

        Returns:
            Type with markers replaced by integers
        """
        # If this is a dict type
        if isinstance(t, dict) and "type" in t:
            tag = t["type"]

            # If this IS the marker object (passed as placeholder), replace with integer
            if tag == "Recursive":
                if isinstance(t["value"], RecursiveTypeMarker) and t["value"] is marker:
                    # scope_id is the current type_ctx stack depth
                    return {"type": "Recursive", "value": stack_depth}
                # Already has integer or different marker, leave as is
                return t

            # Array, Set, Ref: These push to type_ctx
            if tag in ("Array", "Set", "Ref"):
                new_value = replace_markers(t["value"], stack_depth + 1)
                return {**t, "value": new_value}

            # Dict: Pushes to type_ctx once
            if tag == "Dict":
                dict_value = t["value"]
                new_key = replace_markers(dict_value["key"], stack_depth + 1)
                new_value_type = replace_markers(dict_value["value"], stack_depth + 1)
                return {
                    "type": "Dict",
                    "value": {"key": new_key, "value": new_value_type},
                }

            # Struct: Pushes to type_ctx
            if tag == "Struct":
                fields = t["value"]
                new_fields = []
                for field in fields:
                    new_type = replace_markers(field["type"], stack_depth + 1)
                    new_fields.append({"name": field["name"], "type": new_type})
                return {"type": "Struct", "value": new_fields}

            # Variant: Pushes to type_ctx
            if tag == "Variant":
                cases = t["value"]
                new_cases = []
                for case in cases:
                    new_type = replace_markers(case["type"], stack_depth + 1)
                    new_cases.append({"name": case["name"], "type": new_type})
                return {"type": "Variant", "value": new_cases}

            # Function: Does NOT push to type_ctx
            if tag == "Function":
                func_value = t["value"]
                new_inputs = [replace_markers(inp, stack_depth) for inp in func_value["inputs"]]
                new_output = replace_markers(func_value["output"], stack_depth)
                return {
                    "type": "Function",
                    "value": {
                        "inputs": new_inputs,
                        "output": new_output,
                        "platforms": func_value["platforms"],
                    },
                }

        # Not a composite type or marker, return as-is
        return t

    # Start with stack_depth=0; the root type hasn't pushed to type_ctx yet
    return replace_markers(node, stack_depth=0)  # type: ignore


# =============================================================================
# Type Comparison
# =============================================================================


def type_equal(
    t1: EastType, t2: EastType, r1: EastType | None = None, r2: EastType | None = None
) -> EastType:
    """Check if two types are structurally equal and return the unified type.

    This is a port of TypeScript's TypeEqual function.

    Args:
        t1: First type
        t2: Second type
        r1: Root type for t1 (for recursive type handling)
        r2: Root type for t2 (for recursive type handling)

    Returns:
        The unified type (t1) if types are equal

    Raises:
        TypeMismatchError: If types are not structurally equal
    """
    from east.serialization.east_printer import print_type

    if r1 is None:
        r1 = t1
    if r2 is None:
        r2 = t2

    # Handle Ref types
    if t1["type"] == "Ref":
        if t2["type"] == "Ref":
            return RefType(type_equal(t1["value"], t2["value"], r1, r2))  # type: ignore[arg-type]
        raise TypeMismatchError(
            f"{print_type(t1)} is not equal to {print_type(t2)}: incompatible types"
        )

    # Handle Array types
    if t1["type"] == "Array":
        if t2["type"] == "Array":
            return ArrayType(type_equal(t1["value"], t2["value"], r1, r2))  # type: ignore[arg-type]
        raise TypeMismatchError(
            f"{print_type(t1)} is not equal to {print_type(t2)}: incompatible types"
        )

    # Handle Set types
    if t1["type"] == "Set":
        if t2["type"] == "Set":
            return SetType(type_equal(t1["value"], t2["value"], r1, r2))  # type: ignore[arg-type]
        raise TypeMismatchError(
            f"{print_type(t1)} is not equal to {print_type(t2)}: incompatible types"
        )

    # Handle Dict types
    if t1["type"] == "Dict":
        if t2["type"] == "Dict":
            dict1 = t1["value"]  # type: ignore[typeddict-item]
            dict2 = t2["value"]  # type: ignore[typeddict-item]
            return DictType(
                type_equal(dict1["key"], dict2["key"], r1, r2),
                type_equal(dict1["value"], dict2["value"], r1, r2),
            )
        raise TypeMismatchError(
            f"{print_type(t1)} is not equal to {print_type(t2)}: incompatible types"
        )

    # Handle Struct types
    if t1["type"] == "Struct":
        if t2["type"] == "Struct":
            fields1 = t1["value"]  # type: ignore[typeddict-item]
            fields2 = t2["value"]  # type: ignore[typeddict-item]
            if len(fields1) != len(fields2):  # type: ignore[arg-type]
                raise TypeMismatchError(
                    f"{print_type(t1)} is not equal to {print_type(t2)}: structs contain different number of fields"
                )

            unified_fields = []
            for i, (f1, f2) in enumerate(zip(fields1, fields2, strict=False)):  # type: ignore[arg-type]
                if f1["name"] != f2["name"]:
                    raise TypeMismatchError(
                        f"{print_type(t1)} is not equal to {print_type(t2)}: struct field {i} has mismatched names {f1['name']} and {f2['name']}"
                    )
                unified_fields.append((f1["name"], type_equal(f1["type"], f2["type"], r1, r2)))

            return StructType(unified_fields)
        raise TypeMismatchError(
            f"{print_type(t1)} is not equal to {print_type(t2)}: incompatible types"
        )

    # Handle Variant types
    if t1["type"] == "Variant":
        if t2["type"] == "Variant":
            cases1 = t1["value"]  # type: ignore[typeddict-item]
            cases2 = t2["value"]  # type: ignore[typeddict-item]
            if len(cases1) != len(cases2):  # type: ignore[arg-type]
                raise TypeMismatchError(
                    f"{print_type(t1)} is not equal to {print_type(t2)}: variants contain different number of cases"
                )

            unified_cases = []
            for c1, c2 in zip(cases1, cases2, strict=False):  # type: ignore[arg-type]
                if c1["name"] != c2["name"]:
                    # Report which case is missing
                    if c1["name"] < c2["name"]:
                        raise TypeMismatchError(
                            f"{print_type(t1)} is not equal to {print_type(t2)}: variant case {c1['name']} is not present in both variants"
                        )
                    raise TypeMismatchError(
                        f"{print_type(t1)} is not equal to {print_type(t2)}: variant case {c2['name']} is not present in both variants"
                    )
                unified_cases.append((c1["name"], type_equal(c1["type"], c2["type"], r1, r2)))

            return VariantType(unified_cases)
        raise TypeMismatchError(
            f"{print_type(t1)} is not equal to {print_type(t2)}: incompatible types"
        )

    # Handle Function types
    if t1["type"] == "Function":
        if t2["type"] == "Function":
            func1 = t1["value"]  # type: ignore[typeddict-item]
            func2 = t2["value"]  # type: ignore[typeddict-item]

            # Check input types
            inputs1 = func1["inputs"]
            inputs2 = func2["inputs"]
            output1 = func1["output"]
            output2 = func2["output"]
            platforms1 = func1["platforms"]
            platforms2 = func2["platforms"]

            if len(inputs1) != len(inputs2):
                raise TypeMismatchError(
                    f"{print_type(t1)} is not equal to {print_type(t2)}: functions have different number of inputs"
                )

            unified_inputs = [
                type_equal(i1, i2, r1, r2) for i1, i2 in zip(inputs1, inputs2, strict=False)
            ]  # type: ignore[misc]
            unified_output = type_equal(output1, output2, r1, r2)

            # Check platforms match
            if platforms1 != platforms2:
                raise TypeMismatchError(
                    f"{print_type(t1)} is not equal to {print_type(t2)}: functions have different platform requirements"
                )

            return FunctionType(unified_inputs, unified_output, platforms1)
        raise TypeMismatchError(
            f"{print_type(t1)} is not equal to {print_type(t2)}: incompatible types"
        )

    # Handle Recursive types
    if t1["type"] == "Recursive":
        if t2["type"] == "Recursive":
            # Check if we're at the recursive reference point
            if t1 is r1:
                if t2 is r2:
                    return t1  # Both are references to the same recursive type
                raise TypeMismatchError(
                    f"{print_type(t1)} is not equal to {print_type(t2)}: recursive types do not match"
                )
            if t2 is r2:
                raise TypeMismatchError(
                    f"{print_type(t1)} is not equal to {print_type(t2)}: recursive types do not match"
                )

            # Root of new recursive type - check node types are equal
            return type_equal(t1["value"], t2["value"], t1, t2)  # type: ignore[typeddict-item]
        # Recursive type wrapper is transparent
        return type_equal(t1["value"], t2, t1, r2)  # type: ignore[typeddict-item]
    if t2["type"] == "Recursive":
        # Recursive type wrapper is transparent
        return type_equal(t1, t2["value"], r1, t2)  # type: ignore[typeddict-item]

    # Handle primitive types - they must match exactly
    if t1["type"] == t2["type"]:
        # For primitives (Never, Null, Boolean, Integer, Float, String, DateTime, Blob)
        # they're equal if they have the same kind
        return t1
    raise TypeMismatchError(
        f"{print_type(t1)} is not equal to {print_type(t2)}: incompatible types"
    )


def is_type_equal(
    t1: EastType, t2: EastType, r1: EastType | None = None, r2: EastType | None = None
) -> bool:
    """Check if two types are structurally equal (boolean version).

    This is a boolean wrapper around type_equal that returns True/False instead of throwing.

    Args:
        t1: First type to compare
        t2: Second type to compare
        r1: Recursive type root for t1 (internal, optional)
        r2: Recursive type root for t2 (internal, optional)

    Returns:
        True if types are structurally equal, False otherwise
    """
    if r1 is None:
        r1 = t1
    if r2 is None:
        r2 = t2

    # Handle Recursive types
    if t1["type"] == "Recursive":
        if t2["type"] == "Recursive":
            # Both recursive
            v1 = t1["value"]  # type: ignore[typeddict-item]
            v2 = t2["value"]  # type: ignore[typeddict-item]

            # Integer scope_ids (from deserialized types): compare directly
            if isinstance(v1, int) and isinstance(v2, int):
                return v1 == v2

            # Object references (from in-memory types): check identity
            if v1 is r1:
                return v2 is r2
            if v2 is r2:
                return False
            # Both are nested types, compare recursively
            return is_type_equal(v1, v2, v1, v2)  # type: ignore[arg-type]
        # Recursive type wrapper is transparent
        v1 = t1["value"]  # type: ignore[typeddict-item]
        # Integer scope_id can't equal non-recursive type
        if isinstance(v1, int):
            return False
        return is_type_equal(v1, t2, v1, r2)  # type: ignore[arg-type]
    if t2["type"] == "Recursive":
        # Recursive type wrapper is transparent
        v2 = t2["value"]  # type: ignore[typeddict-item]
        # Integer scope_id can't equal non-recursive type
        if isinstance(v2, int):
            return False
        return is_type_equal(t1, v2, r1, v2)  # type: ignore[arg-type]

    # Handle primitive types
    if t1["type"] in ("Never", "Null", "Boolean", "Integer", "Float", "String", "DateTime", "Blob"):
        return t1["type"] == t2["type"]

    # Handle Ref types
    if t1["type"] == "Ref":
        return t2["type"] == "Ref" and is_type_equal(t1["value"], t2["value"], r1, r2)  # type: ignore[typeddict-item]

    # Handle Array types
    if t1["type"] == "Array":
        return t2["type"] == "Array" and is_type_equal(t1["value"], t2["value"], r1, r2)  # type: ignore[typeddict-item]

    # Handle Set types
    if t1["type"] == "Set":
        return t2["type"] == "Set" and is_type_equal(t1["value"], t2["value"], r1, r2)  # type: ignore[typeddict-item]

    # Handle Dict types
    if t1["type"] == "Dict":
        if t2["type"] != "Dict":
            return False
        dict1 = t1["value"]
        dict2 = t2["value"]
        return is_type_equal(dict1["key"], dict2["key"], r1, r2) and is_type_equal(
            dict1["value"], dict2["value"], r1, r2
        )

    # Handle Struct types
    if t1["type"] == "Struct":
        if t2["type"] != "Struct":
            return False
        fields1 = t1["value"]
        fields2 = t2["value"]
        if len(fields1) != len(fields2):
            return False
        for f1, f2 in zip(fields1, fields2, strict=False):  # type: ignore[arg-type]
            if f1["name"] != f2["name"]:
                return False
            if not is_type_equal(f1["type"], f2["type"], r1, r2):
                return False
        return True

    # Handle Variant types
    if t1["type"] == "Variant":
        if t2["type"] != "Variant":
            return False
        cases1 = t1["value"]
        cases2 = t2["value"]
        if len(cases1) != len(cases2):
            return False
        for c1, c2 in zip(cases1, cases2, strict=False):  # type: ignore[arg-type]
            if c1["name"] != c2["name"]:
                return False
            if not is_type_equal(c1["type"], c2["type"], r1, r2):
                return False
        return True

    # Handle Function types
    if t1["type"] == "Function":
        if t2["type"] != "Function":
            return False
        func1 = t1["value"]
        func2 = t2["value"]

        # Check input types
        inputs1 = func1["inputs"]
        inputs2 = func2["inputs"]
        if len(inputs1) != len(inputs2):
            return False
        for i1, i2 in zip(inputs1, inputs2, strict=False):
            if not is_type_equal(i1, i2, r1, r2):
                return False

        # Check output type
        if not is_type_equal(func1["output"], func2["output"], r1, r2):
            return False

        # Check platform requirements
        plat1 = func1["platforms"]
        plat2 = func2["platforms"]
        if plat1 is None and plat2 is None:
            return True
        if plat1 is None or plat2 is None:
            return False
        if len(plat1) != len(plat2):
            return False
        for p1, p2 in zip(plat1, plat2, strict=False):
            if p1 != p2:
                return False
        return True

    # Unknown type
    raise NotImplementedError(f"is_type_equal not implemented for type kind: {t1}")


def is_value_of(
    value: Any,
    typ: EastType,
    node_type: EastType | None = None,
    nodes_visited: set[int] | None = None,
) -> bool:
    """Check if a value conforms to an East type.

    Args:
        value: The value to check
        typ: The East type to validate against
        node_type: Internal parameter for tracking recursive type node
        nodes_visited: Internal parameter for cycle detection

    Returns:
        True if value matches type, False otherwise
    """
    from datetime import datetime

    from east.types.containers import EastArray, EastDict, EastSet
    from east.types.primitives import Blob
    from east.types.ref import Ref

    # Handle Never type
    if typ["type"] == "Never":
        return False

    # Handle primitive types
    if typ["type"] == "Null":
        return value is None
    if typ["type"] == "Boolean":
        return isinstance(value, bool)
    if typ["type"] == "Integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if typ["type"] == "Float":
        return isinstance(value, float)
    if typ["type"] == "String":
        return isinstance(value, str)
    if typ["type"] == "DateTime":
        return isinstance(value, datetime)
    if typ["type"] == "Blob":
        return isinstance(value, (bytes, bytearray, Blob))

    # Handle Ref type
    if typ["type"] == "Ref":
        if not isinstance(value, Ref):
            return False
        return is_value_of(value.value, typ["value"], node_type, nodes_visited)  # type: ignore[typeddict-item]

    # Handle Array type
    if typ["type"] == "Array":
        if not isinstance(value, EastArray):
            return False
        for elem in value:
            if not is_value_of(elem, typ["value"], node_type, nodes_visited):  # type: ignore[typeddict-item]
                return False
        return True

    # Handle Set type
    if typ["type"] == "Set":
        if not isinstance(value, EastSet):
            return False
        for elem in value:
            if not is_value_of(elem, typ["value"], node_type, nodes_visited):  # type: ignore[typeddict-item]
                return False
        return True

    # Handle Dict type
    if typ["type"] == "Dict":
        if not isinstance(value, EastDict):
            return False
        dict_type = typ["value"]
        for k, v in value.items():
            if not is_value_of(k, dict_type["key"], node_type, nodes_visited):
                return False
            if not is_value_of(v, dict_type["value"], node_type, nodes_visited):
                return False
        return True

    # Handle Struct type
    if typ["type"] == "Struct":
        if not is_struct_value(value):
            return False
        # Check fields match
        value_fields = list(value.items())
        type_fields = typ["value"]
        if len(value_fields) != len(type_fields):
            return False
        for i, field_def in enumerate(type_fields):
            field_name = field_def["name"]
            field_type = field_def["type"]
            if i >= len(value_fields):
                return False
            val_name, val_value = value_fields[i]
            if val_name != field_name:
                return False
            if not is_value_of(val_value, field_type, node_type, nodes_visited):
                return False
        return True

    # Handle Variant type
    if typ["type"] == "Variant":
        if not is_variant_value(value):
            return False
        variant_tag = value["type"]
        variant_value = value["value"]
        # Find the case type
        cases = typ["value"]
        for case in cases:
            if case["name"] == variant_tag:
                return is_value_of(variant_value, case["type"], node_type, nodes_visited)
        return False  # Case not found

    # Handle Recursive type
    if typ["type"] == "Recursive":
        recursive_node = typ["value"]
        if node_type is recursive_node:
            # Already tracking this recursive type
            value_id = id(value)
            if nodes_visited is None:
                nodes_visited = set()
            if value_id in nodes_visited:
                return True  # Already seen this object
            nodes_visited.add(value_id)
            return is_value_of(value, recursive_node, node_type, nodes_visited)
        # New recursive type, reset tracking
        return is_value_of(value, recursive_node, recursive_node, {id(value)})

    # Handle Function type
    if typ["type"] == "Function":
        raise TypeError("JavaScript/Python functions cannot be converted to East functions")

    # Unknown type
    raise NotImplementedError(f"is_value_of not implemented for type: {typ}")


def is_subtype(t1: EastType, t2: EastType) -> bool:
    """Check if t1 is a subtype of t2.

    Args:
        t1: The potential subtype
        t2: The potential supertype

    Returns:
        True if t1 is a subtype of t2, False otherwise
    """
    # Handle Recursive types
    if t1["type"] == "Recursive":
        if t2["type"] == "Recursive":
            # Recursive types are invariant for heap layout compatibility
            return is_type_equal(t1["value"], t2["value"])  # type: ignore[typeddict-item]
        # Recursive type wrapper is transparent but invariant
        return is_type_equal(t1["value"], t2)  # type: ignore[typeddict-item]
    if t2["type"] == "Recursive":
        # Recursive type wrapper is transparent
        # Head covariance by unfolding once
        return is_subtype(t1, t2["value"])  # type: ignore[typeddict-item]

    # Never is a subtype of everything
    if t1["type"] == "Never":
        return True

    # Primitive types are only subtypes of themselves
    if t1["type"] in ("Null", "Boolean", "Integer", "Float", "String", "DateTime", "Blob"):
        return t1["type"] == t2["type"]

    # Handle Ref types (invariant)
    if t1["type"] == "Ref":
        return t2["type"] == "Ref" and is_type_equal(t1["value"], t2["value"])  # type: ignore[typeddict-item]

    # Handle Array types (invariant)
    if t1["type"] == "Array":
        return t2["type"] == "Array" and is_type_equal(t1["value"], t2["value"])  # type: ignore[typeddict-item]

    # Handle Set types (invariant)
    if t1["type"] == "Set":
        return t2["type"] == "Set" and is_type_equal(t1["value"], t2["value"])

    # Handle Dict types (invariant)
    if t1["type"] == "Dict":
        if t2["type"] != "Dict":
            return False
        dict1 = t1["value"]
        dict2 = t2["value"]
        return is_type_equal(dict1["key"], dict2["key"]) and is_type_equal(
            dict1["value"], dict2["value"]
        )

    # Handle Struct types (structural subtyping)
    if t1["type"] == "Struct":
        if t2["type"] != "Struct":
            return False
        fields1 = t1["value"]
        fields2 = t2["value"]
        if len(fields1) != len(fields2):
            return False
        for f1, f2 in zip(fields1, fields2, strict=False):  # type: ignore[arg-type]
            if f1["name"] != f2["name"]:
                return False
            if not is_subtype(f1["type"], f2["type"]):
                return False
        return True

    # Handle Variant types (subset of cases)
    if t1["type"] == "Variant":
        if t2["type"] != "Variant":
            return False
        cases1 = t1["value"]
        cases2 = t2["value"]

        # Build case map for t2
        cases2_map = {case["name"]: case["type"] for case in cases2}

        # Check each case in t1 is in t2 with compatible type
        for case1 in cases1:
            case_name = case1["name"]
            case_type1 = case1["type"]
            case_type2 = cases2_map.get(case_name, NeverType)
            if not is_subtype(case_type1, case_type2):
                return False
        return True

    # Handle Function types (contravariant inputs, covariant output)
    if t1["type"] == "Function":
        if t2["type"] != "Function":
            return False
        func1 = t1["value"]
        func2 = t2["value"]

        inputs1 = func1["inputs"]
        inputs2 = func2["inputs"]
        if len(inputs1) != len(inputs2):
            return False

        # Contravariant inputs (t2 input subtypes of t1 inputs)
        for i1, i2 in zip(inputs1, inputs2, strict=False):
            if not is_subtype(i2, i1):
                return False

        # Covariant output
        return is_subtype(func1["output"], func2["output"])

    # Unknown type
    raise NotImplementedError(f"is_subtype not implemented for type: {t1}")


def type_union(t1: EastType, t2: EastType) -> EastType:
    """Compute the union of two East types.

    Args:
        t1: First type
        t2: Second type

    Returns:
        The union type

    Raises:
        TypeMismatchError: When the types cannot be unioned
    """
    from east.serialization.east_printer import print_identifier, print_type

    try:
        # Never is identity for union
        if t1["type"] == "Never":
            return t2
        if t2["type"] == "Never":
            return t1

        # Recursive types
        if t1["type"] == "Recursive":
            if t2["type"] == "Recursive":
                # Both recursive - require exact match
                return type_equal(t1, t2)
            # Rec(A) ∪ NonRec: If NonRec <: A, union is Rec(A)
            if is_subtype(t2, t1["value"]["node"]):
                return t1
            raise TypeMismatchError(
                f"Cannot union {print_type(t1)} with {print_type(t2)}: incompatible types"
            )
        if t2["type"] == "Recursive":
            # NonRec ∪ Rec(B): If NonRec <: B, union is Rec(B)
            if is_subtype(t1, t2["value"]["node"]):
                return t2
            raise TypeMismatchError(
                f"Cannot union {print_type(t1)} with {print_type(t2)}: incompatible types"
            )

        # Ref types
        if t1["type"] == "Ref":
            if t2["type"] == "Ref":
                return RefType(type_equal(t1["value"], t2["value"]))
            raise TypeMismatchError(
                f"Cannot union {print_type(t1)} with {print_type(t2)}: incompatible types"
            )

        # Array types
        if t1["type"] == "Array":
            if t2["type"] == "Array":
                return ArrayType(type_equal(t1["value"], t2["value"]))
            raise TypeMismatchError(
                f"Cannot union {print_type(t1)} with {print_type(t2)}: incompatible types"
            )

        # Set types
        if t1["type"] == "Set":
            if t2["type"] == "Set":
                return SetType(type_equal(t1["value"], t2["value"]))
            raise TypeMismatchError(
                f"Cannot union {print_type(t1)} with {print_type(t2)}: incompatible types"
            )

        # Dict types
        if t1["type"] == "Dict":
            if t2["type"] == "Dict":
                return DictType(
                    type_equal(t1["value"]["key"], t2["value"]["key"]),
                    type_equal(t1["value"]["value"], t2["value"]["value"]),
                )
            raise TypeMismatchError(
                f"Cannot union {print_type(t1)} with {print_type(t2)}: incompatible types"
            )

        # Struct types
        if t1["type"] == "Struct":
            if t2["type"] == "Struct":
                fields1 = t1["value"]  # List of StructFieldDef
                fields2 = t2["value"]
                if len(fields1) != len(fields2):
                    raise TypeMismatchError(
                        f"Cannot union {print_type(t1)} with {print_type(t2)}: "
                        "structs contain different number of fields"
                    )
                result_fields = []
                for i, field1 in enumerate(fields1):
                    field2 = fields2[i]
                    k1, f1 = field1["name"], field1["type"]
                    k2, f2 = field2["name"], field2["type"]
                    if k1 != k2:
                        raise TypeMismatchError(
                            f"Cannot union {print_type(t1)} with {print_type(t2)}: "
                            f"struct field {i} has mismatched names {print_identifier(k1)} and {print_identifier(k2)}"
                        )
                    result_fields.append((k1, type_union(f1, f2)))
                return StructType(result_fields)
            raise TypeMismatchError(
                f"Cannot union {print_type(t1)} with {print_type(t2)}: incompatible types"
            )

        # Variant types
        if t1["type"] == "Variant":
            if t2["type"] == "Variant":
                # Build dict from case lists for easier lookup
                cases1 = {c["name"]: c["type"] for c in t1["value"]}
                cases2 = {c["name"]: c["type"] for c in t2["value"]}
                result_cases = {}
                # Add all cases from t1
                for k1, f1 in cases1.items():
                    f2 = cases2.get(k1)
                    if f2 is None:
                        result_cases[k1] = f1
                    else:
                        result_cases[k1] = type_union(f1, f2)
                # Add cases from t2 not in t1
                for k2, f2 in cases2.items():
                    if k2 not in cases1:
                        result_cases[k2] = f2
                return VariantType(list(result_cases.items()))
            raise TypeMismatchError(
                f"Cannot union {print_type(t1)} with {print_type(t2)}: incompatible types"
            )

        # Function types
        if t1["type"] == "Function":
            if t2["type"] == "Function":
                if len(t1["value"]["inputs"]) != len(t2["value"]["inputs"]):
                    raise TypeMismatchError(
                        f"Cannot union {print_type(t1)} with {print_type(t2)}: "
                        "functions take different number of arguments"
                    )
                # Union platforms
                platforms1 = t1["value"]["platforms"]
                platforms2 = t2["value"]["platforms"]
                if platforms1 is None or platforms2 is None:
                    platforms = []
                else:
                    platforms = sorted(set(platforms1 + platforms2))

                # Contravariant inputs, covariant output
                inputs = [
                    type_intersect(t1["value"]["inputs"][i], t2["value"]["inputs"][i])
                    for i in range(len(t1["value"]["inputs"]))
                ]
                output = type_union(t1["value"]["output"], t2["value"]["output"])
                return FunctionType(inputs, output, platforms)
            raise TypeMismatchError(
                f"Cannot union {print_type(t1)} with {print_type(t2)}: incompatible types"
            )

        # Primitive types
        if t1["type"] == t2["type"]:
            return t1
        raise TypeMismatchError(
            f"Cannot union {print_type(t1)} with {print_type(t2)}: incompatible types"
        )

    except TypeMismatchError:
        raise
    except Exception as e:
        raise TypeMismatchError(f"Cannot union {print_type(t1)} with {print_type(t2)}") from e


def type_intersect(t1: EastType, t2: EastType) -> EastType:
    """Compute the intersection of two East types.

    Args:
        t1: First type
        t2: Second type

    Returns:
        The intersection type

    Raises:
        TypeMismatchError: When the types cannot be intersected
    """
    from east.serialization.east_printer import print_identifier, print_type

    try:
        # Never is absorbing for intersection
        if t1["type"] == "Never":
            return NeverType
        if t2["type"] == "Never":
            return NeverType

        # Recursive types
        if t1["type"] == "Recursive":
            if t2["type"] == "Recursive":
                # Both recursive - require exact match
                return recursive_type(
                    lambda: type_equal(
                        t1["value"]["node"],
                        t2["value"]["node"],
                        t1["value"]["node"],
                        t2["value"]["node"],
                    )
                )
            # Rec(A) ∩ NonRec: If NonRec <: A, intersection is NonRec
            if is_subtype(t2, t1["value"]["node"]):
                return t2
            raise TypeMismatchError(
                f"Cannot intersect {print_type(t1)} with {print_type(t2)}: incompatible types"
            )
        if t2["type"] == "Recursive":
            # NonRec ∩ Rec(B): If NonRec <: B, intersection is NonRec
            if is_subtype(t1, t2["value"]["node"]):
                return t1
            raise TypeMismatchError(
                f"Cannot intersect {print_type(t1)} with {print_type(t2)}: incompatible types"
            )

        # Ref types
        if t1["type"] == "Ref":
            if t2["type"] == "Ref":
                return RefType(type_equal(t1["value"], t2["value"]))
            raise TypeMismatchError(
                f"Cannot intersect {print_type(t1)} with {print_type(t2)}: incompatible types"
            )

        # Array types
        if t1["type"] == "Array":
            if t2["type"] == "Array":
                return ArrayType(type_equal(t1["value"], t2["value"]))
            raise TypeMismatchError(
                f"Cannot intersect {print_type(t1)} with {print_type(t2)}: incompatible types"
            )

        # Set types
        if t1["type"] == "Set":
            if t2["type"] == "Set":
                return SetType(type_equal(t1["value"], t2["value"]))
            raise TypeMismatchError(
                f"Cannot intersect {print_type(t1)} with {print_type(t2)}: incompatible types"
            )

        # Dict types
        if t1["type"] == "Dict":
            if t2["type"] == "Dict":
                return DictType(
                    type_equal(t1["value"]["key"], t2["value"]["key"]),
                    type_equal(t1["value"]["value"], t2["value"]["value"]),
                )
            raise TypeMismatchError(
                f"Cannot intersect {print_type(t1)} with {print_type(t2)}: incompatible types"
            )

        # Struct types
        if t1["type"] == "Struct":
            if t2["type"] == "Struct":
                fields1 = t1["value"]  # List of StructFieldDef
                fields2 = t2["value"]
                if len(fields1) != len(fields2):
                    raise TypeMismatchError(
                        f"Cannot intersect {print_type(t1)} with {print_type(t2)}: "
                        "structs contain different number of fields"
                    )
                result_fields = []
                for i, field1 in enumerate(fields1):
                    field2 = fields2[i]
                    k1, f1 = field1["name"], field1["type"]
                    k2, f2 = field2["name"], field2["type"]
                    if k1 != k2:
                        raise TypeMismatchError(
                            f"Cannot intersect {print_type(t1)} with {print_type(t2)}: "
                            f"struct field {i} has mismatched names {print_identifier(k1)} and {print_identifier(k2)}"
                        )
                    result_fields.append((k1, type_intersect(f1, f2)))
                return StructType(result_fields)
            raise TypeMismatchError(
                f"Cannot intersect {print_type(t1)} with {print_type(t2)}: incompatible types"
            )

        # Variant types
        if t1["type"] == "Variant":
            if t2["type"] == "Variant":
                # Build dict from case lists for easier lookup
                cases1 = {c["name"]: c["type"] for c in t1["value"]}
                cases2 = {c["name"]: c["type"] for c in t2["value"]}
                result_cases = {}
                # Only include common cases
                for k1, f1 in cases1.items():
                    f2 = cases2.get(k1)
                    if f2 is not None:
                        result_cases[k1] = type_intersect(f1, f2)
                if not result_cases:
                    raise TypeMismatchError(
                        f"Cannot intersect {print_type(t1)} with {print_type(t2)}: "
                        "variants have no overlapping cases"
                    )
                return VariantType(list(result_cases.items()))
            raise TypeMismatchError(
                f"Cannot intersect {print_type(t1)} with {print_type(t2)}: incompatible types"
            )

        # Function types
        if t1["type"] == "Function":
            if t2["type"] == "Function":
                if len(t1["value"]["inputs"]) != len(t2["value"]["inputs"]):
                    raise TypeMismatchError(
                        f"Cannot intersect {print_type(t1)} with {print_type(t2)}: "
                        "functions take different number of arguments"
                    )
                # Intersect platforms
                platforms1 = t1["value"]["platforms"]
                platforms2 = t2["value"]["platforms"]
                if platforms1 is None:
                    platforms = platforms2
                elif platforms2 is None:
                    platforms = platforms1
                else:
                    platforms = [p for p in platforms1 if p in platforms2]

                # Contravariant inputs, covariant output
                inputs = [
                    type_union(t1["value"]["inputs"][i], t2["value"]["inputs"][i])
                    for i in range(len(t1["value"]["inputs"]))
                ]
                output = type_intersect(t1["value"]["output"], t2["value"]["output"])
                return FunctionType(inputs, output, platforms)
            raise TypeMismatchError(
                f"Cannot intersect {print_type(t1)} with {print_type(t2)}: incompatible types"
            )

        # Primitive types
        if t1["type"] == t2["type"]:
            return t1
        raise TypeMismatchError(
            f"Cannot intersect {print_type(t1)} with {print_type(t2)}: incompatible types"
        )

    except TypeMismatchError:
        raise
    except Exception as e:
        raise TypeMismatchError(f"Cannot intersect {print_type(t1)} with {print_type(t2)}") from e


# =============================================================================
# Type Inference
# =============================================================================


def type_of(value: Any) -> EastType:
    """Infer the East type of a Python value.

    Args:
        value: Python value

    Returns:
        East type

    Raises:
        TypeError: If value type cannot be inferred
    """
    import datetime

    from east.types.containers import EastArray, EastDict, EastSet
    from east.types.ref import Ref

    if value is None or isinstance(value, Null):
        return NullType
    if isinstance(value, bool):
        return BooleanType
    if isinstance(value, int):
        return IntegerType
    if isinstance(value, float):
        return FloatType
    if isinstance(value, str):
        return StringType
    if isinstance(value, bytes):
        return BlobType
    if isinstance(value, datetime.datetime):
        return DateTimeType
    if isinstance(value, EastArray):
        return ArrayType(value.element_type)
    if isinstance(value, EastSet):
        return SetType(value.element_type)
    if isinstance(value, EastDict):
        return DictType(value.key_type, value.value_type)
    if isinstance(value, Ref):
        # Ref doesn't store type info at runtime - infer from contained value
        return RefType(type_of(value.value))
    if isinstance(value, dict):
        # Check if it's a variant value
        if "type" in value and "value" in value and len(value) == 2:
            # It's a variant - but we don't know the full variant type
            # Return a generic variant with just this case
            case_value_type = type_of(value["value"])
            return VariantType([(value["type"], case_value_type)])
        # It's a struct value
        field_types_list = []
        for key, val in value.items():
            field_types_list.append((key, type_of(val)))
        return StructType(field_types_list)
    if callable(value):
        # Can't infer function types from Python callables
        raise TypeError(f"Cannot infer type of callable {value}")

    raise TypeError(f"Cannot infer type of {type(value).__name__}")


# =============================================================================
# EastTypeType - The type of East types (defined after recursive_type)
# =============================================================================

# The type of all East types - matches TypeScript's EastTypeType
EastTypeType = recursive_type(
    lambda type: VariantType(
        [
            ("Never", NullType),
            ("Null", NullType),
            ("Boolean", NullType),
            ("Integer", NullType),
            ("Float", NullType),
            ("String", NullType),
            ("DateTime", NullType),
            ("Blob", NullType),
            ("Ref", type),
            ("Array", type),
            ("Set", type),
            ("Dict", StructType([("key", type), ("value", type)])),
            ("Struct", ArrayType(StructType([("name", StringType), ("type", type)]))),
            ("Variant", ArrayType(StructType([("name", StringType), ("type", type)]))),
            ("Recursive", IntegerType),
            (
                "Function",
                StructType(
                    [
                        ("inputs", ArrayType(type)),
                        ("output", type),
                        ("platforms", ArrayType(StringType)),
                    ]
                ),
            ),
        ]
    )
)

# Helper types for IR nodes - defined after EastTypeType
LocationType = StructType(
    [
        ("filename", StringType),
        ("line", IntegerType),
        ("column", IntegerType),
    ]
)

IRLabelType = StructType(
    [
        ("name", StringType),
        ("location", LocationType),
    ]
)

VariableType = StructType(
    [
        ("type", EastTypeType),
        ("location", LocationType),
        ("name", StringType),
        ("mutable", BooleanType),
        ("captured", BooleanType),
    ]
)

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


# =============================================================================
# IRType - The type of all IR nodes (defined after recursive_type)
# =============================================================================

# The type of all IR nodes - defined as a variant with all IR node types
# For simplicity, we define this as a recursive variant type
IRType: EastType = recursive_type(
    lambda self: VariantType(
        [
            (
                "Error",
                StructType([("type", EastTypeType), ("location", LocationType), ("message", self)]),
            ),
            (
                "TryCatch",
                StructType(
                    [
                        ("type", EastTypeType),
                        ("location", LocationType),
                        ("try_body", self),
                        ("catch_body", self),
                        ("message", self),
                        ("stack", self),
                        ("finally_body", self),
                    ]
                ),
            ),
            (
                "Value",
                StructType(
                    [
                        ("type", EastTypeType),
                        ("location", LocationType),
                        ("value", LiteralValueType),
                    ]
                ),
            ),
            ("Variable", VariableType),
            (
                "Let",
                StructType(
                    [
                        ("type", EastTypeType),
                        ("location", LocationType),
                        ("variable", self),
                        ("value", self),
                    ]
                ),
            ),
            (
                "Assign",
                StructType(
                    [
                        ("type", EastTypeType),
                        ("location", LocationType),
                        ("variable", self),
                        ("value", self),
                    ]
                ),
            ),
            (
                "As",
                StructType([("type", EastTypeType), ("value", self), ("location", LocationType)]),
            ),
            (
                "Function",
                StructType(
                    [
                        ("type", EastTypeType),
                        ("location", LocationType),
                        ("captures", ArrayType(self)),
                        ("parameters", ArrayType(self)),
                        ("body", self),
                    ]
                ),
            ),
            (
                "Call",
                StructType(
                    [
                        ("type", EastTypeType),
                        ("location", LocationType),
                        ("function", self),
                        ("arguments", ArrayType(self)),
                    ]
                ),
            ),
            (
                "NewRef",
                StructType([("type", EastTypeType), ("location", LocationType), ("value", self)]),
            ),
            (
                "NewArray",
                StructType(
                    [
                        ("type", EastTypeType),
                        ("location", LocationType),
                        ("values", ArrayType(self)),
                    ]
                ),
            ),
            (
                "NewSet",
                StructType(
                    [
                        ("type", EastTypeType),
                        ("location", LocationType),
                        ("values", ArrayType(self)),
                    ]
                ),
            ),
            (
                "NewDict",
                StructType(
                    [
                        ("type", EastTypeType),
                        ("location", LocationType),
                        ("values", ArrayType(StructType([("key", self), ("value", self)]))),
                    ]
                ),
            ),
            (
                "Struct",
                StructType(
                    [
                        ("type", EastTypeType),
                        ("location", LocationType),
                        ("fields", ArrayType(StructType([("name", StringType), ("value", self)]))),
                    ]
                ),
            ),
            (
                "GetField",
                StructType(
                    [
                        ("type", EastTypeType),
                        ("location", LocationType),
                        ("field", StringType),
                        ("struct", self),
                    ]
                ),
            ),
            (
                "Variant",
                StructType(
                    [
                        ("type", EastTypeType),
                        ("location", LocationType),
                        ("case", StringType),
                        ("value", self),
                    ]
                ),
            ),
            (
                "Block",
                StructType(
                    [
                        ("type", EastTypeType),
                        ("location", LocationType),
                        ("statements", ArrayType(self)),
                    ]
                ),
            ),
            (
                "IfElse",
                StructType(
                    [
                        ("type", EastTypeType),
                        ("location", LocationType),
                        ("ifs", ArrayType(StructType([("predicate", self), ("body", self)]))),
                        ("else_body", self),
                    ]
                ),
            ),
            (
                "Match",
                StructType(
                    [
                        ("type", EastTypeType),
                        ("location", LocationType),
                        ("variant", self),
                        (
                            "cases",
                            ArrayType(
                                StructType(
                                    [("case", StringType), ("variable", self), ("body", self)]
                                )
                            ),
                        ),
                    ]
                ),
            ),
            (
                "UnwrapRecursive",
                StructType([("type", EastTypeType), ("location", LocationType), ("value", self)]),
            ),
            (
                "WrapRecursive",
                StructType([("type", EastTypeType), ("location", LocationType), ("value", self)]),
            ),
            (
                "While",
                StructType(
                    [
                        ("type", EastTypeType),
                        ("location", LocationType),
                        ("predicate", self),
                        ("label", IRLabelType),
                        ("body", self),
                    ]
                ),
            ),
            (
                "ForArray",
                StructType(
                    [
                        ("type", EastTypeType),
                        ("location", LocationType),
                        ("array", self),
                        ("label", IRLabelType),
                        ("key", self),
                        ("value", self),
                        ("body", self),
                    ]
                ),
            ),
            (
                "ForSet",
                StructType(
                    [
                        ("type", EastTypeType),
                        ("location", LocationType),
                        ("set", self),
                        ("label", IRLabelType),
                        ("key", self),
                        ("body", self),
                    ]
                ),
            ),
            (
                "ForDict",
                StructType(
                    [
                        ("type", EastTypeType),
                        ("location", LocationType),
                        ("dict", self),
                        ("label", IRLabelType),
                        ("key", self),
                        ("value", self),
                        ("body", self),
                    ]
                ),
            ),
            (
                "Return",
                StructType([("type", EastTypeType), ("location", LocationType), ("value", self)]),
            ),
            (
                "Continue",
                StructType(
                    [("type", EastTypeType), ("location", LocationType), ("label", IRLabelType)]
                ),
            ),
            (
                "Break",
                StructType(
                    [("type", EastTypeType), ("location", LocationType), ("label", IRLabelType)]
                ),
            ),
            (
                "Builtin",
                StructType(
                    [
                        ("type", EastTypeType),
                        ("location", LocationType),
                        ("builtin", StringType),
                        ("type_parameters", ArrayType(EastTypeType)),
                        ("arguments", ArrayType(self)),
                    ]
                ),
            ),
            (
                "Platform",
                StructType(
                    [
                        ("type", EastTypeType),
                        ("location", LocationType),
                        ("name", StringType),
                        ("arguments", ArrayType(self)),
                    ]
                ),
            ),
        ]
    )
)

# IfCaseType - The type of IfCase structs used in IfElse IR
# Each IfCase has predicate and body fields, both of which are IRNodes
IfCaseType: EastType = StructType([("predicate", IRType), ("body", IRType)])


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Common struct TypedDicts
    "Location",
    "IRLabel",
    # IR node value TypedDicts (29 IR types)
    "ValueIRValue",
    "VariableIRValue",
    "BuiltinIRValue",
    "PlatformIRValue",
    "LetIRValue",
    "AssignIRValue",
    "AsIRValue",
    "FunctionIRValue",
    "CallIRValue",
    "NewRefIRValue",
    "NewArrayIRValue",
    "NewSetIRValue",
    "NewDictIRValue",
    "StructIRValue",
    "GetFieldIRValue",
    "VariantIRValue",
    "BlockIRValue",
    "IfElseIRValue",
    "MatchIRValue",
    "UnwrapRecursiveIRValue",
    "WrapRecursiveIRValue",
    "WhileIRValue",
    "ForArrayIRValue",
    "ForSetIRValue",
    "ForDictIRValue",
    "ReturnIRValue",
    "ContinueIRValue",
    "BreakIRValue",
    "ErrorIRValue",
    "TryCatchIRValue",
    # Helper TypedDicts for IR nodes
    "DictEntry",
    "StructField",
    "IfCase",
    "MatchCase",
    "IRNode",
    "LiteralValueVariant",
    # Type TypedDict definitions
    "EastType",
    "NullTypeDef",
    "BooleanTypeDef",
    "IntegerTypeDef",
    "FloatTypeDef",
    "StringTypeDef",
    "BlobTypeDef",
    "DateTimeTypeDef",
    "NeverTypeDef",
    "ArrayTypeDef",
    "SetTypeDef",
    "DictTypeDef",
    "RefTypeDef",
    "StructTypeDef",
    "StructFieldDef",
    "VariantTypeDef",
    "VariantCaseDef",
    "FunctionTypeDef",
    "RecursiveTypeDef",
    # Type constructors
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
    # Helper functions
    "field_names",
    "field_types",
    "field_index",
    "case_names",
    "case_types",
    "case_type",
    "is_struct_type",
    "is_variant_type",
    "is_array_type",
    "is_function_type",
    "is_struct_value",
    "is_variant_value",
    # Type predicates
    "is_data_type",
    "is_immutable_type",
    # Common types
    "SomeType",
    "OptionType",
    # Meta types
    "EastTypeType",
    "LocationType",
    "IRLabelType",
    "VariableType",
    "LiteralValueType",
    "IRType",
    "IfCaseType",
    # Recursive types
    "RecursiveTypeMarker",
    "recursive_type",
    # Type comparison
    "type_equal",
    "is_type_equal",
    "is_subtype",
    "type_union",
    "type_intersect",
    # Type inference
    "type_of",
    "is_value_of",
    # Exceptions
    "TypeMismatchError",
]
