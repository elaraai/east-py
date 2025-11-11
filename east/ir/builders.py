"""Helper functions for building IR nodes as East variants.

IR nodes are East values (variants), not Python dataclasses. These helper functions
make it easier to construct IR variants programmatically.
"""

from datetime import datetime
from typing import Any

from east.types.containers import EastArray
from east.types.primitives import null
from east.types.structural import Case, EastStruct, EastVariant
from east.types.type_system import (
    BlockIR,
    BuiltinIR,
    EastType,
    EastTypeType,
    FunctionIR,
    IfCase,
    IfElseIR,
    IRLabelType,
    IRType,
    LiteralValueType,
    LocationType,
    PlatformIR,
    ValueIR,
    VariableIR,
    WhileIR,
    _StructTypeClass,
)


def _struct_class_from_type(east_type: EastType) -> _StructTypeClass:
    """Extract _StructTypeClass from an EastType representing a struct.

    Args:
        east_type: EastType with tag "Struct"

    Returns:
        _StructTypeClass that can create instances
    """
    assert east_type.tag == "Struct", f"Expected Struct type, got {east_type.tag}"
    field_structs = east_type.value
    fields = tuple((f.name, f.type) for f in field_structs)
    return _StructTypeClass(fields)


def location(filename: str, line: int, column: int) -> EastStruct:
    """Create a Location struct.

    Args:
        filename: Source filename
        line: Line number
        column: Column number

    Returns:
        Location struct
    """
    loc_class = _struct_class_from_type(LocationType)
    return loc_class.create(filename=filename, line=line, column=column)


def ir_label(name: str, loc: EastStruct) -> EastStruct:
    """Create an IR label struct.

    Args:
        name: Label name
        loc: Location

    Returns:
        IRLabel struct
    """
    label_class = _struct_class_from_type(IRLabelType)
    return label_class.create(name=name, location=loc)


def literal_value(value: Any) -> EastVariant:
    """Create a LiteralValue variant from a Python value.

    Args:
        value: Python value (None, bool, int, float, str, bytes, or datetime)

    Returns:
        LiteralValue variant
    """
    if value is None or value is null:
        return EastVariant(LiteralValueType, Case("Null", null))
    if isinstance(value, bool):
        return EastVariant(LiteralValueType, Case("Boolean", value))
    if isinstance(value, int):
        return EastVariant(LiteralValueType, Case("Integer", value))
    if isinstance(value, float):
        return EastVariant(LiteralValueType, Case("Float", value))
    if isinstance(value, str):
        return EastVariant(LiteralValueType, Case("String", value))
    if isinstance(value, bytes):
        return EastVariant(LiteralValueType, Case("Blob", value))
    if isinstance(value, datetime):
        return EastVariant(LiteralValueType, Case("DateTime", value))
    raise TypeError(f"Cannot convert {type(value)} to LiteralValue")


def ir_value(typ: EastType, loc: EastStruct, value: Any) -> EastVariant:
    """Create a Value IR node.

    Args:
        typ: East type of the value
        loc: Location
        value: The literal value (will be wrapped in LiteralValue variant)

    Returns:
        Value IR variant
    """
    lit_val = literal_value(value)
    value_class = _struct_class_from_type(ValueIR)
    value_struct = value_class.create(type=typ, location=loc, value=lit_val)
    return EastVariant(IRType, Case("Value", value_struct))


def ir_variable(
    typ: EastType, name: str, loc: EastStruct, mutable: bool = False, captured: bool = False
) -> EastVariant:
    """Create a Variable IR node.

    Args:
        typ: East type of the variable
        name: Variable name
        loc: Location
        mutable: Whether variable is mutable
        captured: Whether variable is captured by closure

    Returns:
        Variable IR variant
    """
    var_class = _struct_class_from_type(VariableIR)
    var_struct = var_class.create(
        type=typ, name=name, location=loc, mutable=mutable, captured=captured
    )
    return EastVariant(IRType, Case("Variable", var_struct))


def ir_builtin(
    typ: EastType,
    loc: EastStruct,
    builtin_name: str,
    type_parameters: list[EastType],
    arguments: list[EastVariant],
) -> EastVariant:
    """Create a Builtin IR node.

    Args:
        typ: Return type of the builtin
        loc: Location
        builtin_name: Name of the builtin function
        type_parameters: Type parameters for the builtin
        arguments: IR arguments

    Returns:
        Builtin IR variant
    """
    type_params_array = EastArray(EastTypeType, type_parameters)
    args_array = EastArray(IRType, arguments)

    builtin_class = _struct_class_from_type(BuiltinIR)
    builtin_struct = builtin_class.create(
        type=typ,
        location=loc,
        builtin=builtin_name,
        type_parameters=type_params_array,
        arguments=args_array,
    )
    return EastVariant(IRType, Case("Builtin", builtin_struct))


def ir_platform(
    typ: EastType,
    loc: EastStruct,
    platform_name: str,
    arguments: list[EastVariant],
) -> EastVariant:
    """Create a Platform IR node.

    Args:
        typ: Return type of the platform function
        loc: Location
        platform_name: Name of the platform function
        arguments: IR arguments

    Returns:
        Platform IR variant
    """
    args_array = EastArray(IRType, arguments)

    platform_class = _struct_class_from_type(PlatformIR)
    platform_struct = platform_class.create(
        type=typ,
        location=loc,
        name=platform_name,
        arguments=args_array,
    )
    return EastVariant(IRType, Case("Platform", platform_struct))


def ir_function(
    typ: EastType,
    loc: EastStruct,
    captures: list[EastVariant],
    parameters: list[EastVariant],
    body: EastVariant,
) -> EastVariant:
    """Create a Function IR node.

    Args:
        typ: Function type
        loc: Location
        captures: List of captured variables (Variable IR nodes)
        parameters: List of parameter variables (Variable IR nodes)
        body: Function body (IR node)

    Returns:
        Function IR variant
    """
    captures_array = EastArray(IRType, captures)
    params_array = EastArray(IRType, parameters)

    function_class = _struct_class_from_type(FunctionIR)
    function_struct = function_class.create(
        type=typ, location=loc, captures=captures_array, parameters=params_array, body=body
    )
    return EastVariant(IRType, Case("Function", function_struct))


def ir_new_ref(typ: EastType, loc: EastStruct, value: EastVariant) -> EastVariant:
    """Create a NewRef IR node (creates a reference cell).

    Args:
        typ: RefType for the reference cell
        loc: Location
        value: IR node for the initial value

    Returns:
        NewRef IR variant

    Example:
        >>> loc = location("test.east", 1, 1)
        >>> value_ir = ir_value(IntegerType, loc, 42)
        >>> ref_ir = ir_new_ref(RefType(IntegerType), loc, value_ir)
    """
    # Get the NewRef struct type from IRType
    # We need to extract it from the IRType variant cases
    newref_type = None
    for case in IRType.value:
        if case.name == "NewRef":
            newref_type = case.type
            break

    if newref_type is None:
        raise ValueError("NewRef case not found in IRType")

    newref_class = _struct_class_from_type(newref_type)
    newref_struct = newref_class.create(type=typ, location=loc, value=value)
    return EastVariant(IRType, Case("NewRef", newref_struct))


def ir_block(typ: EastType, loc: EastStruct, statements: list[EastVariant]) -> EastVariant:
    """Create a Block IR node.

    Args:
        typ: Type of the block (type of last statement)
        loc: Location
        statements: List of statement IR nodes

    Returns:
        Block IR variant
    """
    stmts_array = EastArray(IRType, statements)

    block_class = _struct_class_from_type(BlockIR)
    block_struct = block_class.create(type=typ, location=loc, statements=stmts_array)
    return EastVariant(IRType, Case("Block", block_struct))


def ir_ifelse(
    typ: EastType,
    loc: EastStruct,
    ifs: list[tuple[EastVariant, EastVariant]],
    else_body: EastVariant,
) -> EastVariant:
    """Create an IfElse IR node.

    Args:
        typ: Type of the if-else expression
        loc: Location
        ifs: List of (predicate, body) tuples
        else_body: Else branch body

    Returns:
        IfElse IR variant
    """
    # Create ifs array
    ifcase_class = _struct_class_from_type(IfCase)
    if_cases = []
    for predicate, body in ifs:
        if_cases.append(ifcase_class.create(predicate=predicate, body=body))
    ifs_array = EastArray(IfCase, if_cases)

    ifelse_class = _struct_class_from_type(IfElseIR)
    ifelse_struct = ifelse_class.create(type=typ, location=loc, ifs=ifs_array, else_body=else_body)
    return EastVariant(IRType, Case("IfElse", ifelse_struct))


def ir_while(
    typ: EastType, loc: EastStruct, predicate: EastVariant, label: EastStruct, body: EastVariant
) -> EastVariant:
    """Create a While IR node.

    Args:
        typ: Type of the while expression (usually Null)
        loc: Location
        predicate: Loop condition
        label: Loop label
        body: Loop body

    Returns:
        While IR variant
    """
    while_class = _struct_class_from_type(WhileIR)
    while_struct = while_class.create(
        type=typ, location=loc, predicate=predicate, label=label, body=body
    )
    return EastVariant(IRType, Case("While", while_struct))


def ir_trycatch(
    typ: EastType,
    loc: EastStruct,
    try_body: EastVariant,
    catch_body: EastVariant,
    message_var: EastVariant,
    stack_var: EastVariant,
    finally_body: EastVariant | None = None,
) -> EastVariant:
    """Create a TryCatch IR node.

    Args:
        typ: Return type (union of try and catch types)
        loc: Location
        try_body: IR for try block
        catch_body: IR for catch block
        message_var: Variable IR for error message (String)
        stack_var: Variable IR for stack trace (Array of location structs)
        finally_body: Optional IR for finally block (if None, creates dummy Value node)

    Returns:
        TryCatch IR variant

    Note:
        When finally_body is None, a dummy Value node with Null is created. This allows
        the compiler to detect trivial finally blocks and optimize them away at compile-time.
    """
    from east.types.primitives import Null
    from east.types.type_system import IRType, NullType

    # If no finally_body provided, create a dummy Value node (Null)
    # This allows compiler to detect and optimize away trivial finally blocks
    if finally_body is None:
        finally_body = ir_value(NullType, loc, Null())

    # Get TryCatch struct type from IRType variant
    # We need to extract the struct type for TryCatch case
    trycatch_cases = IRType.value  # Get variant cases
    trycatch_struct_type = None
    for case in trycatch_cases:
        if case.name == "TryCatch":
            trycatch_struct_type = case.type
            break

    if trycatch_struct_type is None:
        raise ValueError("TryCatch case not found in IRType")

    # Create struct class
    trycatch_class = _struct_class_from_type(trycatch_struct_type)

    # Build struct - finally_body is always present in the struct definition
    trycatch_struct = trycatch_class.create(
        type=typ,
        location=loc,
        try_body=try_body,
        catch_body=catch_body,
        message=message_var,
        stack=stack_var,
        finally_body=finally_body,
    )

    return EastVariant(IRType, Case("TryCatch", trycatch_struct))


__all__ = [
    "location",
    "ir_label",
    "literal_value",
    "ir_value",
    "ir_variable",
    "ir_builtin",
    "ir_function",
    "ir_block",
    "ir_ifelse",
    "ir_while",
    "ir_trycatch",
]
