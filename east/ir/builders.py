"""Helper functions for building IR nodes as East variants.

IR nodes are East values (variants), not Python dataclasses. These helper functions
make it easier to construct IR variants programmatically.
"""

from datetime import datetime
from typing import Any

from east.types.containers import EastArray
from east.types.primitives import null
from east.types.types import (
    BlockIRValue,
    BuiltinIRValue,
    EastType,
    FunctionIRValue,
    IfElseIRValue,
    IRLabel,
    IRNode,
    LiteralValueVariant,
    Location,
    NewRefIRValue,
    PlatformIRValue,
    TryCatchIRValue,
    ValueIRValue,
    VariableIRValue,
    WhileIRValue,
)


def location(filename: str, line: int, column: int) -> Location:
    """Create a Location struct.

    Args:
        filename: Source filename
        line: Line number
        column: Column number

    Returns:
        Location struct
    """
    return {"filename": filename, "line": line, "column": column}


def ir_label(name: str, loc: Location) -> IRLabel:
    """Create an IR label struct.

    Args:
        name: Label name
        loc: Location

    Returns:
        IRLabel struct
    """
    return {"name": name, "location": loc}


def literal_value(value: Any) -> LiteralValueVariant:
    """Create a LiteralValue variant from a Python value.

    Args:
        value: Python value (None, bool, int, float, str, bytes, or datetime)

    Returns:
        LiteralValue variant
    """
    if value is None or value is null:
        return {"type": "Null", "value": null}
    if isinstance(value, bool):
        return {"type": "Boolean", "value": value}
    if isinstance(value, int):
        return {"type": "Integer", "value": value}
    if isinstance(value, float):
        return {"type": "Float", "value": value}
    if isinstance(value, str):
        return {"type": "String", "value": value}
    if isinstance(value, bytes):
        return {"type": "Blob", "value": value}
    if isinstance(value, datetime):
        return {"type": "DateTime", "value": value}
    raise TypeError(f"Cannot convert {type(value)} to LiteralValue")


def ir_value(typ: EastType, loc: Location, value: Any) -> IRNode:
    """Create a Value IR node.

    Args:
        typ: East type of the value
        loc: Location
        value: The literal value (will be wrapped in LiteralValue variant)

    Returns:
        Value IR variant (plain dict)
    """
    lit_val = literal_value(value)
    value_struct: ValueIRValue = {
        "type": typ,
        "location": loc,
        "value": lit_val,
    }
    return {"type": "Value", "value": value_struct}


def ir_variable(
    typ: EastType, name: str, loc: Location, mutable: bool = False, captured: bool = False
) -> IRNode:
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
    var_struct: VariableIRValue = {
        "type": typ,
        "name": name,
        "location": loc,
        "mutable": mutable,
        "captured": captured,
    }
    return {"type": "Variable", "value": var_struct}


def ir_builtin(
    typ: EastType,
    loc: Location,
    builtin_name: str,
    type_parameters: list[EastType],
    arguments: list[IRNode],
) -> IRNode:
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
    from east.types.types import EastTypeType, IRType

    type_params_array = EastArray(EastTypeType, type_parameters)
    args_array = EastArray(IRType, arguments)

    builtin_struct: BuiltinIRValue = {
        "type": typ,
        "location": loc,
        "builtin": builtin_name,
        "type_parameters": type_params_array,
        "arguments": args_array,
    }
    return {"type": "Builtin", "value": builtin_struct}


def ir_platform(
    typ: EastType,
    loc: Location,
    platform_name: str,
    arguments: list[IRNode],
) -> IRNode:
    """Create a Platform IR node.

    Args:
        typ: Return type of the platform function
        loc: Location
        platform_name: Name of the platform function
        arguments: IR arguments

    Returns:
        Platform IR variant
    """
    from east.types.types import IRType

    args_array = EastArray(IRType, arguments)

    platform_struct: PlatformIRValue = {
        "type": typ,
        "location": loc,
        "name": platform_name,
        "arguments": args_array,
    }
    return {"type": "Platform", "value": platform_struct}


def ir_function(
    typ: EastType,
    loc: Location,
    captures: list[IRNode],
    parameters: list[IRNode],
    body: IRNode,
) -> IRNode:
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
    from east.types.types import IRType

    captures_array = EastArray(IRType, captures)
    params_array = EastArray(IRType, parameters)

    function_struct: FunctionIRValue = {
        "type": typ,
        "location": loc,
        "captures": captures_array,
        "parameters": params_array,
        "body": body,
    }
    return {"type": "Function", "value": function_struct}


def ir_new_ref(typ: EastType, loc: Location, value: IRNode) -> IRNode:
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
    newref_struct: NewRefIRValue = {
        "type": typ,
        "location": loc,
        "value": value,
    }
    return {"type": "NewRef", "value": newref_struct}


def ir_block(typ: EastType, loc: Location, statements: list[IRNode]) -> IRNode:
    """Create a Block IR node.

    Args:
        typ: Type of the block (type of last statement)
        loc: Location
        statements: List of statement IR nodes

    Returns:
        Block IR variant
    """
    from east.types.types import IRType

    stmts_array = EastArray(IRType, statements)

    block_struct: BlockIRValue = {
        "type": typ,
        "location": loc,
        "statements": stmts_array,
    }
    return {"type": "Block", "value": block_struct}


def ir_ifelse(
    typ: EastType,
    loc: Location,
    ifs: list[tuple[IRNode, IRNode]],
    else_body: IRNode,
) -> IRNode:
    """Create an IfElse IR node.

    Args:
        typ: Type of the if-else expression
        loc: Location
        ifs: List of (predicate, body) tuples
        else_body: Else branch body

    Returns:
        IfElse IR variant
    """
    from east.types.types import IfCase, IfCaseType

    # Create if cases as plain dicts
    if_cases: list[IfCase] = []
    for predicate, body in ifs:
        if_cases.append({"predicate": predicate, "body": body})

    ifs_array = EastArray(IfCaseType, if_cases)

    ifelse_struct: IfElseIRValue = {
        "type": typ,
        "location": loc,
        "ifs": ifs_array,
        "else_body": else_body,
    }
    return {"type": "IfElse", "value": ifelse_struct}


def ir_while(
    typ: EastType, loc: Location, predicate: IRNode, label: IRLabel, body: IRNode
) -> IRNode:
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
    while_struct: WhileIRValue = {
        "type": typ,
        "location": loc,
        "predicate": predicate,
        "label": label,
        "body": body,
    }
    return {"type": "While", "value": while_struct}


def ir_trycatch(
    typ: EastType,
    loc: Location,
    try_body: IRNode,
    catch_body: IRNode,
    message_var: IRNode,
    stack_var: IRNode,
    finally_body: IRNode | None = None,
) -> IRNode:
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
    from east.types.types import NullType

    # If no finally_body provided, create a dummy Value node (Null)
    # This allows compiler to detect and optimize away trivial finally blocks
    if finally_body is None:
        finally_body = ir_value(NullType, loc, Null())

    trycatch_struct: TryCatchIRValue = {
        "type": typ,
        "location": loc,
        "try_body": try_body,
        "catch_body": catch_body,
        "message": message_var,
        "stack": stack_var,
        "finally_body": finally_body,
    }

    return {"type": "TryCatch", "value": trycatch_struct}


__all__ = [
    "location",
    "ir_label",
    "literal_value",
    "ir_value",
    "ir_variable",
    "ir_builtin",
    "ir_platform",
    "ir_function",
    "ir_new_ref",
    "ir_block",
    "ir_ifelse",
    "ir_while",
    "ir_trycatch",
]
