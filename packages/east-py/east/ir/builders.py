"""Helper functions for building IR nodes as East variants.

IR nodes are East values (variants), not Python dataclasses. These helper functions
make it easier to construct IR variants programmatically.
"""

from datetime import datetime
from typing import Any

from east.types.ir import (
    IR,
    AsyncFunctionIRValue,
    BlockIRValue,
    BuiltinIRValue,
    CallAsyncIRValue,
    FunctionIRValue,
    IfCaseValue,
    IfElseIRValue,
    IRLabelValue,
    LocationValue,
    NewRefIRValue,
    PlatformIRValue,
    TryCatchIRValue,
    ValueIRValue,
    VariableIRValue,
    WhileIRValue,
)
from east.types.type_of_type import (
    EastTypeType,
    EastTypeValue,
    IfCaseType,
    IRType,
    LiteralValue,
)
from east.types.values import EastArray, EastVariant, east_null


def location(filename: str, line: int, column: int) -> LocationValue:
    """Create a Location struct.

    Args:
        filename: Source filename
        line: Line number
        column: Column number

    Returns:
        Location struct
    """
    return {"filename": filename, "line": line, "column": column}


def ir_label(name: str, loc: LocationValue) -> IRLabelValue:
    """Create an IR label struct.

    Args:
        name: Label name
        loc: Location

    Returns:
        IRLabel struct
    """
    return {"name": name, "location": loc}


def literal_value(value: Any) -> LiteralValue:
    """Create a LiteralValue variant from a Python value.

    Args:
        value: Python value (None, bool, int, float, str, bytes, or datetime)

    Returns:
        LiteralValue variant
    """
    if value is None or value is east_null:
        return EastVariant("Null", east_null)
    if isinstance(value, bool):
        return EastVariant("Boolean", value)
    if isinstance(value, int):
        return EastVariant("Integer", value)
    if isinstance(value, float):
        return EastVariant("Float", value)
    if isinstance(value, str):
        return EastVariant("String", value)
    if isinstance(value, bytes):
        return EastVariant("Blob", value)
    if isinstance(value, datetime):
        return EastVariant("DateTime", value)
    raise TypeError(f"Cannot convert {type(value)} to LiteralValue")


def ir_value(typ: EastTypeValue, loc: LocationValue, value: Any) -> IR:
    """Create a Value IR node.

    Args:
        typ: East type of the value
        loc: Location
        value: The literal value (will be wrapped in LiteralValue variant)

    Returns:
        Value IR variant
    """
    lit_val = literal_value(value)
    value_struct: ValueIRValue = {
        "type": typ,
        "location": loc,
        "value": lit_val,
    }
    return EastVariant("Value", value_struct)


def ir_variable(
    typ: EastTypeValue, name: str, loc: LocationValue, mutable: bool = False, captured: bool = False
) -> IR:
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
    return EastVariant("Variable", var_struct)


def ir_builtin(
    typ: EastTypeValue,
    loc: LocationValue,
    builtin_name: str,
    type_parameters: list[EastTypeValue],
    arguments: list[IR],
) -> IR:
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

    builtin_struct: BuiltinIRValue = {
        "type": typ,
        "location": loc,
        "builtin": builtin_name,
        "type_parameters": type_params_array,
        "arguments": args_array,
    }
    return EastVariant("Builtin", builtin_struct)


def ir_platform(
    typ: EastTypeValue,
    loc: LocationValue,
    platform_name: str,
    arguments: list[IR],
    async_: bool = False,
) -> IR:
    """Create a Platform IR node.

    Args:
        typ: Return type of the platform function
        loc: Location
        platform_name: Name of the platform function
        arguments: IR arguments
        async_: Whether this platform function is async

    Returns:
        Platform IR variant
    """
    args_array = EastArray(IRType, arguments)

    platform_struct: PlatformIRValue = {
        "type": typ,
        "location": loc,
        "name": platform_name,
        "arguments": args_array,
        "async": async_,
    }
    return EastVariant("Platform", platform_struct)


def ir_function(
    typ: EastTypeValue,
    loc: LocationValue,
    captures: list[IR],
    parameters: list[IR],
    body: IR,
) -> IR:
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

    function_struct: FunctionIRValue = {
        "type": typ,
        "location": loc,
        "captures": captures_array,
        "parameters": params_array,
        "body": body,
    }
    return EastVariant("Function", function_struct)


def ir_async_function(
    typ: EastTypeValue,
    loc: LocationValue,
    captures: list[IR],
    parameters: list[IR],
    body: IR,
) -> IR:
    """Create an AsyncFunction IR node.

    Args:
        typ: AsyncFunction type
        loc: Location
        captures: List of captured variables (Variable IR nodes)
        parameters: List of parameter variables (Variable IR nodes)
        body: Function body (IR node)

    Returns:
        AsyncFunction IR variant
    """
    captures_array = EastArray(IRType, captures)
    params_array = EastArray(IRType, parameters)

    function_struct: AsyncFunctionIRValue = {
        "type": typ,
        "location": loc,
        "captures": captures_array,
        "parameters": params_array,
        "body": body,
    }
    return EastVariant("AsyncFunction", function_struct)


def ir_call_async(
    typ: EastTypeValue,
    loc: LocationValue,
    function: IR,
    arguments: list[IR],
) -> IR:
    """Create a CallAsync IR node (calls an async function and awaits the result).

    Args:
        typ: Return type of the call
        loc: Location
        function: IR node for the function to call
        arguments: List of IR argument nodes

    Returns:
        CallAsync IR variant
    """
    args_array = EastArray(IRType, arguments)

    call_struct: CallAsyncIRValue = {
        "type": typ,
        "location": loc,
        "function": function,
        "arguments": args_array,
    }
    return EastVariant("CallAsync", call_struct)


def ir_new_ref(typ: EastTypeValue, loc: LocationValue, value: IR) -> IR:
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
    return EastVariant("NewRef", newref_struct)


def ir_block(typ: EastTypeValue, loc: LocationValue, statements: list[IR]) -> IR:
    """Create a Block IR node.

    Args:
        typ: Type of the block (type of last statement)
        loc: Location
        statements: List of statement IR nodes

    Returns:
        Block IR variant
    """
    stmts_array = EastArray(IRType, statements)

    block_struct: BlockIRValue = {
        "type": typ,
        "location": loc,
        "statements": stmts_array,
    }
    return EastVariant("Block", block_struct)


def ir_ifelse(
    typ: EastTypeValue,
    loc: LocationValue,
    ifs: list[tuple[IR, IR]],
    else_body: IR,
) -> IR:
    """Create an IfElse IR node.

    Args:
        typ: Type of the if-else expression
        loc: Location
        ifs: List of (predicate, body) tuples
        else_body: Else branch body

    Returns:
        IfElse IR variant
    """
    # Create if cases as plain dicts
    if_cases: list[IfCaseValue] = []
    for predicate, body in ifs:
        if_cases.append({"predicate": predicate, "body": body})

    ifs_array = EastArray(IfCaseType, if_cases)

    ifelse_struct: IfElseIRValue = {
        "type": typ,
        "location": loc,
        "ifs": ifs_array,
        "else_body": else_body,
    }
    return EastVariant("IfElse", ifelse_struct)


def ir_while(
    typ: EastTypeValue, loc: LocationValue, predicate: IR, label: IRLabelValue, body: IR
) -> IR:
    """Create a While IR node.

    Args:
        typ: Type of the while expression (usually EastNull)
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
    return EastVariant("While", while_struct)


def ir_trycatch(
    typ: EastTypeValue,
    loc: LocationValue,
    try_body: IR,
    catch_body: IR,
    message_var: IR,
    stack_var: IR,
    finally_body: IR | None = None,
) -> IR:
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
        When finally_body is None, a dummy Value node with EastNull is created. This allows
        the compiler to detect trivial finally blocks and optimize them away at compile-time.
    """
    from east.types.types import NullType
    from east.types.values import EastNull

    # If no finally_body provided, create a dummy Value node (EastNull)
    # This allows compiler to detect and optimize away trivial finally blocks
    if finally_body is None:
        finally_body = ir_value(NullType, loc, EastNull())

    trycatch_struct: TryCatchIRValue = {
        "type": typ,
        "location": loc,
        "try_body": try_body,
        "catch_body": catch_body,
        "message": message_var,
        "stack": stack_var,
        "finally_body": finally_body,
    }

    return EastVariant("TryCatch", trycatch_struct)


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
