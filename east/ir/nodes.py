"""East IR (Intermediate Representation) node definitions.

The IR is a tree structure representing East programs.
Each node is a variant with a specific structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from east.types.type_system import EastType


@dataclass(frozen=True)
class Location:
    """Source code location for error reporting.

    Attributes:
        file: Source file name
        line: Line number (1-indexed)
        column: Column number (1-indexed)
    """

    file: str
    line: int
    column: int

    def __repr__(self) -> str:
        """Return readable representation."""
        return f"{self.file}:{self.line}:{self.column}"


# IR Node Types
# Each node is represented as a dataclass with frozen=True for immutability


@dataclass(frozen=True)
class Value:
    """Literal value node.

    Attributes:
        value: The literal value
        value_type: Type of the value
        location: Source location
    """

    value: Any
    value_type: EastType
    location: Location


@dataclass(frozen=True)
class Variable:
    """Variable reference node.

    Attributes:
        name: Variable name
        location: Source location
    """

    name: str
    location: Location


@dataclass(frozen=True)
class Block:
    """Block of statements node.

    Executes statements in sequence, returns last value.

    Attributes:
        statements: List of IR nodes
        location: Source location
    """

    statements: tuple[IR, ...]
    location: Location


@dataclass(frozen=True)
class IfElse:
    """If-else conditional node.

    Attributes:
        condition: Condition IR node
        then_branch: IR node to execute if condition is true
        else_branch: IR node to execute if condition is false
        location: Source location
    """

    condition: IR
    then_branch: IR
    else_branch: IR
    location: Location


@dataclass(frozen=True)
class While:
    """While loop node.

    Attributes:
        label: Optional label for break/continue
        condition: Condition IR node
        body: Loop body IR node
        location: Source location
    """

    label: str | None
    condition: IR
    body: IR
    location: Location


@dataclass(frozen=True)
class Break:
    """Break statement node.

    Attributes:
        label: Optional loop label to break from
        location: Source location
    """

    label: str | None
    location: Location


@dataclass(frozen=True)
class Continue:
    """Continue statement node.

    Attributes:
        label: Optional loop label to continue
        location: Source location
    """

    label: str | None
    location: Location


@dataclass(frozen=True)
class Return:
    """Return statement node.

    Attributes:
        value: Value to return
        location: Source location
    """

    value: IR
    location: Location


@dataclass(frozen=True)
class Let:
    """Variable declaration node.

    Attributes:
        name: Variable name
        mutable: Whether variable is mutable
        value: Initial value IR node
        location: Source location
    """

    name: str
    mutable: bool
    value: IR
    location: Location


@dataclass(frozen=True)
class Assign:
    """Variable assignment node.

    Attributes:
        name: Variable name
        value: New value IR node
        location: Source location
    """

    name: str
    value: IR
    location: Location


@dataclass(frozen=True)
class NewArray:
    """Array construction node.

    Attributes:
        element_type: Type of array elements
        elements: List of element IR nodes
        location: Source location
    """

    element_type: EastType
    elements: tuple[IR, ...]
    location: Location


@dataclass(frozen=True)
class NewSet:
    """Set construction node.

    Attributes:
        element_type: Type of set elements
        elements: List of element IR nodes
        location: Source location
    """

    element_type: EastType
    elements: tuple[IR, ...]
    location: Location


@dataclass(frozen=True)
class NewDict:
    """Dict construction node.

    Attributes:
        key_type: Type of dict keys
        value_type: Type of dict values
        entries: List of (key, value) IR node pairs
        location: Source location
    """

    key_type: EastType
    value_type: EastType
    entries: tuple[tuple[IR, IR], ...]
    location: Location


@dataclass(frozen=True)
class ForArray:
    """For loop over array node.

    Attributes:
        label: Optional loop label
        index_var: Index variable name
        element_var: Element variable name
        array: Array IR node
        body: Loop body IR node
        location: Source location
    """

    label: str | None
    index_var: str
    element_var: str
    array: IR
    body: IR
    location: Location


@dataclass(frozen=True)
class ForSet:
    """For loop over set node.

    Attributes:
        label: Optional loop label
        element_var: Element variable name
        set_expr: Set IR node
        body: Loop body IR node
        location: Source location
    """

    label: str | None
    element_var: str
    set_expr: IR
    body: IR
    location: Location


@dataclass(frozen=True)
class ForDict:
    """For loop over dict node.

    Attributes:
        label: Optional loop label
        key_var: Key variable name
        value_var: Value variable name
        dict_expr: Dict IR node
        body: Loop body IR node
        location: Source location
    """

    label: str | None
    key_var: str
    value_var: str
    dict_expr: IR
    body: IR
    location: Location


@dataclass(frozen=True)
class StructNode:
    """Struct construction node.

    Attributes:
        struct_type: Type of struct
        fields: Dict of field name to IR node
        location: Source location
    """

    struct_type: EastType
    fields: tuple[tuple[str, IR], ...]
    location: Location


@dataclass(frozen=True)
class GetField:
    """Get struct field node.

    Attributes:
        struct: Struct IR node
        field_name: Name of field to get
        location: Source location
    """

    struct: IR
    field_name: str
    location: Location


@dataclass(frozen=True)
class VariantNode:
    """Variant construction node.

    Attributes:
        variant_type: Type of variant
        tag: Variant case tag
        value: Value IR node
        location: Source location
    """

    variant_type: EastType
    tag: str
    value: IR
    location: Location


@dataclass(frozen=True)
class MatchCase:
    """Match case for pattern matching.

    Attributes:
        tag: Variant tag to match
        var_name: Variable name to bind variant value
        body: IR node to execute if matched
    """

    tag: str
    var_name: str
    body: IR


@dataclass(frozen=True)
class Match:
    """Pattern matching node.

    Attributes:
        value: Variant value to match
        cases: List of match cases
        location: Source location
    """

    value: IR
    cases: tuple[MatchCase, ...]
    location: Location


@dataclass(frozen=True)
class Function:
    """Function definition node.

    Attributes:
        param_names: List of parameter names
        param_types: List of parameter types
        return_type: Return type
        body: Function body IR node
        location: Source location
    """

    param_names: tuple[str, ...]
    param_types: tuple[EastType, ...]
    return_type: EastType
    body: IR
    location: Location


@dataclass(frozen=True)
class Call:
    """Function call node.

    Attributes:
        function: Function IR node
        arguments: List of argument IR nodes
        location: Source location
    """

    function: IR
    arguments: tuple[IR, ...]
    location: Location


@dataclass(frozen=True)
class Platform:
    """Platform function call node.

    Attributes:
        platform_name: Name of platform
        function_name: Name of platform function
        arguments: List of argument IR nodes
        location: Source location
    """

    platform_name: str
    function_name: str
    arguments: tuple[IR, ...]
    location: Location


@dataclass(frozen=True)
class Builtin:
    """Builtin function call node.

    Attributes:
        builtin_name: Name of builtin function
        arguments: List of argument IR nodes
        location: Source location
    """

    builtin_name: str
    arguments: tuple[IR, ...]
    location: Location


@dataclass(frozen=True)
class Error:
    """Error throw node.

    Attributes:
        message: Error message IR node
        location: Source location
    """

    message: IR
    location: Location


@dataclass(frozen=True)
class TryCatch:
    """Try-catch error handling node.

    Attributes:
        try_body: IR node to try
        error_var: Variable name to bind error message
        catch_body: IR node to execute on error
        location: Source location
    """

    try_body: IR
    error_var: str
    catch_body: IR
    location: Location


@dataclass(frozen=True)
class As:
    """Type assertion node.

    Attributes:
        value: Value IR node
        target_type: Type to assert
        location: Source location
    """

    value: IR
    target_type: EastType
    location: Location


@dataclass(frozen=True)
class UnwrapRecursive:
    """Unwrap recursive type node.

    Attributes:
        value: Recursive value IR node
        location: Source location
    """

    value: IR
    location: Location


@dataclass(frozen=True)
class WrapRecursive:
    """Wrap value in recursive type node.

    Attributes:
        value: Value IR node
        recursive_type: Recursive type to wrap into
        location: Source location
    """

    value: IR
    recursive_type: EastType
    location: Location


# Union type for all IR nodes
IR = (
    Value
    | Variable
    | Block
    | IfElse
    | While
    | Break
    | Continue
    | Return
    | Let
    | Assign
    | NewArray
    | NewSet
    | NewDict
    | ForArray
    | ForSet
    | ForDict
    | StructNode
    | GetField
    | VariantNode
    | Match
    | Function
    | Call
    | Platform
    | Builtin
    | Error
    | TryCatch
    | As
    | UnwrapRecursive
    | WrapRecursive
)


__all__: list[str] = [
    "Location",
    "Value",
    "Variable",
    "Block",
    "IfElse",
    "While",
    "Break",
    "Continue",
    "Return",
    "Let",
    "Assign",
    "NewArray",
    "NewSet",
    "NewDict",
    "ForArray",
    "ForSet",
    "ForDict",
    "StructNode",
    "GetField",
    "VariantNode",
    "MatchCase",
    "Match",
    "Function",
    "Call",
    "Platform",
    "Builtin",
    "Error",
    "TryCatch",
    "As",
    "UnwrapRecursive",
    "WrapRecursive",
    "IR",
]
