#
# Copyright (c) 2025 Elara AI Pty Ltd
# Licensed under the Business Source License 1.1. See LICENSE.md for details.
#
"""East Intermediate Representation (IR) types.

IR is the intermediate representation for East code. It has been processed
from AST and checked for type safety and variable resolution. The code is
ready to be serialized, evaluated or compiled.

IR is homoiconic - IR nodes are East values (variants) that can be serialized
with standard East value serialization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeAlias, TypedDict

from east.types.values import EastVariant

if TYPE_CHECKING:
    from east.types.type_of_type import EastTypeValue, LiteralValue


# =============================================================================
# Location type
# =============================================================================


class LocationValue(TypedDict):
    """Source code location."""

    filename: str
    line: int
    column: int


class IRLabelValue(TypedDict):
    """IR loop label."""

    name: str
    location: LocationValue


# =============================================================================
# IR Node Value Types
# =============================================================================

# Forward reference for recursive IR types
IR: TypeAlias = EastVariant[Any]


class ErrorIRValue(TypedDict):
    """Value inside Error IR variant."""

    type: EastTypeValue
    location: LocationValue
    message: IR


class TryCatchIRValue(TypedDict):
    """Value inside TryCatch IR variant."""

    type: EastTypeValue
    location: LocationValue
    try_body: IR
    catch_body: IR
    message: IR  # VariableIR
    stack: IR  # VariableIR
    finally_body: IR


class ValueIRValue(TypedDict):
    """Value inside Value IR variant (literal value)."""

    type: EastTypeValue
    location: LocationValue
    value: LiteralValue


class VariableIRValue(TypedDict):
    """Value inside Variable IR variant."""

    type: EastTypeValue
    name: str
    location: LocationValue
    mutable: bool
    captured: bool


class LetIRValue(TypedDict):
    """Value inside Let IR variant."""

    type: EastTypeValue
    location: LocationValue
    variable: IR  # VariableIR
    value: IR


class AssignIRValue(TypedDict):
    """Value inside Assign IR variant."""

    type: EastTypeValue
    location: LocationValue
    variable: IR  # VariableIR
    value: IR


class AsIRValue(TypedDict):
    """Value inside As IR variant (type cast)."""

    type: EastTypeValue
    location: LocationValue
    value: IR


class FunctionIRValue(TypedDict):
    """Value inside Function IR variant."""

    type: EastTypeValue
    location: LocationValue
    captures: list[IR]  # VariableIR[]
    parameters: list[IR]  # VariableIR[]
    body: IR


class AsyncFunctionIRValue(TypedDict):
    """Value inside AsyncFunction IR variant."""

    type: EastTypeValue
    location: LocationValue
    captures: list[IR]  # VariableIR[]
    parameters: list[IR]  # VariableIR[]
    body: IR


class CallIRValue(TypedDict):
    """Value inside Call IR variant."""

    type: EastTypeValue
    location: LocationValue
    function: IR
    arguments: list[IR]


class CallAsyncIRValue(TypedDict):
    """Value inside CallAsync IR variant."""

    type: EastTypeValue
    location: LocationValue
    function: IR
    arguments: list[IR]


class NewRefIRValue(TypedDict):
    """Value inside NewRef IR variant."""

    type: EastTypeValue
    location: LocationValue
    value: IR


class NewArrayIRValue(TypedDict):
    """Value inside NewArray IR variant."""

    type: EastTypeValue
    location: LocationValue
    values: list[IR]


class NewSetIRValue(TypedDict):
    """Value inside NewSet IR variant."""

    type: EastTypeValue
    location: LocationValue
    values: list[IR]


class DictEntryValue(TypedDict):
    """Dict entry for NewDict IR."""

    key: IR
    value: IR


class NewDictIRValue(TypedDict):
    """Value inside NewDict IR variant."""

    type: EastTypeValue
    location: LocationValue
    values: list[DictEntryValue]


class StructFieldValue(TypedDict):
    """Struct field for Struct IR."""

    name: str
    value: IR


class StructIRValue(TypedDict):
    """Value inside Struct IR variant."""

    type: EastTypeValue
    location: LocationValue
    fields: list[StructFieldValue]


class GetFieldIRValue(TypedDict):
    """Value inside GetField IR variant."""

    type: EastTypeValue
    location: LocationValue
    field: str
    struct: IR


class VariantIRValue(TypedDict):
    """Value inside Variant IR variant."""

    type: EastTypeValue
    location: LocationValue
    case: str
    value: IR


class BlockIRValue(TypedDict):
    """Value inside Block IR variant."""

    type: EastTypeValue
    location: LocationValue
    statements: list[IR]


class IfCaseValue(TypedDict):
    """If case for IfElse IR."""

    predicate: IR
    body: IR


class IfElseIRValue(TypedDict):
    """Value inside IfElse IR variant."""

    type: EastTypeValue
    location: LocationValue
    ifs: list[IfCaseValue]
    else_body: IR


class MatchCaseValue(TypedDict):
    """Match case for Match IR."""

    case: str
    variable: IR  # VariableIR
    body: IR


class MatchIRValue(TypedDict):
    """Value inside Match IR variant."""

    type: EastTypeValue
    location: LocationValue
    variant: IR
    cases: list[MatchCaseValue]


class UnwrapRecursiveIRValue(TypedDict):
    """Value inside UnwrapRecursive IR variant."""

    type: EastTypeValue
    location: LocationValue
    value: IR


class WrapRecursiveIRValue(TypedDict):
    """Value inside WrapRecursive IR variant."""

    type: EastTypeValue
    location: LocationValue
    value: IR


class WhileIRValue(TypedDict):
    """Value inside While IR variant."""

    type: EastTypeValue
    location: LocationValue
    predicate: IR
    label: IRLabelValue
    body: IR


class ForArrayIRValue(TypedDict):
    """Value inside ForArray IR variant."""

    type: EastTypeValue
    location: LocationValue
    array: IR
    label: IRLabelValue
    key: IR  # VariableIR
    value: IR  # VariableIR
    body: IR


class ForSetIRValue(TypedDict):
    """Value inside ForSet IR variant."""

    type: EastTypeValue
    location: LocationValue
    set: IR
    label: IRLabelValue
    key: IR  # VariableIR
    body: IR


class ForDictIRValue(TypedDict):
    """Value inside ForDict IR variant."""

    type: EastTypeValue
    location: LocationValue
    dict: IR
    label: IRLabelValue
    key: IR  # VariableIR
    value: IR  # VariableIR
    body: IR


class ReturnIRValue(TypedDict):
    """Value inside Return IR variant."""

    type: EastTypeValue
    location: LocationValue
    value: IR


class ContinueIRValue(TypedDict):
    """Value inside Continue IR variant."""

    type: EastTypeValue
    location: LocationValue
    label: IRLabelValue


class BreakIRValue(TypedDict):
    """Value inside Break IR variant."""

    type: EastTypeValue
    location: LocationValue
    label: IRLabelValue


class BuiltinIRValue(TypedDict):
    """Value inside Builtin IR variant."""

    type: EastTypeValue
    location: LocationValue
    builtin: str
    type_parameters: list[EastTypeValue]
    arguments: list[IR]


class PlatformIRValue(TypedDict):
    """Value inside Platform IR variant."""

    type: EastTypeValue
    location: LocationValue
    name: str
    arguments: list[IR]
    async_: bool  # Named async_ to avoid keyword conflict; serialized as "async"


# =============================================================================
# IR Type Aliases - Generic EastVariant types
# =============================================================================

ErrorIR: TypeAlias = EastVariant[ErrorIRValue]
TryCatchIR: TypeAlias = EastVariant[TryCatchIRValue]
ValueIR: TypeAlias = EastVariant[ValueIRValue]
VariableIR: TypeAlias = EastVariant[VariableIRValue]
LetIR: TypeAlias = EastVariant[LetIRValue]
AssignIR: TypeAlias = EastVariant[AssignIRValue]
AsIR: TypeAlias = EastVariant[AsIRValue]
FunctionIR: TypeAlias = EastVariant[FunctionIRValue]
AsyncFunctionIR: TypeAlias = EastVariant[AsyncFunctionIRValue]
CallIR: TypeAlias = EastVariant[CallIRValue]
CallAsyncIR: TypeAlias = EastVariant[CallAsyncIRValue]
NewRefIR: TypeAlias = EastVariant[NewRefIRValue]
NewArrayIR: TypeAlias = EastVariant[NewArrayIRValue]
NewSetIR: TypeAlias = EastVariant[NewSetIRValue]
NewDictIR: TypeAlias = EastVariant[NewDictIRValue]
StructIR: TypeAlias = EastVariant[StructIRValue]
GetFieldIR: TypeAlias = EastVariant[GetFieldIRValue]
VariantIR: TypeAlias = EastVariant[VariantIRValue]
BlockIR: TypeAlias = EastVariant[BlockIRValue]
IfElseIR: TypeAlias = EastVariant[IfElseIRValue]
MatchIR: TypeAlias = EastVariant[MatchIRValue]
UnwrapRecursiveIR: TypeAlias = EastVariant[UnwrapRecursiveIRValue]
WrapRecursiveIR: TypeAlias = EastVariant[WrapRecursiveIRValue]
WhileIR: TypeAlias = EastVariant[WhileIRValue]
ForArrayIR: TypeAlias = EastVariant[ForArrayIRValue]
ForSetIR: TypeAlias = EastVariant[ForSetIRValue]
ForDictIR: TypeAlias = EastVariant[ForDictIRValue]
ReturnIR: TypeAlias = EastVariant[ReturnIRValue]
ContinueIR: TypeAlias = EastVariant[ContinueIRValue]
BreakIR: TypeAlias = EastVariant[BreakIRValue]
BuiltinIR: TypeAlias = EastVariant[BuiltinIRValue]
PlatformIR: TypeAlias = EastVariant[PlatformIRValue]


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Location types
    "LocationValue",
    "IRLabelValue",
    # IR value TypedDicts
    "ErrorIRValue",
    "TryCatchIRValue",
    "ValueIRValue",
    "VariableIRValue",
    "LetIRValue",
    "AssignIRValue",
    "AsIRValue",
    "FunctionIRValue",
    "AsyncFunctionIRValue",
    "CallIRValue",
    "CallAsyncIRValue",
    "NewRefIRValue",
    "NewArrayIRValue",
    "NewSetIRValue",
    "DictEntryValue",
    "NewDictIRValue",
    "StructFieldValue",
    "StructIRValue",
    "GetFieldIRValue",
    "VariantIRValue",
    "BlockIRValue",
    "IfCaseValue",
    "IfElseIRValue",
    "MatchCaseValue",
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
    "BuiltinIRValue",
    "PlatformIRValue",
    # IR type aliases
    "IR",
    "ErrorIR",
    "TryCatchIR",
    "ValueIR",
    "VariableIR",
    "LetIR",
    "AssignIR",
    "AsIR",
    "FunctionIR",
    "AsyncFunctionIR",
    "CallIR",
    "CallAsyncIR",
    "NewRefIR",
    "NewArrayIR",
    "NewSetIR",
    "NewDictIR",
    "StructIR",
    "GetFieldIR",
    "VariantIR",
    "BlockIR",
    "IfElseIR",
    "MatchIR",
    "UnwrapRecursiveIR",
    "WrapRecursiveIR",
    "WhileIR",
    "ForArrayIR",
    "ForSetIR",
    "ForDictIR",
    "ReturnIR",
    "ContinueIR",
    "BreakIR",
    "BuiltinIR",
    "PlatformIR",
]
