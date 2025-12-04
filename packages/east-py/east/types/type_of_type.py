"""Type-of-type definitions for East's homoiconic type system.

This module defines the meta-types that represent East types as East values:
- LiteralValueType: The type of primitive literal values in IR
- LiteralValue: Python type alias for literal value variants
- EastTypeType: The recursive type that represents all East types
- EastTypeValue: Python type alias for serialized East type values

These enable types to be serialized, transmitted, and reflected upon within East.
"""

from __future__ import annotations

from datetime import datetime
from typing import TypeAlias

# Import type constructors - these will be updated when types.py is cleaned up
from east.types.types import (
    ArrayType,
    BlobType,
    BooleanType,
    DateTimeType,
    FloatType,
    IntegerType,
    NullType,
    StringType,
    StructType,
    VariantType,
    recursive_type,
)
from east.types.values import EastBlob, EastVariant

# =============================================================================
# LiteralValueType - The type of primitive literal values in IR
# =============================================================================

# Used to represent the values in ValueIR nodes
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
# LiteralValue - Python type for literal value variants
# =============================================================================

# The Python type of literal values in IR
# Each case is an EastVariant with the case name and Python value type
LiteralValue: TypeAlias = (
    EastVariant[None]  # Null
    | EastVariant[bool]  # Boolean
    | EastVariant[int]  # Integer
    | EastVariant[float]  # Float
    | EastVariant[str]  # String
    | EastVariant[datetime]  # DateTime
    | EastVariant[EastBlob]  # Blob
)


# =============================================================================
# EastTypeType - The type of East types (meta-type)
# =============================================================================

# The type of East values, represented as an EastType.
# This format is used for serialization of types, IR, etc.
# It also opens the door to type reflection and meta-programming within East.
EastTypeType = recursive_type(
    lambda type_ref: VariantType(
        [
            ("Never", NullType),
            ("Null", NullType),
            ("Boolean", NullType),
            ("Integer", NullType),
            ("Float", NullType),
            ("String", NullType),
            ("DateTime", NullType),
            ("Blob", NullType),
            ("Ref", type_ref),
            ("Array", type_ref),
            ("Set", type_ref),
            ("Dict", StructType([("key", type_ref), ("value", type_ref)])),
            ("Struct", ArrayType(StructType([("name", StringType), ("type", type_ref)]))),
            ("Variant", ArrayType(StructType([("name", StringType), ("type", type_ref)]))),
            ("Recursive", IntegerType),
            (
                "Function",
                StructType(
                    [
                        ("inputs", ArrayType(type_ref)),
                        ("output", type_ref),
                    ]
                ),
            ),
            (
                "AsyncFunction",
                StructType(
                    [
                        ("inputs", ArrayType(type_ref)),
                        ("output", type_ref),
                    ]
                ),
            ),
        ]
    )
)


# =============================================================================
# EastTypeValue - Python type for serialized East types
# =============================================================================

# A serializable representation of East types.
# This is what EastType values look like when serialized as East values.
EastTypeValue: TypeAlias = EastVariant


# =============================================================================
# IRType - The type of IR nodes (meta-type)
# =============================================================================

# The East type that represents IR nodes.
# IR nodes are homoiconic - they are East values themselves.
# This is a recursive type because IR nodes can contain other IR nodes.

# Location struct type used in IR
LocationType = StructType(
    [
        ("filename", StringType),
        ("line", IntegerType),
        ("column", IntegerType),
    ]
)

# IRLabel struct type used in While/For loops
IRLabelType = StructType(
    [
        ("name", StringType),
        ("location", LocationType),
    ]
)

# IfCase struct type used in IfElse IR
IfCaseType = recursive_type(
    lambda ir_ref: StructType(
        [
            ("predicate", ir_ref),
            ("body", ir_ref),
        ]
    )
)

# MatchCase struct type used in Match IR
MatchCaseType = recursive_type(
    lambda ir_ref: StructType(
        [
            ("case", StringType),
            ("variable", ir_ref),
            ("body", ir_ref),
        ]
    )
)

# DictEntry struct type used in NewDict IR
DictEntryType = recursive_type(
    lambda ir_ref: StructType(
        [
            ("key", ir_ref),
            ("value", ir_ref),
        ]
    )
)

# StructField struct type used in Struct IR
StructFieldIRType = recursive_type(
    lambda ir_ref: StructType(
        [
            ("name", StringType),
            ("value", ir_ref),
        ]
    )
)

# The full IR type - all possible IR node variants
IRType = recursive_type(
    lambda ir_ref: VariantType(
        [
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
            (
                "Variable",
                StructType(
                    [
                        ("type", EastTypeType),
                        ("name", StringType),
                        ("location", LocationType),
                        ("mutable", BooleanType),
                        ("captured", BooleanType),
                    ]
                ),
            ),
            (
                "Let",
                StructType(
                    [
                        ("type", EastTypeType),
                        ("location", LocationType),
                        ("variable", ir_ref),
                        ("value", ir_ref),
                    ]
                ),
            ),
            (
                "Assign",
                StructType(
                    [
                        ("type", EastTypeType),
                        ("location", LocationType),
                        ("variable", ir_ref),
                        ("value", ir_ref),
                    ]
                ),
            ),
            (
                "As",
                StructType(
                    [
                        ("type", EastTypeType),
                        ("location", LocationType),
                        ("value", ir_ref),
                    ]
                ),
            ),
            (
                "Function",
                StructType(
                    [
                        ("type", EastTypeType),
                        ("location", LocationType),
                        ("captures", ArrayType(ir_ref)),
                        ("parameters", ArrayType(ir_ref)),
                        ("body", ir_ref),
                    ]
                ),
            ),
            (
                "AsyncFunction",
                StructType(
                    [
                        ("type", EastTypeType),
                        ("location", LocationType),
                        ("captures", ArrayType(ir_ref)),
                        ("parameters", ArrayType(ir_ref)),
                        ("body", ir_ref),
                    ]
                ),
            ),
            (
                "Call",
                StructType(
                    [
                        ("type", EastTypeType),
                        ("location", LocationType),
                        ("function", ir_ref),
                        ("arguments", ArrayType(ir_ref)),
                    ]
                ),
            ),
            (
                "CallAsync",
                StructType(
                    [
                        ("type", EastTypeType),
                        ("location", LocationType),
                        ("function", ir_ref),
                        ("arguments", ArrayType(ir_ref)),
                    ]
                ),
            ),
            (
                "NewRef",
                StructType(
                    [
                        ("type", EastTypeType),
                        ("location", LocationType),
                        ("value", ir_ref),
                    ]
                ),
            ),
            (
                "NewArray",
                StructType(
                    [
                        ("type", EastTypeType),
                        ("location", LocationType),
                        ("values", ArrayType(ir_ref)),
                    ]
                ),
            ),
            (
                "NewSet",
                StructType(
                    [
                        ("type", EastTypeType),
                        ("location", LocationType),
                        ("values", ArrayType(ir_ref)),
                    ]
                ),
            ),
            (
                "NewDict",
                StructType(
                    [
                        ("type", EastTypeType),
                        ("location", LocationType),
                        ("values", ArrayType(StructType([("key", ir_ref), ("value", ir_ref)]))),
                    ]
                ),
            ),
            (
                "Struct",
                StructType(
                    [
                        ("type", EastTypeType),
                        ("location", LocationType),
                        (
                            "fields",
                            ArrayType(StructType([("name", StringType), ("value", ir_ref)])),
                        ),
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
                        ("struct", ir_ref),
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
                        ("value", ir_ref),
                    ]
                ),
            ),
            (
                "Block",
                StructType(
                    [
                        ("type", EastTypeType),
                        ("location", LocationType),
                        ("statements", ArrayType(ir_ref)),
                    ]
                ),
            ),
            (
                "IfElse",
                StructType(
                    [
                        ("type", EastTypeType),
                        ("location", LocationType),
                        ("ifs", ArrayType(StructType([("predicate", ir_ref), ("body", ir_ref)]))),
                        ("else_body", ir_ref),
                    ]
                ),
            ),
            (
                "Match",
                StructType(
                    [
                        ("type", EastTypeType),
                        ("location", LocationType),
                        ("variant", ir_ref),
                        (
                            "cases",
                            ArrayType(
                                StructType(
                                    [("case", StringType), ("variable", ir_ref), ("body", ir_ref)]
                                )
                            ),
                        ),
                    ]
                ),
            ),
            (
                "UnwrapRecursive",
                StructType(
                    [
                        ("type", EastTypeType),
                        ("location", LocationType),
                        ("value", ir_ref),
                    ]
                ),
            ),
            (
                "WrapRecursive",
                StructType(
                    [
                        ("type", EastTypeType),
                        ("location", LocationType),
                        ("value", ir_ref),
                    ]
                ),
            ),
            (
                "While",
                StructType(
                    [
                        ("type", EastTypeType),
                        ("location", LocationType),
                        ("predicate", ir_ref),
                        ("label", IRLabelType),
                        ("body", ir_ref),
                    ]
                ),
            ),
            (
                "ForArray",
                StructType(
                    [
                        ("type", EastTypeType),
                        ("location", LocationType),
                        ("array", ir_ref),
                        ("label", IRLabelType),
                        ("key", ir_ref),
                        ("value", ir_ref),
                        ("body", ir_ref),
                    ]
                ),
            ),
            (
                "ForSet",
                StructType(
                    [
                        ("type", EastTypeType),
                        ("location", LocationType),
                        ("set", ir_ref),
                        ("label", IRLabelType),
                        ("key", ir_ref),
                        ("body", ir_ref),
                    ]
                ),
            ),
            (
                "ForDict",
                StructType(
                    [
                        ("type", EastTypeType),
                        ("location", LocationType),
                        ("dict", ir_ref),
                        ("label", IRLabelType),
                        ("key", ir_ref),
                        ("value", ir_ref),
                        ("body", ir_ref),
                    ]
                ),
            ),
            (
                "Return",
                StructType(
                    [
                        ("type", EastTypeType),
                        ("location", LocationType),
                        ("value", ir_ref),
                    ]
                ),
            ),
            (
                "Continue",
                StructType(
                    [
                        ("type", EastTypeType),
                        ("location", LocationType),
                        ("label", IRLabelType),
                    ]
                ),
            ),
            (
                "Break",
                StructType(
                    [
                        ("type", EastTypeType),
                        ("location", LocationType),
                        ("label", IRLabelType),
                    ]
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
                        ("arguments", ArrayType(ir_ref)),
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
                        ("arguments", ArrayType(ir_ref)),
                        ("async", BooleanType),
                    ]
                ),
            ),
            (
                "Error",
                StructType(
                    [
                        ("type", EastTypeType),
                        ("location", LocationType),
                        ("message", ir_ref),
                    ]
                ),
            ),
            (
                "TryCatch",
                StructType(
                    [
                        ("type", EastTypeType),
                        ("location", LocationType),
                        ("try_body", ir_ref),
                        ("catch_body", ir_ref),
                        ("message", ir_ref),
                        ("stack", ir_ref),
                        ("finally_body", ir_ref),
                    ]
                ),
            ),
        ]
    )
)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "LiteralValueType",
    "LiteralValue",
    "EastTypeType",
    "EastTypeValue",
    "LocationType",
    "IRLabelType",
    "IfCaseType",
    "MatchCaseType",
    "DictEntryType",
    "StructFieldIRType",
    "IRType",
]
