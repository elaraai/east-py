"""Format type definitions for East Python IO.

Provides East type definitions for CSV, XLSX, and XML operations.
"""

from east.types.types import (
    ArrayType,
    BlobType,
    BooleanType,
    DateTimeType,
    DictType,
    FloatType,
    IntegerType,
    NullType,
    OptionType,
    RecursiveType,
    StringType,
    StructType,
    VariantType,
)

# LiteralValueType - represents any primitive value
# Matches TypeScript's LiteralValueType
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

# CSV Column type hint - specifies expected type for a column
CsvColumnType = VariantType(
    [
        ("Null", NullType),
        ("Boolean", NullType),
        ("Integer", NullType),
        ("Float", NullType),
        ("String", NullType),
        ("DateTime", NullType),
        ("Blob", NullType),
    ]
)

# CSV Types
CsvRowType = DictType(StringType, LiteralValueType)
CsvDataType = ArrayType(CsvRowType)

CsvParseConfigType = StructType(
    [
        ("columns", OptionType(DictType(StringType, CsvColumnType))),
        ("delimiter", OptionType(StringType)),
        ("quoteChar", OptionType(StringType)),
        ("escapeChar", OptionType(StringType)),
        ("newline", OptionType(StringType)),
        ("hasHeader", BooleanType),
        ("nullString", OptionType(StringType)),
        ("skipEmptyLines", BooleanType),
        ("trimFields", BooleanType),
    ]
)

CsvSerializeConfigType = StructType(
    [
        ("delimiter", StringType),
        ("quoteChar", StringType),
        ("escapeChar", StringType),
        ("newline", StringType),
        ("includeHeader", BooleanType),
        ("nullString", StringType),
        ("alwaysQuote", BooleanType),
    ]
)

# XLSX Types
XlsxCellType = LiteralValueType
XlsxRowType = ArrayType(XlsxCellType)
XlsxSheetType = ArrayType(XlsxRowType)

XlsxReadOptionsType = StructType(
    [
        ("sheetName", OptionType(StringType)),
    ]
)

XlsxWriteOptionsType = StructType(
    [
        ("sheetName", OptionType(StringType)),
    ]
)

XlsxSheetInfoType = StructType(
    [
        ("name", StringType),
        ("rowCount", IntegerType),
        ("columnCount", IntegerType),
    ]
)

XlsxInfoType = StructType(
    [
        ("sheets", ArrayType(XlsxSheetInfoType)),
    ]
)

# XML Types - using RecursiveType for nested elements
XmlNodeType = RecursiveType(
    lambda self: StructType(
        [
            ("tag", StringType),
            ("attributes", DictType(StringType, StringType)),
            (
                "children",
                ArrayType(
                    VariantType(
                        [
                            ("TEXT", StringType),
                            ("ELEMENT", self),
                        ]
                    )
                ),
            ),
        ]
    )
)

XmlParseConfigType = StructType(
    [
        ("preserveWhitespace", BooleanType),
        ("decodeEntities", BooleanType),
    ]
)

XmlSerializeConfigType = StructType(
    [
        ("indent", OptionType(StringType)),
        ("includeXmlDeclaration", BooleanType),
        ("encodeEntities", BooleanType),
        ("selfClosingTags", BooleanType),
    ]
)

__all__ = [
    "LiteralValueType",
    "CsvColumnType",
    "CsvRowType",
    "CsvDataType",
    "CsvParseConfigType",
    "CsvSerializeConfigType",
    "XlsxCellType",
    "XlsxRowType",
    "XlsxSheetType",
    "XlsxReadOptionsType",
    "XlsxWriteOptionsType",
    "XlsxSheetInfoType",
    "XlsxInfoType",
    "XmlNodeType",
    "XmlParseConfigType",
    "XmlSerializeConfigType",
]
