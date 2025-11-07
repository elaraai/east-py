"""Tests for East type printing functions.

Ported from East/src/types.spec.ts - printType and printIdentifier test suites
"""

from east.serialization.east_printer import print_identifier, print_type
from east.types.type_system import (
    ArrayType,
    BlobType,
    BooleanType,
    DateTimeType,
    DictType,
    FloatType,
    FunctionType,
    IntegerType,
    NeverType,
    NullType,
    SetType,
    StringType,
    StructTypeFromFields,
    VariantTypeFromCases,
)


class TestPrintType:
    """Test suite for print_type function."""

    def test_should_print_primitive_types(self):
        """should print primitive types."""
        assert print_type(NeverType) == ".Never"
        assert print_type(NullType) == ".Null"
        assert print_type(BooleanType) == ".Boolean"
        assert print_type(IntegerType) == ".Integer"
        assert print_type(FloatType) == ".Float"
        assert print_type(StringType) == ".String"
        assert print_type(DateTimeType) == ".DateTime"
        assert print_type(BlobType) == ".Blob"

    def test_should_print_collection_types(self):
        """should print collection types."""
        assert print_type(ArrayType(IntegerType)) == ".Array .Integer"
        assert print_type(SetType(StringType)) == ".Set .String"
        assert (
            print_type(DictType(StringType, IntegerType)) == ".Dict (key=.String, value=.Integer)"
        )

    def test_should_print_struct_types(self):
        """should print struct types."""
        assert (
            print_type(StructTypeFromFields([("x", IntegerType)]))
            == '.Struct [(name="x", type=.Integer)]'
        )
        assert (
            print_type(StructTypeFromFields([("x", IntegerType), ("y", FloatType)]))
            == '.Struct [(name="x", type=.Integer), (name="y", type=.Float)]'
        )

    def test_should_print_variant_types(self):
        """should print variant types."""
        assert (
            print_type(VariantTypeFromCases([("none", NullType)]))
            == '.Variant [(name="none", type=.Null)]'
        )
        assert (
            print_type(VariantTypeFromCases([("none", NullType), ("some", IntegerType)]))
            == '.Variant [(name="none", type=.Null), (name="some", type=.Integer)]'
        )

    def test_should_print_function_types(self):
        """should print function types."""
        assert (
            print_type(FunctionType([], NullType, []))
            == ".Function (inputs=[], output=.Null, platforms=[])"
        )
        assert (
            print_type(FunctionType([IntegerType, StringType], BooleanType, []))
            == ".Function (inputs=[.Integer, .String], output=.Boolean, platforms=[])"
        )


class TestPrintIdentifier:
    """Test suite for print_identifier function."""

    def test_should_print_valid_identifiers_as_is(self):
        """should print valid identifiers as-is."""
        assert print_identifier("foo") == "foo"
        assert print_identifier("_bar") == "_bar"
        assert print_identifier("foo123") == "foo123"

    def test_should_escape_invalid_identifiers(self):
        """should escape invalid identifiers."""
        assert print_identifier("foo bar") == "`foo bar`"
        assert print_identifier("123") == "`123`"
        assert print_identifier("foo-bar") == "`foo-bar`"

    def test_should_escape_special_characters_in_identifiers(self):
        """should escape special characters in identifiers."""
        assert print_identifier("foo`bar") == "`foo\\`bar`"
        assert print_identifier("foo\\bar") == "`foo\\\\bar`"
