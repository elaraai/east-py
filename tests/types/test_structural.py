"""Tests for East structural types (Struct and Variant)."""

import pytest

from east.types.primitives import null
from east.types.structural import Case, EastStruct, EastVariant, make_case
from east.types.type_system import IntegerType, StringType, StructType, VariantType


class TestStructType:
    """Tests for StructType."""

    def test_create(self):
        """Create a struct type."""
        st = StructType((("name", StringType), ("age", IntegerType)))
        assert len(st.fields) == 2
        assert st.field_names() == ["name", "age"]
        assert st.field_types() == [StringType, IntegerType]

    def test_field_index(self):
        """Get field index by name."""
        st = StructType((("name", StringType), ("age", IntegerType)))
        assert st.field_index("name") == 0
        assert st.field_index("age") == 1

    def test_field_index_missing(self):
        """Field index raises KeyError for missing field."""
        st = StructType((("name", StringType),))
        with pytest.raises(KeyError):
            st.field_index("missing")

    def test_create_instance(self):
        """Create struct instance."""
        st = StructType((("name", StringType), ("age", IntegerType)))
        instance = st.create(name="Alice", age=30)
        assert isinstance(instance, EastStruct)
        assert instance._east_type == st
        assert instance._values == ("Alice", 30)

    def test_create_instance_missing_field(self):
        """Creating instance without all fields raises ValueError."""
        st = StructType((("name", StringType), ("age", IntegerType)))
        with pytest.raises(ValueError, match="Expected 2 fields"):
            st.create(name="Alice")

    def test_create_instance_extra_field(self):
        """Creating instance with extra field raises ValueError."""
        st = StructType((("name", StringType), ("age", IntegerType)))
        with pytest.raises(ValueError, match="Expected 2 fields"):
            st.create(name="Alice", age=30, extra="value")

    def test_empty_struct_type(self):
        """Create empty struct type."""
        st = StructType(())
        assert len(st.fields) == 0
        assert st.field_names() == []


class TestEastStruct:
    """Tests for EastStruct."""

    def test_field_access(self):
        """Access fields by name."""
        st = StructType((("name", StringType), ("age", IntegerType)))
        instance = st.create(name="Alice", age=30)
        assert instance.name == "Alice"
        assert instance.age == 30

    def test_field_access_missing(self):
        """Accessing missing field raises AttributeError."""
        st = StructType((("name", StringType),))
        instance = st.create(name="Alice")
        with pytest.raises(AttributeError):
            _ = instance.missing

    def test_immutable(self):
        """Structs are immutable."""
        st = StructType((("name", StringType), ("age", IntegerType)))
        instance = st.create(name="Alice", age=30)
        with pytest.raises((AttributeError, TypeError)):
            instance.name = "Bob"  # type: ignore

    def test_equality(self):
        """Structural equality."""
        st = StructType((("name", StringType), ("age", IntegerType)))
        s1 = st.create(name="Alice", age=30)
        s2 = st.create(name="Alice", age=30)
        s3 = st.create(name="Bob", age=30)
        assert s1 == s2
        assert s1 != s3

    def test_equality_different_types(self):
        """Structs with different types aren't equal."""
        st1 = StructType((("name", StringType),))
        st2 = StructType((("name", StringType), ("age", IntegerType)))
        s1 = st1.create(name="Alice")
        s2 = st2.create(name="Alice", age=30)
        assert s1 != s2

    def test_ordering(self):
        """Structural ordering."""
        st = StructType((("name", StringType), ("age", IntegerType)))
        s1 = st.create(name="Alice", age=25)
        s2 = st.create(name="Alice", age=30)
        s3 = st.create(name="Bob", age=20)
        assert s1 < s2  # Same name, age 25 < 30
        assert s1 < s3  # Alice < Bob

    def test_hash(self):
        """Structs are hashable."""
        st = StructType((("name", StringType), ("age", IntegerType)))
        s1 = st.create(name="Alice", age=30)
        s2 = st.create(name="Alice", age=30)
        assert hash(s1) == hash(s2)
        # Can be used in sets
        structs = {s1, s2}
        assert len(structs) == 1

    def test_repr_empty(self):
        """Empty struct repr."""
        st = StructType(())
        instance = st.create()
        assert repr(instance) == "()"

    def test_repr(self):
        """Struct repr in East format."""
        st = StructType((("name", StringType), ("age", IntegerType)))
        instance = st.create(name="Alice", age=30)
        assert repr(instance) == "(name='Alice', age=30)"


class TestVariantType:
    """Tests for VariantType."""

    def test_create(self):
        """Create a variant type."""
        vt = VariantType((("Some", IntegerType), ("None", StringType)))
        assert len(vt.cases) == 2
        assert vt.case_names() == ["Some", "None"]
        assert vt.case_types() == [IntegerType, StringType]

    def test_case_type(self):
        """Get case type by name."""
        vt = VariantType((("Some", IntegerType), ("None", StringType)))
        assert vt.case_type("Some") == IntegerType
        assert vt.case_type("None") == StringType

    def test_case_type_missing(self):
        """Case type raises KeyError for missing case."""
        vt = VariantType((("Some", IntegerType),))
        with pytest.raises(KeyError):
            vt.case_type("Missing")

    def test_create_instance(self):
        """Create variant instance."""
        vt = VariantType((("Some", IntegerType), ("None", StringType)))
        instance = vt.create("Some", 42)
        assert isinstance(instance, EastVariant)
        assert instance._east_type == vt
        assert instance.tag == "Some"
        assert instance.value == 42

    def test_create_instance_with_null(self):
        """Create variant instance with default null value."""
        vt = VariantType((("Some", IntegerType), ("None", StringType)))
        instance = vt.create("None")
        assert instance.tag == "None"
        assert instance.value == null

    def test_create_instance_invalid_case(self):
        """Creating instance with invalid case raises KeyError."""
        vt = VariantType((("Some", IntegerType),))
        with pytest.raises(KeyError):
            vt.create("Invalid", 42)


class TestCase:
    """Tests for Case."""

    def test_create(self):
        """Create a case."""
        c = Case("Some", 42)
        assert c.tag == "Some"
        assert c.value == 42

    def test_equality(self):
        """Cases equal if tag and value equal."""
        c1 = Case("Some", 42)
        c2 = Case("Some", 42)
        c3 = Case("Some", 43)
        c4 = Case("Other", 42)
        assert c1 == c2
        assert c1 != c3
        assert c1 != c4

    def test_ordering(self):
        """Cases ordered by tag, then value."""
        c1 = Case("A", 2)
        c2 = Case("A", 3)
        c3 = Case("B", 1)
        assert c1 < c2  # Same tag, value 2 < 3
        assert c1 < c3  # Tag A < B
        assert c2 < c3  # Tag A < B

    def test_hash(self):
        """Cases are hashable."""
        c1 = Case("Some", 42)
        c2 = Case("Some", 42)
        assert hash(c1) == hash(c2)

    def test_repr_with_value(self):
        """Case repr with value."""
        c = Case("Some", 42)
        assert repr(c) == ".Some 42"

    def test_repr_null_value(self):
        """Case repr with null value."""
        c = Case("None", null)
        assert repr(c) == ".None"


class TestEastVariant:
    """Tests for EastVariant."""

    def test_tag_value_access(self):
        """Access tag and value."""
        vt = VariantType((("Some", IntegerType), ("None", StringType)))
        instance = vt.create("Some", 42)
        assert instance.tag == "Some"
        assert instance.value == 42

    def test_equality(self):
        """Structural equality."""
        vt = VariantType((("Some", IntegerType), ("None", StringType)))
        v1 = vt.create("Some", 42)
        v2 = vt.create("Some", 42)
        v3 = vt.create("Some", 43)
        v4 = vt.create("None", "test")
        assert v1 == v2
        assert v1 != v3
        assert v1 != v4

    def test_equality_different_types(self):
        """Variants with different types aren't equal."""
        vt1 = VariantType((("Some", IntegerType),))
        vt2 = VariantType((("Some", IntegerType), ("None", StringType)))
        v1 = vt1.create("Some", 42)
        v2 = vt2.create("Some", 42)
        assert v1 != v2

    def test_ordering(self):
        """Structural ordering."""
        vt = VariantType((("A", IntegerType), ("B", IntegerType)))
        v1 = vt.create("A", 1)
        v2 = vt.create("A", 2)
        v3 = vt.create("B", 0)
        assert v1 < v2  # Same tag, value 1 < 2
        assert v1 < v3  # Tag A < B

    def test_hash(self):
        """Variants are hashable."""
        vt = VariantType((("Some", IntegerType),))
        v1 = vt.create("Some", 42)
        v2 = vt.create("Some", 42)
        assert hash(v1) == hash(v2)
        # Can be used in sets
        variants = {v1, v2}
        assert len(variants) == 1

    def test_repr(self):
        """Variant repr in East format."""
        vt = VariantType((("Some", IntegerType), ("None", StringType)))
        v1 = vt.create("Some", 42)
        v2 = vt.create("None")
        assert repr(v1) == ".Some 42"
        assert repr(v2) == ".None"


class TestMakeCase:
    """Tests for make_case helper."""

    def test_with_value(self):
        """Make case with value."""
        c = make_case("Some", 42)
        assert c.tag == "Some"
        assert c.value == 42

    def test_without_value(self):
        """Make case defaults to null."""
        c = make_case("None")
        assert c.tag == "None"
        assert c.value == null

    def test_explicit_none(self):
        """Explicit None becomes null."""
        c = make_case("None", None)
        assert c.tag == "None"
        assert c.value == null


class TestOptionPattern:
    """Test the common Option pattern with variants."""

    def test_option_some(self):
        """Option Some case."""
        option_type = VariantType((("Some", IntegerType), ("None", IntegerType)))
        some = option_type.create("Some", 42)
        assert some.tag == "Some"
        assert some.value == 42

    def test_option_none(self):
        """Option None case."""
        option_type = VariantType((("Some", IntegerType), ("None", IntegerType)))
        none = option_type.create("None")
        assert none.tag == "None"
        assert none.value == null

    def test_pattern_matching_with_tag(self):
        """Pattern match using tag."""
        option_type = VariantType((("Some", IntegerType), ("None", IntegerType)))

        def unwrap_or(opt: EastVariant, default: int) -> int:
            if opt.tag == "Some":
                return opt.value
            return default

        some = option_type.create("Some", 42)
        none = option_type.create("None")

        assert unwrap_or(some, 0) == 42
        assert unwrap_or(none, 0) == 0
