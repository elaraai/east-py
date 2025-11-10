"""Tests for ref type functionality."""

from east.types.ref import Ref, deref, is_ref, ref, set_ref
from east.types.type_system import ArrayType, IntegerType, RefType, StringType


def test_ref_creation():
    """Create a ref and check type."""
    r = ref(42)
    assert isinstance(r, Ref)
    assert is_ref(r)
    assert deref(r) == 42


def test_ref_mutation():
    """Mutate a ref's value."""
    r = ref(0)
    assert deref(r) == 0

    set_ref(r, 1)
    assert deref(r) == 1

    set_ref(r, deref(r) + 1)
    assert deref(r) == 2


def test_ref_aliasing():
    """Test that refs have identity semantics."""
    r1 = ref([1, 2, 3])
    r2 = r1  # Same ref

    set_ref(r2, [4, 5, 6])
    assert deref(r1) == [4, 5, 6]  # r1 sees the change
    assert r1 is r2


def test_ref_distinct():
    """Test that different refs are distinct."""
    r1 = ref(42)
    r2 = ref(42)

    assert r1 is not r2  # Different refs
    assert deref(r1) == deref(r2)  # Same value

    set_ref(r1, 99)
    assert deref(r1) == 99
    assert deref(r2) == 42  # r2 unchanged


def test_ref_nested():
    """Test nested refs."""
    inner = ref(10)
    outer = ref(inner)

    assert deref(deref(outer)) == 10

    set_ref(inner, 20)
    assert deref(deref(outer)) == 20


def test_ref_type_creation():
    """Test RefType constructor."""
    int_ref_type = RefType(IntegerType)
    assert int_ref_type.tag == "Ref"
    assert int_ref_type.value == IntegerType

    array_ref_type = RefType(ArrayType(StringType))
    assert array_ref_type.tag == "Ref"
    assert array_ref_type.value.tag == "Array"


def test_ref_type_requires_data_type():
    """Test that refs can only contain data types."""
    from east.types.type_system import FunctionType

    # Should work with data types
    RefType(IntegerType)
    RefType(ArrayType(IntegerType))

    # Should fail with function types
    func_type = FunctionType([IntegerType], IntegerType, [])
    try:
        RefType(func_type)
        raise AssertionError("Should have raised TypeError")
    except TypeError as e:
        assert "data type" in str(e)


def test_is_ref_false_for_non_refs():
    """Test is_ref returns False for non-refs."""
    assert not is_ref(42)
    assert not is_ref([1, 2, 3])
    assert not is_ref({"value": 42})
    assert not is_ref(None)


def test_ref_repr():
    """Test ref string representation."""
    r = ref(42)
    assert repr(r) == "ref(42)"

    r2 = ref([1, 2])
    assert "ref" in repr(r2)


def test_ref_type_of():
    """Test type_of with refs."""
    from east.types.type_system import type_of

    r = ref(42)
    ref_type = type_of(r)
    assert ref_type.tag == "Ref"
    assert ref_type.value == IntegerType


def test_ref_default_value():
    """Test default_value for ref type."""
    from east.utils.default import default_value

    typ = RefType(IntegerType)
    val = default_value(typ)

    assert is_ref(val)
    assert deref(val) == 0  # Default int is 0


def test_ref_comparison_identity():
    """Test ref identity comparison."""
    from east.utils.ordering import is_for

    ref_type = RefType(IntegerType)

    # Identity - same ref
    r1 = ref(42)
    r2 = r1

    is_comparer = is_for(ref_type)
    assert is_comparer(r1, r2)

    # Different refs
    r3 = ref(42)
    assert not is_comparer(r1, r3)


def test_ref_comparison_equality():
    """Test ref structural equality."""
    from east.utils.ordering import equal_for

    ref_type = RefType(IntegerType)
    equal_comparer = equal_for(ref_type)

    # Same identity
    r1 = ref(42)
    r2 = r1
    assert equal_comparer(r1, r2)

    # Different identity, same value
    r3 = ref(42)
    r4 = ref(42)
    assert equal_comparer(r3, r4)

    # Different values
    r5 = ref(42)
    r6 = ref(99)
    assert not equal_comparer(r5, r6)


def test_ref_comparison_ordering():
    """Test ref ordering comparison."""
    from east.utils.ordering import compare_for

    ref_type = RefType(IntegerType)
    comparer = compare_for(ref_type)

    r1 = ref(10)
    r2 = ref(20)
    r3 = ref(10)

    assert comparer(r1, r2) < 0  # 10 < 20
    assert comparer(r2, r1) > 0  # 20 > 10
    assert comparer(r1, r3) == 0  # Same value


def test_ref_circular_equality():
    """Test circular refs can be compared."""
    from east.utils.ordering import equal_for

    ref_type = RefType(RefType(IntegerType))
    equal_comparer = equal_for(ref_type)

    r1: Ref[Ref | None] = ref(None)
    r2: Ref[Ref | None] = ref(None)

    set_ref(r1, r1)  # Self-reference
    set_ref(r2, r2)  # Self-reference

    # Should not infinite loop
    assert equal_comparer(r1, r2)  # Both are circular
