"""Tests for East primitive types."""

from datetime import UTC, datetime, timedelta, timezone

import pytest

from east.types.primitives import Blob, Null, ensure_utc_datetime, null, validate_east_value


class TestNull:
    """Tests for the Null type."""

    def test_singleton(self):
        """Null is a singleton."""
        n1 = Null()
        n2 = Null()
        assert n1 is n2
        assert n1 is null

    def test_repr(self):
        """Null repr is 'null'."""
        assert repr(null) == "null"
        assert str(null) == "null"

    def test_equality(self):
        """Null equals only itself."""
        assert null == null
        assert null == Null()
        assert null != None  # noqa: E711
        assert null != 0
        assert null != False  # noqa: E712

    def test_hash(self):
        """Null is hashable."""
        assert hash(null) == hash(None)
        # Can be used in sets/dicts
        s = {null}
        assert null in s

    def test_ordering(self):
        """Null is not less than itself."""
        assert not (null < null)
        assert null <= null
        assert not (null > null)
        assert null >= null


class TestBlob:
    """Tests for the Blob type."""

    def test_from_bytes(self):
        """Create Blob from bytes."""
        b = Blob(b"hello")
        assert len(b) == 5
        assert b[0] == ord("h")

    def test_from_list(self):
        """Create Blob from list of integers."""
        b = Blob([0x00, 0xFF, 0xAA])
        assert len(b) == 3
        assert b[0] == 0x00
        assert b[1] == 0xFF
        assert b[2] == 0xAA

    def test_from_blob(self):
        """Create Blob from another Blob."""
        b1 = Blob(b"test")
        b2 = Blob(b1)
        assert b1 == b2
        assert b1 is not b2  # Different objects
        assert b1.data is b2.data  # But share underlying bytes

    def test_empty(self):
        """Empty Blob."""
        b = Blob(b"")
        assert len(b) == 0
        assert repr(b) == "0x"

    def test_indexing(self):
        """Index into Blob."""
        b = Blob(b"abc")
        assert b[0] == ord("a")
        assert b[1] == ord("b")
        assert b[2] == ord("c")
        assert b[-1] == ord("c")

    def test_slicing(self):
        """Slice Blob returns new Blob."""
        b = Blob(b"hello")
        b2 = b[1:4]
        assert isinstance(b2, Blob)
        assert len(b2) == 3
        assert b2[0] == ord("e")

    def test_immutable(self):
        """Blob is immutable."""
        b = Blob(b"test")
        with pytest.raises((TypeError, AttributeError)):
            b[0] = 65  # type: ignore

    def test_equality(self):
        """Blobs with same bytes are equal."""
        b1 = Blob(b"test")
        b2 = Blob(b"test")
        b3 = Blob(b"different")
        assert b1 == b2
        assert b1 != b3
        assert b1 != "test"

    def test_ordering(self):
        """Blobs have lexicographic ordering."""
        b1 = Blob(b"aaa")
        b2 = Blob(b"aab")
        b3 = Blob(b"aba")
        assert b1 < b2 < b3
        assert not (b2 < b1)

    def test_hash(self):
        """Blobs are hashable."""
        b1 = Blob(b"test")
        b2 = Blob(b"test")
        assert hash(b1) == hash(b2)
        # Can be used in sets
        s = {b1, b2}
        assert len(s) == 1

    def test_repr_short(self):
        """Short blob repr in hex."""
        b = Blob(b"\x00\xff\xaa")
        assert repr(b) == "0x00ffaa"

    def test_repr_long(self):
        """Long blob repr is truncated."""
        b = Blob(bytes(range(256)) + b"\xff")
        r = repr(b)
        assert r.startswith("0x")
        assert r.endswith("...")
        assert len(r) < 600  # Truncated


class TestDateTime:
    """Tests for DateTime handling."""

    def test_ensure_utc_naive(self):
        """Naive datetime assumed to be UTC."""
        dt = datetime(2025, 1, 1, 12, 0, 0)
        result = ensure_utc_datetime(dt)
        assert result.tzinfo == UTC
        assert result.year == 2025
        assert result.hour == 12

    def test_ensure_utc_already_utc(self):
        """UTC datetime unchanged."""
        dt = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
        result = ensure_utc_datetime(dt)
        assert result is dt  # Same object

    def test_ensure_utc_conversion(self):
        """Other timezone converted to UTC."""
        # UTC-5 timezone
        tz_minus_5 = timezone(timedelta(hours=-5))
        dt = datetime(2025, 1, 1, 12, 0, 0, tzinfo=tz_minus_5)
        result = ensure_utc_datetime(dt)
        assert result.tzinfo == UTC
        assert result.hour == 17  # 12 + 5 = 17 UTC


class TestValidation:
    """Tests for value validation."""

    def test_validate_null(self):
        """Validate Null type."""
        validate_east_value(null, "Null")
        validate_east_value(Null(), "Null")
        with pytest.raises(TypeError):
            validate_east_value(None, "Null")

    def test_validate_boolean(self):
        """Validate Boolean type."""
        validate_east_value(True, "Boolean")
        validate_east_value(False, "Boolean")
        with pytest.raises(TypeError):
            validate_east_value(1, "Boolean")
        with pytest.raises(TypeError):
            validate_east_value(0, "Boolean")

    def test_validate_integer(self):
        """Validate Integer type."""
        validate_east_value(42, "Integer")
        validate_east_value(-10, "Integer")
        validate_east_value(0, "Integer")
        # Booleans are not integers in East
        with pytest.raises(TypeError):
            validate_east_value(True, "Integer")
        with pytest.raises(TypeError):
            validate_east_value(3.14, "Integer")

    def test_validate_float(self):
        """Validate Float type."""
        validate_east_value(3.14, "Float")
        validate_east_value(0.0, "Float")
        validate_east_value(float("inf"), "Float")
        validate_east_value(float("-inf"), "Float")
        validate_east_value(float("nan"), "Float")
        with pytest.raises(TypeError):
            validate_east_value(42, "Float")

    def test_validate_string(self):
        """Validate String type."""
        validate_east_value("hello", "String")
        validate_east_value("", "String")
        validate_east_value("unicode: \u2603", "String")
        with pytest.raises(TypeError):
            validate_east_value(b"bytes", "String")

    def test_validate_blob(self):
        """Validate Blob type."""
        validate_east_value(Blob(b"test"), "Blob")
        with pytest.raises(TypeError):
            validate_east_value(b"bytes", "Blob")

    def test_validate_datetime(self):
        """Validate DateTime type."""
        validate_east_value(datetime.now(), "DateTime")
        validate_east_value(datetime(2025, 1, 1, tzinfo=UTC), "DateTime")
        with pytest.raises(TypeError):
            validate_east_value("2025-01-01", "DateTime")

    def test_validate_unknown_type(self):
        """Unknown type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown East primitive type"):
            validate_east_value(42, "UnknownType")


class TestPrimitiveTypes:
    """Integration tests for primitive type interactions."""

    def test_bool_is_not_int(self):
        """In East, Boolean is distinct from Integer."""
        # Python's bool is a subclass of int, but East treats them separately
        assert isinstance(True, bool)
        assert isinstance(True, int)  # Python quirk
        # Our validation correctly distinguishes them
        validate_east_value(True, "Boolean")
        with pytest.raises(TypeError):
            validate_east_value(True, "Integer")

    def test_special_floats(self):
        """Special float values work correctly."""
        inf = float("inf")
        neg_inf = float("-inf")
        nan = float("nan")

        # All are valid floats
        validate_east_value(inf, "Float")
        validate_east_value(neg_inf, "Float")
        validate_east_value(nan, "Float")

        # Ordering works
        assert neg_inf < 0.0 < inf
        # NaN is not equal to anything, including itself
        assert nan != nan
        assert not (nan < nan)
        assert not (nan > nan)
