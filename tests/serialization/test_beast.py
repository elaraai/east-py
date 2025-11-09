"""Tests for Beast v1 binary format serialization.

Ported from East/src/serialization/beast.spec.ts
"""

from datetime import UTC, datetime

import pytest

from east.serialization.beast import (
    decode_beast_for,
    decode_beast_value_for,
    encode_beast_for,
    encode_beast_value_to_buffer_for,
)
from east.types.containers import EastArray, EastDict, EastSet
from east.types.primitives import Blob
from east.types.type_system import (
    ArrayType,
    BlobType,
    BooleanType,
    DateTimeType,
    DictType,
    FloatType,
    IntegerType,
    NullType,
    SetType,
    StringType,
    StructType,
    VariantType,
)
from east.utils.ordering import equal_for, less_for

# =============================================================================
# Byte ordering utilities (memcmp simulation)
# =============================================================================


def memcmp(a: bytes, b: bytes) -> int:
    """Compare two byte arrays lexicographically (simulates memcmp).

    Returns:
        -1 if a < b, 0 if a == b, 1 if a > b
    """
    min_len = min(len(a), len(b))
    for i in range(min_len):
        if a[i] < b[i]:
            return -1
        if a[i] > b[i]:
            return 1
    if len(a) < len(b):
        return -1
    if len(a) == len(b):
        return 0
    return 1


# =============================================================================
# Comprehensive test helpers
# =============================================================================


def _test_round_trip(type_val, values):
    """Test that values round-trip correctly (without checking byte ordering)."""
    from east.serialization.binary_utils import BufferWriter

    encode = encode_beast_value_to_buffer_for(type_val)
    decode = decode_beast_value_for(type_val)
    equal = equal_for(type_val)

    # Test each value round-trips
    for v in values:
        writer = BufferWriter()
        encode(v, writer)
        encoded = writer.to_bytes()
        decoded, offset = decode(encoded, 0)
        assert offset == len(encoded), f"Did not consume all bytes for {v}"
        assert equal(decoded, v), f"Value did not round-trip: {v}"


def _test_round_trip_and_ordering(type_val, values):
    """Test that values round-trip correctly AND that byte ordering matches value ordering.

    This is the critical property of beast v1 format.
    """
    from east.serialization.binary_utils import BufferWriter

    encode = encode_beast_value_to_buffer_for(type_val)
    decode = decode_beast_value_for(type_val)
    equal = equal_for(type_val)
    less = less_for(type_val)

    # Test each value round-trips
    for v in values:
        writer = BufferWriter()
        encode(v, writer)
        encoded = writer.to_bytes()
        decoded, offset = decode(encoded, 0)
        assert offset == len(encoded), f"Did not consume all bytes for {v}"
        assert equal(decoded, v), f"Value did not round-trip: {v}"

    # Test byte ordering matches value ordering (the key property of beast v1)
    for v1 in values:
        writer1 = BufferWriter()
        encode(v1, writer1)
        encoded1 = writer1.to_bytes()

        for v2 in values:
            writer2 = BufferWriter()
            encode(v2, writer2)
            encoded2 = writer2.to_bytes()

            byte_cmp = memcmp(encoded1, encoded2)

            if equal(v1, v2):
                assert byte_cmp == 0, f"Equal values should have equal bytes: {v1} vs {v2}"
            elif less(v1, v2):
                assert byte_cmp == -1, f"Less value should have less bytes: {v1} < {v2}"
            else:
                assert byte_cmp == 1, f"Greater value should have greater bytes: {v1} > {v2}"


# =============================================================================
# Primitive types
# =============================================================================


class TestBeastNull:
    """Tests for Null type."""

    def test_should_round_trip_and_maintain_ordering(self):
        """Should round-trip and maintain ordering."""
        _test_round_trip_and_ordering(NullType, [None])


class TestBeastBoolean:
    """Tests for Boolean type."""

    def test_should_round_trip_and_maintain_ordering(self):
        """Should round-trip and maintain ordering."""
        _test_round_trip_and_ordering(BooleanType, [False, True])


class TestBeastInteger:
    """Tests for Integer type."""

    def test_should_round_trip_and_maintain_ordering(self):
        """Should round-trip and maintain ordering."""
        _test_round_trip_and_ordering(
            IntegerType,
            [
                -9223372036854775808,  # MIN_INT64
                -1,
                0,
                42,
                90071992547409919,
                9223372036854775807,  # MAX_INT64
            ],
        )


class TestBeastFloat:
    """Tests for Float type."""

    def test_should_round_trip_and_maintain_ordering(self):
        """Should round-trip and maintain ordering."""
        _test_round_trip_and_ordering(
            FloatType,
            [
                float("-inf"),
                -1e8,
                -3.14,
                -1.0,
                -1e-8,
                -1e-16,
                -0.0,
                0.0,
                1e-16,
                1e-8,
                1.0,
                3.14,
                1e8,
                float("inf"),
                float("nan"),
            ],
        )


class TestBeastString:
    """Tests for String type."""

    def test_should_round_trip_and_maintain_ordering(self):
        """Should round-trip and maintain ordering."""
        _test_round_trip_and_ordering(
            StringType,
            [
                "",
                "a",
                "ab",
                "abc",
                "abd",
                "def",
                "いろはにほへとちりぬるを",  # UTF-8 Japanese
            ],
        )


class TestBeastDateTime:
    """Tests for DateTime type."""

    def test_should_round_trip_and_maintain_ordering(self):
        """Should round-trip and maintain ordering."""
        _test_round_trip_and_ordering(
            DateTimeType,
            [
                datetime.fromtimestamp(0, tz=UTC),  # Unix epoch
                datetime.fromisoformat("2022-06-29T13:43:00.123+00:00"),
                datetime.fromisoformat("2025-01-01T00:00:00.000+00:00"),
            ],
        )


class TestBeastBlob:
    """Tests for Blob type."""

    def test_should_round_trip(self):
        """Should round-trip.

        Note: Blobs do NOT preserve ordering in beast v1 binary format.
        The format uses length-first encoding (8-byte length prefix), but comparison is lexicographic.
        So byte ordering (length-first) doesn't match value ordering (lexicographic).
        """
        _test_round_trip(
            BlobType,
            [
                Blob(b""),
                Blob(b"\x01"),
                Blob(b"\x01\x03"),
                Blob(b"\x01\x03\x03"),
                Blob(b"\x01\x03\x04"),
                Blob(b"\x09"),
                Blob(b"\xff"),
                Blob(b"\xff\xff"),
            ],
        )


# =============================================================================
# Collection types
# =============================================================================


class TestBeastArray:
    """Tests for Array type."""

    def test_should_round_trip_and_maintain_ordering(self):
        """Should round-trip and maintain ordering."""
        _test_round_trip_and_ordering(
            ArrayType(IntegerType),
            [
                EastArray(IntegerType, []),
                EastArray(IntegerType, [0]),
                EastArray(IntegerType, [0, 1]),
                EastArray(IntegerType, [0, 2, 3]),
                EastArray(IntegerType, [0, 2, 4]),
                EastArray(IntegerType, [1]),
            ],
        )


class TestBeastSet:
    """Tests for Set type."""

    def test_should_round_trip_and_maintain_ordering(self):
        """Should round-trip and maintain ordering."""
        _test_round_trip_and_ordering(
            SetType(StringType),
            [
                EastSet(StringType, set()),
                EastSet(StringType, {"abc"}),
                EastSet(StringType, {"abc", "def"}),
                EastSet(StringType, {"def"}),
            ],
        )


class TestBeastDict:
    """Tests for Dict type."""

    def test_should_round_trip_and_maintain_ordering(self):
        """Should round-trip and maintain ordering."""
        _test_round_trip_and_ordering(
            DictType(StringType, IntegerType),
            [
                EastDict(StringType, IntegerType, {}),
                EastDict(StringType, IntegerType, {"abc": 0}),
                EastDict(StringType, IntegerType, {"abc": 0, "def": 1}),
                EastDict(StringType, IntegerType, {"abc": 1}),
                EastDict(StringType, IntegerType, {"def": 1}),
            ],
        )


# =============================================================================
# Compound types
# =============================================================================


class TestBeastStruct:
    """Tests for Struct type."""

    def test_should_round_trip_and_maintain_ordering(self):
        """Should round-trip and maintain ordering."""
        type_val = StructType(
            [
                ("boolean", BooleanType),
                ("string", StringType),
            ]
        )
        _test_round_trip_and_ordering(
            type_val,
            [
                {"boolean": False, "string": "good"},
                {"boolean": True, "string": "bad"},
                {"boolean": True, "string": "ok"},
            ],
        )


class TestBeastVariant:
    """Tests for Variant type."""

    def test_should_round_trip_and_maintain_ordering(self):
        """Should round-trip and maintain ordering."""
        type_val = VariantType(
            [
                ("none", NullType),
                ("some", IntegerType),
            ]
        )
        _test_round_trip_and_ordering(
            type_val,
            [
                {"type": "none", "value": None},
                {"type": "some", "value": 0},
                {"type": "some", "value": 1},
            ],
        )


# =============================================================================
# Complex real-world structures
# =============================================================================


class TestBeastComplexStructs:
    """Tests for complex real-world structures."""

    def test_should_round_trip_complex_production_like_struct(self):
        """Should round-trip complex production-like struct."""
        # This is based on a real production struct from the old tests
        type_val = StructType(
            [
                ("active", BooleanType),
                ("count", IntegerType),
                ("id", StringType),
                ("metadata", DictType(StringType, StringType)),
                ("score", FloatType),
                ("tags", ArrayType(StringType)),
                ("timestamp", DateTimeType),
            ]
        )

        value = {
            "id": "35932005329",
            "active": True,
            "timestamp": datetime.fromisoformat("2022-03-01T00:00:00.000+00:00"),
            "score": 3.14,
            "count": 42,
            "tags": EastArray(StringType, ["alpha", "beta"]),
            "metadata": EastDict(
                StringType,
                StringType,
                {
                    "key1": "value1",
                    "key2": "value2",
                },
            ),
        }

        from east.serialization.binary_utils import BufferWriter

        encode = encode_beast_value_to_buffer_for(type_val)
        decode = decode_beast_value_for(type_val)

        writer = BufferWriter()
        encode(value, writer)
        encoded = writer.to_bytes()
        decoded, offset = decode(encoded, 0)

        assert offset == len(encoded), "Did not consume all bytes"

        equal = equal_for(type_val)
        assert equal(decoded, value), "Complex struct did not round-trip"


# =============================================================================
# Nested structures
# =============================================================================


class TestBeastNestedStructures:
    """Tests for nested structures."""

    def test_should_round_trip_nested_arrays(self):
        """Should round-trip nested arrays."""
        type_val = ArrayType(ArrayType(IntegerType))
        values = [
            EastArray(ArrayType(IntegerType), []),
            EastArray(ArrayType(IntegerType), [EastArray(IntegerType, [])]),
            EastArray(
                ArrayType(IntegerType),
                [
                    EastArray(IntegerType, [1, 2]),
                    EastArray(IntegerType, [3]),
                ],
            ),
            EastArray(
                ArrayType(IntegerType),
                [
                    EastArray(IntegerType, [1, 2]),
                    EastArray(IntegerType, [3, 4]),
                ],
            ),
        ]
        _test_round_trip_and_ordering(type_val, values)

    def test_should_round_trip_nested_structs(self):
        """Should round-trip nested structs."""
        inner_type = StructType(
            [
                ("x", IntegerType),
                ("y", IntegerType),
            ]
        )
        outer_type = StructType(
            [
                ("label", StringType),
                ("point", inner_type),
            ]
        )

        values = [
            {"point": {"x": 0, "y": 0}, "label": "origin"},
            {"point": {"x": 1, "y": 2}, "label": "a"},
            {"point": {"x": 1, "y": 2}, "label": "b"},
        ]

        _test_round_trip_and_ordering(outer_type, values)


# =============================================================================
# Byte ordering verification for critical types
# =============================================================================


class TestBeastByteOrdering:
    """Tests for byte ordering properties."""

    def test_integer_byte_ordering_should_match_numeric_ordering(self):
        """Integer byte ordering should match numeric ordering."""
        from east.serialization.binary_utils import BufferWriter

        encode = encode_beast_value_to_buffer_for(IntegerType)

        def encode_int(val):
            writer = BufferWriter()
            encode(val, writer)
            return writer.to_bytes()

        # Negative numbers should come before positive
        neg_encoded = encode_int(-100)
        pos_encoded = encode_int(100)
        assert memcmp(neg_encoded, pos_encoded) == -1, "Negative should sort before positive"

        # More negative should come before less negative
        more_neg_encoded = encode_int(-200)
        assert (
            memcmp(more_neg_encoded, neg_encoded) == -1
        ), "More negative should sort before less negative"

        # Larger positive should come after smaller positive
        larger_pos_encoded = encode_int(200)
        assert (
            memcmp(pos_encoded, larger_pos_encoded) == -1
        ), "Smaller positive should sort before larger positive"

    def test_float_byte_ordering_should_match_numeric_ordering_including_special_values(self):
        """Float byte ordering should match numeric ordering (including special values)."""
        from east.serialization.binary_utils import BufferWriter

        encode = encode_beast_value_to_buffer_for(FloatType)

        def encode_float(val):
            writer = BufferWriter()
            encode(val, writer)
            return writer.to_bytes()

        # Test key ordering properties
        neg_inf = encode_float(float("-inf"))
        neg_one = encode_float(-1.0)
        neg_zero = encode_float(-0.0)
        pos_zero = encode_float(0.0)
        pos_one = encode_float(1.0)
        pos_inf = encode_float(float("inf"))
        nan = encode_float(float("nan"))

        # Verify total ordering: -Inf < -1 < -0 == 0 < 1 < +Inf < NaN
        assert memcmp(neg_inf, neg_one) == -1, "-Infinity < -1"
        assert memcmp(neg_one, neg_zero) == -1, "-1 < -0"
        assert memcmp(neg_zero, pos_zero) == -1, "-0 < 0 (distinct in byte representation)"
        assert memcmp(pos_zero, pos_one) == -1, "0 < 1"
        assert memcmp(pos_one, pos_inf) == -1, "1 < +Infinity"
        assert memcmp(pos_inf, nan) == -1, "+Infinity < NaN"

    def test_string_byte_ordering_should_match_lexicographic_ordering(self):
        """String byte ordering should match lexicographic ordering."""
        from east.serialization.binary_utils import BufferWriter

        encode = encode_beast_value_to_buffer_for(StringType)

        def encode_str(val):
            writer = BufferWriter()
            encode(val, writer)
            return writer.to_bytes()

        empty = encode_str("")
        a = encode_str("a")
        ab = encode_str("ab")
        b = encode_str("b")

        assert memcmp(empty, a) == -1, "empty string < 'a'"
        assert memcmp(a, ab) == -1, "'a' < 'ab'"
        assert memcmp(ab, b) == -1, "'ab' < 'b'"

    def test_array_byte_ordering_should_match_lexicographic_ordering(self):
        """Array byte ordering should match lexicographic ordering."""
        from east.serialization.binary_utils import BufferWriter

        encode = encode_beast_value_to_buffer_for(ArrayType(IntegerType))

        def encode_arr(val):
            writer = BufferWriter()
            encode(val, writer)
            return writer.to_bytes()

        empty = encode_arr(EastArray(IntegerType, []))
        zero = encode_arr(EastArray(IntegerType, [0]))
        zero_one = encode_arr(EastArray(IntegerType, [0, 1]))
        one = encode_arr(EastArray(IntegerType, [1]))

        assert memcmp(empty, zero) == -1, "[] < [0]"
        assert memcmp(zero, zero_one) == -1, "[0] < [0, 1]"
        assert memcmp(zero_one, one) == -1, "[0, 1] < [1]"


# =============================================================================
# Error handling
# =============================================================================


class TestBeastErrorHandling:
    """Tests for error handling."""

    def test_should_throw_on_truncated_integer_data(self):
        """Should throw on truncated integer data."""
        decode = decode_beast_value_for(IntegerType)

        truncated = bytes([0x00, 0x00, 0x00])  # Only 3 bytes, need 8
        with pytest.raises(ValueError, match=r"Buffer underflow"):
            decode(truncated, 0)

    def test_should_throw_on_truncated_string_missing_null_terminator(self):
        """Should throw on truncated string (missing null terminator)."""
        decode = decode_beast_value_for(StringType)

        no_terminator = bytes([0x68, 0x65, 0x6C, 0x6C, 0x6F])  # "hello" without null
        with pytest.raises(ValueError, match=r"Missing null terminator"):
            decode(no_terminator, 0)

    def test_should_throw_on_invalid_array_continuation_byte(self):
        """Should throw on invalid array continuation byte."""
        decode = decode_beast_value_for(ArrayType(IntegerType))

        invalid = bytes([0x02])  # Invalid continuation byte (not 0x00 or 0x01)
        with pytest.raises(ValueError, match=r"Invalid continuation byte"):
            decode(invalid, 0)

    def test_should_throw_on_invalid_variant_tag(self):
        """Should throw on invalid variant tag."""
        type_val = VariantType(
            [
                ("none", NullType),
                ("some", IntegerType),
            ]
        )
        decode = decode_beast_value_for(type_val)

        invalid_tag = bytes([0x02])  # Tag 2, but only 0 and 1 are valid
        with pytest.raises(ValueError, match=r"Invalid variant tag"):
            decode(invalid_tag, 0)

    def test_should_throw_on_excess_data_after_value(self):
        """Should throw on excess data after value."""
        from east.serialization.binary_utils import BufferWriter

        encode = encode_beast_value_to_buffer_for(IntegerType)
        decode = decode_beast_value_for(IntegerType)

        writer = BufferWriter()
        encode(42, writer)
        encoded = writer.to_bytes()

        with_excess = bytearray(encoded)
        with_excess.extend(bytes(10))
        with_excess[len(encoded)] = 0xFF  # Extra byte

        # This test checks that the decoder properly reports where it stopped
        value, offset = decode(bytes(with_excess), 0)
        assert (
            offset != len(with_excess)
        ), f"Should have excess data: decoded to offset {offset}, but buffer has {len(with_excess)} bytes"


class TestBeastFuzz:
    """Fuzz tests for BEAST binary serialization."""

    def test_fuzz_round_trip_random_types(self):
        """Test that random types and values round-trip correctly through BEAST encoding."""
        import asyncio

        from east.testing.fuzz import fuzzer_test
        from east.utils.ordering import equal_for

        async def run_fuzz():
            def test_factory(type_val):
                encode = encode_beast_for(type_val)
                decode = decode_beast_for(type_val)
                equal = equal_for(type_val)

                def test_value(value):
                    # Encode and decode
                    encoded = encode(value)
                    decoded = decode(encoded)

                    # Check value equality
                    if not equal(decoded, value):
                        raise AssertionError("Round-trip failed: values not equal")

                return test_value

            result = await fuzzer_test(test_factory, n_types=100, n_samples=10)
            assert result is True, "Fuzz test failed"

        asyncio.run(run_fuzz())
