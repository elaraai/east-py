"""Tests for JSON serialization.

Based on /home/crambelsoupy/src/East/src/serialization/json.spec.ts
"""

from datetime import UTC, datetime

import pytest

from east.serialization.json import (
    JSONDecodeError,
    from_json_for,
    to_json_for,
)
from east.types.containers import EastArray, EastDict, EastSet
from east.types.primitives import Blob, null
from east.types.type_system import (
    ArrayType,
    BlobType,
    BooleanType,
    DateTimeType,
    DictType,
    FloatType,
    IntegerType,
    IRType,
    NullType,
    SetType,
    StringType,
    StructType,
    VariantType,
    recursive_type,
)


class TestJSONEncoding:
    """Test JSON encoding/decoding of East values."""

    def test_encode_decode_null(self):
        """Test null encoding/decoding."""
        type_val = NullType
        to_json = to_json_for(type_val)
        from_json = from_json_for(type_val)

        # Valid values
        assert to_json(null) is None
        assert from_json(None) == null

        # Invalid values
        with pytest.raises(JSONDecodeError):
            from_json(True)
        with pytest.raises(JSONDecodeError):
            from_json(1)
        with pytest.raises(JSONDecodeError):
            from_json("")
        with pytest.raises(JSONDecodeError):
            from_json([])

    def test_encode_decode_boolean(self):
        """Test boolean encoding/decoding."""
        type_val = BooleanType
        to_json = to_json_for(type_val)
        from_json = from_json_for(type_val)

        # Valid values
        assert to_json(True) is True
        assert to_json(False) is False
        assert from_json(True) is True
        assert from_json(False) is False

        # Invalid values
        with pytest.raises(JSONDecodeError):
            from_json(None)
        with pytest.raises(JSONDecodeError):
            from_json(1)
        with pytest.raises(JSONDecodeError):
            from_json("")

    def test_encode_decode_integer(self):
        """Test integer encoding/decoding."""
        type_val = IntegerType
        to_json = to_json_for(type_val)
        from_json = from_json_for(type_val)

        # Valid values
        assert to_json(0) == "0"
        assert to_json(42) == "42"
        assert to_json(-1) == "-1"
        assert to_json(90071992547409919) == "90071992547409919"
        assert to_json(2**63 - 1) == "9223372036854775807"
        assert to_json(-(2**63)) == "-9223372036854775808"

        assert from_json("0") == 0
        assert from_json("42") == 42
        assert from_json("-1") == -1
        assert from_json("90071992547409919") == 90071992547409919

        # Invalid values
        with pytest.raises(JSONDecodeError):
            from_json(42)  # Not a string
        with pytest.raises(JSONDecodeError):
            from_json("")
        with pytest.raises(JSONDecodeError):
            from_json("abc")
        # Out of range
        with pytest.raises(JSONDecodeError):
            from_json("9223372036854775808")
        with pytest.raises(JSONDecodeError):
            from_json("-9223372036854775809")

    def test_encode_decode_float(self):
        """Test float encoding/decoding."""
        type_val = FloatType
        to_json = to_json_for(type_val)
        from_json = from_json_for(type_val)

        # Normal floats
        assert to_json(0.0) == 0.0
        assert to_json(3.14) == 3.14
        assert to_json(-1e6) == -1e6

        # Special values
        assert to_json(float("inf")) == "Infinity"
        assert to_json(float("-inf")) == "-Infinity"
        assert to_json(float("nan")) == "NaN"

        # Decode
        assert from_json(0.0) == 0.0
        assert from_json(3.14) == 3.14
        assert from_json(-1e6) == -1e6
        assert from_json("Infinity") == float("inf")
        assert from_json("-Infinity") == float("-inf")
        assert str(from_json("NaN")) == "nan"

    def test_encode_decode_string(self):
        """Test string encoding/decoding."""
        type_val = StringType
        to_json = to_json_for(type_val)
        from_json = from_json_for(type_val)

        assert to_json("") == ""
        assert to_json("abc") == "abc"
        # Unicode string (Japanese hiragana)
        assert to_json("いろはにほへとちりぬるを") == "いろはにほへとちりぬるを"

        assert from_json("") == ""
        assert from_json("abc") == "abc"
        assert from_json("いろはにほへとちりぬるを") == "いろはにほへとちりぬるを"

        with pytest.raises(JSONDecodeError):
            from_json(123)

    def test_encode_decode_datetime(self):
        """Test DateTime encoding/decoding."""
        type_val = DateTimeType
        to_json = to_json_for(type_val)
        from_json = from_json_for(type_val)

        # Create datetimes in UTC
        dt1 = datetime(1970, 1, 1, 0, 0, 0, 0, tzinfo=UTC)
        dt2 = datetime(2022, 6, 29, 13, 43, 0, 123000, tzinfo=UTC)

        assert to_json(dt1) == "1970-01-01T00:00:00.000+00:00"
        assert to_json(dt2) == "2022-06-29T13:43:00.123+00:00"

        # Decode
        decoded1 = from_json("1970-01-01T00:00:00.000+00:00")
        assert decoded1.year == 1970
        assert decoded1.month == 1
        assert decoded1.day == 1

        decoded2 = from_json("2022-06-29T13:43:00.123+00:00")
        assert decoded2.year == 2022
        assert decoded2.month == 6
        assert decoded2.day == 29
        assert decoded2.hour == 13
        assert decoded2.minute == 43
        assert decoded2.second == 0
        assert decoded2.microsecond == 123000

        # Test Z timezone
        decoded_z = from_json("2022-06-29T13:43:00.123Z")
        assert decoded_z.year == 2022

        # Test offset timezone (should convert to UTC)
        decoded_offset = from_json("2022-06-29T13:43:00.123+05:00")
        assert decoded_offset.hour == 8  # 13:43 +05:00 = 08:43 +00:00

        # Invalid formats
        with pytest.raises(JSONDecodeError):
            from_json("1970-13-01T00:00:00.000+00:00")  # Invalid month
        with pytest.raises(JSONDecodeError):
            from_json("1970-01-01T00:00:00.000")  # Missing timezone
        with pytest.raises(JSONDecodeError):
            from_json("1970-01-01 00:00:00.000Z")  # Space instead of T
        with pytest.raises(JSONDecodeError):
            from_json("1970-01-01T00:00:00Z")  # Missing milliseconds
        with pytest.raises(JSONDecodeError):
            from_json("2022-06-29T13:43:00.123+5:00")  # Invalid offset format

    def test_encode_decode_blob(self):
        """Test Blob encoding/decoding."""
        type_val = BlobType
        to_json = to_json_for(type_val)
        from_json = from_json_for(type_val)

        # Empty blob
        blob = Blob(b"")
        assert to_json(blob) == "0x"
        assert from_json("0x").data == b""

        # Non-empty blob
        blob = Blob(bytes([1, 3, 3, 7]))
        assert to_json(blob) == "0x01030307"
        assert from_json("0x01030307").data == bytes([1, 3, 3, 7])

        # Invalid
        with pytest.raises(JSONDecodeError):
            from_json("abc")  # Missing 0x
        with pytest.raises(JSONDecodeError):
            from_json("0xgg")  # Invalid hex
        with pytest.raises(JSONDecodeError):
            from_json("0x123")  # Odd length

    def test_encode_decode_array(self):
        """Test Array encoding/decoding."""
        type_val = ArrayType(DateTimeType)
        to_json = to_json_for(type_val)
        from_json = from_json_for(type_val)

        # Empty array
        arr = EastArray(DateTimeType, [])
        assert to_json(arr) == []
        assert list(from_json([])) == []

        # Non-empty array
        dt1 = datetime(1970, 1, 1, 0, 0, 0, 0, tzinfo=UTC)
        dt2 = datetime(2022, 6, 29, 13, 43, 0, 123000, tzinfo=UTC)
        arr = EastArray(DateTimeType, [dt1, dt2])
        encoded = to_json(arr)
        assert encoded == ["1970-01-01T00:00:00.000+00:00", "2022-06-29T13:43:00.123+00:00"]

        decoded = from_json(encoded)
        assert len(decoded) == 2

        # Invalid
        with pytest.raises(JSONDecodeError):
            from_json("not an array")
        with pytest.raises(JSONDecodeError):
            from_json([None])
        with pytest.raises(JSONDecodeError):
            from_json([1])
        with pytest.raises(JSONDecodeError):
            from_json([[]])
        with pytest.raises(JSONDecodeError):
            from_json([{}])
        with pytest.raises(JSONDecodeError):
            from_json(["1970-01-01T00:00:00.000"])  # Missing timezone

    def test_encode_decode_set(self):
        """Test Set encoding/decoding."""
        type_val = SetType(StringType)
        to_json = to_json_for(type_val)
        from_json = from_json_for(type_val)

        # Empty set
        s = EastSet(StringType, [])
        assert to_json(s) == []
        assert list(from_json([])) == []

        # Non-empty set
        s = EastSet(StringType, ["abc", "def"])
        encoded = to_json(s)
        assert sorted(encoded) == ["abc", "def"]

        decoded = from_json(["abc", "def"])
        assert sorted(decoded) == ["abc", "def"]

        # Invalid
        with pytest.raises(JSONDecodeError):
            from_json("not an array")
        with pytest.raises(JSONDecodeError):
            from_json([None])
        with pytest.raises(JSONDecodeError):
            from_json([1])
        with pytest.raises(JSONDecodeError):
            from_json([[]])
        with pytest.raises(JSONDecodeError):
            from_json([{}])

    def test_encode_decode_dict(self):
        """Test Dict encoding/decoding."""
        type_val = DictType(StringType, DateTimeType)
        to_json = to_json_for(type_val)
        from_json = from_json_for(type_val)

        # Empty dict
        d = EastDict(StringType, DateTimeType, {})
        assert to_json(d) == []
        decoded = from_json([])
        assert len(decoded) == 0

        # Non-empty dict
        dt1 = datetime(1970, 1, 1, 0, 0, 0, 0, tzinfo=UTC)
        dt2 = datetime(2022, 6, 29, 13, 43, 0, 123000, tzinfo=UTC)
        d = EastDict(StringType, DateTimeType, {"abc": dt1, "def": dt2})
        encoded = to_json(d)
        assert len(encoded) == 2
        assert {"key": "abc", "value": "1970-01-01T00:00:00.000+00:00"} in encoded
        assert {"key": "def", "value": "2022-06-29T13:43:00.123+00:00"} in encoded

        # Decode
        decoded = from_json(encoded)
        assert len(decoded) == 2

        # Invalid
        with pytest.raises(JSONDecodeError):
            from_json([None])
        with pytest.raises(JSONDecodeError):
            from_json([1])
        with pytest.raises(JSONDecodeError):
            from_json(["abc"])
        with pytest.raises(JSONDecodeError):
            from_json([[]])
        with pytest.raises(JSONDecodeError):
            from_json([{}])
        with pytest.raises(JSONDecodeError):
            from_json([{"key": "abc"}])  # Missing value
        with pytest.raises(JSONDecodeError):
            from_json([{"value": "1970-01-01T00:00:00.000+00:00"}])  # Missing key
        with pytest.raises(JSONDecodeError):
            from_json([{"key": 1, "value": "1970-01-01T00:00:00.000+00:00"}])  # Wrong key type
        with pytest.raises(JSONDecodeError):
            from_json([{"key": "abc", "value": "1970-01-01T00:00:00.000"}])  # Missing timezone
        with pytest.raises(JSONDecodeError):
            from_json(
                [{"key": "abc", "value": "1970-01-01T00:00:00.000+00:00", "extra": "naughty"}]
            )  # Extra field

    def test_encode_decode_struct(self):
        """Test Struct encoding/decoding."""
        type_val = StructType(
            [
                ("boolean", BooleanType),
                ("string", StringType),
                ("date", DateTimeType),
            ]
        )
        to_json = to_json_for(type_val)
        from_json = from_json_for(type_val)

        # Struct instances
        dt1 = datetime(1970, 1, 1, 0, 0, 0, 0, tzinfo=UTC)
        dt2 = datetime(2022, 6, 29, 13, 43, 0, 123000, tzinfo=UTC)

        obj1 = {"boolean": True, "string": "good", "date": dt1}
        encoded1 = to_json(obj1)
        assert encoded1 == {
            "boolean": True,
            "string": "good",
            "date": "1970-01-01T00:00:00.000+00:00",
        }

        obj2 = {"boolean": False, "string": "bad", "date": dt2}
        encoded2 = to_json(obj2)
        assert encoded2 == {
            "boolean": False,
            "string": "bad",
            "date": "2022-06-29T13:43:00.123+00:00",
        }

        # Decode
        decoded1 = from_json(encoded1)
        assert decoded1.boolean is True
        assert decoded1.string == "good"

        # Invalid
        with pytest.raises(JSONDecodeError):
            from_json({"boolean": True, "string": "good"})  # Missing field
        with pytest.raises(JSONDecodeError):
            from_json(
                {"boolean": True, "string": "good", "date": "1970-01-01T00:00:00.000"}
            )  # Missing timezone
        with pytest.raises(JSONDecodeError):
            from_json(
                {
                    "boolean": True,
                    "string": "good",
                    "date": "1970-01-01T00:00:00.000+00:00",
                    "extra": "naughty",
                }
            )  # Extra field

    def test_encode_decode_variant(self):
        """Test Variant encoding/decoding."""
        type_val = VariantType(
            [
                ("none", NullType),
                ("some", DateTimeType),
            ]
        )
        to_json = to_json_for(type_val)
        from_json = from_json_for(type_val)

        # Variant instances
        v1 = {"type": "none", "value": null}
        encoded1 = to_json(v1)
        assert encoded1 == {"type": "none", "value": None}

        dt = datetime(2022, 6, 29, 13, 43, 0, 123000, tzinfo=UTC)
        v2 = {"type": "some", "value": dt}
        encoded2 = to_json(v2)
        assert encoded2 == {"type": "some", "value": "2022-06-29T13:43:00.123+00:00"}

        # Decode
        decoded1 = from_json({"type": "none", "value": None})
        assert decoded1.tag == "none"

        decoded2 = from_json({"type": "some", "value": "2022-06-29T13:43:00.123+00:00"})
        assert decoded2.tag == "some"

        # Invalid
        with pytest.raises(JSONDecodeError):
            from_json({"type": "none"})  # Missing value
        with pytest.raises(JSONDecodeError):
            from_json({"value": None})  # Missing type
        with pytest.raises(JSONDecodeError):
            from_json({"type": "nothing", "value": None})  # Unknown variant
        with pytest.raises(JSONDecodeError):
            from_json({"type": "none", "value": 1})  # Wrong value type

    def test_encode_decode_simple_linked_list(self):
        """Test simple linked list (recursive type)."""
        # LinkedList = Variant<nil: Null, cons: Struct<head: Integer, tail: LinkedList>>
        linked_list_type = recursive_type(
            lambda self: VariantType(
                [  # type: ignore[arg-type]
                    ("nil", NullType),
                    (
                        "cons",
                        StructType(
                            [
                                ("head", IntegerType),
                                ("tail", self),
                            ]
                        ),
                    ),
                ]
            )
        )

        to_json = to_json_for(linked_list_type)
        from_json = from_json_for(linked_list_type)

        # nil
        nil = {"type": "nil", "value": null}
        encoded_nil = to_json(nil)
        assert encoded_nil == {"type": "nil", "value": None}
        decoded_nil = from_json({"type": "nil", "value": None})
        assert decoded_nil.tag == "nil"

        # cons(1, nil)
        list1 = {"type": "cons", "value": {"head": 1, "tail": {"type": "nil", "value": null}}}
        encoded_list1 = to_json(list1)
        assert encoded_list1 == {
            "type": "cons",
            "value": {"head": "1", "tail": {"type": "nil", "value": None}},
        }
        decoded_list1 = from_json(encoded_list1)
        assert decoded_list1.tag == "cons"
        assert decoded_list1.value.head == 1

        # cons(1, cons(2, cons(3, nil)))
        list3 = {
            "type": "cons",
            "value": {
                "head": 1,
                "tail": {
                    "type": "cons",
                    "value": {
                        "head": 2,
                        "tail": {
                            "type": "cons",
                            "value": {"head": 3, "tail": {"type": "nil", "value": null}},
                        },
                    },
                },
            },
        }
        encoded_list3 = to_json(list3)
        expected = {
            "type": "cons",
            "value": {
                "head": "1",
                "tail": {
                    "type": "cons",
                    "value": {
                        "head": "2",
                        "tail": {
                            "type": "cons",
                            "value": {"head": "3", "tail": {"type": "nil", "value": None}},
                        },
                    },
                },
            },
        }
        assert encoded_list3 == expected

        decoded_list3 = from_json(encoded_list3)
        assert decoded_list3.tag == "cons"
        assert decoded_list3.value.head == 1

        # Invalid
        with pytest.raises(JSONDecodeError):
            from_json({"type": "cons"})  # Missing value
        with pytest.raises(JSONDecodeError):
            from_json({"type": "cons", "value": {}})  # Missing head
        with pytest.raises(JSONDecodeError):
            from_json({"type": "cons", "value": {"head": "1"}})  # Missing tail
        with pytest.raises(JSONDecodeError):
            from_json(
                {
                    "type": "cons",
                    "value": {"head": "not an int", "tail": {"type": "nil", "value": None}},
                }
            )

    def test_encode_decode_binary_tree(self):
        """Test binary tree (recursive type)."""
        # Tree = Variant<leaf: Integer, node: Struct<left: Tree, right: Tree>>
        tree_type = recursive_type(
            lambda self: VariantType(
                [  # type: ignore[arg-type]
                    ("leaf", IntegerType),
                    (
                        "node",
                        StructType(
                            [
                                ("left", self),
                                ("right", self),
                            ]
                        ),
                    ),
                ]
            )
        )

        to_json = to_json_for(tree_type)
        from_json = from_json_for(tree_type)

        # leaf(42)
        leaf = {"type": "leaf", "value": 42}
        encoded_leaf = to_json(leaf)
        assert encoded_leaf == {"type": "leaf", "value": "42"}
        decoded_leaf = from_json({"type": "leaf", "value": "42"})
        assert decoded_leaf.tag == "leaf"
        assert decoded_leaf.value == 42

        # node(leaf(1), leaf(2))
        tree1 = {
            "type": "node",
            "value": {"left": {"type": "leaf", "value": 1}, "right": {"type": "leaf", "value": 2}},
        }
        encoded_tree1 = to_json(tree1)
        assert encoded_tree1 == {
            "type": "node",
            "value": {
                "left": {"type": "leaf", "value": "1"},
                "right": {"type": "leaf", "value": "2"},
            },
        }
        decoded_tree1 = from_json(encoded_tree1)
        assert decoded_tree1.tag == "node"

        # node(node(leaf(1), leaf(2)), leaf(3))
        tree2 = {
            "type": "node",
            "value": {
                "left": {
                    "type": "node",
                    "value": {
                        "left": {"type": "leaf", "value": 1},
                        "right": {"type": "leaf", "value": 2},
                    },
                },
                "right": {"type": "leaf", "value": 3},
            },
        }
        encoded_tree2 = to_json(tree2)
        expected = {
            "type": "node",
            "value": {
                "left": {
                    "type": "node",
                    "value": {
                        "left": {"type": "leaf", "value": "1"},
                        "right": {"type": "leaf", "value": "2"},
                    },
                },
                "right": {"type": "leaf", "value": "3"},
            },
        }
        assert encoded_tree2 == expected
        decoded_tree2 = from_json(encoded_tree2)
        assert decoded_tree2.tag == "node"

        # Invalid
        with pytest.raises(JSONDecodeError):
            from_json({"type": "leaf"})  # Missing value
        with pytest.raises(JSONDecodeError):
            from_json({"type": "leaf", "value": "not an int"})
        with pytest.raises(JSONDecodeError):
            from_json({"type": "node", "value": {}})  # Missing left/right
        with pytest.raises(JSONDecodeError):
            from_json(
                {"type": "node", "value": {"left": {"type": "leaf", "value": "1"}}}
            )  # Missing right

    def test_encode_decode_tree_with_array_children(self):
        """Test tree with array children (recursive type)."""
        # Node = Struct<value: Integer, children: Array<Node>>
        node_type = recursive_type(
            lambda self: StructType(
                [
                    ("value", IntegerType),
                    ("children", ArrayType(self)),
                ]
            )
        )

        to_json = to_json_for(node_type)
        from_json = from_json_for(node_type)

        # Leaf node
        leaf = {"value": 1, "children": []}
        encoded_leaf = to_json(leaf)
        assert encoded_leaf == {"value": "1", "children": []}
        decoded_leaf = from_json({"value": "1", "children": []})
        assert decoded_leaf.value == 1
        assert len(decoded_leaf.children) == 0

        # Node with 2 children
        node1 = {
            "value": 1,
            "children": [{"value": 2, "children": []}, {"value": 3, "children": []}],
        }
        encoded_node1 = to_json(node1)
        assert encoded_node1 == {
            "value": "1",
            "children": [{"value": "2", "children": []}, {"value": "3", "children": []}],
        }
        decoded_node1 = from_json(encoded_node1)
        assert decoded_node1.value == 1
        assert len(decoded_node1.children) == 2

        # Nested tree
        node2 = {
            "value": 1,
            "children": [
                {
                    "value": 2,
                    "children": [{"value": 4, "children": []}, {"value": 5, "children": []}],
                },
                {"value": 3, "children": []},
            ],
        }
        encoded_node2 = to_json(node2)
        expected = {
            "value": "1",
            "children": [
                {
                    "value": "2",
                    "children": [{"value": "4", "children": []}, {"value": "5", "children": []}],
                },
                {"value": "3", "children": []},
            ],
        }
        assert encoded_node2 == expected
        decoded_node2 = from_json(encoded_node2)
        assert decoded_node2.value == 1

        # Invalid
        with pytest.raises(JSONDecodeError):
            from_json({"value": "1"})  # Missing children
        with pytest.raises(JSONDecodeError):
            from_json({"children": []})  # Missing value
        with pytest.raises(JSONDecodeError):
            from_json({"value": "not an int", "children": []})
        with pytest.raises(JSONDecodeError):
            from_json({"value": "1", "children": [{}]})  # Invalid child

    def test_encode_decode_graph_with_string_labels(self):
        """Test graph with string labels (recursive type)."""
        # GraphNode = Struct<label: String, edges: Array<GraphNode>>
        graph_node_type = recursive_type(
            lambda self: StructType(
                [
                    ("label", StringType),
                    ("edges", ArrayType(self)),
                ]
            )
        )

        to_json = to_json_for(graph_node_type)
        from_json = from_json_for(graph_node_type)

        # Single node
        node = {"label": "A", "edges": []}
        encoded = to_json(node)
        assert encoded == {"label": "A", "edges": []}
        decoded = from_json({"label": "A", "edges": []})
        assert decoded.label == "A"

        # Node with edges
        graph = {"label": "A", "edges": [{"label": "B", "edges": []}, {"label": "C", "edges": []}]}
        encoded_graph = to_json(graph)
        assert encoded_graph == {
            "label": "A",
            "edges": [{"label": "B", "edges": []}, {"label": "C", "edges": []}],
        }
        decoded_graph = from_json(encoded_graph)
        assert decoded_graph.label == "A"
        assert len(decoded_graph.edges) == 2

        # Invalid
        with pytest.raises(JSONDecodeError):
            from_json({"label": "A"})  # Missing edges
        with pytest.raises(JSONDecodeError):
            from_json({"edges": []})  # Missing label
        with pytest.raises(JSONDecodeError):
            from_json({"label": 123, "edges": []})  # Wrong label type
        with pytest.raises(JSONDecodeError):
            from_json({"label": "A", "edges": [{"label": "B"}]})  # Invalid edge

    def test_encode_decode_nested_variant_structures(self):
        """Test nested variant structures (recursive type) - expression trees."""
        # Expr = Variant<num: Integer, add: Struct<left: Expr, right: Expr>, mul: Struct<left: Expr, right: Expr>>
        expr_type = recursive_type(
            lambda self: VariantType(
                [  # type: ignore[arg-type]
                    ("num", IntegerType),
                    (
                        "add",
                        StructType(
                            [
                                ("left", self),
                                ("right", self),
                            ]
                        ),
                    ),
                    (
                        "mul",
                        StructType(
                            [
                                ("left", self),
                                ("right", self),
                            ]
                        ),
                    ),
                ]
            )
        )

        to_json = to_json_for(expr_type)
        from_json = from_json_for(expr_type)

        # num(42)
        num = {"type": "num", "value": 42}
        encoded_num = to_json(num)
        assert encoded_num == {"type": "num", "value": "42"}
        decoded_num = from_json({"type": "num", "value": "42"})
        assert decoded_num.tag == "num"
        assert decoded_num.value == 42

        # add(num(1), num(2))
        add = {
            "type": "add",
            "value": {"left": {"type": "num", "value": 1}, "right": {"type": "num", "value": 2}},
        }
        encoded_add = to_json(add)
        assert encoded_add == {
            "type": "add",
            "value": {
                "left": {"type": "num", "value": "1"},
                "right": {"type": "num", "value": "2"},
            },
        }
        decoded_add = from_json(encoded_add)
        assert decoded_add.tag == "add"

        # mul(add(num(2), num(3)), num(4))
        mul = {
            "type": "mul",
            "value": {
                "left": {
                    "type": "add",
                    "value": {
                        "left": {"type": "num", "value": 2},
                        "right": {"type": "num", "value": 3},
                    },
                },
                "right": {"type": "num", "value": 4},
            },
        }
        encoded_mul = to_json(mul)
        expected = {
            "type": "mul",
            "value": {
                "left": {
                    "type": "add",
                    "value": {
                        "left": {"type": "num", "value": "2"},
                        "right": {"type": "num", "value": "3"},
                    },
                },
                "right": {"type": "num", "value": "4"},
            },
        }
        assert encoded_mul == expected
        decoded_mul = from_json(encoded_mul)
        assert decoded_mul.tag == "mul"

        # Invalid
        with pytest.raises(JSONDecodeError):
            from_json({"type": "num"})  # Missing value
        with pytest.raises(JSONDecodeError):
            from_json({"type": "add", "value": {}})  # Missing left/right
        with pytest.raises(JSONDecodeError):
            from_json(
                {"type": "add", "value": {"left": {"type": "num", "value": "1"}}}
            )  # Missing right
        with pytest.raises(JSONDecodeError):
            from_json({"type": "unknown", "value": "1"})  # Unknown variant

    def test_fuzz_round_trip_random_types(self):
        """Test that random types and values round-trip correctly."""
        import asyncio

        from east.testing.fuzz import fuzzer_test
        from east.utils.ordering import equal_for

        async def run_fuzz():
            def test_factory(type_val):
                to_json = to_json_for(type_val)
                from_json = from_json_for(type_val)
                equal = equal_for(type_val)

                def test_value(value):
                    # Encode and decode
                    encoded = to_json(value)
                    decoded = from_json(encoded)

                    # Check value equality
                    if not equal(decoded, value):
                        raise AssertionError("Round-trip failed: values not equal")

                return test_value

            result = await fuzzer_test(test_factory, n_types=100, n_samples=10)
            assert result is True, "Fuzz test failed"

        asyncio.run(run_fuzz())


class TestJSONErrorMessages:
    """Test error message formatting during JSON decoding - EXACT message checks."""

    def test_error_wrong_type_boolean(self):
        """Test 1: Error for wrong type at root."""
        from_json = from_json_for(BooleanType)
        with pytest.raises(JSONDecodeError) as exc_info:
            from_json("not a boolean")
        assert (
            str(exc_info.value)
            == 'Error occurred because expected boolean, got "not a boolean" (line 1, col 1) while parsing value of type ".Boolean"'
        )

    def test_error_array_element(self):
        """Test 2: Error in array element with path [1]."""
        from_json = from_json_for(ArrayType(IntegerType))
        with pytest.raises(JSONDecodeError) as exc_info:
            from_json(["42", "not an integer"])
        assert (
            str(exc_info.value)
            == 'Error occurred because expected string representing integer, got "not an integer" at [1] (line 1, col 1) while parsing value of type ".Array .Integer"'
        )

    def test_error_struct_field(self):
        """Test 3: Error in struct field with path .age."""
        from_json = from_json_for(StructType([("name", StringType), ("age", IntegerType)]))
        with pytest.raises(JSONDecodeError) as exc_info:
            from_json({"name": "Alice", "age": 42})
        assert (
            str(exc_info.value)
            == 'Error occurred because expected string representing integer, got 42 at .age (line 1, col 1) while parsing value of type ".Struct [(name="name", type=.String), (name="age", type=.Integer)]"'
        )

    def test_error_missing_struct_field(self):
        """Test 4: Missing struct field "age"."""
        from_json = from_json_for(StructType([("name", StringType), ("age", IntegerType)]))
        with pytest.raises(JSONDecodeError) as exc_info:
            from_json({"name": "Alice"})
        assert (
            str(exc_info.value)
            == 'Error occurred because missing field "age" in Struct, got {"name": "Alice"} (line 1, col 1) while parsing value of type ".Struct [(name="name", type=.String), (name="age", type=.Integer)]"'
        )

    def test_error_unexpected_struct_field(self):
        """Test 5: Unexpected struct field "extra"."""
        from_json = from_json_for(StructType([("name", StringType)]))
        with pytest.raises(JSONDecodeError) as exc_info:
            from_json({"name": "Alice", "extra": "unexpected"})
        assert (
            str(exc_info.value)
            == 'Error occurred because unexpected field "extra" in Struct, got {"name": "Alice", "extra": "unexpected"} (line 1, col 1) while parsing value of type ".Struct [(name="name", type=.String)]"'
        )

    def test_error_dict_value(self):
        """Test 6: Error in dict entry value at [0].value."""
        from_json = from_json_for(DictType(StringType, IntegerType))
        with pytest.raises(JSONDecodeError) as exc_info:
            from_json([{"key": "a", "value": "not an integer"}])
        assert (
            str(exc_info.value)
            == 'Error occurred because expected string representing integer, got "not an integer" at [0].value (line 1, col 1) while parsing value of type ".Dict (key=.String, value=.Integer)"'
        )

    def test_error_variant_value(self):
        """Test 7: Error in variant case value at .some."""
        from_json = from_json_for(VariantType([("none", NullType), ("some", IntegerType)]))
        with pytest.raises(JSONDecodeError) as exc_info:
            from_json({"type": "some", "value": "not an integer"})
        assert (
            str(exc_info.value)
            == 'Error occurred because expected string representing integer, got "not an integer" at .some (line 1, col 1) while parsing value of type ".Variant [(name="none", type=.Null), (name="some", type=.Integer)]"'
        )

    def test_error_unknown_variant(self):
        """Test 8: Unknown variant type "unknown"."""
        from_json = from_json_for(VariantType([("none", NullType), ("some", IntegerType)]))
        with pytest.raises(JSONDecodeError) as exc_info:
            from_json({"type": "unknown", "value": None})
        assert (
            str(exc_info.value)
            == 'Error occurred because unknown variant type "unknown", got {"type": "unknown", "value": null} (line 1, col 1) while parsing value of type ".Variant [(name="none", type=.Null), (name="some", type=.Integer)]"'
        )

    def test_error_nested_array_struct(self):
        """Test 9: Deeply nested - array of structs error at [1].id."""
        from_json = from_json_for(
            ArrayType(StructType([("id", IntegerType), ("name", StringType)]))
        )
        with pytest.raises(JSONDecodeError) as exc_info:
            from_json([{"id": "1", "name": "Alice"}, {"id": "not an int", "name": "Bob"}])
        assert (
            str(exc_info.value)
            == 'Error occurred because expected string representing integer, got "not an int" at [1].id (line 1, col 1) while parsing value of type ".Array .Struct [(name="id", type=.Integer), (name="name", type=.String)]"'
        )

    def test_error_set_element(self):
        """Test 10: Set element error at [2]."""
        from_json = from_json_for(SetType(IntegerType))
        with pytest.raises(JSONDecodeError) as exc_info:
            from_json(["1", "2", "not an int"])
        assert (
            str(exc_info.value)
            == 'Error occurred because expected string representing integer, got "not an int" at [2] (line 1, col 1) while parsing value of type ".Set .Integer"'
        )

    def test_error_dict_array_nested(self):
        """Test 11: Dict with array values - error in array element at [0].value[2]."""
        from_json = from_json_for(DictType(StringType, ArrayType(IntegerType)))
        with pytest.raises(JSONDecodeError) as exc_info:
            from_json([{"key": "nums", "value": ["1", "2", "not an int"]}])
        assert (
            str(exc_info.value)
            == 'Error occurred because expected string representing integer, got "not an int" at [0].value[2] (line 1, col 1) while parsing value of type ".Dict (key=.String, value=.Array .Integer)"'
        )

    def test_error_struct_variant_nested(self):
        """Test 12: Struct containing variant with error at .result.ok."""
        from_json = from_json_for(
            StructType([("result", VariantType([("ok", IntegerType), ("error", StringType)]))])
        )
        with pytest.raises(JSONDecodeError) as exc_info:
            from_json({"result": {"type": "ok", "value": "not an int"}})
        assert (
            str(exc_info.value)
            == 'Error occurred because expected string representing integer, got "not an int" at .result.ok (line 1, col 1) while parsing value of type ".Struct [(name="result", type=.Variant [(name="error", type=.String), (name="ok", type=.Integer)])]"'
        )

    def test_error_dict_missing_key(self):
        """Test 13: Dict missing key field at [0]."""
        from_json = from_json_for(DictType(StringType, IntegerType))
        with pytest.raises(JSONDecodeError) as exc_info:
            from_json([{"value": "123"}])
        assert (
            str(exc_info.value)
            == 'Error occurred because expected object with key and value for Dict entry, got {"value": "123"} at [0] (line 1, col 1) while parsing value of type ".Dict (key=.String, value=.Integer)"'
        )

    def test_error_dict_extra_field(self):
        """Test 14: Dict extra field at [0]."""
        from_json = from_json_for(DictType(StringType, IntegerType))
        with pytest.raises(JSONDecodeError) as exc_info:
            from_json([{"key": "a", "value": "123", "extra": "bad"}])
        assert (
            str(exc_info.value)
            == 'Error occurred because unexpected field "extra" in Dict entry, got {"key": "a", "value": "123", "extra": "bad"} at [0] (line 1, col 1) while parsing value of type ".Dict (key=.String, value=.Integer)"'
        )

    def test_error_very_complex_nested(self):
        """Test 15: Very complex nested structure - error at [0].value[0].results[0].ok.items[2]."""
        very_complex_type = DictType(
            StringType,
            ArrayType(
                StructType(
                    [
                        (
                            "results",
                            ArrayType(
                                VariantType(
                                    [
                                        ("ok", StructType([("items", SetType(IntegerType))])),
                                        ("error", StringType),
                                    ]
                                )
                            ),
                        )
                    ]
                )
            ),
        )
        from_json = from_json_for(very_complex_type)
        with pytest.raises(JSONDecodeError) as exc_info:
            from_json(
                [
                    {
                        "key": "batch1",
                        "value": [
                            {
                                "results": [
                                    {"type": "ok", "value": {"items": ["1", "2", "not an int"]}}
                                ]
                            }
                        ],
                    }
                ]
            )
        assert (
            str(exc_info.value)
            == 'Error occurred because expected string representing integer, got "not an int" at [0].value[0].results[0].ok.items[2] (line 1, col 1) while parsing value of type ".Dict (key=.String, value=.Array .Struct [(name="results", type=.Array .Variant [(name="error", type=.String), (name="ok", type=.Struct [(name="items", type=.Set .Integer)])])])"'
        )

    def test_error_very_complex_missing_field(self):
        """Test 16: Very complex nested structure - missing field at [0].value[0].results[0].ok."""
        very_complex_type = DictType(
            StringType,
            ArrayType(
                StructType(
                    [
                        (
                            "results",
                            ArrayType(
                                VariantType(
                                    [
                                        ("ok", StructType([("items", SetType(IntegerType))])),
                                        ("error", StringType),
                                    ]
                                )
                            ),
                        )
                    ]
                )
            ),
        )
        from_json = from_json_for(very_complex_type)
        with pytest.raises(JSONDecodeError) as exc_info:
            from_json(
                [
                    {
                        "key": "batch1",
                        "value": [{"results": [{"type": "ok", "value": {"wrong": ["1"]}}]}],
                    }
                ]
            )
        assert (
            str(exc_info.value)
            == 'Error occurred because unexpected field "wrong" in Struct, got {"wrong": ["1"]} at [0].value[0].results[0].ok (line 1, col 1) while parsing value of type ".Dict (key=.String, value=.Array .Struct [(name="results", type=.Array .Variant [(name="error", type=.String), (name="ok", type=.Struct [(name="items", type=.Set .Integer)])])])"'
        )


class TestNeverAndFunctionTypes:
    """Test Never and Function type handling."""

    def test_should_throw_when_encoding_never_type(self):
        """Should throw when encoding Never type."""
        from east.types.type_system import NeverType

        with pytest.raises((ValueError, TypeError), match=r"[Cc]annot encode Never"):
            to_json_for(NeverType)

    def test_should_throw_when_decoding_never_type_with_from_json_for(self):
        """Should throw when decoding Never type with from_json_for."""
        from east.types.type_system import NeverType

        with pytest.raises((ValueError, TypeError), match=r"[Cc]annot decode Never"):
            from_json_for(NeverType)

    def test_should_throw_when_decoding_never_type_with_decode_json_for(self):
        """Should throw when decoding Never type with decode_json_for."""
        from east.serialization.json import decode_json_for
        from east.types.type_system import NeverType

        with pytest.raises((ValueError, TypeError), match=r"[Cc]annot decode Never"):
            decode_json_for(NeverType)

    def test_should_throw_when_encoding_function_type(self):
        """Should throw when encoding Function type."""
        from east.types.type_system import FunctionType

        func_type = FunctionType((), IntegerType, ())
        with pytest.raises(ValueError, match=r"[Cc]annot encode function"):
            to_json_for(func_type)

    def test_should_throw_when_decoding_function_type_with_from_json_for(self):
        """Should throw when decoding Function type with from_json_for."""
        from east.types.type_system import FunctionType

        func_type = FunctionType((), IntegerType, ())
        with pytest.raises(ValueError, match=r"[Cc]annot decode function"):
            from_json_for(func_type)

    def test_should_throw_when_creating_decoder_for_function_type(self):
        """Should throw when creating decoder for Function type with decode_json_for."""
        from east.serialization.json import decode_json_for
        from east.types.type_system import FunctionType

        func_type = FunctionType((), IntegerType, ())
        with pytest.raises(ValueError, match=r"[Cc]annot decode function"):
            decode_json_for(func_type)


class TestErrorPropagation:
    """Test that non-JSONDecodeError exceptions are properly re-thrown."""

    def test_should_rethrow_non_json_decode_error_in_array_decoding(self):
        """Should re-throw non-JSONDecodeError exceptions in Array decoding."""
        from east.serialization.json import from_json_for

        # Create a custom exception that isn't JSONDecodeError
        class CustomError(Exception):
            pass

        # We need to trigger a non-JSONDecodeError exception
        # One way is to pass something that causes Python's type system to fail
        # For example, passing an object that can't be compared
        array_type = ArrayType(IntegerType)
        from_json = from_json_for(array_type)

        # Pass something that will cause a non-JSONDecodeError
        # Using a complex nested object that triggers unexpected behavior
        with pytest.raises((TypeError, AttributeError, KeyError)):
            # This should cause an error that ISN'T JSONDecodeError
            from_json([object()])

    def test_should_rethrow_non_json_decode_error_in_set_decoding(self):
        """Should re-throw non-JSONDecodeError exceptions in Set decoding."""
        from east.serialization.json import from_json_for

        set_type = SetType(IntegerType)
        from_json = from_json_for(set_type)

        with pytest.raises((TypeError, AttributeError, KeyError)):
            from_json([object()])

    def test_should_rethrow_non_json_decode_error_in_dict_key_decoding(self):
        """Should re-throw non-JSONDecodeError exceptions in Dict key decoding."""
        from east.serialization.json import from_json_for

        dict_type = DictType(IntegerType, StringType)
        from_json = from_json_for(dict_type)

        with pytest.raises((TypeError, AttributeError, KeyError, JSONDecodeError)):
            from_json([{"key": object(), "value": "test"}])

    def test_should_rethrow_non_json_decode_error_in_dict_value_decoding(self):
        """Should re-throw non-JSONDecodeError exceptions in Dict value decoding."""
        from east.serialization.json import from_json_for

        dict_type = DictType(StringType, IntegerType)
        from_json = from_json_for(dict_type)

        with pytest.raises((TypeError, AttributeError, KeyError)):
            from_json([{"key": "test", "value": object()}])

    def test_should_rethrow_non_json_decode_error_in_struct_decoding(self):
        """Should re-throw non-JSONDecodeError exceptions in Struct decoding."""
        from east.serialization.json import from_json_for

        struct_type = StructType([("name", StringType), ("age", IntegerType)])
        from_json = from_json_for(struct_type)

        with pytest.raises((TypeError, AttributeError, KeyError)):
            from_json({"name": object(), "age": 30})

    def test_should_rethrow_non_json_decode_error_in_variant_decoding(self):
        """Should re-throw non-JSONDecodeError exceptions in Variant decoding."""
        from east.serialization.json import from_json_for

        variant_type = VariantType([("some", IntegerType), ("none", NullType)])
        from_json = from_json_for(variant_type)

        with pytest.raises((TypeError, AttributeError, KeyError)):
            from_json({"type": "some", "value": object()})


class TestJSONParseErrors:
    """Test error handling for malformed JSON input."""

    def test_should_handle_json_parse_syntax_errors(self):
        """Should handle JSON.parse syntax errors in decodeJSONFor."""
        import json

        from east.serialization.json import decode_json_for

        decode = decode_json_for(IntegerType)

        # Malformed JSON should raise an error
        with pytest.raises((json.JSONDecodeError, ValueError)):
            decode(b"{invalid json}")

    def test_should_handle_json_parse_errors_with_position(self):
        """Should track line and column numbers in JSON parse errors."""
        from east.serialization.json import decode_json_for

        decode = decode_json_for(ArrayType(IntegerType))

        # Malformed JSON with syntax error
        with pytest.raises(ValueError) as exc_info:
            decode(b'["1", "2",]')  # Trailing comma

        # Error message should contain position info
        error_msg = str(exc_info.value)
        assert "line" in error_msg.lower() and "col" in error_msg.lower()

    def test_should_handle_json_parse_errors_without_position_info(self):
        """Should handle JSON parse errors without position info."""
        import json

        from east.serialization.json import decode_json_for

        decode = decode_json_for(StringType)

        # Various malformed JSON inputs
        malformed_inputs = [
            b'{"unclosed": ',
            b"[1, 2, ",
            b"null null",
            b'"unterminated',
        ]

        for input_data in malformed_inputs:
            with pytest.raises((json.JSONDecodeError, ValueError)):
                decode(input_data)

    def test_should_handle_non_syntax_error_exceptions(self):
        """Should handle non-SyntaxError exceptions in decodeJSONFor."""
        from east.serialization.json import decode_json_for

        decode = decode_json_for(IntegerType)

        # Pass invalid input type (not bytes)
        with pytest.raises((TypeError, AttributeError)):
            decode(12345)  # Not bytes


class TestSharedReferences:
    """Test shared reference encoding with $ref JSON pointers."""

    def test_should_encode_shared_array_references(self):
        """Should encode shared array references within RecursiveType."""
        from east.serialization.json import from_json_for, to_json_for

        # RecursiveType imported as recursive_type

        # Create a type that can hold shared arrays
        node_type = recursive_type(
            lambda self: StructType(
                [
                    ("value", IntegerType),
                    ("children", ArrayType(self)),
                    ("metadata", ArrayType(StringType)),
                ]
            )
        )

        # Create a shared array that appears in multiple places
        from east.types.containers import EastArray

        shared_metadata = EastArray(StringType, ["tag1", "tag2"])
        value = {
            "value": 1,
            "children": EastArray(
                node_type,
                [
                    {
                        "value": 2,
                        "children": EastArray(node_type, []),
                        "metadata": shared_metadata,  # First reference
                    },
                    {
                        "value": 3,
                        "children": EastArray(node_type, []),
                        "metadata": shared_metadata,  # Second reference (same instance)
                    },
                ],
            ),
            "metadata": shared_metadata,  # Third reference (same instance)
        }

        to_json = to_json_for(node_type)
        from_json = from_json_for(node_type)
        encoded = to_json(value)

        # Check that references appear in the encoding
        assert encoded["children"][0]["metadata"] == ["tag1", "tag2"]
        assert encoded["children"][1]["metadata"] == {"$ref": "2#0/metadata"}
        assert encoded["metadata"] == {"$ref": "1#children/0/metadata"}

        # Check that decoding preserves shared references
        decoded = from_json(encoded)
        assert (
            decoded.metadata is decoded.children[0].metadata
        ), "metadata should be same instance as children[0].metadata"
        assert (
            decoded.metadata is decoded.children[1].metadata
        ), "metadata should be same instance as children[1].metadata"

    def test_should_encode_shared_set_references(self):
        """Should encode shared set references within RecursiveType."""
        from east.serialization.json import from_json_for, to_json_for
        from east.types.containers import EastArray, EastSet

        # RecursiveType imported as recursive_type

        node_type = recursive_type(
            lambda self: StructType(
                [
                    ("value", IntegerType),
                    ("children", ArrayType(self)),
                    ("tags", SetType(StringType)),
                ]
            )
        )

        shared_tags = EastSet(StringType, ["a", "b", "c"])

        value = {
            "value": 1,
            "children": EastArray(
                node_type,
                [
                    {
                        "value": 2,
                        "children": EastArray(node_type, []),
                        "tags": shared_tags,  # First reference
                    },
                ],
            ),
            "tags": shared_tags,  # Second reference (same instance)
        }

        to_json = to_json_for(node_type)
        from_json = from_json_for(node_type)
        encoded = to_json(value)

        # Check encoding (first occurrence is real data, rest are refs)
        assert set(encoded["children"][0]["tags"]) == {"a", "b", "c"}
        assert encoded["tags"] == {"$ref": "1#children/0/tags"}

        # Check decoding preserves shared references
        decoded = from_json(encoded)
        assert decoded.tags is decoded.children[0].tags, "tags should be same instance"

    def test_should_encode_shared_dict_references(self):
        """Should encode shared dict references within RecursiveType."""
        from east.serialization.json import from_json_for, to_json_for
        from east.types.containers import EastArray, EastDict

        # RecursiveType imported as recursive_type

        node_type = recursive_type(
            lambda self: StructType(
                [
                    ("value", IntegerType),
                    ("children", ArrayType(self)),
                    ("properties", DictType(StringType, IntegerType)),
                ]
            )
        )

        shared_props = EastDict(StringType, IntegerType, {"x": 10, "y": 20})

        value = {
            "value": 1,
            "children": EastArray(
                node_type,
                [
                    {
                        "value": 2,
                        "children": EastArray(node_type, []),
                        "properties": shared_props,
                    },
                    {
                        "value": 3,
                        "children": EastArray(node_type, []),
                        "properties": shared_props,
                    },
                ],
            ),
            "properties": shared_props,
        }

        to_json = to_json_for(node_type)
        from_json = from_json_for(node_type)
        encoded = to_json(value)

        # Check that references appear in the encoding
        assert encoded["children"][0]["properties"] == [
            {"key": "x", "value": "10"},
            {"key": "y", "value": "20"},
        ]
        assert encoded["children"][1]["properties"] == {"$ref": "2#0/properties"}
        assert encoded["properties"] == {"$ref": "1#children/0/properties"}

        # Check that decoding preserves shared references
        decoded = from_json(encoded)
        assert (
            decoded.properties is decoded.children[0].properties
        ), "properties should be same instance as children[0].properties"
        assert (
            decoded.properties is decoded.children[1].properties
        ), "properties should be same instance as children[1].properties"


class TestJSONPointerEscaping:
    """Test JSON Pointer escaping in field names."""

    def test_should_handle_json_pointer_escaping_in_field_names(self):
        """Should handle JSON Pointer escaping in field names with special characters."""
        from east.serialization.json import from_json_for, to_json_for
        from east.types.containers import EastArray

        # RecursiveType imported as recursive_type

        # Create a struct with field names that need escaping (~ and /)
        node_type = recursive_type(
            lambda self: StructType(
                [
                    ("field/with/slashes", IntegerType),
                    ("field~with~tildes", IntegerType),
                    ("normal", ArrayType(IntegerType)),
                    ("children", ArrayType(self)),
                ]
            )
        )

        shared_array = EastArray(IntegerType, [1, 2, 3])
        value = {
            "field/with/slashes": 42,
            "field~with~tildes": 99,
            "normal": shared_array,
            "children": EastArray(
                node_type,
                [
                    {
                        "field/with/slashes": 10,
                        "field~with~tildes": 20,
                        "normal": shared_array,  # Shared reference
                        "children": EastArray(node_type, []),
                    },
                ],
            ),
        }

        to_json = to_json_for(node_type)
        from_json = from_json_for(node_type)
        encoded = to_json(value)

        # Check encoding - shared array should have reference with escaped field names
        assert encoded["normal"] == ["1", "2", "3"]
        # The reference path should have properly escaped special chars:
        # ~ becomes ~0, / becomes ~1
        assert encoded["children"][0]["normal"] == {"$ref": "3#normal"}

        # Check decoding preserves shared references
        decoded = from_json(encoded)
        assert decoded.normal is decoded.children[0].normal, "normal should be same instance"

        # Check values are correct
        assert getattr(decoded, "field/with/slashes") == 42
        assert getattr(decoded, "field~with~tildes") == 99


class TestIRTypes:
    """Test IR type handling."""

    def test_deserialize_increment_function(self):
        """Test deserializing a simple increment function from JSON.

        The function is: (x: Integer) -> x + 1
        """
        # JSON generated from TypeScript East compiler
        increment_ir_json = {
            "type": "Function",
            "value": {
                "type": {
                    "type": "Function",
                    "value": {
                        "inputs": [{"type": "Integer", "value": None}],
                        "output": {"type": "Integer", "value": None},
                        "platforms": [],
                    },
                },
                "location": {
                    "filename": "node:internal/modules/esm/loader",
                    "line": "651",
                    "column": "26",
                },
                "captures": [],
                "parameters": [
                    {
                        "type": "Variable",
                        "value": {
                            "type": {"type": "Integer", "value": None},
                            "name": "_0",
                            "location": {
                                "filename": "node:internal/modules/esm/loader",
                                "line": "651",
                                "column": "26",
                            },
                            "mutable": False,
                            "captured": False,
                        },
                    }
                ],
                "body": {
                    "type": "Builtin",
                    "value": {
                        "type": {"type": "Integer", "value": None},
                        "location": {
                            "filename": "node:internal/modules/esm/loader",
                            "line": "651",
                            "column": "26",
                        },
                        "builtin": "IntegerAdd",
                        "type_parameters": [],
                        "arguments": [
                            {
                                "type": "Variable",
                                "value": {
                                    "type": {"type": "Integer", "value": None},
                                    "name": "_0",
                                    "location": {
                                        "filename": "node:internal/modules/esm/loader",
                                        "line": "651",
                                        "column": "26",
                                    },
                                    "mutable": False,
                                    "captured": False,
                                },
                            },
                            {
                                "type": "Value",
                                "value": {
                                    "type": {"type": "Integer", "value": None},
                                    "location": {
                                        "filename": "node:internal/modules/esm/loader",
                                        "line": "651",
                                        "column": "26",
                                    },
                                    "value": {"type": "Integer", "value": "1"},
                                },
                            },
                        ],
                    },
                },
            },
        }

        # Create decoder for IR type
        decode_ir = from_json_for(IRType)

        # Deserialize the JSON to IR
        ir = decode_ir(increment_ir_json)

        # Construct expected IR using builders
        from east.ir.builders import ir_builtin, ir_function, ir_value, ir_variable, location
        from east.types.type_system import FunctionType, IntegerType
        from east.utils.ordering import equal_for

        loc = location("node:internal/modules/esm/loader", 651, 26)

        # Parameter: _0
        param = ir_variable(IntegerType, "_0", loc, mutable=False, captured=False)

        # Body: IntegerAdd(_0, 1)
        value_1 = ir_value(IntegerType, loc, 1)
        body = ir_builtin(IntegerType, loc, "IntegerAdd", [], [param, value_1])

        # Function type
        func_type = FunctionType([IntegerType], IntegerType, [])

        # Complete function
        expected = ir_function(func_type, loc, [], [param], body)

        # Deep comparison using type-aware equality
        equal_fn = equal_for(IRType)
        assert equal_fn(ir, expected), f"IR values not equal:\nGot: {ir}\nExpected: {expected}"

    def test_serialize_increment_function(self):
        """Test serializing a simple increment function to JSON.

        The function is: (x: Integer) -> x + 1
        """
        from east.ir.builders import ir_builtin, ir_function, ir_value, ir_variable, location
        from east.types.type_system import FunctionType, IntegerType

        # Build the IR using builders
        loc = location("node:internal/modules/esm/loader", 651, 26)

        # Parameter: _0
        param = ir_variable(IntegerType, "_0", loc, mutable=False, captured=False)

        # Body: IntegerAdd(_0, 1)
        value_1 = ir_value(IntegerType, loc, 1)
        body = ir_builtin(IntegerType, loc, "IntegerAdd", [], [param, value_1])

        # Function type
        func_type = FunctionType([IntegerType], IntegerType, [])

        # Complete function
        ir = ir_function(func_type, loc, [], [param], body)

        # Serialize to JSON
        encode_ir = to_json_for(IRType)
        json_output = encode_ir(ir)

        # Expected JSON (same as in test_deserialize_increment_function)
        expected_json = {
            "type": "Function",
            "value": {
                "type": {
                    "type": "Function",
                    "value": {
                        "inputs": [{"type": "Integer", "value": None}],
                        "output": {"type": "Integer", "value": None},
                        "platforms": [],
                    },
                },
                "location": {
                    "filename": "node:internal/modules/esm/loader",
                    "line": "651",
                    "column": "26",
                },
                "captures": [],
                "parameters": [
                    {
                        "type": "Variable",
                        "value": {
                            "type": {"type": "Integer", "value": None},
                            "name": "_0",
                            "location": {
                                "filename": "node:internal/modules/esm/loader",
                                "line": "651",
                                "column": "26",
                            },
                            "mutable": False,
                            "captured": False,
                        },
                    }
                ],
                "body": {
                    "type": "Builtin",
                    "value": {
                        "type": {"type": "Integer", "value": None},
                        "location": {
                            "filename": "node:internal/modules/esm/loader",
                            "line": "651",
                            "column": "26",
                        },
                        "builtin": "IntegerAdd",
                        "type_parameters": [],
                        "arguments": [
                            {
                                "type": "Variable",
                                "value": {
                                    "type": {"type": "Integer", "value": None},
                                    "name": "_0",
                                    "location": {
                                        "filename": "node:internal/modules/esm/loader",
                                        "line": "651",
                                        "column": "26",
                                    },
                                    "mutable": False,
                                    "captured": False,
                                },
                            },
                            {
                                "type": "Value",
                                "value": {
                                    "type": {"type": "Integer", "value": None},
                                    "location": {
                                        "filename": "node:internal/modules/esm/loader",
                                        "line": "651",
                                        "column": "26",
                                    },
                                    "value": {"type": "Integer", "value": "1"},
                                },
                            },
                        ],
                    },
                },
            },
        }

        # Deep comparison
        assert json_output == expected_json
