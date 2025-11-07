"""Tests for JSON serialization.

Based on /home/crambelsoupy/src/East/src/serialization/json.spec.ts
"""

import pytest
from datetime import datetime, timezone

from east.serialization.json import to_json_for, from_json_for, JSONDecodeError, encode_json_for, decode_json_for
from east.types.containers import EastArray, EastSet, EastDict
from east.types.primitives import Blob, null
from east.types.type_system import (
    NullType,
    BooleanType,
    IntegerType,
    FloatType,
    StringType,
    DateTimeType,
    BlobType,
    ArrayType,
    SetType,
    DictType,
    StructType,
    VariantType,
    recursive_type,
    NeverType,
    FunctionType,
    RecursiveTypeRef,
)
from east.types.structural import make_case


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
        dt1 = datetime(1970, 1, 1, 0, 0, 0, 0, tzinfo=timezone.utc)
        dt2 = datetime(2022, 6, 29, 13, 43, 0, 123000, tzinfo=timezone.utc)

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
        dt1 = datetime(1970, 1, 1, 0, 0, 0, 0, tzinfo=timezone.utc)
        dt2 = datetime(2022, 6, 29, 13, 43, 0, 123000, tzinfo=timezone.utc)
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
        assert sorted(list(decoded)) == ["abc", "def"]

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
        dt1 = datetime(1970, 1, 1, 0, 0, 0, 0, tzinfo=timezone.utc)
        dt2 = datetime(2022, 6, 29, 13, 43, 0, 123000, tzinfo=timezone.utc)
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
            from_json([{"key": "abc", "value": "1970-01-01T00:00:00.000+00:00", "extra": "naughty"}])  # Extra field

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
        dt1 = datetime(1970, 1, 1, 0, 0, 0, 0, tzinfo=timezone.utc)
        dt2 = datetime(2022, 6, 29, 13, 43, 0, 123000, tzinfo=timezone.utc)

        obj1 = {"boolean": True, "string": "good", "date": dt1}
        encoded1 = to_json(obj1)
        assert encoded1 == {"boolean": True, "string": "good", "date": "1970-01-01T00:00:00.000+00:00"}

        obj2 = {"boolean": False, "string": "bad", "date": dt2}
        encoded2 = to_json(obj2)
        assert encoded2 == {"boolean": False, "string": "bad", "date": "2022-06-29T13:43:00.123+00:00"}

        # Decode
        decoded1 = from_json(encoded1)
        assert decoded1["boolean"] is True
        assert decoded1["string"] == "good"

        # Invalid
        with pytest.raises(JSONDecodeError):
            from_json({"boolean": True, "string": "good"})  # Missing field
        with pytest.raises(JSONDecodeError):
            from_json({"boolean": True, "string": "good", "date": "1970-01-01T00:00:00.000"})  # Missing timezone
        with pytest.raises(JSONDecodeError):
            from_json({"boolean": True, "string": "good", "date": "1970-01-01T00:00:00.000+00:00", "extra": "naughty"})  # Extra field

    def test_encode_decode_variant(self):
        """Test Variant encoding/decoding."""
        type_val = VariantType([("none", NullType), ("some", DateTimeType)])
        to_json = to_json_for(type_val)
        from_json = from_json_for(type_val)

        # Variant instances
        v1 = {"type": "none", "value": null}
        encoded1 = to_json(v1)
        assert encoded1 == {"type": "none", "value": None}

        dt = datetime(2022, 6, 29, 13, 43, 0, 123000, tzinfo=timezone.utc)
        v2 = {"type": "some", "value": dt}
        encoded2 = to_json(v2)
        assert encoded2 == {"type": "some", "value": "2022-06-29T13:43:00.123+00:00"}

        # Decode
        decoded1 = from_json({"type": "none", "value": None})
        assert decoded1["type"] == "none"

        decoded2 = from_json({"type": "some", "value": "2022-06-29T13:43:00.123+00:00"})
        assert decoded2["type"] == "some"

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
        LinkedListType = recursive_type(
            lambda self: VariantType(
                [
                    ("nil", NullType),
                    ("cons", StructType([("head", IntegerType), ("tail", self)])),
                ]
            )
        )

        to_json = to_json_for(LinkedListType)
        from_json = from_json_for(LinkedListType)

        # nil
        nil = {"type": "nil", "value": null}
        encoded_nil = to_json(nil)
        assert encoded_nil == {"type": "nil", "value": None}
        decoded_nil = from_json({"type": "nil", "value": None})
        assert decoded_nil["type"] == "nil"

        # cons(1, nil)
        list1 = {"type": "cons", "value": {"head": 1, "tail": {"type": "nil", "value": null}}}
        encoded_list1 = to_json(list1)
        assert encoded_list1 == {"type": "cons", "value": {"head": "1", "tail": {"type": "nil", "value": None}}}
        decoded_list1 = from_json(encoded_list1)
        assert decoded_list1["type"] == "cons"
        assert decoded_list1["value"]["head"] == 1

        # cons(1, cons(2, cons(3, nil)))
        list3 = {
            "type": "cons",
            "value": {
                "head": 1,
                "tail": {
                    "type": "cons",
                    "value": {
                        "head": 2,
                        "tail": {"type": "cons", "value": {"head": 3, "tail": {"type": "nil", "value": null}}},
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
                        "tail": {"type": "cons", "value": {"head": "3", "tail": {"type": "nil", "value": None}}},
                    },
                },
            },
        }
        assert encoded_list3 == expected

        decoded_list3 = from_json(encoded_list3)
        assert decoded_list3["type"] == "cons"
        assert decoded_list3["value"]["head"] == 1

        # Invalid
        with pytest.raises(JSONDecodeError):
            from_json({"type": "cons"})  # Missing value
        with pytest.raises(JSONDecodeError):
            from_json({"type": "cons", "value": {}})  # Missing head
        with pytest.raises(JSONDecodeError):
            from_json({"type": "cons", "value": {"head": "1"}})  # Missing tail
        with pytest.raises(JSONDecodeError):
            from_json({"type": "cons", "value": {"head": "not an int", "tail": {"type": "nil", "value": None}}})

    def test_encode_decode_binary_tree(self):
        """Test binary tree (recursive type)."""
        # Tree = Variant<leaf: Integer, node: Struct<left: Tree, right: Tree>>
        TreeType = recursive_type(
            lambda self: VariantType(
                [("leaf", IntegerType), ("node", StructType([("left", self), ("right", self)]))]
            )
        )

        to_json = to_json_for(TreeType)
        from_json = from_json_for(TreeType)

        # leaf(42)
        leaf = {"type": "leaf", "value": 42}
        encoded_leaf = to_json(leaf)
        assert encoded_leaf == {"type": "leaf", "value": "42"}
        decoded_leaf = from_json({"type": "leaf", "value": "42"})
        assert decoded_leaf["type"] == "leaf"
        assert decoded_leaf["value"] == 42

        # node(leaf(1), leaf(2))
        tree1 = {"type": "node", "value": {"left": {"type": "leaf", "value": 1}, "right": {"type": "leaf", "value": 2}}}
        encoded_tree1 = to_json(tree1)
        assert encoded_tree1 == {
            "type": "node",
            "value": {"left": {"type": "leaf", "value": "1"}, "right": {"type": "leaf", "value": "2"}},
        }
        decoded_tree1 = from_json(encoded_tree1)
        assert decoded_tree1["type"] == "node"

        # node(node(leaf(1), leaf(2)), leaf(3))
        tree2 = {
            "type": "node",
            "value": {
                "left": {
                    "type": "node",
                    "value": {"left": {"type": "leaf", "value": 1}, "right": {"type": "leaf", "value": 2}},
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
                    "value": {"left": {"type": "leaf", "value": "1"}, "right": {"type": "leaf", "value": "2"}},
                },
                "right": {"type": "leaf", "value": "3"},
            },
        }
        assert encoded_tree2 == expected
        decoded_tree2 = from_json(encoded_tree2)
        assert decoded_tree2["type"] == "node"

        # Invalid
        with pytest.raises(JSONDecodeError):
            from_json({"type": "leaf"})  # Missing value
        with pytest.raises(JSONDecodeError):
            from_json({"type": "leaf", "value": "not an int"})
        with pytest.raises(JSONDecodeError):
            from_json({"type": "node", "value": {}})  # Missing left/right
        with pytest.raises(JSONDecodeError):
            from_json({"type": "node", "value": {"left": {"type": "leaf", "value": "1"}}})  # Missing right

    def test_encode_decode_tree_with_array_children(self):
        """Test tree with array children (recursive type)."""
        # Node = Struct<value: Integer, children: Array<Node>>
        NodeType = recursive_type(
            lambda self: StructType([("value", IntegerType), ("children", ArrayType(self))])
        )

        to_json = to_json_for(NodeType)
        from_json = from_json_for(NodeType)

        # Leaf node
        leaf = {"value": 1, "children": []}
        encoded_leaf = to_json(leaf)
        assert encoded_leaf == {"value": "1", "children": []}
        decoded_leaf = from_json({"value": "1", "children": []})
        assert decoded_leaf["value"] == 1
        assert len(decoded_leaf["children"]) == 0

        # Node with 2 children
        node1 = {"value": 1, "children": [{"value": 2, "children": []}, {"value": 3, "children": []}]}
        encoded_node1 = to_json(node1)
        assert encoded_node1 == {"value": "1", "children": [{"value": "2", "children": []}, {"value": "3", "children": []}]}
        decoded_node1 = from_json(encoded_node1)
        assert decoded_node1["value"] == 1
        assert len(decoded_node1["children"]) == 2

        # Nested tree
        node2 = {
            "value": 1,
            "children": [
                {"value": 2, "children": [{"value": 4, "children": []}, {"value": 5, "children": []}]},
                {"value": 3, "children": []},
            ],
        }
        encoded_node2 = to_json(node2)
        expected = {
            "value": "1",
            "children": [
                {"value": "2", "children": [{"value": "4", "children": []}, {"value": "5", "children": []}]},
                {"value": "3", "children": []},
            ],
        }
        assert encoded_node2 == expected
        decoded_node2 = from_json(encoded_node2)
        assert decoded_node2["value"] == 1

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
        GraphNodeType = recursive_type(
            lambda self: StructType([("label", StringType), ("edges", ArrayType(self))])
        )

        to_json = to_json_for(GraphNodeType)
        from_json = from_json_for(GraphNodeType)

        # Single node
        node = {"label": "A", "edges": []}
        encoded = to_json(node)
        assert encoded == {"label": "A", "edges": []}
        decoded = from_json({"label": "A", "edges": []})
        assert decoded["label"] == "A"

        # Node with edges
        graph = {"label": "A", "edges": [{"label": "B", "edges": []}, {"label": "C", "edges": []}]}
        encoded_graph = to_json(graph)
        assert encoded_graph == {"label": "A", "edges": [{"label": "B", "edges": []}, {"label": "C", "edges": []}]}
        decoded_graph = from_json(encoded_graph)
        assert decoded_graph["label"] == "A"
        assert len(decoded_graph["edges"]) == 2

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
        ExprType = recursive_type(
            lambda self: VariantType(
                [
                    ("num", IntegerType),
                    ("add", StructType([("left", self), ("right", self)])),
                    ("mul", StructType([("left", self), ("right", self)])),
                ]
            )
        )

        to_json = to_json_for(ExprType)
        from_json = from_json_for(ExprType)

        # num(42)
        num = {"type": "num", "value": 42}
        encoded_num = to_json(num)
        assert encoded_num == {"type": "num", "value": "42"}
        decoded_num = from_json({"type": "num", "value": "42"})
        assert decoded_num["type"] == "num"
        assert decoded_num["value"] == 42

        # add(num(1), num(2))
        add = {"type": "add", "value": {"left": {"type": "num", "value": 1}, "right": {"type": "num", "value": 2}}}
        encoded_add = to_json(add)
        assert encoded_add == {
            "type": "add",
            "value": {"left": {"type": "num", "value": "1"}, "right": {"type": "num", "value": "2"}},
        }
        decoded_add = from_json(encoded_add)
        assert decoded_add["type"] == "add"

        # mul(add(num(2), num(3)), num(4))
        mul = {
            "type": "mul",
            "value": {
                "left": {
                    "type": "add",
                    "value": {"left": {"type": "num", "value": 2}, "right": {"type": "num", "value": 3}},
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
                    "value": {"left": {"type": "num", "value": "2"}, "right": {"type": "num", "value": "3"}},
                },
                "right": {"type": "num", "value": "4"},
            },
        }
        assert encoded_mul == expected
        decoded_mul = from_json(encoded_mul)
        assert decoded_mul["type"] == "mul"

        # Invalid
        with pytest.raises(JSONDecodeError):
            from_json({"type": "num"})  # Missing value
        with pytest.raises(JSONDecodeError):
            from_json({"type": "add", "value": {}})  # Missing left/right
        with pytest.raises(JSONDecodeError):
            from_json({"type": "add", "value": {"left": {"type": "num", "value": "1"}}})  # Missing right
        with pytest.raises(JSONDecodeError):
            from_json({"type": "unknown", "value": "1"})  # Unknown variant
