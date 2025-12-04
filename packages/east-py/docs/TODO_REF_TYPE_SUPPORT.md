# TODO: Implement Ref Type Support

**Reference Commit:** 466b99d7b09319f5ea9446e1eabfd46cccd64d52 in ../East
**Base Commit:** 792c7bd9de2df38698b6fa3b2d2d3a63fce39121 in ../East
**Feature:** Add `ref` type - mutable reference cells with identity semantics

## Overview

The East TypeScript implementation added a new `ref` type for mutable reference cells. This type provides:
- **Mutable containers** with single-value storage
- **Identity semantics** (refs are compared by identity, not structure)
- **Aliasing support** in serialization (multiple references to same cell)
- **Nominal typing** via brand symbol
- **Invariant type system** (no subtyping)

**Use cases:**
- Mutable state in functional code
- Shared mutable references (aliasing)
- Circular data structures
- State cells that need identity

**Similar to:**
- OCaml/ML `ref` type
- Scheme/Racket boxes
- Clojure atoms (but simpler - no transactional semantics)
- C++ `std::shared_ptr` (but for mutability, not memory management)

---

## 1. Create Ref Container Module

### File: `east/types/ref.py` (NEW FILE)

**Location:** New module in `east/types/`

#### Task 1.1: Create Ref class with brand symbol

**Required implementation:**

```python
"""East ref type - mutable reference cells with identity semantics.

Ref-cells provide mutable reference containers with identity semantics.
Similar to OCaml's ref type or Scheme boxes.
"""

from __future__ import annotations

from typing import Any, Generic, TypeVar

T = TypeVar('T')

# Symbol for nominal typing (brand)
REF_SYMBOL = object()


class Ref(Generic[T]):
    """Mutable reference cell containing a value.

    Ref-cells are mutable containers with identity semantics:
    - Two refs are equal only if they're the same object (Object.is)
    - Or if their contents are deeply equal and not circular
    - Refs support aliasing in serialization
    - Refs are invariant in the type system

    Examples:
        >>> counter = ref(0)
        >>> set_ref(counter, deref(counter) + 1)
        >>> deref(counter)
        1

        >>> # Aliasing - both variables point to same ref-cell
        >>> r1 = ref([1, 2, 3])
        >>> r2 = r1  # Same ref-cell
        >>> set_ref(r2, [4, 5, 6])
        >>> deref(r1)
        [4, 5, 6]

    Attributes:
        value: The mutable value contained in this ref-cell
        _brand: Brand symbol for nominal typing (private)
    """

    __slots__ = ('value', '_brand')

    def __init__(self, value: T):
        """Create a new ref-cell (use ref() function instead)."""
        self.value: T = value
        self._brand = REF_SYMBOL

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"ref({self.value!r})"

    def __str__(self) -> str:
        """Human-readable string."""
        return f"&{self.value}"


def ref(value: T) -> Ref[T]:
    """Create a new mutable reference cell containing the specified value.

    Args:
        value: The initial value to store in the ref-cell

    Returns:
        A branded ref-cell object

    Examples:
        >>> counter = ref(0)
        >>> set_ref(counter, deref(counter) + 1)
        >>> deref(counter)
        1

        >>> # Refs have identity semantics
        >>> original = ref([1, 2, 3])
        >>> alias = original  # Same ref-cell
        >>> set_ref(alias, [4, 5, 6])
        >>> deref(original)
        [4, 5, 6]
    """
    return Ref(value)


def is_ref(v: Any) -> bool:
    """Check if a value is a ref-cell.

    Args:
        v: The value to check

    Returns:
        True if the value is a ref-cell, False otherwise

    Examples:
        >>> r = ref(42)
        >>> is_ref(r)
        True
        >>> is_ref(42)
        False
        >>> is_ref({'value': 42})
        False
    """
    return isinstance(v, Ref) and hasattr(v, '_brand') and v._brand is REF_SYMBOL


def deref(r: Ref[T]) -> T:
    """Retrieve the current value from a ref-cell.

    Args:
        r: The ref-cell to dereference

    Returns:
        The current value stored in the ref-cell

    Examples:
        >>> counter = ref(10)
        >>> deref(counter)
        10
    """
    return r.value


def set_ref(r: Ref[T], value: T) -> None:
    """Update the value stored in a ref-cell.

    This mutates the ref-cell in place. All aliases to the same
    ref-cell will see the updated value.

    Args:
        r: The ref-cell to update
        value: The new value to store

    Examples:
        >>> counter = ref(0)
        >>> set_ref(counter, 1)
        >>> set_ref(counter, deref(counter) + 1)
        >>> deref(counter)
        2

        >>> # Aliasing - both variables point to same ref-cell
        >>> r1 = ref("hello")
        >>> r2 = r1
        >>> set_ref(r2, "world")
        >>> deref(r1)
        'world'
    """
    r.value = value


__all__ = ['Ref', 'ref', 'is_ref', 'deref', 'set_ref', 'REF_SYMBOL']
```

**Notes:**
- Use `__slots__` for memory efficiency
- Brand symbol ensures nominal typing (can't create ref by accident)
- Generic type hints for type safety
- Follow Python conventions (snake_case)

---

## 2. Type System Changes

### File: `east/types/type_system.py`

**Location:** Type constructor functions and type checking

#### Task 2.1: Add RefType constructor

**Location:** After DictType (~line 346)

**Required addition:**

```python
def RefType(value_type: EastType) -> EastType:
    """Create a ref type.

    Args:
        value_type: Type of the referenced value (must be data type)

    Returns:
        Ref type

    Raises:
        TypeError: If value_type is not a data type

    Examples:
        >>> RefType(IntegerType)  # ref<Integer>
        >>> RefType(ArrayType(StringType))  # ref<Array<String>>
    """
    if not is_data_type(value_type):
        from east.serialization.east_printer import print_type

        raise TypeError(
            f"Ref value type must be a (non-function) data type, got {print_type(value_type)}"
        )
    return EastType(make_case("Ref", value_type))
```

**Location:** After line 198 (primitive types)

**Notes:**
- Refs can only contain data types (not functions)
- Refs themselves are mutable, but that's okay
- Refs are invariant (no subtyping)

#### Task 2.2: Update is_data_type to handle Ref

**Location:** is_data_type function (~line 202-242)

**Required change:**

Add after line 219 (`if tag == "Function"`):

```python
if tag == "Ref":
    # Refs are data types (serializable)
    # Constructor already validates inner type is data type
    return True
```

#### Task 2.3: Update is_immutable_type to handle Ref

**Location:** is_immutable_type function (~line 245-276)

**Required change:**

Change line 264:

```python
# Before:
if tag in ("Array", "Set", "Dict", "Function"):
    return False

# After:
if tag in ("Array", "Set", "Dict", "Ref", "Function"):
    return False
```

**Rationale:** Refs are mutable like arrays/sets/dicts

#### Task 2.4: Update EastTypeType definition

**Location:** EastTypeType recursive definition (~line 559-588)

**Required change:**

Add "Ref" case to the variant (alphabetically between "Null" and "Recursive"):

```python
EastTypeType = recursive_type(
    lambda self: VariantType(
        [
            ("Array", self),
            ("Blob", NullType),
            ("Boolean", NullType),
            ("DateTime", NullType),
            ("Dict", StructType([("key", self), ("value", self)])),
            ("Float", NullType),
            (
                "Function",
                StructType(
                    [
                        ("inputs", ArrayType(self)),
                        ("output", self),
                        ("platforms", ArrayType(StringType)),
                    ]
                ),
            ),
            ("Integer", NullType),
            ("Never", NullType),
            ("Null", NullType),
            ("Recursive", IntegerType),
            ("Ref", self),  # ADD THIS LINE
            ("Set", self),
            ("String", NullType),
            ("Struct", ArrayType(StructType([("name", StringType), ("type", self)]))),
            ("Variant", ArrayType(StructType([("name", StringType), ("type", self)]))),
        ]
    )
)
```

#### Task 2.5: Update is_subtype for Ref (invariance)

**Location:** is_subtype function (~line 730-832)

**Required addition:**

Add before "Mutable collections are invariant" comment (~line 769):

```python
# Ref type - invariant (mutable)
if t1.tag == "Ref":
    if t2.tag == "Ref":
        return is_type_equal(t1.value, t2.value)
    return False
```

**Rationale:** Refs are invariant - `ref<Subtype>` is NOT a subtype of `ref<Supertype>`

#### Task 2.6: Update is_value_of for Ref

**Location:** is_value_of function (~line 835-944)

**Required addition:**

Add after "Blob" case (~line 875):

```python
if tag == "Ref":
    from east.types.ref import Ref

    if not isinstance(value, Ref):
        return False
    value_type = typ.value
    return is_value_of(value.value, value_type, node_type, nodes_visited)
```

#### Task 2.7: Update type_union for Ref

**Location:** type_union function (~line 947-1089)

**Required addition:**

Add after Recursive case (~line 992):

```python
if t1.tag == "Ref":
    if t2.tag == "Ref":
        return RefType(type_equal(t1.value, t2.value))
    raise TypeMismatchError(
        f"Cannot union {print_type(t1)} with {print_type(t2)}: incompatible types"
    )
```

**Rationale:** Refs are invariant, inner types must be exactly equal

#### Task 2.8: Update type_intersect for Ref

**Location:** type_intersect function (~line 1092-1238)

**Required addition:**

Add after Recursive case (~line 1137):

```python
if t1.tag == "Ref":
    if t2.tag == "Ref":
        return RefType(type_equal(t1.value, t2.value))
    raise TypeMismatchError(
        f"Cannot intersect {print_type(t1)} with {print_type(t2)}: incompatible types"
    )
```

#### Task 2.9: Update type_equal for Ref

**Location:** type_equal function (~line 1241-1403)

**Required addition:**

Add after Recursive case (~line 1357):

```python
if t1.tag == "Ref":
    if t2.tag == "Ref":
        return RefType(type_equal(t1.value, t2.value, r1, r2))
    raise TypeMismatchError(
        f"{print_type(t1)} is not equal to {print_type(t2)}: incompatible types"
    )
```

#### Task 2.10: Update type_of for Ref

**Location:** type_of function (~line 1406-1471)

**Required addition:**

Add after EastDict case (~line 1463):

```python
# Ref
if isinstance(value, Ref):
    return RefType(type_of(value.value))
```

**Import needed at top:**
```python
from east.types.ref import Ref
```

#### Task 2.11: Update east_type_of for Ref

**Location:** east_type_of function (~line 1474-1576)

**Required addition:**

Add after EastDict case (~line 1550):

```python
# Ref
if isinstance(value, Ref):
    return RefType(east_type_of(value.value))
```

**Import needed at top:**
```python
from east.types.ref import Ref
```

#### Task 2.12: Export RefType in __all__

**Location:** __all__ list at end of file (~line 1970-2017)

**Required addition:**

```python
__all__ = [
    # ... existing exports ...
    "RefType",  # ADD THIS
    # ... rest of exports ...
]
```

---

## 3. Comparison Functions

### File: `east/builtins/comparison.py`

**Location:** Comparison builtin functions

#### Task 3.1: Add Ref support to is_equal

**Location:** Find the type dispatch in is_equal implementation

**Required changes:**

Add Ref case to handle identity comparison and cycle detection:

```python
def _is_equal_for_type(typ: EastType) -> Callable[[Any, Any], bool]:
    """Create equality comparison function for a type."""
    tag = typ.tag

    # ... existing cases ...

    if tag == "Ref":
        from east.types.ref import Ref

        # Get inner value comparer
        inner_comparer = _is_equal_for_type(typ.value)

        def compare_refs(x: Ref, y: Ref, ctx: dict | None = None) -> bool:
            # Fast path - same identity
            if x is y:
                return True

            # Create context for cycle detection
            if ctx is None:
                ctx = {}

            # Check if we've visited this pair
            if id(x) in ctx:
                visited_set = ctx[id(x)]
                if id(y) in visited_set:
                    return True  # Cycle - already comparing
            else:
                ctx[id(x)] = set()

            # Mark as visited
            ctx[id(x)].add(id(y))

            # Compare inner values
            return inner_comparer(x.value, y.value, ctx)

        return compare_refs

    # ... rest of cases ...
```

**Notes:**
- Refs use identity first (fast path)
- Then structural comparison with cycle detection
- Context tracks visited pairs to detect cycles

#### Task 3.2: Add Ref support to compare (ordering)

**Location:** Find compare implementation

**Required changes:**

Add Ref case for ordering comparison:

```python
def _compare_for_type(typ: EastType) -> Callable[[Any, Any], int]:
    """Create ordering comparison function for a type."""
    tag = typ.tag

    # ... existing cases ...

    if tag == "Ref":
        from east.types.ref import Ref

        # Get inner value comparer
        inner_comparer = _compare_for_type(typ.value)

        def compare_refs(x: Ref, y: Ref, ctx: dict | None = None) -> int:
            # Fast path - same identity
            if x is y:
                return 0

            # Create context for cycle detection
            if ctx is None:
                ctx = {}

            # Check if we've visited this pair
            if id(x) in ctx:
                visited_set = ctx[id(x)]
                if id(y) in visited_set:
                    return 0  # Cycle - treat as equal
            else:
                ctx[id(x)] = set()

            # Mark as visited
            ctx[id(x)].add(id(y))

            # Compare inner values
            return inner_comparer(x.value, y.value, ctx)

        return compare_refs

    # ... rest of cases ...
```

#### Task 3.3: Add Ref support to is (identity check)

**Location:** Find is implementation

**Required addition:**

```python
def _is_for_type(typ: EastType) -> Callable[[Any, Any], bool]:
    """Create identity comparison function for a type."""
    tag = typ.tag

    # ... existing cases ...

    if tag == "Ref":
        # Mutable types compared by identity
        return lambda x, y, _ctx=None: x is y

    # ... rest of cases ...
```

---

## 4. Default Values

### File: `east/utils/default.py`

**Location:** Default value generation functions

#### Task 4.1: Add Ref to default_value

**Location:** default_value function

**Required addition:**

```python
def default_value(typ: EastType) -> Any:
    """Provide a default value for a given EastType.

    Returns sensible defaults:
    - 0 for integers
    - 0.0 for floats
    - "" for strings
    - ref(default) for refs
    - etc.
    """
    from east.types.ref import ref

    tag = typ.tag

    # ... existing cases ...

    if tag == "Ref":
        inner_default = default_value(typ.value)
        return ref(inner_default)

    # ... rest of cases ...
```

#### Task 4.2: Add Ref to minimal_value

**Location:** minimal_value function (if exists)

**Required addition:**

```python
def minimal_value(typ: EastType) -> Any:
    """Provide a minimal value for a given EastType.

    Similar to default_value but prefers smallest representation.
    """
    from east.types.ref import ref

    tag = typ.tag

    # ... existing cases ...

    if tag == "Ref":
        inner_minimal = minimal_value(typ.value)
        return ref(inner_minimal)

    # ... rest of cases ...
```

---

## 5. Fuzzing Support

### File: `east/testing/fuzz.py`

**Location:** Random value generation for testing

#### Task 5.1: Add Ref to random value generation

**Location:** Find random value generator function

**Required addition:**

```python
def random_value_for_type(typ: EastType, max_depth: int = 5) -> Any:
    """Generate random value of given type for fuzz testing."""
    from east.types.ref import ref

    tag = typ.tag

    # ... existing cases ...

    if tag == "Ref":
        # Generate random value for inner type
        inner_value = random_value_for_type(typ.value, max_depth - 1)
        return ref(inner_value)

    # ... rest of cases ...
```

---

## 6. Serialization - Beast v1 (Not Supported)

### File: `east/serialization/beast.py`

**Location:** Beast v1 binary encoding/decoding

#### Task 6.1: Add error for Ref in encode

**Location:** Find encode function

**Required addition:**

```python
def encode_beast_value(typ: EastType, value: Any, writer: BinaryWriter) -> None:
    """Encode value in Beast v1 format."""
    tag = typ.tag

    # ... existing cases ...

    if tag == "Ref":
        raise ValueError("Beast v1 format does not support ref types")

    # ... rest of cases ...
```

#### Task 6.2: Add error for Ref in decode

**Location:** Find decode function

**Required addition:**

```python
def decode_beast_value(typ: EastType, reader: BinaryReader) -> Any:
    """Decode value from Beast v1 format."""
    tag = typ.tag

    # ... existing cases ...

    if tag == "Ref":
        raise ValueError("Beast v1 format does not support ref types")

    # ... rest of cases ...
```

---

## 7. Serialization - Beast v2 (Full Support)

### File: `east/serialization/binary_utils.py`

**Location:** Binary encoding/decoding utilities

#### Task 7.1: Ensure backreference support exists

**Verify:** Check if backreference tracking infrastructure exists

**Required:** Context object for tracking refs:

```python
@dataclass
class Beast2EncodeContext:
    """Context for Beast v2 encoding with backreference support."""
    refs: dict[int, int]  # Maps id(object) -> offset
```

### File: `east/serialization/beast.py` (or separate beast2.py)

**Location:** Beast v2 encoding/decoding

#### Task 7.2: Add Ref encoding

**Required implementation:**

```python
def encode_beast2_ref(
    value_type: EastType,
    value: Ref,
    writer: BinaryWriter,
    ctx: Beast2EncodeContext
) -> None:
    """Encode a ref in Beast v2 format.

    Format:
    - If backreference: varint(offset_delta) where delta > 0
    - If inline: varint(0) + encoded_value
    """
    from east.types.ref import Ref

    ref_id = id(value)

    # Check for backreference
    if ref_id in ctx.refs:
        # Write backreference
        target_offset = ctx.refs[ref_id]
        offset_delta = writer.current_offset - target_offset
        writer.write_varint(offset_delta)
        return

    # Inline ref - write marker and register
    writer.write_varint(0)  # Inline marker
    ctx.refs[ref_id] = writer.current_offset

    # Encode inner value
    encode_beast2_value(value_type.value, value.value, writer, ctx)
```

#### Task 7.3: Add Ref decoding

**Required implementation:**

```python
def decode_beast2_ref(
    value_type: EastType,
    reader: BinaryReader,
    ctx: Beast2DecodeContext
) -> Ref:
    """Decode a ref from Beast v2 format.

    Format:
    - If backreference: varint(offset_delta) where delta > 0
    - If inline: varint(0) + encoded_value
    """
    from east.types.ref import ref

    start_offset = reader.current_offset
    ref_or_zero = reader.read_varint()

    # Check if backreference
    if ref_or_zero > 0:
        target_offset = start_offset - ref_or_zero
        if target_offset not in ctx.refs:
            raise ValueError(
                f"Undefined backreference at offset {start_offset}, "
                f"target {target_offset}"
            )
        return ctx.refs[target_offset]

    # Inline ref - decode value
    result = ref(None)  # Create ref first (for circular refs)
    ctx.refs[start_offset] = result

    # Decode inner value
    result.value = decode_beast2_value(value_type.value, reader, ctx)

    return result
```

**Notes:**
- Backreferences allow sharing same ref-cell
- Pre-register ref before decoding value (handles circular refs)
- Offset delta encoding saves space

---

## 8. Serialization - JSON (With References)

### File: `east/serialization/json.py`

**Location:** JSON encoding/decoding

#### Task 8.1: Add Ref JSON encoding

**Required implementation:**

```python
def encode_json_ref(
    value_type: EastType,
    value: Ref,
    ctx: JSONEncodeContext
) -> Any:
    """Encode ref to JSON with reference support.

    Format:
    - First occurrence: [value]  (array with single element)
    - Backreference: {"$ref": "relative_path"}

    Example:
        ref(42) -> [42]
        ref([1,2]) -> [[1,2]]
        backreference -> {"$ref": "../0"}
    """
    from east.types.ref import Ref

    ref_id = id(value)

    # Check for backreference
    if ref_id in ctx.refs:
        target_path = ctx.refs[ref_id]
        ref_str = encode_relative_ref(ctx.current_path, target_path)
        return {"$ref": ref_str}

    # First occurrence - register path
    ctx.refs[ref_id] = list(ctx.current_path)

    # Encode as array of one element
    ctx.current_path.append("0")
    encoded_value = encode_json_value(value_type.value, value.value, ctx)
    ctx.current_path.pop()

    return [encoded_value]


def encode_relative_ref(current_path: list[str], target_path: list[str]) -> str:
    """Encode a relative reference path.

    Examples:
        current: ["a", "b", "c"]
        target: ["a", "b", "d"]
        result: "../d"

        current: ["a", "b"]
        target: ["x", "y"]
        result: "../../x/y"
    """
    # Find common prefix
    common_len = 0
    for i in range(min(len(current_path), len(target_path))):
        if current_path[i] == target_path[i]:
            common_len += 1
        else:
            break

    # Build relative path
    ups = len(current_path) - common_len
    downs = target_path[common_len:]

    parts = [".."] * ups + downs
    return "/".join(parts)
```

#### Task 8.2: Add Ref JSON decoding

**Required implementation:**

```python
def decode_json_ref(
    value_type: EastType,
    json_value: Any,
    ctx: JSONDecodeContext
) -> Ref:
    """Decode ref from JSON with reference support.

    Format:
    - Array: [value] - inline ref
    - Object with $ref: {"$ref": "path"} - backreference
    """
    from east.types.ref import ref

    # Check for backreference first
    if isinstance(json_value, dict) and "$ref" in json_value and len(json_value) == 1:
        ref_str = json_value["$ref"]
        target_path = decode_relative_ref(ref_str, ctx.current_path)
        path_key = "/" + "/".join(encode_json_pointer_component(p) for p in target_path)

        if path_key not in ctx.refs:
            raise ValueError(f"Undefined reference: {ref_str}")

        return ctx.refs[path_key]

    # Inline ref
    if not isinstance(json_value, list) or len(json_value) != 1:
        raise ValueError(f"Expected array with 1 element for ref, got {json_value}")

    # Create ref and pre-register
    result = ref(None)
    path_key = "/" + "/".join(encode_json_pointer_component(p) for p in ctx.current_path)
    ctx.refs[path_key] = result

    # Decode inner value
    result.value = decode_json_value(value_type.value, json_value[0], ctx)

    return result
```

---

## 9. Serialization - East Format (Text)

### File: `east/serialization/east_printer.py`

**Location:** Text format encoding

#### Task 9.1: Add Ref printing

**Required implementation:**

```python
def print_ref(
    value_type: EastType,
    value: Ref,
    ctx: EastPrintContext
) -> str:
    """Print ref in East text format.

    Format:
    - Inline: &value
    - Backreference: ../path

    Examples:
        ref(42) -> &42
        ref([1,2,3]) -> &[1, 2, 3]
        backreference -> ../[]
    """
    from east.types.ref import Ref

    ref_id = id(value)

    # Check for backreference
    if ref_id in ctx.refs:
        target_path = ctx.refs[ref_id]
        return encode_relative_ref(ctx.current_path, target_path)

    # First occurrence - register
    ctx.refs[ref_id] = list(ctx.current_path)

    # Print as &value
    ctx.current_path.append("[]")
    value_str = print_value(value_type.value, value.value, ctx)
    ctx.current_path.pop()

    return f"&{value_str}"
```

### File: `east/serialization/east_parser.py`

**Location:** Text format parsing

#### Task 9.2: Add Ref parsing

**Required implementation:**

```python
def parse_ref(
    value_type: EastType,
    text: str,
    pos: int,
    ctx: EastParseContext
) -> tuple[Ref, int]:
    """Parse ref from East text format.

    Format:
    - &value - inline ref
    - ../path - backreference
    """
    from east.types.ref import ref

    pos = skip_whitespace(text, pos)

    # Create context if needed
    if ctx is None:
        ctx = EastParseContext()

    # Check for backreference
    if text[pos:pos+3] in ('./', '../'):
        # Parse reference path
        result, pos = parse_reference(text, pos, ctx)
        return result, pos

    # Inline ref
    if pos >= len(text) or text[pos] != '&':
        raise ParseError(f"Expected '&' to start ref at position {pos}")

    pos += 1
    pos = skip_whitespace(text, pos)

    # Create ref and pre-register
    result = ref(None)
    path_str = path_to_punctuated(ctx.current_path)
    ctx.refs[path_str] = result

    # Parse inner value
    ctx.current_path.append("[]")
    try:
        result.value, pos = parse_value(value_type.value, text, pos, ctx)
    finally:
        ctx.current_path.pop()

    return result, pos
```

---

## 10. Runtime Compiler Changes

### File: `east/runtime/compiler.py`

**Location:** Type parameter application and compilation

#### Task 10.1: Add Ref to apply_type_parameters

**Location:** Find apply_type_parameters function

**Required addition:**

```python
def apply_type_parameters(
    typ: EastType,
    params: dict[str, EastType]
) -> EastType:
    """Apply type parameter substitutions to a type.

    Used for generic functions and polymorphism.
    """
    tag = typ.tag

    # ... existing cases ...

    if tag == "Ref":
        inner_type = apply_type_parameters(typ.value, params)
        return RefType(inner_type)

    # ... rest of cases ...
```

---

## 11. Testing

### Directory: `tests/`

#### Task 11.1: Create basic ref tests

**File:** `tests/test_ref.py`

**Required tests:**

```python
"""Tests for ref type functionality."""

import pytest
from east.types.ref import ref, is_ref, deref, set_ref, Ref
from east.types.type_system import RefType, IntegerType, ArrayType, StringType


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
    with pytest.raises(TypeError, match="data type"):
        RefType(func_type)


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
```

#### Task 11.2: Create ref comparison tests

**File:** `tests/test_ref_comparison.py`

**Required tests:**

```python
"""Tests for ref comparison operations."""

import pytest
from east.types.ref import ref, deref, set_ref
from east.builtins.comparison import is_equal, compare, is_identical


def test_ref_identity_equal():
    """Test refs are equal by identity."""
    r1 = ref(42)
    r2 = r1

    assert is_identical(r1, r2)
    assert is_equal(r1, r2)


def test_ref_structural_equal():
    """Test refs with equal contents are structurally equal."""
    r1 = ref(42)
    r2 = ref(42)

    assert not is_identical(r1, r2)  # Different identity
    assert is_equal(r1, r2)  # Same contents


def test_ref_not_equal():
    """Test refs with different contents are not equal."""
    r1 = ref(42)
    r2 = ref(99)

    assert not is_equal(r1, r2)


def test_ref_circular_equal():
    """Test circular refs can be compared."""
    r1 = ref(None)
    r2 = ref(None)

    set_ref(r1, r1)  # Self-reference
    set_ref(r2, r2)  # Self-reference

    # Should not infinite loop
    assert is_equal(r1, r2)  # Both are circular


def test_ref_ordering():
    """Test ref ordering comparison."""
    r1 = ref(10)
    r2 = ref(20)
    r3 = ref(10)

    assert compare(r1, r2) < 0  # 10 < 20
    assert compare(r2, r1) > 0  # 20 > 10
    assert compare(r1, r3) == 0  # Same value


def test_ref_circular_ordering():
    """Test circular refs in ordering."""
    r1 = ref(None)
    set_ref(r1, r1)

    r2 = ref(None)
    set_ref(r2, r2)

    # Should not infinite loop
    assert compare(r1, r2) == 0
```

#### Task 11.3: Create ref serialization tests

**File:** `tests/test_ref_serialization.py`

**Required tests:**

```python
"""Tests for ref serialization in various formats."""

import pytest
from east.types.ref import ref, deref, set_ref
from east.types.type_system import RefType, IntegerType, ArrayType, StringType
from east.serialization.beast import encode_beast, decode_beast
from east.serialization.beast2 import encode_beast2, decode_beast2
from east.serialization.json import to_json, from_json
from east.serialization.east_printer import print_value
from east.serialization.east_parser import parse_value


def test_ref_beast_v1_unsupported():
    """Beast v1 should reject refs."""
    r = ref(42)
    typ = RefType(IntegerType)

    with pytest.raises(ValueError, match="Beast v1.*not support ref"):
        encode_beast(typ, r)


def test_ref_beast2_roundtrip():
    """Test ref encoding/decoding in Beast v2."""
    r = ref(42)
    typ = RefType(IntegerType)

    encoded = encode_beast2(typ, r)
    decoded = decode_beast2(typ, encoded)

    assert deref(decoded) == 42


def test_ref_beast2_aliasing():
    """Test ref aliasing in Beast v2."""
    r = ref(42)
    typ = ArrayType(RefType(IntegerType))

    # Array with two refs to same cell
    value = [r, r]

    encoded = encode_beast2(typ, value)
    decoded = decode_beast2(typ, encoded)

    # Should be same ref
    assert decoded[0] is decoded[1]

    # Mutation should be visible in both
    set_ref(decoded[0], 99)
    assert deref(decoded[1]) == 99


def test_ref_json_roundtrip():
    """Test ref encoding/decoding in JSON."""
    r = ref(42)
    typ = RefType(IntegerType)

    json_str = to_json(typ, r)
    decoded = from_json(typ, json_str)

    assert deref(decoded) == 42


def test_ref_json_format():
    """Test ref JSON format is array."""
    r = ref(42)
    typ = RefType(IntegerType)

    json_str = to_json(typ, r)
    # Should be [42]
    assert "[42]" in json_str


def test_ref_json_aliasing():
    """Test ref aliasing in JSON with $ref."""
    r = ref(42)
    typ = ArrayType(RefType(IntegerType))

    value = [r, r]

    json_str = to_json(typ, value)
    decoded = from_json(typ, json_str)

    # Should be same ref
    assert decoded[0] is decoded[1]


def test_ref_east_format_roundtrip():
    """Test ref in East text format."""
    r = ref(42)
    typ = RefType(IntegerType)

    text = print_value(typ, r)
    decoded = parse_value(typ, text)

    assert deref(decoded) == 42


def test_ref_east_format_syntax():
    """Test ref East format uses & syntax."""
    r = ref(42)
    typ = RefType(IntegerType)

    text = print_value(typ, r)
    assert text == "&42"


def test_ref_east_format_nested():
    """Test nested ref in East format."""
    r = ref([1, 2, 3])
    typ = RefType(ArrayType(IntegerType))

    text = print_value(typ, r)
    # Should be &[1, 2, 3]
    assert text.startswith("&[")

    decoded = parse_value(typ, text)
    assert deref(decoded) == [1, 2, 3]


def test_ref_circular_serialization():
    """Test circular ref serialization."""
    r = ref(None)
    set_ref(r, r)  # Self-reference

    typ = RefType(RefType(IntegerType))  # Approximate type

    # Should not infinite loop
    # Beast v2 should handle with backreferences
    encoded = encode_beast2(typ, r)
    decoded = decode_beast2(typ, encoded)

    # Should be circular
    assert decoded.value is decoded
```

#### Task 11.4: Create ref integration tests

**File:** `tests/test_ref_integration.py`

**Required tests:**

```python
"""Integration tests for ref with other East features."""

import pytest
from east.types.ref import ref, deref, set_ref
from east.types.type_system import (
    RefType, IntegerType, ArrayType, StructType, VariantType
)


def test_ref_in_struct():
    """Test ref as struct field."""
    counter_type = StructType([
        ("value", RefType(IntegerType)),
        ("name", StringType),
    ])

    r = ref(0)
    counter = {"value": r, "name": "counter"}

    # Increment via ref
    set_ref(counter["value"], deref(counter["value"]) + 1)
    assert deref(counter["value"]) == 1


def test_ref_in_variant():
    """Test ref in variant case."""
    result_type = VariantType([
        ("ok", RefType(IntegerType)),
        ("error", StringType),
    ])

    r = ref(42)
    result = {"type": "ok", "value": r}

    assert deref(result["value"]) == 42


def test_ref_default_value():
    """Test default_value for ref type."""
    from east.utils.default import default_value

    typ = RefType(IntegerType)
    val = default_value(typ)

    assert is_ref(val)
    assert deref(val) == 0  # Default int is 0


def test_ref_type_operations():
    """Test type operations on refs."""
    from east.types.type_system import is_subtype, type_equal, type_of

    ref_int = RefType(IntegerType)
    ref_int2 = RefType(IntegerType)
    ref_float = RefType(FloatType)

    # Refs are invariant
    assert is_subtype(ref_int, ref_int2)  # Same type
    assert not is_subtype(ref_int, ref_float)  # Different inner type

    # Type equality
    assert type_equal(ref_int, ref_int2)

    # Type of value
    r = ref(42)
    assert type_of(r) == ref_int


def test_ref_shared_mutation():
    """Test shared mutable state via refs."""
    # Simulate counter shared between multiple contexts
    counter = ref(0)

    def increment():
        set_ref(counter, deref(counter) + 1)

    def decrement():
        set_ref(counter, deref(counter) - 1)

    increment()
    increment()
    increment()
    assert deref(counter) == 3

    decrement()
    assert deref(counter) == 2


def test_ref_complex_aliasing():
    """Test complex aliasing patterns."""
    # Multiple arrays sharing same refs
    r1 = ref(1)
    r2 = ref(2)

    arr1 = [r1, r2, r1]  # r1 appears twice
    arr2 = [r2, r1, r2]  # Different order

    # Mutation visible everywhere
    set_ref(r1, 10)

    assert deref(arr1[0]) == 10
    assert deref(arr1[2]) == 10
    assert deref(arr2[1]) == 10
```

---

## 12. Documentation

### File: `docs/REF_TYPE.md` (NEW)

**Create comprehensive documentation:**

```markdown
# Ref Type - Mutable Reference Cells

## Overview

The `ref` type provides mutable reference cells with identity semantics. Refs are similar to:
- OCaml/ML `ref` type
- Scheme/Racket boxes
- Clojure atoms (but simpler)

## Basic Usage

### Creating Refs

```python
from east.types.ref import ref, deref, set_ref

# Create a ref containing an integer
counter = ref(0)

# Access the value
print(deref(counter))  # 0

# Update the value
set_ref(counter, 1)
print(deref(counter))  # 1

# Update based on current value
set_ref(counter, deref(counter) + 1)
print(deref(counter))  # 2
```

### Identity Semantics

Refs have identity semantics - two refs are the same only if they're literally
the same object:

```python
r1 = ref(42)
r2 = ref(42)

print(r1 is r2)  # False - different refs
print(deref(r1) == deref(r2))  # True - same contents
```

### Aliasing

Multiple variables can point to the same ref-cell (aliasing):

```python
r1 = ref([1, 2, 3])
r2 = r1  # Same ref-cell

set_ref(r2, [4, 5, 6])
print(deref(r1))  # [4, 5, 6] - r1 sees the change
print(r1 is r2)  # True - same object
```

## Type System

### Creating Ref Types

```python
from east.types.type_system import RefType, IntegerType, ArrayType

# ref<Integer>
int_ref_type = RefType(IntegerType)

# ref<Array<String>>
array_ref_type = RefType(ArrayType(StringType))
```

### Type Constraints

- Refs can only contain **data types** (not functions)
- Refs are **invariant** - no subtyping
  - `ref<Subtype>` is NOT a subtype of `ref<Supertype>`
- Refs are **mutable** (like arrays, sets, dicts)
- Refs cannot be dict keys (not immutable)

### Invariance Example

```python
from east.types.type_system import is_subtype

ref_int = RefType(IntegerType)
ref_num = RefType(NumberType)

# Even though Integer <: Number,
# ref<Integer> is NOT <: ref<Number>
assert not is_subtype(ref_int, ref_num)
```

**Why invariance?** If refs were covariant, you could:
```python
ref_int: ref<Integer> = ref(0)
ref_num: ref<Number> = ref_int  # If this were allowed...
set_ref(ref_num, 3.14)  # ...we could put float in integer ref!
deref(ref_int)  # Type violation!
```

## Comparison

### Equality

Refs can be compared two ways:

1. **Identity** - Same object?
```python
r1 = ref(42)
r2 = r1
print(r1 is r2)  # True
```

2. **Structural** - Same contents?
```python
r1 = ref(42)
r2 = ref(42)
print(is_equal(r1, r2))  # True (same contents)
print(r1 is r2)  # False (different objects)
```

### Circular Refs

Refs can be circular:

```python
r = ref(None)
set_ref(r, r)  # Self-reference

# Comparison handles cycles
r2 = ref(None)
set_ref(r2, r2)

print(is_equal(r, r2))  # True (both circular)
```

## Serialization

### Beast v1

Beast v1 **does not support** refs:

```python
r = ref(42)
typ = RefType(IntegerType)

encode_beast(typ, r)  # ValueError: Beast v1 does not support ref types
```

### Beast v2

Beast v2 **fully supports** refs with backreferences:

```python
r = ref(42)
typ = RefType(IntegerType)

# Encode and decode
encoded = encode_beast2(typ, r)
decoded = decode_beast2(typ, encoded)

print(deref(decoded))  # 42
```

**Aliasing preserved:**
```python
r = ref(42)
arr = [r, r]  # Same ref twice

encoded = encode_beast2(ArrayType(RefType(IntegerType)), arr)
decoded = decode_beast2(ArrayType(RefType(IntegerType)), encoded)

# Aliasing preserved
assert decoded[0] is decoded[1]
```

### JSON

JSON format uses arrays:

```python
r = ref(42)
typ = RefType(IntegerType)

json_str = to_json(typ, r)
# Result: [42]

decoded = from_json(typ, json_str)
print(deref(decoded))  # 42
```

**Backreferences use `$ref`:**
```python
r = ref(42)
arr = [r, r]

json_str = to_json(ArrayType(RefType(IntegerType)), arr)
# Result: [[42], {"$ref": "../0"}]
```

### East Text Format

East format uses `&` syntax:

```python
r = ref(42)
text = print_value(RefType(IntegerType), r)
# Result: &42

r2 = ref([1, 2, 3])
text2 = print_value(RefType(ArrayType(IntegerType)), r2)
# Result: &[1, 2, 3]
```

**Backreferences:**
```python
r = ref(42)
arr = [r, r]
text = print_value(ArrayType(RefType(IntegerType)), arr)
# Result: [&42, ../0/[]]
```

## Use Cases

### Mutable Counters

```python
counter = ref(0)

def increment():
    set_ref(counter, deref(counter) + 1)

increment()
increment()
print(deref(counter))  # 2
```

### Shared State

```python
# Multiple contexts sharing mutable state
state = ref({"count": 0, "total": 0})

def update(value):
    current = deref(state)
    set_ref(state, {
        "count": current["count"] + 1,
        "total": current["total"] + value,
    })

update(10)
update(20)
print(deref(state))  # {"count": 2, "total": 30}
```

### Mutable Records

```python
from east.types.type_system import StructType

# Type: { name: String, age: ref<Integer> }
person_type = StructType([
    ("name", StringType),
    ("age", RefType(IntegerType)),
])

age_ref = ref(25)
person = {"name": "Alice", "age": age_ref}

# Birthday - update age
set_ref(person["age"], deref(person["age"]) + 1)
print(deref(person["age"]))  # 26
```

### Circular Structures

```python
# Circular linked list node
node1 = ref({"value": 1, "next": None})
node2 = ref({"value": 2, "next": node1})
node3 = ref({"value": 3, "next": node2})

# Make circular
set_ref(node1, {"value": 1, "next": node3})

# Can traverse infinitely
current = node1
for _ in range(10):
    print(deref(current)["value"])
    current = deref(current)["next"]
```

## API Reference

### Functions

#### `ref(value: T) -> Ref[T]`
Create a new mutable reference cell.

#### `is_ref(v: Any) -> bool`
Check if a value is a ref-cell.

#### `deref(r: Ref[T]) -> T`
Retrieve the current value from a ref-cell.

#### `set_ref(r: Ref[T], value: T) -> None`
Update the value stored in a ref-cell.

### Types

#### `RefType(value_type: EastType) -> EastType`
Create a ref type.

**Constraints:**
- `value_type` must be a data type (not function)
- Refs are invariant
- Refs are mutable

## Design Notes

### Why Refs?

Refs solve the problem of **mutable state** in functional code:

- **Without refs:** Need to return new values and thread state through code
- **With refs:** Can update state in-place while maintaining functional style

### Refs vs. Mutable Variables

| Feature | Refs | Mutable Variables |
|---------|------|-------------------|
| Identity | Yes | Depends on language |
| Aliasing | Explicit | Implicit |
| Serializable | Yes | No |
| Type checking | Explicit in type | Implicit |

### Implementation Details

- **Brand symbol:** Ensures nominal typing (can't create ref by accident)
- **Identity semantics:** Fast equality check (object identity)
- **Cycle detection:** Prevents infinite loops in comparison
- **Backreferences:** Preserves sharing in serialization

---

**See Also:**
- [Type System Documentation](TYPE_SYSTEM.md)
- [Serialization Guide](SERIALIZATION.md)
- [Comparison Functions](COMPARISON.md)
```

---

## 13. Examples

### File: `examples/ref_examples.py` (NEW)

**Create practical examples:**

```python
"""Examples demonstrating ref type usage in East Python."""

from east.types.ref import ref, deref, set_ref, is_ref
from east.types.type_system import (
    RefType, IntegerType, ArrayType, StringType, StructType
)


# Example 1: Simple Counter
def counter_example():
    """Basic mutable counter using refs."""
    counter = ref(0)

    def increment():
        set_ref(counter, deref(counter) + 1)

    def get_value():
        return deref(counter)

    print("Counter example:")
    print(f"Initial: {get_value()}")  # 0

    increment()
    print(f"After increment: {get_value()}")  # 1

    increment()
    increment()
    print(f"After 2 more: {get_value()}")  # 3


# Example 2: Shared State
def shared_state_example():
    """Multiple contexts sharing mutable state."""
    # Shared accumulator
    total = ref(0)
    count = ref(0)

    def add_value(value):
        set_ref(total, deref(total) + value)
        set_ref(count, deref(count) + 1)

    def get_average():
        if deref(count) == 0:
            return 0.0
        return deref(total) / deref(count)

    print("\nShared state example:")
    add_value(10)
    add_value(20)
    add_value(30)

    print(f"Total: {deref(total)}")  # 60
    print(f"Count: {deref(count)}")  # 3
    print(f"Average: {get_average()}")  # 20.0


# Example 3: Aliasing
def aliasing_example():
    """Demonstrate ref aliasing."""
    print("\nAliasing example:")

    r1 = ref([1, 2, 3])
    r2 = r1  # Alias - same ref-cell

    print(f"r1: {deref(r1)}")  # [1, 2, 3]
    print(f"r2: {deref(r2)}")  # [1, 2, 3]
    print(f"Same object? {r1 is r2}")  # True

    # Modify through r2
    set_ref(r2, [4, 5, 6])

    print(f"After modifying r2:")
    print(f"r1: {deref(r1)}")  # [4, 5, 6] - sees the change
    print(f"r2: {deref(r2)}")  # [4, 5, 6]


# Example 4: Mutable Record Fields
def mutable_record_example():
    """Use refs for mutable fields in records."""
    print("\nMutable record example:")

    # Person with mutable age
    person = {
        "name": "Alice",
        "age": ref(25),
        "email": "alice@example.com",
    }

    print(f"Name: {person['name']}")
    print(f"Age: {deref(person['age'])}")  # 25

    # Birthday - increment age
    set_ref(person["age"], deref(person["age"]) + 1)

    print(f"After birthday:")
    print(f"Age: {deref(person['age'])}")  # 26


# Example 5: Circular Structures
def circular_structure_example():
    """Create circular data structures with refs."""
    print("\nCircular structure example:")

    # Circular linked list: 1 -> 2 -> 3 -> 1
    node1 = ref(None)
    node2 = ref(None)
    node3 = ref(None)

    set_ref(node1, {"value": 1, "next": node2})
    set_ref(node2, {"value": 2, "next": node3})
    set_ref(node3, {"value": 3, "next": node1})

    # Traverse circular list
    current = node1
    print("Circular list (10 steps):")
    for i in range(10):
        node_data = deref(current)
        print(f"  Step {i}: {node_data['value']}")
        current = node_data["next"]


# Example 6: Multiple References
def multiple_refs_example():
    """Array with multiple refs to same cell."""
    print("\nMultiple references example:")

    shared = ref(0)
    counters = [shared, shared, shared]  # Three refs to same cell

    print(f"Initial: {[deref(c) for c in counters]}")  # [0, 0, 0]

    # Update through first counter
    set_ref(counters[0], 10)

    print(f"After update: {[deref(c) for c in counters]}")  # [10, 10, 10]
    print(f"All same object? {counters[0] is counters[1] is counters[2]}")  # True


# Example 7: State Machine
def state_machine_example():
    """Simple state machine using refs."""
    print("\nState machine example:")

    # States: "idle", "running", "paused", "stopped"
    state = ref("idle")

    def transition(new_state):
        old_state = deref(state)
        set_ref(state, new_state)
        print(f"  {old_state} -> {new_state}")

    def get_state():
        return deref(state)

    print(f"Initial state: {get_state()}")

    transition("running")
    transition("paused")
    transition("running")
    transition("stopped")

    print(f"Final state: {get_state()}")


# Example 8: Cache with Refs
def cache_example():
    """Simple cache using refs."""
    print("\nCache example:")

    cache = ref({})
    hits = ref(0)
    misses = ref(0)

    def get_or_compute(key, compute_fn):
        current_cache = deref(cache)

        if key in current_cache:
            # Cache hit
            set_ref(hits, deref(hits) + 1)
            return current_cache[key]
        else:
            # Cache miss
            set_ref(misses, deref(misses) + 1)
            value = compute_fn()
            # Update cache
            new_cache = {**current_cache, key: value}
            set_ref(cache, new_cache)
            return value

    def expensive_computation(x):
        print(f"  Computing {x}^2...")
        return x * x

    # Use cache
    print(get_or_compute("a", lambda: expensive_computation(5)))  # Miss: 25
    print(get_or_compute("b", lambda: expensive_computation(7)))  # Miss: 49
    print(get_or_compute("a", lambda: expensive_computation(5)))  # Hit: 25
    print(get_or_compute("b", lambda: expensive_computation(7)))  # Hit: 49

    print(f"Stats: {deref(hits)} hits, {deref(misses)} misses")


if __name__ == "__main__":
    counter_example()
    shared_state_example()
    aliasing_example()
    mutable_record_example()
    circular_structure_example()
    multiple_refs_example()
    state_machine_example()
    cache_example()
```

---

## 14. Implementation Order & Dependencies

**Recommended implementation order:**

### Phase 1: Core Ref Module (2-3 hours)
1. ✅ Create `east/types/ref.py` (Task 1.1)
   - Ref class with brand symbol
   - ref(), is_ref(), deref(), set_ref() functions

### Phase 2: Type System Integration (3-4 hours)
2. ✅ Add RefType constructor (Task 2.1)
3. ✅ Update is_data_type (Task 2.2)
4. ✅ Update is_immutable_type (Task 2.3)
5. ✅ Update EastTypeType (Task 2.4)
6. ✅ Update is_subtype (Task 2.5)
7. ✅ Update is_value_of (Task 2.6)
8. ✅ Update type_union (Task 2.7)
9. ✅ Update type_intersect (Task 2.8)
10. ✅ Update type_equal (Task 2.9)
11. ✅ Update type_of (Task 2.10)
12. ✅ Update east_type_of (Task 2.11)
13. ✅ Export RefType (Task 2.12)

### Phase 3: Comparison Functions (2-3 hours)
14. ✅ Add Ref to is_equal (Task 3.1)
15. ✅ Add Ref to compare (Task 3.2)
16. ✅ Add Ref to is (Task 3.3)

### Phase 4: Utility Functions (1-2 hours)
17. ✅ Add Ref to default_value (Task 4.1)
18. ✅ Add Ref to minimal_value (Task 4.2)
19. ✅ Add Ref to fuzzing (Task 5.1)

### Phase 5: Serialization - Beast (4-6 hours)
20. ✅ Beast v1 error handling (Tasks 6.1-6.2)
21. ✅ Beast v2 encoding (Task 7.2)
22. ✅ Beast v2 decoding (Task 7.3)

### Phase 6: Serialization - JSON & East (6-8 hours)
23. ✅ JSON encoding (Task 8.1)
24. ✅ JSON decoding (Task 8.2)
25. ✅ East format printing (Task 9.1)
26. ✅ East format parsing (Task 9.2)

### Phase 7: Runtime Compiler (1 hour)
27. ✅ Type parameter application (Task 10.1)

### Phase 8: Testing (8-10 hours)
28. ✅ Basic ref tests (Task 11.1)
29. ✅ Comparison tests (Task 11.2)
30. ✅ Serialization tests (Task 11.3)
31. ✅ Integration tests (Task 11.4)

### Phase 9: Documentation & Examples (4-6 hours)
32. ✅ Create REF_TYPE.md (Task 12)
33. ✅ Create examples (Task 13)

**Total Estimated Time: 31-43 hours (4-6 days)**

---

## 15. Verification Checklist

After implementation, verify:

- [ ] Ref class created with brand symbol
- [ ] ref(), is_ref(), deref(), set_ref() functions work
- [ ] RefType added to type system
- [ ] Refs are invariant (no subtyping)
- [ ] Refs are mutable (not immutable)
- [ ] Refs can only contain data types
- [ ] Identity comparison works (fast path)
- [ ] Structural comparison works
- [ ] Cycle detection in comparison works
- [ ] Default/minimal values for refs
- [ ] Random value generation for fuzzing
- [ ] Beast v1 throws error
- [ ] Beast v2 encoding/decoding works
- [ ] Beast v2 preserves aliasing
- [ ] JSON encoding/decoding works
- [ ] JSON backreferences with $ref
- [ ] East format & syntax works
- [ ] East format backreferences work
- [ ] Type parameter application works
- [ ] All 40+ tests pass
- [ ] Documentation complete
- [ ] Examples run successfully
- [ ] No regressions in existing tests

---

## 16. Known Issues & Open Questions

### Questions to resolve:

1. **Brand symbol approach:**
   - Use object() or custom sentinel?
   - Should brand be private (_brand) or use name mangling?
   - How to handle pickle/deepcopy?

2. **Type checking:**
   - Should is_ref check brand or just isinstance?
   - How strict should type validation be?

3. **Comparison cycle detection:**
   - Use id-based dict or WeakSet?
   - Performance implications for large structures?
   - Should cycles short-circuit or compare structure?

4. **Serialization format:**
   - JSON uses arrays - is this optimal?
   - Should Beast v2 have special ref tag?
   - How to handle refs in stream processing?

5. **Backreference encoding:**
   - Relative paths vs absolute offsets?
   - Forward references allowed?
   - Maximum backreference distance?

6. **Circular refs:**
   - Should serialize error or handle gracefully?
   - Depth limits for circular structures?
   - How to print circular refs?

7. **Memory management:**
   - Do refs need __del__ for cleanup?
   - Weak references for caches?
   - GC implications of circular refs?

### Python-specific considerations:

- **Pickle support:** Need __reduce__ for pickling?
- **Copy module:** How should copy.copy vs copy.deepcopy work?
- **Hash:** Should refs be hashable (by identity)?
- **Slots:** Using __slots__ for efficiency?
- **Type hints:** Generic[T] support?
- **Threading:** Thread-safe mutation needed?

---

## 17. Success Criteria

Implementation is complete when:

1. ✅ Ref module created with all functions
2. ✅ RefType integrated into type system
3. ✅ All type operations handle Ref correctly
4. ✅ Comparison functions work with cycle detection
5. ✅ Default/minimal values work
6. ✅ Fuzzing generates valid refs
7. ✅ Beast v1 rejects refs properly
8. ✅ Beast v2 fully supports refs with aliasing
9. ✅ JSON supports refs with $ref backreferences
10. ✅ East format uses & syntax correctly
11. ✅ All 40+ tests pass
12. ✅ Documentation complete
13. ✅ Examples demonstrate all features
14. ✅ No regressions in existing tests
15. ✅ Code review passed

---

## References

- **East TypeScript Commit:** 466b99d7b09319f5ea9446e1eabfd46cccd64d52
- **Key Files Changed:**
  - `src/containers/ref.ts` (122 lines) - NEW FILE
  - `src/comparison.ts` (+68 lines) - Ref comparison support
  - `src/type_of_type.ts` (+18 lines) - RefTypeValue
  - `src/default.ts` (+3 lines) - Default values
  - `src/fuzz.ts` (+4 lines) - Random refs
  - `src/serialization/beast.ts` (+4 lines) - v1 error
  - `src/serialization/beast2.ts` (+major refactoring) - v2 support
  - `src/serialization/beast2-stream.ts` (+refactoring) - Stream support
  - `src/serialization/json.ts` (+92 lines) - JSON with $ref
  - `src/serialization/east.ts` (+81 lines) - & syntax
  - Multiple test files

**Related Concepts:**
- OCaml ref type: https://ocaml.org/docs/if-statements-loops-and-recursion#refs
- Scheme boxes: https://docs.racket-lang.org/guide/boxes.html
- Clojure atoms: https://clojure.org/reference/atoms

---

**Document Version:** 1.0
**Created:** 2025-11-10
**Last Updated:** 2025-11-10
**Status:** Ready for implementation
