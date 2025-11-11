# TODO: Implement Ref Expression Support (AST, IR, Builtins, Compiler)

**Reference Commit:** 3c3b001db79e7f4487eb57fb6ce1c7ffe5e0145a in ../East
**Base Commit:** 466b99d8e4c743ce31f0e2e1c92cd5b20ad15e43 in ../East
**Python Implementation:** Commit 5f72851 (RefType and serialization complete)
**Feature:** Add Ref expression support to enable refs in compiled East programs

## Current Status

**✅ ALREADY IMPLEMENTED (Commit 5f72851):**
- ✅ Core type system (RefType, type operations) in `east/types/type_system.py`
- ✅ Ref container class in `east/types/ref.py`
- ✅ Serialization support:
  - ✅ East format (`&<value>`) in `east_parser.py` and `east_printer.py`
  - ✅ JSON with backreferences in `json.py`
  - ✅ Beast v2 with backreferences (new file `beast2.py`)
  - ✅ Beast v1 rejection (refs not supported)
- ✅ Comparison and ordering with cycle detection in `ordering.py`
- ✅ Default value support in `default.py`
- ✅ Fuzzing support in `fuzz.py`
- ✅ 15 ref tests in `tests/test_ref.py`
- ✅ 1707 Beast2 tests in `tests/serialization/test_beast2.py`

**✅ COMPLETED (all RefType expression support):**
- ✅ IR support (NewRefIR) - added to type_system.py IRType definition
- ✅ IR builders (ir_new_ref) - implemented in builders.py
- ✅ IR analysis (NewRef handling) - added to analyze.py
- ✅ Runtime compilation (_compile_new_ref) - implemented in compiler.py
- ✅ Builtins (Ref.Get, Ref.Update, Ref.Merge) - implemented in ref_ops.py
- ✅ End-to-end compiler tests - added to test_compiler.py
- ✅ Builtin tests - added to test_builtins.py
- ✅ Array.merge return type fix - changed to NullType

**Note on AST:** Python east-py works directly with IR (no AST layer like TypeScript). The TypeScript version has both AST (NewRefAST) and IR (NewRefIR), but Python only needs IR.

## Overview

The East TypeScript implementation added `RefExpr` - the expression-level API for working with mutable reference cells in compiled East programs. This enables refs to be created and manipulated in East code, not just in pure Python.

**TypeScript changes:**
- AST layer: `NewRefAST` for expression-level API
- Expression API: `RefExpr<T>` class with methods:
  - `.get()` - Get current value from ref
  - `.update(value)` - Replace ref value
  - `.merge(value, updateFn)` - Update based on current value
- IR layer: `NewRefIR` for compilation
- AST-to-IR: Conversion from NewRefAST to NewRefIR
- Builtins: Three functions (Ref.Get, Ref.Update, Ref.Merge)
- Compiler: Compiles NewRefIR to JavaScript

**Python architecture difference:**
- Python east-py **does not have an AST layer** - it works directly with IR
- No need for NewRefAST or AST-to-IR conversion
- Only need IR layer (NewRefIR) + builtins + compiler

**Python status:**
- ✅ Type system (RefType) complete
- ✅ Ref container class complete
- ✅ Serialization complete
- ❌ IR layer (NewRefIR, builders, analysis) **not implemented**
- ❌ Builtins (Ref.Get, Ref.Update, Ref.Merge) **not implemented**
- ❌ Compiler (_compile_new_ref) **not implemented**

**Key Requirements for Python Implementation:**
- Add NewRefIR to IR type system
- Create ir_new_ref builder function
- Add NewRefIR analysis
- Implement Ref.Get, Ref.Update, Ref.Merge builtins
- Compile NewRefIR nodes to create ref containers
- Enable refs to work in compiled East functions

**Related Changes:**
- Fix Array.merge() return type from element type to NullType (TypeScript has this fix)

---

## Quick Summary

**What works:**
- ✅ Type system fully supports RefType
- ✅ Ref container class with identity semantics
- ✅ Serialization (East format, JSON, Beast v2) with aliasing
- ✅ IR layer (NewRefIR nodes, builders, analysis)
- ✅ Builtins (Ref.Get, Ref.Update, Ref.Merge)
- ✅ Compiler support (_compile_new_ref)
- ✅ End-to-end tests through compiler
- ✅ 1190 tests passing

**RefType expression support is now COMPLETE!**

You can now:
- Create refs in pure Python code
- Serialize and deserialize refs (East format, JSON, Beast v2)
- Use refs in **compiled East programs** via IR (NewRefIR)
- Call ref builtins (Ref.Get, Ref.Update, Ref.Merge)
- Compile East programs that use refs

**Example that now works:**
```python
# This now works in Python east-py!
from east.runtime.compiler import compile
from east.ir.builders import ir_new_ref, ir_value, location
from east.types.type_system import RefType, IntegerType

# Create IR for: ref(42)
loc = location("test.east", 1, 1)
value_ir = ir_value(IntegerType, loc, 42)
ref_ir = ir_new_ref(RefType(IntegerType), loc, value_ir)

# Compile and execute
compiled = compile(ref_ir, [])
result = compiled({})  # Returns a Ref containing 42
```

All ref operations are now fully functional:
- `NewRefIR` creates ref cells in compiled code
- `Ref.Get`, `Ref.Update`, `Ref.Merge` builtins work in IR
- End-to-end compilation and execution works

---

## 1. Type System Changes ✅ COMPLETE

### File: `east/types/type_system.py`

**Status:** ✅ **IMPLEMENTED in commit 5f72851**

The following changes were completed:
- ✅ RefType type definition using variant-based type system
- ✅ RefType constructor with validation (prevents function types)
- ✅ Updated EastTypeType variant to include "Ref" case
- ✅ Updated is_data_type() to return True for RefType
- ✅ Updated is_immutable_type() to return False for RefType (mutable)
- ✅ Updated is_subtype() with invariant checking for Ref
- ✅ Updated is_value_of() to check ref values
- ✅ Updated type_union(), type_intersect(), type_equal() for RefType
- ✅ Updated type_of() and east_type_of() to extract types from Ref values

**Implementation uses variant-based type system (not dataclasses):**
- RefType created via `RefType(value_type)` function
- Returns `EastType(make_case("Ref", value_type))`
- Validates value_type is a data type

---

## 2. Container Implementation ✅ COMPLETE

### File: `east/types/ref.py`

**Status:** ✅ **IMPLEMENTED in commit 5f72851**

The following was implemented:
- ✅ `Ref` class with identity semantics (uses `__slots__`)
- ✅ `ref(value)` constructor function
- ✅ `is_ref(v)` type checking function
- ✅ `deref(r)` to get current value
- ✅ `set_ref(r, value)` to update value
- ✅ Brand symbol (`REF_SYMBOL`) for nominal typing
- ✅ String representation (`__repr__` and `__str__`)

**Additional implementations:**
- ✅ Comparison support with cycle detection in `east/utils/ordering.py`
- ✅ Default value generation in `east/utils/default.py`
- ✅ Fuzzing support in `east/testing/fuzz.py`

**Serialization (also complete):**
- ✅ East format parsing (`&<value>`) in `east/serialization/east_parser.py`
- ✅ East format printing (`&<value>`) in `east/serialization/east_printer.py`
- ✅ JSON encoding with `$ref` backreferences in `east/serialization/json.py`
- ✅ Beast v2 with full aliasing support in `east/serialization/beast2.py`
- ✅ Beast v1 rejection (not supported)
- ✅ Binary utilities for Beast v2 in `east/serialization/binary_utils.py`

---

## 3. Architecture Note: No AST Layer in Python

**Important difference from TypeScript:**

The TypeScript East implementation has two layers:
1. **AST layer** (`src/ast.ts`) - High-level expression API with NewRefAST
2. **IR layer** (`src/ir.ts`) - Intermediate representation with NewRefIR

The Python east-py implementation **only has an IR layer**:
- No AST nodes (no NewRefAST equivalent)
- No AST-to-IR conversion needed
- Work directly with IR builders (ir_new_ref)

**What this means for implementation:**
- Skip all AST-related sections from TypeScript
- Focus only on IR (NewRefIR), builtins, and compiler
- IR builders serve the same purpose as AST constructors in TypeScript

---

## 4. IR Type System Changes

### File: `east/types/type_system.py`

**Location:** IR type definitions (~lines 1580-1960)

#### Task 3.1: Add NewRefIR struct type definition

**Implementation:**

```python
# In the IRType definition section, add:

_NewRefIR = _ir_struct_type(
    lambda ir: [
        ("type", EastTypeType),
        ("location", LocationType),
        ("value", ir),  # IR for the value to wrap in a ref
    ]
)
```

**Location in IRType variant:**
```python
IRType = recursive_type(lambda ir: VariantType([
    # ... existing cases ...
    ("Call", _CallIR(ir)),
    ("NewRef", _NewRefIR(ir)),  # ADD THIS - alphabetically after Call
    ("NewArray", _NewArrayIR(ir)),
    # ... rest of cases ...
]))
```

**Notes:**
- NewRef IR node wraps a single value in a reference cell
- Similar structure to NewArray but for single values
- The value must type-check against the Ref's value type

---

## 4. IR Builder Functions

### File: `east/ir/builders.py`

**Status:** 🆕 **CREATE NEW BUILDER**

#### Task 4.1: Create ir_new_ref builder function

**Implementation:**

```python
def ir_new_ref(
    typ: EastType,
    loc: EastStruct,
    value: EastVariant,
) -> EastVariant:
    """Create a NewRef IR node.

    Args:
        typ: RefType for the reference cell
        loc: Location
        value: IR for the initial value

    Returns:
        NewRef IR variant

    Raises:
        TypeError: If typ is not a RefType
        TypeError: If value type doesn't match ref's value type

    Examples:
        >>> loc = location("test.east", 1, 1)
        >>> value_ir = ir_value(IntegerType, loc, 42)
        >>> ref_ir = ir_new_ref(RefType(IntegerType), loc, value_ir)
    """
    if not isinstance(typ, RefType):
        raise TypeError(f"ir_new_ref requires RefType, got {print_type(typ)}")

    # Get NewRef struct type from IRType
    newref_cases = IRType.value  # Get variant cases
    newref_struct_type = None
    for case in newref_cases:
        if case.name == "NewRef":
            newref_struct_type = case.type
            break

    if newref_struct_type is None:
        raise ValueError("NewRef case not found in IRType")

    # Create struct
    newref_struct = EastStruct(
        newref_struct_type,
        (typ, loc, value)
    )

    return EastVariant(IRType, Case("NewRef", newref_struct))
```

**Notes:**
- Similar pattern to ir_new_array, ir_new_set, etc.
- Type validation happens during IR analysis phase
- Value can be any IR node that produces the correct type

---

## 5. IR Analysis Changes

### File: `east/ir/analyze.py`

**Location:** `analyze_ir` function, visitor dispatch

#### Task 5.1: Add NewRef analysis

**Implementation:**

```python
# In the analyze_ir visitor function, add case for NewRef:

elif tag == "NewRef":
    # Validate type is Ref
    if not isinstance(node.value.type, RefType):
        raise TypeError(
            f"NewRef node must have Ref type, got {print_type(node.value.type)} "
            f"at {print_location(node.value.location)}"
        )

    element_type = node.value.type.value

    # Analyze the value IR
    value_info = visit_ir(node.value.value, var_ctx)
    if value_info:
        is_async = True

    # Validate value type exactly matches ref's value type
    value_type = get_type(node.value.value)
    if value_type.type != "Never" and not is_type_equal(value_type, element_type):
        raise TypeError(
            f"Ref value has type {print_type(value_type)} "
            f"but Ref expects {print_type(element_type)} "
            f"at {print_location(node.value.location)}"
        )
```

**Notes:**
- Similar to NewArray analysis
- Value type must exactly match (no subtyping for invariant Ref)
- Async propagation: NewRef is async if value is async

---

## 6. Runtime Compiler Changes

### File: `east/runtime/compiler.py`

**Location:** Compilation dispatch and node compilers

#### Task 6.1: Implement _compile_new_ref function

**Implementation:**

```python
def _compile_new_ref(
    node: EastVariant,
    platform_fns: dict[str, Callable[..., Any]],
    async_platform_fns: set[str],
    is_async_map: dict[int, bool],
) -> Callable:
    """Compile a NewRef IR node (creates a reference cell).

    Args:
        node: NewRef IR variant
        platform_fns: Available platform functions
        async_platform_fns: Set of async platform function names
        is_async_map: Map from IR node id to async status

    Returns:
        Compiled function that creates a ref cell
    """
    from east.types.ref import ref

    newref_struct = node.value

    # Validate type
    if not isinstance(newref_struct.type, RefType):
        raise TypeError(
            f"Expected Ref output type, got {print_type(newref_struct.type)} "
            f"at {print_location(newref_struct.location)}"
        )

    # Compile the value
    value_fn = _compile_ir(
        newref_struct.value,
        platform_fns,
        async_platform_fns,
        is_async_map
    )

    # Check if async
    value_is_async = is_async_map.get(id(newref_struct.value), False)

    if value_is_async:
        async def execute_new_ref_async(env):
            val = await value_fn(env)
            return ref(val)
        return execute_new_ref_async
    else:
        def execute_new_ref_sync(env):
            val = value_fn(env)
            return ref(val)
        return execute_new_ref_sync
```

**Notes:**
- Import `ref` from east.types.ref
- Creates a new ref cell with the computed value
- Async version awaits the value computation

#### Task 6.2: Add NewRef case to compilation dispatcher

**Location:** Main `_compile_ir` function dispatch

**Required change:**
```python
def _compile_ir(node, platform_fns, async_platform_fns, is_async_map):
    tag = node.tag

    # ... existing cases ...

    if tag == "Call":
        return _compile_call(node, platform_fns, async_platform_fns, is_async_map)
    elif tag == "NewRef":
        return _compile_new_ref(node, platform_fns, async_platform_fns, is_async_map)
    elif tag == "NewArray":
        return _compile_new_array(node, platform_fns, async_platform_fns, is_async_map)

    # ... rest of cases ...
```

---

## 7. Builtin Functions

### File: `east/builtins/ref_ops.py` (NEW FILE)

**Status:** 🆕 **CREATE NEW FILE**

#### Task 7.1: Implement Ref builtin operations

**Implementation:**

```python
"""Builtin functions for Ref operations."""

from typing import Any, Callable
from east.types.ref import EastRef, deref, set_ref
from east.builtins.registry import register_builtin


def ref_get(ref_cell: EastRef, T: Any) -> Any:
    """Get the current value from a reference cell.

    Args:
        ref_cell: The reference cell
        T: Type parameter (element type)

    Returns:
        The current value stored in the ref

    Builtin name: Ref.Get
    Type signature: (Ref<T>) -> T
    """
    return deref(ref_cell)


def ref_update(ref_cell: EastRef, value: Any, T: Any) -> None:
    """Replace the value in a reference cell.

    Args:
        ref_cell: The reference cell to update
        value: The new value
        T: Type parameter (element type)

    Returns:
        Null

    Builtin name: Ref.Update
    Type signature: (Ref<T>, T) -> Null
    """
    set_ref(ref_cell, value)
    return None


def ref_merge(
    ref_cell: EastRef,
    new_value: Any,
    update_fn: Callable[[Any, Any], Any],
    T: Any,
    T2: Any,
) -> None:
    """Modify reference value by merging with a new value using a function.

    This is useful for patterns where you want to update a reference based on its current value,
    e.g. incrementing a number, appending to a string, updating fields in a struct.

    Args:
        ref_cell: The reference cell to update
        new_value: The new value to merge with
        update_fn: Function (current, new) -> merged
        T: Type parameter (current value type)
        T2: Type parameter (new value type)

    Returns:
        Null

    Builtin name: Ref.Merge
    Type signature: (Ref<T>, T2, (T, T2) -> T) -> Null

    Examples:
        # Increment counter
        ref_merge(counter, 5, lambda cur, delta: cur + delta, IntegerType, IntegerType)
    """
    current = deref(ref_cell)
    merged = update_fn(current, new_value)
    set_ref(ref_cell, merged)
    return None


# Register builtins
register_builtin("Ref.Get", ref_get)
register_builtin("Ref.Update", ref_update)
register_builtin("Ref.Merge", ref_merge)
```

#### Task 7.2: Import ref_ops in builtins/__init__.py

**Location:** `east/builtins/__init__.py`

**Add import:**
```python
from east.builtins import (
    array,
    set_ops,
    dict_ops,
    string,
    datetime_ops,
    blob,
    comparison,
    ref_ops,  # ADD THIS
)
```

#### Task 7.3: Update builtin type signatures (if needed)

**Location:** `east/builtins/registry.py` or type definition file

**Add type signatures:**
```python
BUILTIN_SIGNATURES = {
    # ... existing signatures ...

    "Ref.Get": {
        "type_parameters": ["T"],
        "inputs": [RefType("T")],
        "output": "T",
    },
    "Ref.Update": {
        "type_parameters": ["T"],
        "inputs": [RefType("T"), "T"],
        "output": NullType,
    },
    "Ref.Merge": {
        "type_parameters": ["T", "T2"],
        "inputs": [RefType("T"), "T2", FunctionType(["T", "T2"], "T", None)],
        "output": NullType,
    },
}
```

---

## 8. Serialization Changes

### File: `east/serialization/east_parser.py`

**Location:** Type-specific parsing functions

#### Task 8.1: Implement parse_ref function

**Implementation:**

```python
def parse_ref(text: str, typ: RefType, start: int = 0) -> tuple[EastRef, int]:
    """Parse a ref value from East text format.

    Format: &<value>
    Example: &42, &"hello", &[1, 2, 3]

    Args:
        text: The text to parse
        typ: The RefType to parse into
        start: Starting position in text

    Returns:
        Tuple of (parsed ref, end position)

    Raises:
        ParseError: If text doesn't match expected format
    """
    from east.types.ref import ref

    # Skip whitespace
    pos = skip_whitespace(text, start)

    # Expect '&' prefix
    if pos >= len(text) or text[pos] != '&':
        raise ParseError(f"Expected '&' for Ref at position {pos}")
    pos += 1

    # Skip whitespace after '&'
    pos = skip_whitespace(text, pos)

    # Parse the value
    value, pos = parse_value(text, typ.value, pos)

    return ref(value), pos
```

**Notes:**
- Refs use `&` prefix in text format (e.g., `&42`)
- The value is parsed according to the ref's value type
- Similar to parsing other container types

#### Task 8.2: Add RefType case to parse dispatch

**Location:** Main `parse_value` function

**Add case:**
```python
def parse_value(text: str, typ: EastType, start: int = 0) -> tuple[Any, int]:
    """Parse a value of the given type from text."""
    # ... existing type checks ...

    elif isinstance(typ, RefType):
        return parse_ref(text, typ, start)
    elif isinstance(typ, ArrayType):
        return parse_array(text, typ, start)
    # ... rest of cases ...
```

---

### File: `east/serialization/east_printer.py`

**Location:** Type-specific printing functions

#### Task 8.3: Implement print_ref function

**Implementation:**

```python
def print_ref(value: EastRef, typ: RefType, ctx: PrintContext) -> str:
    """Print a ref value to East text format.

    Format: &<value>

    Args:
        value: The ref to print
        typ: The RefType
        ctx: Printing context (for indentation, etc.)

    Returns:
        Text representation of the ref

    Examples:
        >>> print_ref(ref(42), RefType(IntegerType), ctx)
        '&42'
    """
    from east.types.ref import deref

    # Get the value
    inner_value = deref(value)

    # Print the value
    inner_text = print_value(inner_value, typ.value, ctx)

    return f"&{inner_text}"
```

**Notes:**
- Simple prefix notation with `&`
- Delegates to inner value printing
- No special formatting needed

#### Task 8.4: Add RefType case to print dispatch

**Location:** Main `print_value` function

**Add case:**
```python
def print_value(value: Any, typ: EastType, ctx: PrintContext) -> str:
    """Print a value of the given type to text."""
    # ... existing type checks ...

    elif isinstance(typ, RefType):
        return print_ref(value, typ, ctx)
    elif isinstance(typ, ArrayType):
        return print_array(value, typ, ctx)
    # ... rest of cases ...
```

---

### File: `east/serialization/json.py`

**Location:** JSON encoding/decoding

#### Task 8.5: Implement Ref JSON encoding

**Implementation:**

```python
# In encode_json_for function:

def _encode_ref(value: EastRef, typ: RefType, ctx: JSONEncodeContext) -> Any:
    """Encode a ref to JSON format.

    Refs are encoded as single-element arrays to support aliasing.
    The aliasing mechanism allows multiple references to share the same value.

    Format: [<value>]

    Args:
        value: The ref to encode
        typ: The RefType
        ctx: Encoding context (for aliasing tracking)

    Returns:
        JSON-compatible array
    """
    from east.types.ref import deref

    # Serialize the referenced value as array of one element
    ctx.current_path.append("0")
    result = [_encode_value(deref(value), typ.value, ctx)]
    ctx.current_path.pop()

    return result
```

**Note from TypeScript:**
The TypeScript version had a bug where it was encoding `value` directly instead of `value.value` (deref). Make sure to use `deref(value)` in Python.

**Add case to dispatch:**
```python
elif isinstance(typ, RefType):
    return _encode_ref(value, typ, ctx)
```

#### Task 8.6: Implement Ref JSON decoding

**Implementation:**

```python
# In decode_json_for function:

def _decode_ref(data: Any, typ: RefType, ctx: JSONDecodeContext) -> EastRef:
    """Decode a ref from JSON format.

    Refs are encoded as single-element arrays.

    Args:
        data: The JSON data (array)
        typ: The RefType
        ctx: Decoding context (for aliasing)

    Returns:
        Decoded ref

    Raises:
        DecodeError: If data is not an array or has wrong length
    """
    from east.types.ref import ref

    if not isinstance(data, list):
        raise DecodeError(f"Expected array for Ref, got {type(data).__name__}")

    if len(data) != 1:
        raise DecodeError(f"Expected array of length 1 for Ref, got {len(data)}")

    # Decode the value
    ctx.current_path.append("0")
    value = _decode_value(data[0], typ.value, ctx)
    ctx.current_path.pop()

    return ref(value)
```

**Add case to dispatch:**
```python
elif isinstance(typ, RefType):
    return _decode_ref(data, typ, ctx)
```

---

## 9. Testing

### Directory: `tests/`

**Status:** 🆕 **CREATE NEW TEST FILE**

#### Task 9.1: Create test file for RefType

**File:** `tests/types/test_ref.py`

**Required tests (based on TypeScript ref.spec.ts):**

1. ✅ **Construct, get, set** - Basic ref creation and mutation
2. ✅ **Identity semantics** - Multiple variables pointing to same ref
3. ✅ **Ref.Get builtin** - Get value from ref
4. ✅ **Ref.Update builtin** - Update ref value
5. ✅ **Ref.Merge builtin** - Merge with function
6. ✅ **Identity comparison with is()** - Ref identity vs value equality
7. ✅ **Equality comparison** - Ref equality by identity
8. ✅ **Not equal comparison** - Ref inequality
9. ✅ **Less than comparison** - Ref ordering (by value)
10. ✅ **Less than or equal comparison**
11. ✅ **Greater than comparison**
12. ✅ **Greater than or equal comparison**
13. ✅ **Printing** - Format as `&<value>`
14. ✅ **Parsing** - Parse `&42`, `& 42`, etc.
15. ✅ **Parse errors** - Invalid formats

**Test template:**

```python
"""Tests for RefType and Ref operations."""

import pytest
from east.types.type_system import RefType, IntegerType, StringType
from east.types.ref import ref, deref, set_ref, is_ref
from east.builtins.ref_ops import ref_get, ref_update, ref_merge
from east.serialization.east_parser import parse_value
from east.serialization.east_printer import print_value


def test_construct_get_set():
    """Test basic ref construction, get, and set operations."""
    # Create ref
    r1 = ref(42)
    assert is_ref(r1)
    assert deref(r1) == 42

    # Identity semantics - aliases point to same ref
    r2 = r1
    assert r1 is r2
    assert deref(r2) == 42

    # Update via r1
    set_ref(r1, 100)
    assert deref(r1) == 100
    assert deref(r2) == 100  # r2 sees the update

    # Update via r2
    set_ref(r2, 200)
    assert deref(r1) == 200
    assert deref(r2) == 200


def test_ref_get_builtin():
    """Test Ref.Get builtin function."""
    r = ref(42)
    result = ref_get(r, IntegerType)
    assert result == 42


def test_ref_update_builtin():
    """Test Ref.Update builtin function."""
    r = ref(0)
    result = ref_update(r, 100, IntegerType)
    assert result is None
    assert deref(r) == 100


def test_ref_merge_builtin():
    """Test Ref.Merge builtin function."""
    r = ref(10)

    # Merge with addition
    def add(current, delta):
        return current + delta

    result = ref_merge(r, 5, add, IntegerType, IntegerType)
    assert result is None
    assert deref(r) == 15

    # Merge again
    ref_merge(r, 20, add, IntegerType, IntegerType)
    assert deref(r) == 35


def test_identity_comparison():
    """Test ref identity comparison with is vs equality."""
    r1 = ref(10)
    r2 = ref(20)
    r3 = r1  # Alias

    # Identity - same object
    assert r1 is r3
    assert r1 is not r2

    # Equality (structural) - refs with same value are NOT equal unless same identity
    assert r1 == r3  # Same identity
    assert r1 != r2  # Different identity

    # Even if values are equal, refs are different objects
    r4 = ref(10)
    assert deref(r1) == deref(r4)
    assert r1 is not r4
    assert r1 != r4


def test_comparison_operations():
    """Test ref comparison operations (by value)."""
    r1 = ref(10)
    r2 = ref(20)
    r3 = r1

    # Note: Comparisons might be by identity or by value - check TypeScript behavior
    # For now, assume value-based comparisons:

    # Less than
    assert (deref(r1) < deref(r2)) == True
    assert (deref(r2) < deref(r1)) == False

    # Less than or equal
    assert (deref(r1) <= deref(r2)) == True
    assert (deref(r2) <= deref(r1)) == False
    assert (deref(r1) <= deref(r3)) == True

    # Greater than
    assert (deref(r2) > deref(r1)) == True
    assert (deref(r1) > deref(r2)) == False

    # Greater than or equal
    assert (deref(r2) >= deref(r1)) == True
    assert (deref(r1) >= deref(r2)) == False
    assert (deref(r1) >= deref(r3)) == True


def test_printing():
    """Test printing refs to East text format."""
    from east.serialization.east_printer import PrintContext

    r = ref(42)
    ctx = PrintContext()
    result = print_value(r, RefType(IntegerType), ctx)
    assert result == "&42"


def test_parsing():
    """Test parsing refs from East text format."""
    # Valid formats
    r1, _ = parse_value("&42", RefType(IntegerType))
    assert deref(r1) == 42

    r2, _ = parse_value("& 42", RefType(IntegerType))
    assert deref(r2) == 42

    r3, _ = parse_value('&"hello"', RefType(StringType))
    assert deref(r3) == "hello"


def test_parse_errors():
    """Test parsing errors for invalid ref formats."""
    # Missing &
    with pytest.raises(Exception):  # ParseError
        parse_value("42", RefType(IntegerType))

    # Just &
    with pytest.raises(Exception):
        parse_value("&", RefType(IntegerType))

    # Empty string
    with pytest.raises(Exception):
        parse_value("", RefType(IntegerType))

    # Wrong value type
    with pytest.raises(Exception):
        parse_value("&3.14", RefType(IntegerType))


def test_type_validation():
    """Test that RefType validates value types."""
    from east.types.type_system import FunctionType

    # Valid: data types
    RefType(IntegerType)  # OK
    RefType(StringType)  # OK

    # Invalid: function types
    with pytest.raises(TypeError):
        RefType(FunctionType([IntegerType], IntegerType, None))
```

#### Task 9.2: Add integration tests

**File:** `tests/integration/test_ref_integration.py`

**Tests needed:**
- Refs in closures (shared state)
- Refs in recursive functions
- Refs with complex value types (structs, arrays)
- Refs in serialization round-trip (JSON, BEAST)
- Async functions with refs

---

## 10. Additional Changes

### File: `east/builtins/array.py`

**Status:** 🐛 **BUG FIX**

#### Task 10.1: Fix Array.merge() return type

**Location:** Array.merge builtin signature

**Current (incorrect):**
```python
def array_merge(arr: EastArray, index: int, value: Any, merge_fn: Callable, T: Any, T2: Any) -> Any:
    """Merge value at index using function."""
    # ... implementation ...
    return merged_value  # WRONG - should return None
```

**Required change:**
```python
def array_merge(arr: EastArray, index: int, value: Any, merge_fn: Callable, T: Any, T2: Any) -> None:
    """Merge value at index using function.

    This is a mutation operation that returns Null.
    """
    # ... implementation ...
    return None  # CORRECT - side effect only
```

**Also update type signature:**
```python
BUILTIN_SIGNATURES["Array.Merge"] = {
    "type_parameters": ["T", "T2"],
    "inputs": [ArrayType("T"), IntegerType, "T2", FunctionType(["T", "T2", IntegerType], "T", None)],
    "output": NullType,  # CHANGED from "T"
}
```

**Notes:**
- Array.merge is a mutation operation (side effect)
- Should return NullType, not the element type
- This aligns with Ref.Merge and other mutation operations

---

## 11. Documentation

### File: `docs/TYPES.md` or README updates

**Required sections:**

1. **RefType Overview**
   - Mutable reference cells
   - Identity semantics
   - Use cases (shared state, closures)

2. **Basic Usage**
   ```python
   from east.types.type_system import RefType, IntegerType
   from east.types.ref import ref, deref, set_ref

   # Create ref
   counter = ref(0)

   # Get value
   value = deref(counter)

   # Update value
   set_ref(counter, value + 1)
   ```

3. **Operations**
   - `Ref.Get` - Get current value
   - `Ref.Update` - Replace value
   - `Ref.Merge` - Update based on current value

4. **Identity Semantics**
   ```python
   r1 = ref(10)
   r2 = r1  # Same ref
   set_ref(r2, 20)
   assert deref(r1) == 20  # r1 sees the update
   ```

5. **Serialization**
   - Text format: `&<value>`
   - JSON format: single-element array
   - Aliasing support

---

## 12. Implementation Order & Dependencies

**Current Status Summary:**

### ✅ Phase 1: Core Type System - **COMPLETE** (commit 5f72851)
1. ✅ Create `east/types/ref.py` with ref container
2. ✅ Add RefType to `type_system.py`
3. ✅ Update all type operations (subtype, union, intersect, etc.)
4. ✅ Update is_data_type(), is_immutable_type()

### ✅ Phase 6: Serialization - **COMPLETE** (commit 5f72851)
14. ✅ Implement parse_ref (East format with `&`)
15. ✅ Add RefType parse dispatch
16. ✅ Implement print_ref (East format with `&`)
17. ✅ Add RefType print dispatch
18. ✅ Implement Ref JSON encoding (with `$ref` backreferences)
19. ✅ Implement Ref JSON decoding
20. ✅ Implement Beast v2 encoding/decoding (with aliasing)
21. ✅ Update Beast v1 to reject refs

### ✅ Phase 7 (Partial): Basic Testing - **COMPLETE** (commit 5f72851)
22. ✅ Create basic ref tests (15 tests in `tests/test_ref.py`)
23. ✅ Create serialization tests (1707 tests in `tests/serialization/test_beast2.py`)
24. ✅ Add fuzzing support for refs

---

**Remaining Work:**

### ❌ Phase 2: IR System - **NOT STARTED**
5. ❌ Add NewRefIR to IR type definitions (Task 3.1)
6. ❌ Create ir_new_ref builder (Task 4.1)

### ❌ Phase 3: Analysis - **NOT STARTED**
7. ❌ Add NewRef analysis in analyze.py (Task 5.1)

### ❌ Phase 4: Runtime - **NOT STARTED**
8. ❌ Implement _compile_new_ref in compiler.py (Task 6.1)
9. ❌ Add NewRef dispatch case (Task 6.2)

### ❌ Phase 5: Builtins - **NOT STARTED**
10. ❌ Create ref_ops.py with builtins (Task 7.1)
11. ❌ Register builtins (Task 7.2)
12. ❌ Update builtin type signatures (Task 7.3)
13. ❌ Fix Array.merge return type (Task 10.1)

### ❌ Phase 7 (Remaining): Integration Testing - **NOT STARTED**
25. ❌ Create end-to-end compiler tests (Task 9.2)
26. ❌ Test refs with closures and captures
27. ❌ Test async refs

### 📝 Phase 8: Documentation - **PARTIAL**
28. ✅ Basic ref usage documented in tests
29. ❌ High-level documentation (Task 11)
30. ❌ Update CLAUDE.md or README with ref info

---

## 13. Verification Checklist

After implementation, verify:

- [ ] RefType properly defined with validation
- [ ] ref container class with identity semantics
- [ ] NewRefIR added to IR type system
- [ ] IR analysis handles NewRef nodes
- [ ] Compiler generates correct ref creation code
- [ ] All three ref builtins implemented (Get, Update, Merge)
- [ ] Builtins properly registered
- [ ] Array.merge return type fixed to NullType
- [ ] Ref parsing supports `&<value>` format
- [ ] Ref printing outputs `&<value>` format
- [ ] Ref JSON encoding as single-element array
- [ ] Ref JSON decoding from array
- [ ] All 15 tests pass
- [ ] Integration tests pass
- [ ] Documentation complete and accurate
- [ ] Identity semantics work correctly
- [ ] Aliasing works in serialization
- [ ] Type validation rejects function types
- [ ] No regressions in existing tests

---

## 14. Known Issues & Open Questions

### Questions to resolve:

1. **Comparison semantics:**
   - Should ref comparisons use identity or value?
   - TypeScript tests suggest value-based for <, <=, >, >=
   - But identity-based for == and is()
   - Need to verify exact semantics

2. **Frozen refs:**
   - TypeScript checks `Object.isFrozen(ref)` before mutations
   - How to implement freezing in Python?
   - Use `object.__setattr__` restrictions?

3. **Aliasing in serialization:**
   - TypeScript mentions aliasing support in JSON
   - Need to implement alias tracking in JSON context
   - Multiple refs to same object should serialize once

4. **Set/Dict membership:**
   - Are refs allowed as Set elements or Dict keys?
   - TypeScript marks them as immutable (for type system purposes)
   - But they're mutable values - need clarification

5. **Ordering semantics:**
   - How should refs be ordered?
   - By identity (memory address)?
   - By value?
   - TypeScript tests suggest by value

### TypeScript-Python differences:

- **Object identity:** Python `is` vs JavaScript `===`
- **Freezing:** Python doesn't have `Object.freeze()` equivalent
- **Brand symbols:** Python uses dataclass field, TypeScript uses Symbol
- **Hash function:** Python needs `__hash__` for dict/set usage

---

## 15. Success Criteria

**Already achieved (commit 5f72851):**
1. ✅ RefType and ref container properly defined
2. ✅ Parsing supports `&<value>` syntax
3. ✅ Printing outputs `&<value>` format
4. ✅ JSON serialization handles refs (with `$ref` backreferences)
5. ✅ Beast v2 serialization handles refs (with aliasing)
6. ✅ All 15 basic ref tests pass
7. ✅ All 1707 Beast2 tests pass (including refs)
8. ✅ Identity semantics correct
9. ✅ No regressions in existing tests (1182 tests passing)
10. ✅ Comparison and ordering implemented with cycle detection
11. ✅ Default value generation for refs
12. ✅ Fuzzing support for refs

**Still needed for completion:**
13. ❌ IR system includes NewRef nodes
14. ❌ Compiler generates ref creation code
15. ❌ All three builtins implemented (Ref.Get, Ref.Update, Ref.Merge)
16. ❌ Builtins tested
17. ❌ Array.merge bug fix applied (return NullType instead of element type)
18. ❌ Integration tests through compiler pass
19. ❌ End-to-end tests with closures and async
20. ❌ High-level documentation complete
21. ❌ Code review passed

---

## 16. Timeline Estimate

**Already completed (commit 5f72851):**
- ✅ Phase 1 (Core Type System): ~4 hours
- ✅ Phase 6 (Serialization): ~8 hours
- ✅ Phase 7 Partial (Basic Testing): ~5 hours
- **Subtotal completed: ~17 hours**

**Remaining work (1 developer working full-time):**
- ❌ Phase 2 (IR System): 2-3 hours
- ❌ Phase 3 (Analysis): 1-2 hours
- ❌ Phase 4 (Runtime): 4-6 hours
- ❌ Phase 5 (Builtins): 3-4 hours
- ❌ Phase 7 Remaining (Integration Tests): 3-5 hours
- ❌ Phase 8 (Documentation): 2-3 hours

**Total remaining: 15-23 hours (2-3 days)**

**Original estimate: 31-43 hours**
**Completed: ~17 hours (40%)**
**Remaining: ~19 hours (60%)**

---

## References

- **East TypeScript Commit:** 3c3b001db79e7f4487eb57fb6ce1c7ffe5e0145a
- **Key Files Changed:**
  - `src/types.ts` - Added RefType type definition
  - `src/containers/ref.ts` - NEW: Ref container implementation
  - `src/ir.ts` - Added NewRefIR to IR type
  - `src/ast.ts` - Added NewRefAST
  - `src/ast_to_ir.ts` - AST to IR conversion for refs
  - `src/analyze.ts` - IR analysis for NewRef
  - `src/compile.ts` - Compilation of NewRef nodes
  - `src/builtins.ts` - Added Ref.Get, Ref.Update, Ref.Merge
  - `src/expr/ref.ts` - NEW: RefExpr class
  - `src/expr/ast.ts` - Value to AST conversion for refs
  - `src/expr/types.ts` - Type mappings for RefExpr
  - `src/expr/block.ts` - Factory for RefExpr
  - `src/expr/array.ts` - Fixed Array.merge return type
  - `src/serialization/json.ts` - JSON encoding fix
  - `src/type_of_type.ts` - EastTypeValue support for Ref
  - `test/ref.spec.ts` - NEW: Comprehensive ref tests
  - `USAGE.md` - RefType documentation
  - `STANDARDS.md` - NEW: Documentation standards

---

**Document Version:** 1.0
**Created:** 2025-11-11
**Last Updated:** 2025-11-11
**Status:** Ready for implementation
