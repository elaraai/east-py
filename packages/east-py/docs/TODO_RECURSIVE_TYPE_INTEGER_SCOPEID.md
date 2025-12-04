# TODO: Refactor Recursive Types to Use Integer scope_id (TypeScript Compatibility)

**Based on:** ../East TypeScript implementation (src/types.ts, src/comparison.ts, src/internal.ts)
**Status:** Python uses RecursiveTypeMarker objects; TypeScript uses integers
**Goal:** Achieve exact implementation parity with TypeScript for recursive types
**Priority:** CRITICAL - Blocking TypeScript export tests, creating maintenance burden

## Overview

The East Python runtime uses a **different implementation** of recursive types than TypeScript. This divergence causes:
- **Test failures** when comparing TypeScript-exported type values with Python type values
- **Maintenance burden** due to dual implementations that must be kept in sync
- **Complexity** in serialization/deserialization with marker conversion logic
- **Bugs** from mismatched recursive type resolution between implementations

**Current Python approach:**
- Uses `RecursiveTypeMarker` objects as placeholders
- Maintains `marker_map: dict[int, int]` to map marker id() to context indices
- Calls `_find_recursive_marker()` to discover markers in types
- Different code paths for RecursiveTypeMarker vs integer scope_ids

**TypeScript approach (target):**
- Uses **integer scope_id** directly in recursive type nodes
- Formula: `typeCtx[typeCtx.length - scope_id]` to resolve recursive references
- No marker objects, no marker maps, no marker discovery
- Single code path for all recursive type handling

**Why this matters:**
- TypeScript exports contain integer scope_ids in JSON
- Python must handle both RecursiveTypeMarker (native) AND integers (TypeScript imports)
- Creating conversion logic is error-prone and diverges implementations
- Maintenance nightmare: changes to TypeScript require parallel changes to Python

**Impact:**
- **tests/test_typescript_exports.py** - String test fails with "object of type 'EastVariant' has no len()"
- **Cross-language compatibility** - Type comparison fails between Python/TypeScript types
- **Code complexity** - Dual code paths for RecursiveTypeMarker vs integer scope_ids
- **Serialization** - Complex conversion logic in JSON encoder/decoder/printer/parser

---

## Current State Analysis

### Problem Root Cause

When comparing two type values (e.g., `.Array .Integer` parsed from TypeScript JSON), the comparison fails because:

1. **EastTypeType (Python)** uses `RecursiveTypeMarker` objects:
   ```python
   # Array case in EastTypeType has:
   ("Array", type=.Recursive <RecursiveMarker at 0x...>)
   ```

2. **Type values from TypeScript** use integer scope_ids:
   ```python
   # Parsed from JSON:
   ("Array", type=.Recursive 1)  # Integer, not marker
   ```

3. **Comparison code** built for EastTypeType expects markers and builds `marker_map`, but when comparing values with integer scope_ids, the context stack has the wrong length/indices.

4. **The formula** `ctx_index = len(type_ctx) - scope_id` works, but only if the type context stack is built correctly. With RecursiveTypeMarkers, we try to find and register them, leading to mismatches.

### Files Using RecursiveTypeMarker (Current Implementation)

#### `east/types/type_system.py` (Lines 48-148)
**Defines RecursiveTypeMarker class and recursive_type() function:**

```python
class RecursiveTypeMarker:
    """Marker object for recursive type references.

    Created by recursive_type() to mark self-reference points.
    At creation time, node=None. After the type is constructed,
    the marker's node is set to point to the owning type.
    """

    def __init__(self) -> None:
        self.node: EastType | None = None

def recursive_type(f: Callable[[RecursiveTypeMarker], EastType]) -> EastType:
    """Create a recursive type.

    Args:
        f: Function that receives a self-reference marker and returns the node type

    Returns:
        The recursive type with marker.node set to the returned type
    """
    marker = RecursiveTypeMarker()
    node = f(marker)
    marker.node = node
    return node
```

**Problem:** This entire abstraction doesn't exist in TypeScript. TypeScript uses integers directly.

#### `east/serialization/json.py` (Multiple locations)
**Type encoder (lines 543-573):**

```python
if type_kind == "Recursive":
    marker = type_val.value  # Could be RecursiveTypeMarker OR int
    if isinstance(marker, RecursiveTypeMarker):
        marker_id = id(marker)
        if marker_id not in marker_map:
            raise ValueError(f"Recursive type marker not found: marker_id={marker_id}")
        stack_index = marker_map[marker_id]
        if stack_index < 0 or stack_index >= len(type_ctx):
            raise ValueError(f"Invalid type context index: {stack_index}")
        return type_ctx[stack_index]
    # Integer scope_id from TypeScript exports
    if isinstance(marker, int):
        stack_index = len(type_ctx) - marker
        if stack_index < 0 or stack_index >= len(type_ctx):
            raise ValueError(f"Invalid recursive scope_id {marker}")
        return type_ctx[stack_index]
```

**Problem:** Dual code paths. TypeScript has only the integer path.

**Type decoder (lines 1171-1206):** Same dual code path pattern.

**Value encoder (lines 367-495):** For Array/Set/Dict/Ref, tries to register markers:

```python
# Push encoder onto stack before building element encoder
stack_index = len(type_ctx)
type_ctx.append(encode_array)
marker = _find_recursive_marker(type_val)  # ← This is Python-only
if marker is not None:
    marker_map[id(marker)] = stack_index  # ← This is Python-only
```

**Problem:** TypeScript doesn't need `_find_recursive_marker()` or `marker_map`. It just uses integer scope_ids.

#### `east/serialization/east_printer.py` (Multiple locations)
**Same pattern as JSON - dual code paths, marker discovery, marker_map.**

Lines 311-318, 451-461:
```python
elif isinstance(marker, int):
    # Integer scope_id from TypeScript exports
    ctx_index = len(type_ctx) - marker
    if ctx_index < 0 or ctx_index >= len(type_ctx):
        raise ValueError(f"Invalid recursive scope_id {marker}")
    resolved_type = type_ctx[ctx_index]
```

**Problem:** Should ONLY have the integer path, not dual paths.

#### `east/serialization/east_parser.py` (Multiple locations)
**Same dual code path pattern.**

#### `east/utils/ordering.py` (Multiple locations)
**Comparison functions (equal_for, compare_for, is_for) all have dual code paths.**

Lines 612-622 (equal_for):
```python
if isinstance(marker, RecursiveTypeMarker):
    marker_id = id(marker)
    if marker_id not in marker_map:
        raise ValueError(f"Recursive type marker not found: marker_id={marker_id}")
    ctx_index = marker_map[marker_id]
    # ...
    return type_ctx[ctx_index]
# Integer scope_id from TypeScript exports
if isinstance(marker, int):
    ctx_index = len(type_ctx) - marker
    # ...
    return type_ctx[ctx_index]
```

**Problem:** TypeScript only has integer path (see src/comparison.ts:89).

Lines 407-410, 443-447, 459-463, 507-510 (marker registration in Array/Set/Dict/Ref):
```python
# Register marker if this is a recursive type
marker = _find_recursive_marker(type_val)
if marker is not None:
    marker_map[id(marker)] = len(type_ctx) - 1
```

**Problem:** TypeScript doesn't do marker discovery or registration.

#### `east/utils/ordering.py` (Lines 43-112)
**Function `_find_recursive_marker()`:**

```python
def _find_recursive_marker(typ: Any) -> Any | None:
    """Find the RecursiveTypeMarker that this type owns (if any).

    For a Struct/Variant type created with recursive_type(), this returns the marker
    by checking if any Recursive refs in the type point back to this type as their node.
    """
    # ... 70 lines of marker discovery logic ...
```

**Problem:** This entire function doesn't exist in TypeScript. It's Python-specific complexity.

---

## TypeScript Implementation (Target)

### File: `src/types.ts`

**Type definition (lines 40-43):**
```typescript
export type RecursiveTypeMarker = { type: "Recursive" };
export type RecursiveType<Node = any> = { type: "Recursive", node: Node };
```

**Note:** `RecursiveTypeMarker` is just a type placeholder for the builder, not a runtime object.

**Constructor (lines 117-166):**
```typescript
export function RecursiveType<F extends (self: RecursiveTypeMarker) => EastType>(f: F): RecursiveType<ReturnType<F>> {
  // Create a marker object
  const marker: RecursiveTypeMarker = { type: "Recursive" };

  // Call user function to get node type
  const node = f(marker);

  // Replace all instances of marker with integer scope_id
  let depth = 0;

  function replaceMarkers(t: any): any {
    if (t === marker) {
      // Replace marker with integer scope_id
      return { type: "Recursive", value: depth } as EastType;
    }

    if (t.type === "Array" || t.type === "Set" || t.type === "Ref") {
      depth++;
      const result = { ...t, value: replaceMarkers(t.value) };
      depth--;
      return result;
    }

    if (t.type === "Dict") {
      depth++;
      const result = {
        ...t,
        value: {
          key: replaceMarkers(t.value.key),
          value: replaceMarkers(t.value.value),
        },
      };
      depth--;
      return result;
    }

    if (t.type === "Struct") {
      depth++;
      const result = {
        ...t,
        value: t.value.map((field: any) => ({
          name: field.name,
          type: replaceMarkers(field.type),
        })),
      };
      depth--;
      return result;
    }

    if (t.type === "Variant") {
      depth++;
      const result = {
        ...t,
        value: t.value.map((c: any) => ({
          name: c.name,
          type: replaceMarkers(c.type),
        })),
      };
      depth--;
      return result;
    }

    return t;
  }

  const nodeWithIntegers = replaceMarkers(node);

  return { type: "Recursive", node: nodeWithIntegers } as RecursiveType<ReturnType<F>>;
}
```

**Key insight:** TypeScript starts with a marker object during construction, then **traverses the type tree** and replaces all marker references with **integer scope_ids** based on depth. The final type contains only integers.

### File: `src/comparison.ts`

**Recursive type resolution (lines 88-93):**
```typescript
} else if (type.type === "Recursive") {
  const ret = typeCtx[typeCtx.length - Number(type.value)];
  if (ret === undefined) {
    throw new Error(`Internal error: Recursive type context not found`);
  }
  return ret;
}
```

**Key insight:** Simple formula. No marker_map, no marker discovery, no dual code paths.

### File: `src/internal.ts` (JSON encoder/decoder)

**Type encoding (similar pattern in JSON encoder):**
- No `_find_recursive_marker()`
- No `marker_map` maintenance
- Recursive case just uses: `typeCtx[typeCtx.length - type.value]`

---

## Required Changes

### Phase 1: Refactor `recursive_type()` to Generate Integer scope_ids

**File: `east/types/type_system.py`**

**Current code (lines 75-148):**
```python
class RecursiveTypeMarker:
    """Marker object for recursive type references."""
    def __init__(self) -> None:
        self.node: EastType | None = None

def recursive_type(f: Callable[[RecursiveTypeMarker], EastType]) -> EastType:
    """Create a recursive type."""
    marker = RecursiveTypeMarker()
    node = f(marker)
    marker.node = node
    return node
```

**New code:**
```python
class RecursiveTypeMarker:
    """Temporary marker used during recursive type construction.

    This is only used during the build phase. After construction,
    all marker references are replaced with integer scope_ids.
    """
    pass

def recursive_type(f: Callable[[RecursiveTypeMarker], EastType]) -> EastType:
    """Create a recursive type with integer scope_ids.

    Args:
        f: Function that receives a marker and returns the node type

    Returns:
        A type where all self-references use integer scope_ids
    """
    marker = RecursiveTypeMarker()
    node = f(marker)

    # Replace all marker instances with integer scope_ids
    def replace_markers(t: Any, depth: int = 0) -> Any:
        """Recursively replace markers with integer scope_ids."""
        from east.types.structural import EastVariant, EastStruct

        # If this IS the marker, replace with Recursive integer
        if isinstance(t, RecursiveTypeMarker) and t is marker:
            # Create Recursive type with integer scope_id
            return EastVariant(EastTypeType, Case("Recursive", depth))

        # If this is already a Recursive type with a marker, replace it
        if isinstance(t, EastVariant) and t.tag == "Recursive":
            if isinstance(t.value, RecursiveTypeMarker) and t.value is marker:
                return EastVariant(EastTypeType, Case("Recursive", depth))
            # Already has integer or different marker, leave as is
            return t

        # Traverse composite types and increment depth
        if isinstance(t, EastVariant):
            tag = t.tag

            if tag in ("Array", "Set", "Ref"):
                # Increase depth when going into container
                new_value = replace_markers(t.value, depth + 1)
                return EastVariant(t._east_type, Case(tag, new_value))

            if tag == "Dict":
                # Increase depth, replace both key and value
                dict_struct = t.value
                new_key = replace_markers(dict_struct.key, depth + 1)
                new_value = replace_markers(dict_struct.value, depth + 1)
                new_struct = EastStruct(
                    dict_struct._east_type,
                    {"key": new_key, "value": new_value}
                )
                return EastVariant(t._east_type, Case(tag, new_struct))

            if tag == "Struct":
                # Increase depth, replace all field types
                fields = t.value
                new_fields = []
                for field in fields:
                    new_type = replace_markers(field.type, depth + 1)
                    new_field = EastStruct(
                        field._east_type,
                        {"name": field.name, "type": new_type}
                    )
                    new_fields.append(new_field)
                return EastVariant(t._east_type, Case(tag, new_fields))

            if tag == "Variant":
                # Increase depth, replace all case types
                cases = t.value
                new_cases = []
                for case in cases:
                    new_type = replace_markers(case.type, depth + 1)
                    new_case = EastStruct(
                        case._east_type,
                        {"name": case.name, "type": new_type}
                    )
                    new_cases.append(new_case)
                return EastVariant(t._east_type, Case(tag, new_cases))

        # Not a composite type, return as-is
        return t

    return replace_markers(node, depth=0)
```

**Result:** All recursive types created in Python will use integer scope_ids, matching TypeScript.

---

### Phase 2: Remove Marker Discovery and Marker Map Logic

**Files to change:**
- `east/serialization/json.py`
- `east/serialization/east_printer.py`
- `east/serialization/east_parser.py`
- `east/utils/ordering.py`

#### Remove `_find_recursive_marker()` function

**File: `east/utils/ordering.py` (lines 43-112)**

**Action:** DELETE entire function. It's no longer needed.

#### Remove dual code paths for RecursiveTypeMarker vs integer

**Example: `east/utils/ordering.py` (equal_for function)**

**Current code (lines 612-622):**
```python
if type_kind == "Recursive":
    from east.types.type_system import RecursiveTypeMarker

    marker = type_val.value
    if isinstance(marker, RecursiveTypeMarker):
        marker_id = id(marker)
        if marker_id not in marker_map:
            raise ValueError(f"Recursive type marker not found: marker_id={marker_id}")
        ctx_index = marker_map[marker_id]
        if ctx_index < 0 or ctx_index >= len(type_ctx):
            raise ValueError(f"Invalid type context index: {ctx_index}")
        return type_ctx[ctx_index]
    # Integer scope_id from TypeScript exports
    if isinstance(marker, int):
        ctx_index = len(type_ctx) - marker
        if ctx_index < 0 or ctx_index >= len(type_ctx):
            raise ValueError(f"Invalid recursive scope_id {marker}")
        return type_ctx[ctx_index]
    raise ValueError(f"Expected RecursiveTypeMarker or int, got {type(marker)}")
```

**New code:**
```python
if type_kind == "Recursive":
    scope_id = type_val.value
    if not isinstance(scope_id, int):
        raise ValueError(f"Recursive type must have integer scope_id, got {type(scope_id)}")

    ctx_index = len(type_ctx) - scope_id
    if ctx_index < 0 or ctx_index >= len(type_ctx):
        raise ValueError(
            f"Invalid recursive scope_id {scope_id} "
            f"(ctx len={len(type_ctx)}, calculated index={ctx_index})"
        )
    return type_ctx[ctx_index]
```

**Apply same pattern to:**
- `east/serialization/json.py` - Type encoder (lines 543-573)
- `east/serialization/json.py` - Type decoder (lines 1171-1206)
- `east/serialization/east_printer.py` - Type printer (lines 451-461)
- `east/serialization/east_parser.py` - Type parser (lines 459-466)
- `east/utils/ordering.py` - equal_for (lines 612-622)
- `east/utils/ordering.py` - compare_for (lines 760-770)
- `east/utils/ordering.py` - is_for (lines 1088-1098)

#### Remove marker registration in Array/Set/Dict/Ref handlers

**Example: `east/utils/ordering.py` (equal_for, Array case)**

**Current code (lines 407-411):**
```python
if type_kind == "Array":
    type_ctx.append(None)  # Placeholder
    # Register marker if this is a recursive type
    marker = _find_recursive_marker(type_val)
    if marker is not None:
        marker_map[id(marker)] = len(type_ctx) - 1
    value_comparer = equal_for(type_val.value, type_ctx, marker_map)
```

**New code:**
```python
if type_kind == "Array":
    type_ctx.append(None)  # Placeholder
    value_comparer = equal_for(type_val.value, type_ctx, marker_map)
```

**Note:** We still need `type_ctx` (the stack), but we don't need marker discovery or `marker_map` registration.

**Actually, wait** - do we still need `marker_map` at all?

Looking at TypeScript `src/comparison.ts` - NO `marker_map` parameter exists.

So we should:
1. Remove `marker_map` parameter from all functions
2. Remove all `marker_map` creation/passing
3. Remove all `_find_recursive_marker()` calls
4. Keep `type_ctx` (the comparer/encoder/decoder stack)

**Revised approach:**

**Remove `marker_map` parameter entirely from all functions:**
- `equal_for(type_val, type_ctx=None, marker_map=None)` → `equal_for(type_val, type_ctx=None)`
- `compare_for(type_val, type_ctx=None, marker_map=None)` → `compare_for(type_val, type_ctx=None)`
- `is_for(type_val, type_ctx=None, marker_map=None)` → `is_for(type_val, type_ctx=None)`
- `to_json_for(type_val, type_ctx=None, marker_map=None)` → `to_json_for(type_val, type_ctx=None)`
- `from_json_for(type_val, frozen, type_ctx=None, marker_map=None, type_str="")` → `from_json_for(type_val, frozen, type_ctx=None, type_str="")`
- `encode_json_for(type_val, type_ctx=None, marker_map=None)` → `encode_json_for(type_val, type_ctx=None)`
- `decode_json_for(type_val, frozen=False, type_ctx=None, marker_map=None)` → `decode_json_for(type_val, frozen=False, type_ctx=None)`
- `print_type(type_val, type_ctx=None, marker_map=None)` → `print_type(type_val, type_ctx=None)`
- `parse_type(...)` - Remove marker_map parameter
- All parse_* functions - Remove marker_map parameter

**Remove all marker_map initialization:**
```python
# DELETE these lines everywhere:
marker_map: dict[Any, int] = {}
```

**Remove all marker discovery and registration:**
```python
# DELETE these lines everywhere:
marker = _find_recursive_marker(type_val)
if marker is not None:
    marker_map[id(marker)] = len(type_ctx) - 1
```

---

### Phase 3: Update EastTypeType Definition

**File: `east/types/type_system.py`**

**Current definition uses recursive_type() which creates RecursiveTypeMarker:**
```python
EastTypeType = recursive_type(
    lambda self: variant_type(
        [
            ("Array", self),
            ("Set", self),
            # ...
        ]
    )
)
```

**After Phase 1 refactor:** This will automatically use integer scope_ids! The `recursive_type()` function now replaces markers with integers.

**Action:** No change needed. The refactored `recursive_type()` handles this automatically.

---

### Phase 4: Update All Calling Code

**Remove marker_map arguments from all function calls:**

**Files to change (comprehensive search needed):**
- `east/serialization/json.py` - All internal calls
- `east/serialization/east_printer.py` - All internal calls
- `east/serialization/east_parser.py` - All internal calls
- `east/utils/ordering.py` - All internal calls
- `east/builtins/type_system.py` - External calls to print_type, parse_type
- `east/builtins/array.py` - Calls to encode_json_for, etc.
- Any other files that call serialization/comparison functions

**Example changes:**

**Before:**
```python
encoder = to_json_for(element_type, type_ctx, marker_map)
decoder = from_json_for(element_type, frozen, type_ctx, marker_map, type_str)
comparer = equal_for(field_type, type_ctx, marker_map)
```

**After:**
```python
encoder = to_json_for(element_type, type_ctx)
decoder = from_json_for(element_type, frozen, type_ctx, type_str)
comparer = equal_for(field_type, type_ctx)
```

---

### Phase 5: Remove RecursiveTypeMarker Class (Optional Cleanup)

**File: `east/types/type_system.py`**

**Current:**
```python
class RecursiveTypeMarker:
    """Marker object for recursive type references."""
    def __init__(self) -> None:
        self.node: EastType | None = None
```

**After Phase 1:** This class is only used temporarily during `recursive_type()` construction, then all instances are replaced with integers.

**Options:**
1. **Keep it** - It's still used internally by `recursive_type()` during construction
2. **Make it simpler** - Remove the `node` attribute since we don't use it anymore
3. **Replace with sentinel** - Use `object()` as a sentinel instead of a class

**Recommended:** Keep a simple version:
```python
class RecursiveTypeMarker:
    """Temporary marker used during recursive type construction.

    After construction, all marker instances are replaced with integer scope_ids.
    This class should not appear in any final type structures.
    """
    pass
```

---

## Migration Strategy

### Backwards Compatibility

**Question:** Do we need to support old RecursiveTypeMarker-based types?

**Answer:** NO - This is an internal implementation detail. All types are created through `recursive_type()`, which we're refactoring. After refactoring:
1. All NEW types created use integers
2. All EXISTING types (EastTypeType, IRType) are recreated on import using the new `recursive_type()`
3. No old marker-based types persist

**Serialization:** JSON/East serialization already handles both markers and integers. After refactoring, we only generate integers, but the code can still accept old marker-based JSON if needed (though we won't use it).

### Testing During Migration

**Strategy:** Change one subsystem at a time and test:

1. **Phase 1:** Refactor `recursive_type()` only
   - Test: Create recursive types and verify they have integer scope_ids
   - Test: EastTypeType and IRType still work

2. **Phase 2:** Remove marker logic from `equal_for` / `compare_for` / `is_for`
   - Test: Comparison still works with integer-based types
   - Test: tests/utils/test_ordering.py all pass

3. **Phase 3:** Remove marker logic from JSON encoder/decoder
   - Test: JSON serialization round-trip works
   - Test: tests/serialization/test_json.py all pass

4. **Phase 4:** Remove marker logic from East printer/parser
   - Test: East text serialization round-trip works
   - Test: tests/serialization/test_east_parser.py all pass

5. **Phase 5:** Run all tests
   - Test: All 980+ tests pass
   - Test: tests/test_typescript_exports.py String test passes

---

## Detailed File Changes

### File 1: `east/types/type_system.py`

**Line 75-148:** Refactor `recursive_type()`

**Current code:** (shown above in Phase 1)

**New code:** (shown above in Phase 1)

**Lines to delete:**
- Line 48-58: `class RecursiveTypeMarker` (replace with simpler version)

**Lines to add:**
- Import statements: `from east.types.structural import Case` (if not already imported)
- New `replace_markers()` helper function (60 lines)

---

### File 2: `east/utils/ordering.py`

#### Change 1: Delete `_find_recursive_marker()`
**Lines 43-112:** DELETE entire function

#### Change 2: Update function signatures (remove marker_map parameter)
**Lines:**
- 348: `def equal_for(type_val, type_ctx=None, marker_map=None)` → `def equal_for(type_val, type_ctx=None)`
- 630: `def is_for(type_val, type_ctx=None, marker_map=None)` → `def is_for(type_val, type_ctx=None)`
- 801: `def compare_for(type_val, type_ctx=None, marker_map=None)` → `def compare_for(type_val, type_ctx=None)`
- 1134: `def less_for(type_val, type_ctx=None, marker_map=None)` → `def less_for(type_val, type_ctx=None)`
- 1153: `def not_equal_for(type_val, type_ctx=None, marker_map=None)` → `def not_equal_for(type_val, type_ctx=None)`
- 1172: `def less_equal_for(type_val, type_ctx=None, marker_map=None)` → `def less_equal_for(type_val, type_ctx=None)`
- 1191: `def greater_equal_for(type_val, type_ctx=None, marker_map=None)` → `def greater_equal_for(type_val, type_ctx=None)`
- 1210: `def greater_for(type_val, type_ctx=None, marker_map=None)` → `def greater_for(type_val, type_ctx=None)`

#### Change 3: Update function bodies to remove marker_map initialization
**Pattern to find and delete:**
```python
if type_ctx is None:
    type_ctx = []
if marker_map is None:
    marker_map = {}
```

**Replace with:**
```python
if type_ctx is None:
    type_ctx = []
```

**Affected lines:** Multiple locations in equal_for, is_for, compare_for

#### Change 4: Remove marker registration from Array/Set/Dict/Ref handlers
**Lines to change:**
- 407-410 (equal_for, Array)
- 443-447 (equal_for, Set)
- 459-463 (equal_for, Dict)
- 507-510 (equal_for, Ref)
- 922-926 (compare_for, Array)
- 933-937 (compare_for, Set)
- 1030-1034 (compare_for, Dict)
- 1068-1072 (compare_for, Ref)

**Pattern to DELETE:**
```python
# Register marker if this is a recursive type
marker = _find_recursive_marker(type_val)
if marker is not None:
    marker_map[id(marker)] = len(type_ctx) - 1
```

#### Change 5: Simplify Recursive type handling
**Lines to change:**
- 612-622 (equal_for, Recursive case)
- 760-770 (is_for, Recursive case)
- 1088-1098 (compare_for, Recursive case)

**Replace dual code path with single integer path (shown above in Phase 2)**

#### Change 6: Update all recursive function calls to remove marker_map
**Pattern to find:**
```python
some_func(type_val.value, type_ctx, marker_map)
```

**Replace with:**
```python
some_func(type_val.value, type_ctx)
```

**Affected lines:** Dozens of locations throughout the file

---

### File 3: `east/serialization/json.py`

#### Change 1: Update function signatures
**Lines:**
- ~170: `def to_json_for(type_val, type_ctx=None, marker_map=None)` → `def to_json_for(type_val, type_ctx=None)`
- ~690: `def from_json_for(type_val, frozen=False, type_ctx=None, marker_map=None, type_str="")` → `def from_json_for(type_val, frozen=False, type_ctx=None, type_str="")`
- ~1350: `def encode_json_for(type_val, type_ctx=None, marker_map=None)` → `def encode_json_for(type_val, type_ctx=None)`
- ~1450: `def decode_json_for(type_val, frozen=False, type_ctx=None, marker_map=None)` → `def decode_json_for(type_val, frozen=False, type_ctx=None)`

#### Change 2: Remove marker_map initialization
**Pattern to DELETE:**
```python
if marker_map is None:
    marker_map = {}
```

#### Change 3: Remove marker registration from type encoders
**Pattern to DELETE in Array/Set/Dict/Ref type encoders:**
```python
stack_index = len(type_ctx)
type_ctx.append(encode_array)
marker = _find_recursive_marker(type_val)
if marker is not None:
    marker_map[id(marker)] = stack_index
```

**Replace with:**
```python
type_ctx.append(encode_array)
```

**Affected lines:**
- ~375 (to_json_for, Array)
- ~395 (to_json_for, Set)
- ~465 (to_json_for, Dict)
- ~495 (to_json_for, Ref)
- ~375 (encode_json_for, Array)
- ~395 (encode_json_for, Set)
- ~465 (encode_json_for, Dict)
- ~495 (encode_json_for, Ref)

#### Change 4: Remove marker registration from value decoders
**Pattern to DELETE in Array/Set/Dict/Ref decoders:**
```python
stack_index = len(type_ctx)
type_ctx.append(decode_array)
marker = _find_recursive_marker(type_val)
if marker is not None:
    marker_map[id(marker)] = stack_index
```

**Replace with:**
```python
type_ctx.append(decode_array)
```

**Affected lines:**
- ~845 (from_json_for, Array)
- ~905 (from_json_for, Set)
- ~1001 (from_json_for, Dict)
- ~1082 (from_json_for, Ref)
- ~845 (decode_json_for, Array)
- ~905 (decode_json_for, Set)
- ~1001 (decode_json_for, Dict)
- ~1082 (decode_json_for, Ref)

#### Change 5: Simplify Recursive type encoding/decoding
**Lines ~543-573 (to_json_for, Recursive type):**
**Lines ~1171-1206 (from_json_for, Recursive type):**

**Replace dual code path with single integer path (similar to ordering.py pattern)**

#### Change 6: Update all recursive calls
**Pattern:** Remove `marker_map` argument from all function calls throughout the file

---

### File 4: `east/serialization/east_printer.py`

**Similar changes as json.py:**

#### Change 1: Update `print_type()` signature
**Remove marker_map parameter**

#### Change 2: Remove marker_map initialization

#### Change 3: Remove marker registration from Array/Set/Dict/Ref/Struct/Variant handlers
**Pattern to DELETE:**
```python
type_ctx.append(value_type)
marker = _find_recursive_marker(value_type)
if marker is not None and id(marker) not in marker_map:
    marker_map[id(marker)] = len(type_ctx) - 1
```

**Replace with:**
```python
type_ctx.append(value_type)
```

#### Change 4: Simplify Recursive type printing (lines 451-461)

#### Change 5: Update all recursive calls

---

### File 5: `east/serialization/east_parser.py`

**Similar changes as east_printer.py:**

#### Change 1: Update function signatures to remove marker_map

#### Change 2: Remove marker_map initialization

#### Change 3: Remove marker registration from composite type parsers

#### Change 4: Simplify Recursive type parsing (lines 459-466)

#### Change 5: Update all recursive calls

---

### File 6: `east/builtins/type_system.py`

**Update external calls to serialization functions:**

**Lines ~36:** Update `parse_type()` call (if it passes marker_map)
**Lines ~48:** Update `print_type()` call (if it passes marker_map)

**Pattern:** Remove `marker_map` argument from all calls

---

### Files 7-N: All other files using serialization/comparison

**Action:** Search entire codebase for:
- Calls to `equal_for`, `compare_for`, `is_for`, `less_for`, etc.
- Calls to `to_json_for`, `from_json_for`, `encode_json_for`, `decode_json_for`
- Calls to `print_type`, `parse_type`

**For each call:** Remove the `marker_map` argument

---

## Implementation Plan

### Phase 1: Core Refactor (1 day)
**Goal:** Make recursive_type() generate integer scope_ids

**Tasks:**
1. Refactor `recursive_type()` in `east/types/type_system.py`
   - Implement `replace_markers()` helper
   - Test that EastTypeType has integer scope_ids
   - Test that IRType has integer scope_ids

2. Add validation tests
   ```python
   def test_recursive_type_uses_integers():
       """Verify recursive_type() generates integer scope_ids."""
       ListType = recursive_type(
           lambda self: variant_type([
               ("nil", NullType),
               ("cons", struct_type([("head", IntegerType), ("tail", self)]))
           ])
       )
       # Find the "cons" case
       cons_case = [c for c in ListType.value if c.name == "cons"][0]
       cons_struct = cons_case.type
       tail_field = [f for f in cons_struct.value if f.name == "tail"][0]
       tail_type = tail_field.type

       # Verify it's Recursive with integer scope_id
       assert tail_type.tag == "Recursive"
       assert isinstance(tail_type.value, int)
       assert tail_type.value == 2  # 2 levels deep: Variant -> Struct -> Recursive
   ```

**After Phase 1:** All recursive types use integers internally

---

### Phase 2: Simplify Ordering (0.5 days)
**Goal:** Remove marker logic from comparison functions

**Tasks:**
1. Update `east/utils/ordering.py`:
   - Delete `_find_recursive_marker()`
   - Remove marker_map parameters
   - Remove marker_map initialization
   - Remove marker registration
   - Simplify Recursive type handling
   - Update all internal calls

2. Run tests: `pytest tests/utils/test_ordering.py -v`

3. Run tests: `pytest tests/types/ -v`

**After Phase 2:** Comparison functions only handle integers

---

### Phase 3: Simplify JSON Serialization (0.5 days)
**Goal:** Remove marker logic from JSON encoder/decoder

**Tasks:**
1. Update `east/serialization/json.py`:
   - Remove marker_map parameters
   - Remove marker registration
   - Simplify Recursive type handling
   - Update all internal calls

2. Run tests: `pytest tests/serialization/test_json.py -v`

**After Phase 3:** JSON serialization only handles integers

---

### Phase 4: Simplify East Text Serialization (0.5 days)
**Goal:** Remove marker logic from East printer/parser

**Tasks:**
1. Update `east/serialization/east_printer.py`:
   - Remove marker_map parameters
   - Remove marker registration
   - Simplify Recursive type handling
   - Update all internal calls

2. Update `east/serialization/east_parser.py`:
   - Same changes as printer

3. Run tests: `pytest tests/serialization/test_east_parser.py -v`

**After Phase 4:** East text serialization only handles integers

---

### Phase 5: Update All Callers (0.5 days)
**Goal:** Remove marker_map arguments from all external calls

**Tasks:**
1. Search entire codebase for function calls with marker_map
2. Remove marker_map argument from each call
3. Run full test suite: `pytest -v`

**After Phase 5:** No marker_map references remain

---

### Phase 6: Final Testing (0.5 days)
**Goal:** Verify TypeScript compatibility

**Tasks:**
1. Run TypeScript export tests: `pytest tests/test_typescript_exports.py -k String -v`
2. Verify String test passes (all 19 test cases)
3. Run all TypeScript export tests: `pytest tests/test_typescript_exports.py -v`
4. Run full test suite: `pytest -v` (should be 980+ passing)

**After Phase 6:** Full TypeScript compatibility achieved

---

## Testing Strategy

### Unit Tests for recursive_type()

**File:** `tests/types/test_type_system.py`

```python
def test_recursive_type_generates_integers():
    """Verify recursive_type() generates integer scope_ids, not markers."""
    from east.types.type_system import recursive_type, EastTypeType
    from east.types.primitives import IntegerType, NullType
    from east.types.structural import variant_type, struct_type

    # Create a simple recursive type: List<Integer>
    ListType = recursive_type(
        lambda self: variant_type([
            ("nil", NullType),
            ("cons", struct_type([("head", IntegerType), ("tail", self)]))
        ])
    )

    # Verify the structure uses integers
    assert ListType.tag == "Variant"

    # Find "cons" case
    cons_case = next(c for c in ListType.value if c.name == "cons")
    assert cons_case.type.tag == "Struct"

    # Find "tail" field
    tail_field = next(f for f in cons_case.type.value if f.name == "tail")
    assert tail_field.type.tag == "Recursive"

    # CRITICAL: Verify it's an integer, not a RecursiveTypeMarker
    assert isinstance(tail_field.type.value, int)
    # Should be 2 (depth: Variant=0 -> Struct=1 -> Recursive=2)
    assert tail_field.type.value == 2


def test_easttype_uses_integers():
    """Verify EastTypeType itself uses integer scope_ids."""
    from east.types.type_system import EastTypeType

    # Find "Array" case in EastTypeType
    array_case = next(c for c in EastTypeType.value if c.name == "Array")

    # The Array case should have type .Recursive <integer>
    assert array_case.type.tag == "Recursive"
    assert isinstance(array_case.type.value, int)
    assert array_case.type.value == 1  # 1 level deep


def test_nested_recursive_types():
    """Test deeply nested recursive structures."""
    from east.types.type_system import recursive_type
    from east.types.primitives import IntegerType
    from east.types.structural import variant_type, struct_type
    from east.types.containers import ArrayType, DictType

    # Tree with both left and right subtrees
    TreeType = recursive_type(
        lambda self: variant_type([
            ("leaf", IntegerType),
            ("node", struct_type([
                ("value", IntegerType),
                ("left", self),
                ("right", self)
            ]))
        ])
    )

    # Verify both self-references use the same scope_id
    node_case = next(c for c in TreeType.value if c.name == "node")
    left_field = next(f for f in node_case.type.value if f.name == "left")
    right_field = next(f for f in node_case.type.value if f.name == "right")

    assert left_field.type.tag == "Recursive"
    assert right_field.type.tag == "Recursive"
    assert isinstance(left_field.type.value, int)
    assert isinstance(right_field.type.value, int)
    # Both should be same depth
    assert left_field.type.value == right_field.type.value
```

### Integration Tests

**File:** `tests/test_typescript_exports.py`

**Already exists - this is the primary validation that we've achieved parity.**

**Target:** All String tests pass (19/19)

### Regression Tests

**Run full test suite:**
```bash
pytest -v
```

**Expected:** 980+ tests pass, no regressions

---

## Success Criteria

Implementation is complete when:

1. ✅ `recursive_type()` generates integer scope_ids (verified by unit tests)
2. ✅ EastTypeType uses integer scope_ids (verified by unit test)
3. ✅ IRType uses integer scope_ids (verified by unit test)
4. ✅ All `_find_recursive_marker()` calls removed from codebase
5. ✅ All `marker_map` parameters removed from function signatures
6. ✅ All `marker_map` arguments removed from function calls
7. ✅ Recursive type handling uses ONLY integer scope_id path (no dual paths)
8. ✅ All comparison tests pass (`tests/utils/test_ordering.py`)
9. ✅ All JSON serialization tests pass (`tests/serialization/test_json.py`)
10. ✅ All East text tests pass (`tests/serialization/test_east_parser.py`)
11. ✅ TypeScript export tests pass (`tests/test_typescript_exports.py -k String`)
12. ✅ All 980+ tests pass with no regressions (`pytest -v`)
13. ✅ Code is simpler - fewer lines, single code paths, no marker discovery

**Validation command:**
```bash
# Verify no marker_map remains
grep -r "marker_map" east/ | grep -v ".pyc" | grep -v "__pycache__"
# Should return only comments or the word in strings, no actual usage

# Verify no _find_recursive_marker remains
grep -r "_find_recursive_marker" east/ | grep -v ".pyc" | grep -v "__pycache__"
# Should return empty

# Run tests
pytest tests/test_typescript_exports.py -k String -v
# Should show 19/19 passing

pytest -v
# Should show 980+ passing, 0 failures
```

---

## Estimated Effort

**Total:** 3-4 days of focused work

- **Phase 1 (Core Refactor):** 1 day
  - Implement recursive_type() refactor (4 hours)
  - Write/run unit tests (2 hours)
  - Debug edge cases (2 hours)

- **Phase 2 (Simplify Ordering):** 0.5 days
  - Update ordering.py (2 hours)
  - Run tests and fix issues (2 hours)

- **Phase 3 (Simplify JSON):** 0.5 days
  - Update json.py (2 hours)
  - Run tests and fix issues (2 hours)

- **Phase 4 (Simplify East Text):** 0.5 days
  - Update printer/parser (2 hours)
  - Run tests and fix issues (2 hours)

- **Phase 5 (Update Callers):** 0.5 days
  - Search and update all calls (2 hours)
  - Run full test suite (2 hours)

- **Phase 6 (Final Testing):** 0.5 days
  - TypeScript export tests (1 hour)
  - Full regression testing (2 hours)
  - Documentation updates (1 hour)

**Buffer:** +1 day for unexpected issues

---

## Risks and Mitigation

### Risk 1: Breaking Changes in Serialization
**Risk:** Old JSON/East files with RecursiveTypeMarker references may not parse

**Mitigation:**
- Low risk - we don't persist RecursiveTypeMarker objects to JSON (they were already converted to integers)
- If needed, keep legacy parsing support temporarily
- All production data uses TypeScript exports, which already use integers

### Risk 2: Test Failures
**Risk:** Refactoring may break subtle edge cases

**Mitigation:**
- Comprehensive unit tests for recursive_type()
- Test after each phase
- Run full test suite frequently
- Git commits after each working phase

### Risk 3: Performance Regression
**Risk:** New implementation may be slower

**Mitigation:**
- Benchmark before/after
- New approach should be FASTER (simpler, no marker discovery)
- Monitor test suite run time

### Risk 4: Incomplete Migration
**Risk:** Missing some marker_map references

**Mitigation:**
- Use grep to search for all marker_map references
- Use grep to search for all _find_recursive_marker references
- Run tests after each change
- Final validation with grep (shown in Success Criteria)

---

## Benefits

**After completion:**

1. **TypeScript Compatibility** - Python matches TypeScript exactly
2. **Simpler Code** - No marker discovery, no dual code paths, fewer parameters
3. **Fewer Bugs** - Single implementation path means fewer edge cases
4. **Easier Maintenance** - Changes to TypeScript can be ported directly
5. **Better Tests** - TypeScript export tests validate compatibility
6. **Cleaner API** - Functions have fewer parameters
7. **Better Performance** - No marker discovery overhead
8. **Code Reduction** - Delete ~150 lines of marker-related code

---

## References

- **TypeScript Types:** ../East/src/types.ts (RecursiveType constructor)
- **TypeScript Comparison:** ../East/src/comparison.ts (isFor, equalFor, compareFor)
- **TypeScript Internal:** ../East/src/internal.ts (JSON encoder/decoder)
- **Python Type System:** east/types/type_system.py
- **Python Comparison:** east/utils/ordering.py
- **Python JSON:** east/serialization/json.py
- **Python East Text:** east/serialization/east_printer.py, east/serialization/east_parser.py

---

**Document Version:** 1.0
**Created:** 2025-11-13
**Status:** Ready for implementation
**Priority:** CRITICAL - Blocking TypeScript export tests, creating maintenance burden
**Estimated Completion:** 2025-11-17 (4 days)
