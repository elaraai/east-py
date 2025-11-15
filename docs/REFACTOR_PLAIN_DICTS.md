# Refactor: Use Plain Dicts for Types, Structs, and Variants

**Status:** IN PROGRESS
**Priority:** CRITICAL - Fixes type comparison bugs blocking TypeScript/Python parity
**Goal:** Match TypeScript's representation: plain objects for all values

## Summary

Replace custom Python classes (`EastType`, `EastStruct`, `EastVariant`, `_StructTypeClass`, `_VariantTypeClass`) with **plain dicts** and **TypedDict** type hints. This achieves complete TypeScript parity while maintaining Python type safety.

**Key Innovation:** Use `TypedDict` for static type checking without runtime overhead. At runtime, everything is a plain dict - just like TypeScript.

```python
# Before: Classes with complex hierarchy
IntegerType = EastType(make_case("Integer"))
struct = EastStruct(runtime_type, {"name": "Alice"})
variant = EastVariant(_east_type, Case("some", 42))

# After: Plain dicts with TypedDict hints
IntegerType: IntegerTypeDef = {"type": "Integer"}
struct = {"name": "Alice"}  # Plain dict
variant = {"type": "some", "value": 42}  # Plain dict
```

**Benefits:**
- ✅ Fixes type comparison bugs (no more class instance mismatches)
- ✅ Complete TypeScript parity (same representation)
- ✅ Type safety via TypedDict (mypy, IDE autocomplete)
- ✅ No runtime overhead (plain dicts)
- ✅ Natural JSON serialization
- ✅ Eliminates `east/types/structural.py` entirely (redundant)

---

## Problem Statement

The Python implementation created custom classes (`EastType`, `EastStruct`, `EastVariant`, `_StructTypeClass`, `_VariantTypeClass`) to represent types, struct values, and variant values. This caused several problems:

1. **Type comparison failures:** Types deserialized from JSON (plain `EastVariant` instances) don't compare equal to directly-created types (`EastType` instances) because `EastType.__eq__` had `isinstance(other, EastType)` check
2. **Recursive marker issues:** `_StructTypeClass` stores field types from schema (containing `.Recursive N` markers), but actual decoded values have concrete types, causing mismatches
3. **Architectural complexity:** `_StructTypeClass` and `_VariantTypeClass` are runtime type metadata holders that violate type annotations (`_east_type: EastType`)
4. **TypeScript divergence:** TypeScript uses plain objects everywhere - `{type: "Integer"}`, `{field: value}`, `{type: tag, value: val}` - no classes

### Example Failure

Blob tests: 20/46 passing

```
Type mismatch: expected .Dict (key=.String, value=.Integer), got .Dict (key=.String, value=.Integer)
```

Both types print identically but don't compare equal due to class instance mismatches.

---

## TypeScript Approach

### Types (lines 25-90 in types.ts)

```typescript
// Primitive types - plain objects
export type IntegerType = { type: "Integer" };
export const IntegerType: IntegerType = { type: "Integer" };

// Compound types - plain objects with value field
export type ArrayType<T = any> = { type: "Array", value: T };
export function ArrayType<T>(type: T): ArrayType<T> {
  return { type: "Array", value: type };
}

export type DictType<K = any, V = any> = {
  type: "Dict",
  value: { key: K, value: V }
};
```

### Structs (lines 838-842 in json.ts)

```typescript
// Struct values are just plain objects
if (frozen) {
    Object.freeze(obj); // is this overkill?
}
return obj;  // Returns {field1: val1, field2: val2, ...}
```

### Variants (lines 20-27 in variant.ts)

```typescript
export type variant<Type = string, Value = any> = {
    type: Type,      // The tag
    value: Value,    // The associated value
    [variant_symbol]: null,  // Brand symbol for nominal typing
};

export function variant<Type extends string, Value = null>(
    type: Type,
    value: Value = null as Value
): variant<Type, Value> {
    return { type, value, [variant_symbol]: null };
}
```

**Key insight:** Everything is a plain object. No classes, just dicts and functions.

---

## Python Solution

### Use TypedDict for Type Safety

Python has **TypedDict** which provides type hints for dict structures without runtime overhead:

```python
from typing import TypedDict, Literal, Union, Any

# Primitive types
class IntegerType(TypedDict):
    type: Literal["Integer"]

class ArrayType(TypedDict):
    type: Literal["Array"]
    value: "EastType"  # Element type

class StructFieldDef(TypedDict):
    name: str
    type: "EastType"

class StructType(TypedDict):
    type: Literal["Struct"]
    value: list[StructFieldDef]

class VariantCaseDef(TypedDict):
    name: str
    type: "EastType"

class VariantType(TypedDict):
    type: Literal["Variant"]
    value: list[VariantCaseDef]

# Union of all type variants
EastType = Union[IntegerType, ArrayType, StructType, VariantType, ...]

# Now you can use rich type hints with plain dicts!
def field_types(struct_type: StructType) -> list[EastType]:
    """Get field types from a Struct type."""
    return [field["type"] for field in struct_type["value"]]

# At runtime, just plain dicts
my_type: IntegerType = {"type": "Integer"}
```

**Key benefits:**
- Static type checking (mypy, IDE autocomplete)
- Runtime: plain dicts (no classes, no overhead)
- Matches TypeScript exactly
- Natural JSON serialization

### Representation Changes

| Current Python | TypeScript | New Python (TypedDict) |
|----------------|------------|------------------------|
| `EastType(make_case("Integer"))` | `{type: "Integer"}` | `IntegerType = {"type": "Integer"}` |
| `ArrayType(element_type)` creates `EastType(Case("Array", element_type))` | `{type: "Array", value: element_type}` | `ArrayType = {"type": "Array", "value": element_type}` |
| `EastStruct(runtime_type, {"name": "Alice"})` with `_east_type` field | `{name: "Alice", age: 30}` | `{"name": "Alice", "age": 30}` (plain dict) |
| `EastVariant(_east_type, Case("some", 42))` | `{type: "some", value: 42}` | `{"type": "some", "value": 42}` (plain dict) |
| `Case("some", 42)` | N/A (not needed) | Eliminated (use dicts directly) |

### Access Pattern Changes

| Current (Dot notation) | New (Dict notation) |
|------------------------|---------------------|
| `struct.name` | `struct["name"]` |
| `variant.tag` | `variant["type"]` |
| `variant.value` | `variant["value"]` |
| `type_val.tag` | `type_val["type"]` |
| `type_val.value` | `type_val["value"]` |

### Type System Functions

Since types are plain dicts, we use helper functions (not methods). With TypedDict, these can have proper type hints:

```python
# Helper functions for working with types
def field_names(struct_type: StructType) -> list[str]:
    """Get field names from a Struct type."""
    return [field["name"] for field in struct_type["value"]]

def field_types(struct_type: StructType) -> list[EastType]:
    """Get field types from a Struct type."""
    return [field["type"] for field in struct_type["value"]]

def case_names(variant_type: VariantType) -> list[str]:
    """Get case names from a Variant type."""
    return [case["name"] for case in variant_type["value"]]

def case_types(variant_type: VariantType) -> list[EastType]:
    """Get case types from a Variant type."""
    return [case["type"] for case in variant_type["value"]]

# Type checking helpers
def is_struct_type(typ: EastType) -> bool:
    """Check if a type is a Struct type."""
    return typ["type"] == "Struct"

def is_variant_type(typ: EastType) -> bool:
    """Check if a type is a Variant type."""
    return typ["type"] == "Variant"

# Value checking helpers
def is_struct_value(value: Any) -> bool:
    """Check if a value is a struct (plain dict without 'type' key)."""
    return isinstance(value, dict) and "type" not in value

def is_variant_value(value: Any) -> bool:
    """Check if a value is a variant (dict with 'type' and 'value' keys)."""
    return (isinstance(value, dict) and
            "type" in value and
            "value" in value and
            len(value) == 2)
```

**Note:** With TypedDict, mypy will enforce types statically, so runtime checks are often unnecessary.

---

## Implementation Plan

### Phase 1: Core Type System ✅ (PARTIALLY DONE)

**File:** `east/types/type_system.py`

- [x] Remove `_StructTypeClass` class
- [x] Remove `_VariantTypeClass` class
- [x] Remove `_create_struct` helper
- [ ] **Add TypedDict definitions for all type variants**
- [ ] **Convert EastType from class to Union[IntegerType, ArrayType, ...]**
- [ ] Rewrite type constructors to return plain dicts
- [ ] Add helper functions (field_names, case_names, etc.)

**File:** `east/types/structural.py` ⚠️ **REDUNDANT - WILL BE DELETED**

With plain dicts and TypedDict, this entire file becomes redundant:

- [x] Remove `EastStruct` class (already done)
- [x] Remove from `__all__` (already done)
- [ ] **Remove `Case` class** - replaced by plain dicts: `{"type": tag, "value": val}`
- [ ] **Remove `EastVariant` class** - variants are plain dicts
- [ ] **Remove `make_case` helper** - just create dicts directly: `{"type": tag, "value": val}`
- [ ] **Delete entire file after moving any TypedDict definitions to type_system.py**

**Rationale:** TypeScript doesn't have a `structural.ts` file. Variants and structs are just plain objects. Python should match this exactly using TypedDict for type safety.

### Phase 2: Serialization (CRITICAL PATH)

**File:** `east/serialization/json.py`

Current code (lines 1146-1151):
```python
# Build runtime _StructTypeClass and create EastStruct instance
from east.types.type_system import _StructTypeClass

fields = [(field.name, field.type) for field in type_val.value]
runtime_type = _StructTypeClass(tuple(fields))
return runtime_type.create(**obj)
```

New code:
```python
# Struct values are plain dicts (matching TypeScript)
return obj  # That's it!
```

**File:** `east/serialization/beast.py`

- [ ] Update type comparison (already partially done - removed debug output)
- [ ] Ensure deserialized types are plain dicts
- [ ] Update struct/variant decoding to use dicts

**File:** `east/serialization/east_printer.py`

- [ ] Update printing for dict-based types
- [ ] Update printing for dict-based struct values
- [ ] Update printing for dict-based variant values

**File:** `east/serialization/east_parser.py`

- [ ] Parse types as plain dicts
- [ ] Parse struct values as plain dicts
- [ ] Parse variant values as plain dicts

### Phase 3: Runtime Compiler

**File:** `east/runtime/compiler.py`

- [ ] Update Struct compilation to create plain dicts
- [ ] Update Variant compilation to create dicts: `{"type": tag, "value": val}`
- [ ] Update GetField to use dict access: `struct[field_name]`
- [ ] Update Match to use dict access: `variant["type"]`, `variant["value"]`

### Phase 4: Builtins (MASSIVE CHANGE)

All 212+ builtins need updating to:
- Accept plain dicts for struct values: change `struct.field` → `struct["field"]`
- Accept plain dicts for variant values: change `variant.tag` → `variant["type"]`
- Accept plain dicts for types: change `type_val.tag` → `type_val["type"]`

**Files to update:**
- `east/builtins/array.py`
- `east/builtins/blob.py`
- `east/builtins/datetime_ops.py`
- `east/builtins/dict_ops.py`
- `east/builtins/float_ops.py`
- `east/builtins/integer.py`
- `east/builtins/ref_ops.py`
- `east/builtins/set_ops.py`
- `east/builtins/string.py`
- `east/builtins/struct_ops.py`
- `east/builtins/variant_ops.py`
- Many more...

### Phase 5: IR System

**File:** `east/ir/builders.py`

- [ ] Update IR builders to work with dict-based types
- [ ] Ensure type parameters are plain dicts

**File:** `east/ir/analyze.py`

- [ ] Update type checking to work with dict-based types
- [ ] Update recursive type handling

### Phase 6: Type Utilities

**File:** `east/types/containers.py`

- [ ] Update EastArray to store `element_type` as plain dict
- [ ] Update EastSet to store `element_type` as plain dict
- [ ] Update EastDict to store `key_type`, `value_type` as plain dicts

**File:** `east/types/ref.py`

- [ ] Update EastRef to store `value_type` as plain dict

### Phase 7: Tests

- [ ] Update all tests to use dict notation
- [ ] Fix tests that create types/structs/variants
- [ ] Verify 980 tests still pass
- [ ] Get Blob tests passing (currently 20/46)
- [ ] Test remaining TypeScript exports (Array, Dict, Set, etc.)

---

## Bootstrap Problem - SOLVED with TypedDict

**Challenge:** To create type system, need to represent TypeType (the type-of-types). But if types are dicts, how do we bootstrap?

**TypeScript approach:** Uses plain objects from the start. Types are just data.

**Python solution with TypedDict:**

```python
from typing import TypedDict, Literal, Union, Any

# Define type structure with TypedDict
class IntegerTypeDef(TypedDict):
    type: Literal["Integer"]

class ArrayTypeDef(TypedDict):
    type: Literal["Array"]
    value: "EastType"

# ... more type definitions ...

# EastType is a union of all type variants
EastType = Union[IntegerTypeDef, ArrayTypeDef, ...]

# Type constructors return plain dicts
IntegerType: IntegerTypeDef = {"type": "Integer"}

def ArrayType(element_type: EastType) -> ArrayTypeDef:
    return {"type": "Array", "value": element_type}

# TypeType itself is just another variant
class TypeTypeDef(TypedDict):
    type: Literal["Type"]

TypeType: TypeTypeDef = {"type": "Type"}
```

**Key insight:** TypedDict is only for static type checking. At runtime, everything is a plain dict. No bootstrap problem because there are no classes to instantiate!

**Benefits:**
- Complete TypeScript parity
- Simplest representation
- No classes, no bootstrap cycle
- Full static type checking
- Natural JSON serialization

**Decision:** Use TypedDict approach - get both type safety AND plain dicts!

---

## Key Challenges

### 1. Field Access Throughout Codebase

**Scope:** Hundreds of locations need updating

Use find/replace patterns:
- `\.tag\b` → `["type"]`
- `\.value\b` → `["value"]`
- `struct\.(\w+)` → `struct["$1"]` (careful with method calls!)

### 2. Type Equality

Current:
```python
if type_val == IntegerType:
    ...
```

After (works the same):
```python
if type_val == IntegerType:  # Dict equality works!
    ...
```

Dicts compare by value, so this should work automatically.

### 3. Type Checking

Current:
```python
if isinstance(value, EastStruct):
    ...
```

After:
```python
if isinstance(value, dict) and all(k not in value for k in ["type", "_case"]):
    # It's a struct (plain dict without special keys)
    ...
```

Or use helper:
```python
def is_struct_value(value: Any) -> bool:
    """Check if value is a struct (plain dict)."""
    return isinstance(value, dict) and "type" not in value

def is_variant_value(value: Any) -> bool:
    """Check if value is a variant."""
    return isinstance(value, dict) and "type" in value and "value" in value
```

### 4. Immutability

TypeScript uses `Object.freeze()`. Python doesn't have built-in frozen dicts (besides `types.MappingProxyType`).

**Options:**
- Use regular dicts (trust immutability by convention)
- Use `types.MappingProxyType` (read-only view)
- Create custom `frozendict` class

**Decision:** Use regular dicts for now (simplicity). If immutability becomes an issue, revisit.

---

## Migration Strategy

### Step 1: Create New Type System Alongside Old

Create `east/types/type_system_v2.py` with dict-based types. Keep old system working.

### Step 2: Update Serialization First

Update JSON/Beast to use new representation. This is critical path.

### Step 3: Update Compiler

Get compiled code working with dict-based values.

### Step 4: Migrate Builtins Incrementally

Update builtins one file at a time, testing after each.

### Step 5: Replace Old Type System

Once everything works, delete old classes and rename v2 → type_system.py.

---

## Testing Strategy

### Unit Tests

After each phase, run:
```bash
make test
```

Track test count (should stay at 980).

### TypeScript Export Tests

After serialization + compiler phases:
```bash
uv run pytest tests/test_typescript_exports.py -v
```

**Current status:**
- String: 19/19 ✅
- DateTime: 11/11 ✅
- Ref: passing ✅
- Blob: 20/46 (BLOCKED)
- Others: Not yet tested

**Goal:** All tests passing

### Regression Tests

```bash
make check  # lint + typecheck + test
```

Must pass before merging.

---

## Success Criteria

Refactoring is complete when:

1. ✅ All classes removed (`EastType`, `EastStruct`, `_StructTypeClass`, `_VariantTypeClass`)
2. ✅ Types are plain dicts: `{"type": "Integer"}`
3. ✅ Struct values are plain dicts: `{"name": "Alice"}`
4. ✅ Variant values are plain dicts: `{"type": "some", "value": 42}`
5. ✅ All field access uses dict notation: `obj["field"]`
6. ✅ JSON serialization matches TypeScript output
7. ✅ Type comparison works correctly (no more mismatch errors)
8. ✅ All 980 tests pass
9. ✅ Blob tests: 46/46 passing
10. ✅ All TypeScript export tests pass

---

## Estimated Effort

**Total:** 2-3 days of focused work

- **Phase 1 (Core):** 4 hours
  - Remove classes
  - Create helper functions
  - Update type constructors

- **Phase 2 (Serialization):** 6 hours
  - JSON encoder/decoder
  - Beast encoder/decoder
  - East printer/parser

- **Phase 3 (Compiler):** 3 hours
  - Update IR compilation
  - Test with simple programs

- **Phase 4 (Builtins):** 8 hours
  - Update all 212+ builtins
  - Test each module

- **Phase 5 (IR/Containers):** 3 hours
  - Update IR system
  - Update container types

- **Phase 6 (Tests):** 4 hours
  - Fix failing tests
  - Verify all 980 pass
  - Get TypeScript exports working

**Testing:** Continuous throughout

---

## Current Status

### Completed ✅

- Removed `EastStruct` class from structural.py
- Removed `_StructTypeClass` class from type_system.py
- Removed `_VariantTypeClass` class from type_system.py
- Removed `_create_struct` helper
- Updated `EastType.__eq__` to accept EastVariant

### In Progress 🚧

- Understanding TypeScript representation
- Planning full dict conversion
- This design document

### Blocked ⏸️

- Blob tests (20/46 passing) - waiting for refactor
- Other TypeScript export tests - waiting for refactor

### Not Started ❌

- Convert EastType to plain dicts
- Update serialization
- Update compiler
- Update builtins (massive)
- Update IR system
- Update tests

---

## References

- **TypeScript Types:** `../East/src/types.ts`
- **TypeScript Variant:** `../East/src/containers/variant.ts`
- **TypeScript JSON:** `../East/src/serialization/json.ts`
- **Python Type System:** `east/types/type_system.py`
- **Python Serialization:** `east/serialization/json.py`
- **Test Failures:** `tests/test_typescript_exports.py::test_typescript_exported_ir[Blob]`

---

## Notes

### Why This Matters

This refactoring is **critical** for TypeScript/Python parity. Without it:
- Type comparisons fail mysteriously
- JSON deserialization creates wrong type instances
- Cross-language compatibility is broken
- Tests can't verify consistency between implementations

### Lessons Learned

1. **Don't add classes unless TypeScript has them** - Python implementation added unnecessary abstractions
2. **TypedDict is the key to matching TypeScript** - Get type safety without runtime overhead
3. **Homoiconicity is about data, not classes** - Types are values, not metaclasses
4. **Plain dicts are powerful** - No need for custom containers
5. **Match the reference implementation** - TypeScript is the source of truth
6. **Python can match TypeScript exactly** - Using TypedDict, we get identical representations with type safety

---

**Document Version:** 1.0
**Created:** 2025-11-13
**Status:** Design phase - ready to implement
**Author:** Claude Code (with user guidance)
**Priority:** CRITICAL - Blocks TypeScript/Python parity
