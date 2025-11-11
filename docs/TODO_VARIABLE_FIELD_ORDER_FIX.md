# TODO: Fix VariableIR field order

**Reference Commit:** 9abd135547383a772ae407db9575089401732531 in ../East
**Base Commit:** 302d2249503af9efb9d46cd89e75ab5d9d3e7fff in ../East
**Feature:** Fix field order in VariableIR struct type for consistency with TypeScript

## Current Status

**❌ NEEDS FIX:**
- ❌ VariableIR field order in Python doesn't match TypeScript
- ❌ Current order: type, name, location, mutable, captured
- ✅ Correct order: type, location, name, mutable, captured

## Overview

The East TypeScript implementation fixed the field order in VariableType to be more consistent (commit 9abd135). The fields `location` and `name` were swapped so that `location` comes before `name`.

**Why this matters:**
- **Serialization compatibility**: Field order affects binary serialization (Beast/Beast2)
- **Consistency**: Struct field order should be consistent across implementations
- **Convention**: Location typically comes early in IR node definitions

**TypeScript change:**
```typescript
export const VariableType = StructType({
  type: EastTypeType,
  location: LocationType,  // Moved up (was after name)
  name: StringType,        // Moved down (was before location)
  mutable: BooleanType,
  captured: BooleanType,
});
```

**Python current (incorrect) order:**
```python
VariableIR = StructType(
    [
        ("type", EastTypeType),
        ("name", StringType),        # Should be after location
        ("location", LocationType),  # Should be before name
        ("mutable", BooleanType),
        ("captured", BooleanType),
    ]
)
```

**Python corrected order:**
```python
VariableIR = StructType(
    [
        ("type", EastTypeType),
        ("location", LocationType),  # Move before name
        ("name", StringType),        # Move after location
        ("mutable", BooleanType),
        ("captured", BooleanType),
    ]
)
```

---

## Quick Summary

**What needs to change:**
- Swap the order of `name` and `location` fields in VariableIR struct definition

**Impact:**
- **Serialization**: May affect Beast/Beast2 serialization format (need to verify compatibility)
- **Code**: All code using VariableIR should continue to work (field access by name, not position)
- **Tests**: Existing tests should pass without changes

**Risk:**
- Low: Field access is by name, not positional
- Concern: Serialization format compatibility (if field order is significant)

---

## 1. Type System Changes

### File: `east/types/type_system.py`

**Status:** 🔧 **NEEDS UPDATE**

#### Task 1.1: Fix VariableIR field order

**Current code** (lines 1731-1739):
```python
VariableIR = StructType(
    [
        ("type", EastTypeType),
        ("name", StringType),
        ("location", LocationType),
        ("mutable", BooleanType),
        ("captured", BooleanType),
    ]
)
```

**Corrected code:**
```python
VariableIR = StructType(
    [
        ("type", EastTypeType),
        ("location", LocationType),  # Swapped with name
        ("name", StringType),        # Swapped with location
        ("mutable", BooleanType),
        ("captured", BooleanType),
    ]
)
```

**Notes:**
- This is a simple two-line swap
- Field order matters for serialization consistency
- Python struct field access is by name, so existing code should work

---

## 2. Testing

### File: Tests (various)

**Status:** ✅ **NO CHANGES NEEDED (probably)**

Since Python accesses struct fields by name (not position), existing tests should continue to work without modification.

**Verification steps:**
1. Run full test suite: `make test`
2. Verify no test failures
3. Check serialization tests specifically

**If tests fail:**
- Look for positional access to VariableIR fields (unlikely)
- Check serialization format expectations
- Update test expectations if needed

---

## 3. Serialization Compatibility

### Files: `east/serialization/*.py`

**Status:** ⚠️ **VERIFY COMPATIBILITY**

Need to verify if Beast/Beast2 serialization is affected by field order changes.

**Questions to answer:**
1. Does Beast2 serialize struct fields in declaration order?
2. Will this change break deserialization of existing Beast2 data?
3. Do we need to update serialization version or format?

**Investigation:**
- Check if Beast2 format includes field names (order-independent)
- Or if it uses positional encoding (order-dependent)
- Run serialization round-trip tests

**If order-dependent:**
- May need to bump serialization format version
- Or maintain backward compatibility logic

**If order-independent:**
- No changes needed, tests should pass

---

## 4. Implementation Checklist

**Phase 1: Code Change**
- [ ] Update VariableIR field order in `east/types/type_system.py`

**Phase 2: Testing**
- [ ] Run full test suite: `make test`
- [ ] Verify all tests pass
- [ ] Check serialization tests specifically
- [ ] Run round-trip serialization tests

**Phase 3: Verification**
- [ ] Verify Beast2 serialization still works
- [ ] Check if format version needs updating
- [ ] Confirm compatibility with TypeScript implementation

**Phase 4: Documentation**
- [ ] Update TODO_VARIABLE_FIELD_ORDER_FIX.md with findings
- [ ] Note any serialization compatibility considerations

---

## 5. Impact Assessment

**Code Impact:**
- Minimal: One 2-line change in type_system.py
- No code changes needed elsewhere (field access by name)

**Serialization Impact:**
- Unknown: Need to verify Beast2 format handling
- Potential: May require format version bump or compatibility layer

**Testing Impact:**
- Expected: Tests should pass without changes
- Possible: May need to update serialization test expectations

**Risk:**
- Low for code changes
- Medium for serialization compatibility
- Mitigation: Test thoroughly before committing

---

## 6. Timeline Estimate

**Total effort: 30 minutes - 2 hours**

- Code change: 5 minutes
- Testing: 10-30 minutes
- Serialization investigation: 15 minutes - 1 hour
- Documentation: 10 minutes

**Recommended approach:**
1. Make the field order change
2. Run tests and see what breaks (if anything)
3. Investigate serialization if needed
4. Document findings and commit

---

## 7. References

- **East TypeScript Commit:** 9abd135547383a772ae407db9575089401732531
- **Commit Message:** "Fix variable field order (#26)"
- **Files Changed:** src/ir.ts (1 insertion, 1 deletion)
- **Change:** Swapped `name` and `location` field order in VariableType
- **Python Location:** east/types/type_system.py:1731-1739

---

**Document Version:** 1.0
**Created:** 2025-11-11
**Status:** Ready for implementation (simple field order fix)
