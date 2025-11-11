# TODO: Clean up finally_body handling in TryCatch

**Reference Commit:** 302d2249503af9efb9d46cd89e75ab5d9d3e7fff in ../East
**Base Commit:** 3c3b001db79e7f4487eb57fb6ce1c7ffe5e0145a in ../East
**Python Implementation:** Commit 023b02c (finally block support complete)
**Feature:** Optimize and clean up finally block handling in try-catch-finally

## Current Status

**✅ ALREADY IMPLEMENTED:**
- ✅ TryCatchIR has `finally_body` field in `east/types/type_system.py:1719`
- ✅ Compiler handles finally blocks in `east/runtime/compiler.py:625-722`
- ✅ Finally blocks work in compiled East programs
- ✅ Both sync and async finally execution

**❌ OPTIMIZATION NEEDED (based on ../East TypeScript changes):**
- ❌ AST-to-IR conversion doesn't generate dummy Value node when no finally block
- ❌ Compiler doesn't optimize away effect-free finally blocks (Value nodes)
- ❌ Compiler uses runtime checks (`if finally_body_fn is not None`) instead of compile-time branching
- ❌ Code duplication in async vs sync paths for finally handling

## Overview

The East TypeScript implementation cleaned up finally block handling to make it more efficient and elegant. The key insight is that when there's no finally block in the source code, we should generate a dummy `Value` IR node with `NullType` instead of leaving it optional/undefined. This allows:

1. **Simpler type system**: `finally_body` is always required (not optional)
2. **Compile-time optimization**: Skip generating finally code if it's just a Value node (effect-free)
3. **Cleaner runtime**: No runtime checks for finally presence; split into two code paths at compile time

**TypeScript changes:**
- IR: Make `finally_body: any` (required, not `finally_body?: any` optional)
- AST-to-IR: Generate dummy Value node when no finally block present
- Compiler: Check `ir.value.finally_body.type === "Value"` to skip compilation
- Compiler: Split into separate functions with/without finally to avoid runtime checks

**Python current approach:**
- IR: `finally_body` is already required (non-optional)
- AST-to-IR: Python has no AST layer (works directly with IR)
- Compiler: Uses `hasattr` + `is not None` + `not isinstance(..., Null)` checks
- Compiler: Uses runtime `if finally_body_fn is not None` checks in both sync/async paths

**What needs to change:**
- IR builders should create dummy Value nodes when no finally needed
- Compiler should detect Value nodes and skip compilation
- Compiler should split into 4 variants (sync/async × with/without finally) for efficiency

---

## Quick Summary

**What works:**
- ✅ Finally blocks execute correctly in both sync and async contexts
- ✅ TryCatchIR structure is sound

**What's inefficient:**
- ❌ Runtime checks for finally presence (should be compile-time)
- ❌ No optimization for trivial finally blocks
- ❌ Code duplication in sync/async paths

**Impact:**
This is a **performance optimization and code cleanup**, not a bug fix. The current implementation is correct but generates slightly less efficient code with runtime checks.

---

## 1. IR System - No Changes Needed ✅

### File: `east/types/type_system.py`

**Status:** ✅ **ALREADY CORRECT**

The Python implementation already has `finally_body` as a required field (line 1719):

```python
_TryCatchIR = _ir_struct_type(
    lambda ir: [
        ("type", EastTypeType),
        ("location", LocationType),
        ("try_body", ir),
        ("catch_body", ir),
        ("message", ir),
        ("stack", ir),
        ("finally_body", ir),  # Already required, not optional
    ]
)
```

This matches the TypeScript change from `finally_body?: any` to `finally_body: any`.

---

## 2. IR Builder Changes

### File: `east/ir/builders.py`

**Status:** 🔧 **NEEDS UPDATE**

Currently there's no `ir_try_catch` builder function. Need to add one that handles the finally_body requirement.

#### Task 2.1: Create ir_try_catch builder

**Implementation:**

```python
def ir_try_catch(
    typ: EastType,
    loc: EastStruct,
    try_body: EastVariant,
    catch_body: EastVariant,
    message: EastVariant,
    stack: EastVariant,
    finally_body: EastVariant | None = None,
) -> EastVariant:
    """Create a TryCatch IR node with optional finally block.

    Args:
        typ: Result type of the try-catch expression
        loc: Location
        try_body: IR for try block
        catch_body: IR for catch block
        message: Variable IR for error message
        stack: Variable IR for stack trace
        finally_body: Optional IR for finally block (if None, creates dummy Value node)

    Returns:
        TryCatch IR variant

    Example:
        >>> loc = location("test.east", 1, 1)
        >>> try_ir = ir_value(IntegerType, loc, 1)
        >>> catch_ir = ir_value(IntegerType, loc, 0)
        >>> msg_var = ir_variable(StringType, "_msg", loc, False, False)
        >>> stack_var = ir_variable(ArrayType(LocationType), "_stack", loc, False, False)
        >>> tc_ir = ir_try_catch(IntegerType, loc, try_ir, catch_ir, msg_var, stack_var)
    """
    # If no finally_body provided, create a dummy Value node (Null)
    if finally_body is None:
        from east.types.type_system import NullType
        from east.types.primitives import Null

        finally_body = ir_value(NullType, loc, Null())

    # Get TryCatch struct type from IRType
    trycatch_type = None
    for case in IRType.value:
        if case.name == "TryCatch":
            trycatch_type = case.type
            break

    if trycatch_type is None:
        raise ValueError("TryCatch case not found in IRType")

    # Create struct
    trycatch_class = _struct_class_from_type(trycatch_type)
    trycatch_struct = trycatch_class.create(
        type=typ,
        location=loc,
        try_body=try_body,
        catch_body=catch_body,
        message=message,
        stack=stack,
        finally_body=finally_body,
    )

    return EastVariant(IRType, Case("TryCatch", trycatch_struct))
```

**Notes:**
- When `finally_body` is None, automatically create a dummy Value node with Null
- This matches TypeScript's approach of always having a finally_body in IR
- Simplifies compiler logic

---

## 3. Compiler Optimization

### File: `east/runtime/compiler.py`

**Status:** 🔧 **NEEDS OPTIMIZATION**

Current implementation (lines 625-722):
- Uses `hasattr`, `is not None`, and `not isinstance(..., Null)` checks to detect finally presence
- Uses runtime `if finally_body_fn is not None` checks in both sync and async code paths
- Single async and single sync function with conditional finally execution

#### Task 3.1: Optimize finally_body detection

**Current approach** (lines 650-663):
```python
# Compile finally body if present (not null)
from east.types.primitives import Null

finally_body_fn = None
finally_is_async = False
if (
    hasattr(trycatch_struct, "finally_body")
    and trycatch_struct.finally_body is not None
    and not isinstance(trycatch_struct.finally_body, Null)
):
    finally_body_fn = _compile_ir(
        trycatch_struct.finally_body, platform_fns, async_platform_fns, is_async_map
    )
    finally_is_async = is_async_map.get(id(trycatch_struct.finally_body), False)
```

**Optimized approach** (TypeScript-style):
```python
# Don't compile finally_body if it's just a Value node (effect-free)
finally_body_fn = None
finally_is_async = False

# Check if finally_body is a non-trivial operation
# Value nodes are effect-free and can be optimized away
if trycatch_struct.finally_body.tag != "Value":
    finally_body_fn = _compile_ir(
        trycatch_struct.finally_body, platform_fns, async_platform_fns, is_async_map
    )
    finally_is_async = is_async_map.get(id(trycatch_struct.finally_body), False)
```

**Benefits:**
- Simpler: Just check the IR node type
- Cleaner: No isinstance checks or multiple conditions
- Faster: Compile-time optimization instead of runtime checks

#### Task 3.2: Split into 4 compilation variants

**Current approach**: Two functions (async/sync) with runtime finally checks

**Optimized approach**: Four functions (async/sync × with/without finally)

```python
def _compile_trycatch(
    node: EastVariant,
    platform_fns: dict[str, Callable[..., Any]],
    async_platform_fns: set[str],
    is_async_map: dict[int, bool],
) -> Callable:
    """Compile a TryCatch IR node (try-catch-finally error handling)."""
    trycatch_struct = node.value

    # Compile try body
    try_body_fn = _compile_ir(
        trycatch_struct.try_body, platform_fns, async_platform_fns, is_async_map
    )

    # Extract message and stack variable names
    message_var = trycatch_struct.message.value
    message_name = message_var.name
    stack_var = trycatch_struct.stack.value
    stack_name = stack_var.name

    # Compile catch body
    catch_body_fn = _compile_ir(
        trycatch_struct.catch_body, platform_fns, async_platform_fns, is_async_map
    )

    # Don't compile finally_body if it's just a Value node (effect-free)
    finally_body_fn = None
    finally_is_async = False

    if trycatch_struct.finally_body.tag != "Value":
        finally_body_fn = _compile_ir(
            trycatch_struct.finally_body, platform_fns, async_platform_fns, is_async_map
        )
        finally_is_async = is_async_map.get(id(trycatch_struct.finally_body), False)

    # Check if any component is async
    try_is_async = is_async_map.get(id(trycatch_struct.try_body), False)
    catch_is_async = is_async_map.get(id(trycatch_struct.catch_body), False)
    is_async = try_is_async or catch_is_async or finally_is_async

    # Split into 4 variants for optimal code generation
    if is_async:
        if finally_body_fn is None:
            # Async without finally
            async def execute_trycatch_async(env):
                try:
                    if try_is_async:
                        return await try_body_fn(env)
                    return try_body_fn(env)
                except Exception as e:
                    catch_env = {**env}
                    catch_env[message_name] = str(e)
                    catch_env[stack_name] = _extract_stack_trace(e)

                    if catch_is_async:
                        return await catch_body_fn(catch_env)
                    return catch_body_fn(catch_env)

            return execute_trycatch_async

        # Async with finally
        async def execute_trycatch_async_finally(env):
            try:
                if try_is_async:
                    result = await try_body_fn(env)
                else:
                    result = try_body_fn(env)
            except Exception as e:
                catch_env = {**env}
                catch_env[message_name] = str(e)
                catch_env[stack_name] = _extract_stack_trace(e)

                if catch_is_async:
                    result = await catch_body_fn(catch_env)
                else:
                    result = catch_body_fn(catch_env)
            finally:
                if finally_is_async:
                    await finally_body_fn(env)
                else:
                    finally_body_fn(env)

            return result

        return execute_trycatch_async_finally

    # Sync variants
    if finally_body_fn is None:
        # Sync without finally
        def execute_trycatch_sync(env):
            try:
                return try_body_fn(env)
            except Exception as e:
                catch_env = {**env}
                catch_env[message_name] = str(e)
                catch_env[stack_name] = _extract_stack_trace(e)
                return catch_body_fn(catch_env)

        return execute_trycatch_sync

    # Sync with finally
    def execute_trycatch_sync_finally(env):
        try:
            result = try_body_fn(env)
        except Exception as e:
            catch_env = {**env}
            catch_env[message_name] = str(e)
            catch_env[stack_name] = _extract_stack_trace(e)
            result = catch_body_fn(catch_env)
        finally:
            finally_body_fn(env)

        return result

    return execute_trycatch_sync_finally
```

**Benefits:**
- No runtime checks for finally presence
- Slightly faster execution (no conditional branches)
- Cleaner code structure
- Matches TypeScript implementation

---

## 4. Testing

### File: `tests/runtime/test_compiler.py`

**Status:** ✅ **Tests already exist**

The existing tests in `test_compiler.py` already cover try-catch-finally:
- Basic try-catch
- Try-catch-finally with all blocks
- Error handling with message and stack

**Optional enhancement**: Add a test specifically for try-catch WITHOUT finally to verify the optimization works:

```python
def test_compile_try_catch_no_finally(self):
    """Test compiling try-catch without finally block (should optimize away dummy Value node)."""
    from east.ir.builders import ir_try_catch, ir_value, ir_variable, location
    from east.types.type_system import IntegerType, StringType, ArrayType, LocationType

    loc = location("test.east", 1, 1)

    # Try body: 42
    try_body = ir_value(IntegerType, loc, 42)

    # Catch body: 0
    catch_body = ir_value(IntegerType, loc, 0)

    # Message and stack variables
    msg_var = ir_variable(StringType, "_msg", loc, False, False)
    stack_var = ir_variable(ArrayType(LocationType), "_stack", loc, False, False)

    # Create try-catch WITHOUT finally (should create dummy Value node internally)
    tc_ir = ir_try_catch(IntegerType, loc, try_body, catch_body, msg_var, stack_var)

    # Compile
    compiled = compile(tc_ir, [])

    # Execute - should return try body result
    result = compiled({})
    assert result == 42
```

---

## 5. Implementation Checklist

**Phase 1: IR Builder** (Optional but recommended)
- [ ] Add `ir_try_catch` builder in `east/ir/builders.py`
- [ ] Test builder generates dummy Value node when finally_body is None

**Phase 2: Compiler Optimization** (Main work)
- [ ] Update finally_body detection to check `.tag != "Value"` instead of complex conditions
- [ ] Split async path into with/without finally variants
- [ ] Split sync path into with/without finally variants
- [ ] Remove runtime `if finally_body_fn is not None` checks

**Phase 3: Testing**
- [ ] Run existing tests to ensure no regressions
- [ ] Add test for try-catch without finally (optional)
- [ ] Verify performance improvement (optional)

---

## 6. Impact Assessment

**Performance:**
- Small improvement: Removes one runtime check per try-catch execution
- Code size: Slightly larger (4 function variants instead of 2)
- Overall: Minor optimization, mainly for code cleanliness

**Complexity:**
- Reduces: No more complex finally_body presence checks
- Increases: 4 code paths instead of 2 (but simpler individually)
- Overall: Slight improvement in maintainability

**Risk:**
- Low: Changes are localized to compiler
- Testing: Existing tests should catch any issues
- Rollback: Easy to revert if problems arise

---

## 7. Timeline Estimate

**Total effort: 2-4 hours**

- Phase 1 (IR Builder): 1 hour (optional)
- Phase 2 (Compiler Optimization): 1-2 hours (main work)
- Phase 3 (Testing): 0.5-1 hour

**Recommended approach:**
1. Skip Phase 1 initially (IR builder is optional convenience)
2. Focus on Phase 2 (compiler optimization) - the main benefit
3. Run tests and verify (Phase 3)
4. Add IR builder later if needed for ergonomics

---

## 8. References

- **East TypeScript Commit:** 302d2249503af9efb9d46cd89e75ab5d9d3e7fff
- **Commit Message:** "Clean up finally_body"
- **Files Changed:**
  - `src/ir.ts` - Made finally_body required (not optional)
  - `src/ast_to_ir.ts` - Generate dummy Value node when no finally
  - `src/compile.ts` - Optimize compilation and split code paths
- **Python Status:**
  - IR type system already correct (finally_body required)
  - Compiler needs optimization (remove runtime checks, split paths)
  - No AST layer to update (Python works directly with IR)

---

**Document Version:** 1.0
**Created:** 2025-11-11
**Status:** Ready for implementation (low priority optimization)
