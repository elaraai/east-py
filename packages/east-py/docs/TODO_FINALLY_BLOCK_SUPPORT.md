# TODO: Implement Finally Block Support for Try-Catch

**Reference Commit:** 792c7bd9de2df38698b6fa3b2d2d3a63fce39121 in ../East
**Base Commit:** a3ddfd70e83551dd626eef0a68c99b6beee9ef58 in ../East
**Feature:** Add finally block support to try-catch-finally error handling

## Overview

The East TypeScript implementation added support for `finally` blocks in try-catch-finally error handling. This feature allows cleanup code that must execute regardless of whether an error occurred or how control flow exits (early return, break, continue).

**Key Requirements:**
- Finally blocks are optional (can have try-catch, try-finally, or try-catch-finally)
- Finally blocks always execute for cleanup, regardless of control flow
- Finally blocks execute after try or catch completes
- Finally blocks execute even with early returns, breaks, or continues
- Finally blocks do NOT affect the return type (type is union of try and catch only)
- Cannot call `.catch()` twice on the same try block
- `.catch()` now returns `self`/`this` to enable chaining `.finally()`

---

## 1. Type System Changes

### File: `east/types/type_system.py`

**Location:** IR type definitions (lines ~1580-1960)

#### Task 1.1: Update `_TryCatchIR` struct type definition

**Current state:**
```python
_TryCatchIR = _ir_struct_type(
    lambda ir: [
        ("type", EastTypeType),
        ("location", LocationType),
        ("try_body", ir),
        ("catch_body", ir),
        ("message", ir),
        ("stack", ir),
    ]
)
```

**Required change:**
```python
_TryCatchIR = _ir_struct_type(
    lambda ir: [
        ("type", EastTypeType),
        ("location", LocationType),
        ("try_body", ir),
        ("catch_body", ir),
        ("message", ir),
        ("stack", ir),
        ("finally_body", ir),  # ADD THIS FIELD - Optional finally block
    ]
)
```

**Notes:**
- The TypeScript version adds `finally_body?: any` (optional field)
- In Python IR, we still need the field in the struct type, but it can hold a null/None-like value when not present
- The field type is `ir` (recursive IR type) since finally block contains IR statements

**Files to verify consistency:**
- Check if `IRType` variant properly includes the updated `_TryCatchIR`
- Currently at line ~1927: `("TryCatch", _TryCatchIR(ir))`

---

## 2. IR Analysis Changes

### File: `east/ir/analyze.py`

**Location:** `analyze_ir` function, TryCatch handling (lines ~349-356)

#### Task 2.1: Add finally_body analysis in TryCatch visitor

**Current state:**
```python
elif tag == "TryCatch":
    # TryCatch is async if try_body or catch_body is async
    if visit_ir(node.value.try_body, var_ctx):
        is_async = True
    visit_ir(node.value.error_variable, var_ctx)
    if visit_ir(node.value.catch_body, var_ctx):
        is_async = True
```

**Required change:**
```python
elif tag == "TryCatch":
    # TryCatch is async if try_body, catch_body, or finally_body is async
    if visit_ir(node.value.try_body, var_ctx):
        is_async = True
    visit_ir(node.value.message, var_ctx)
    visit_ir(node.value.stack, var_ctx)
    if visit_ir(node.value.catch_body, var_ctx):
        is_async = True
    # Process finally block if present
    if hasattr(node.value, 'finally_body') and node.value.finally_body is not None:
        if visit_ir(node.value.finally_body, var_ctx):
            is_async = True
```

**Notes:**
- Need to handle optional finally_body
- Finally block can make the entire TryCatch async
- Need to visit message and stack variable references (looks like current code has error_variable instead of message/stack?)
- **ACTION REQUIRED:** Verify current TryCatch IR structure in analyze.py matches the type definition

---

## 3. Runtime Compiler Changes

### File: `east/runtime/compiler.py`

**Status:** ⚠️ **CRITICAL - TryCatch compilation not yet implemented!**

**Investigation findings:**
- Searched for `_compile_trycatch` or `TryCatch` in compiler.py
- **No TryCatch compilation implementation found**
- Compilation dispatcher likely missing TryCatch case

#### Task 3.1: Implement `_compile_trycatch` function (NEW IMPLEMENTATION)

**Reference:** TypeScript implementation in `src/compile.ts` (lines 103-152)

**Required implementation:**

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
        trycatch_struct.try_body,
        platform_fns,
        async_platform_fns,
        is_async_map
    )

    # Extract message and stack variable names
    message_var = trycatch_struct.message.value
    message_name = message_var.name
    stack_var = trycatch_struct.stack.value
    stack_name = stack_var.name

    # Compile catch body
    catch_body_fn = _compile_ir(
        trycatch_struct.catch_body,
        platform_fns,
        async_platform_fns,
        is_async_map
    )

    # Compile finally body if present
    finally_body_fn = None
    finally_is_async = False
    if hasattr(trycatch_struct, 'finally_body') and trycatch_struct.finally_body is not None:
        finally_body_fn = _compile_ir(
            trycatch_struct.finally_body,
            platform_fns,
            async_platform_fns,
            is_async_map
        )
        finally_is_async = is_async_map.get(id(trycatch_struct.finally_body), False)

    # Check if any component is async
    try_is_async = is_async_map.get(id(trycatch_struct.try_body), False)
    catch_is_async = is_async_map.get(id(trycatch_struct.catch_body), False)
    is_async = try_is_async or catch_is_async or finally_is_async

    if is_async:
        async def execute_trycatch_async(env):
            try:
                # Execute try body
                if try_is_async:
                    result = await try_body_fn(env)
                else:
                    result = try_body_fn(env)
            except Exception as e:
                # Execute catch body with error info
                # Create child environment with message and stack
                catch_env = {**env}
                catch_env[message_name] = str(e)
                # TODO: Extract stack trace and convert to East array format
                catch_env[stack_name] = []  # Placeholder for stack trace

                if catch_is_async:
                    result = await catch_body_fn(catch_env)
                else:
                    result = catch_body_fn(catch_env)
            finally:
                # Execute finally block if present
                if finally_body_fn is not None:
                    if finally_is_async:
                        await finally_body_fn(env)
                    else:
                        finally_body_fn(env)

            return result

        return execute_trycatch_async
    else:
        def execute_trycatch_sync(env):
            try:
                # Execute try body
                result = try_body_fn(env)
            except Exception as e:
                # Execute catch body with error info
                catch_env = {**env}
                catch_env[message_name] = str(e)
                # TODO: Extract stack trace and convert to East array format
                catch_env[stack_name] = []  # Placeholder

                result = catch_body_fn(catch_env)
            finally:
                # Execute finally block if present
                if finally_body_fn is not None:
                    finally_body_fn(env)

            return result

        return execute_trycatch_sync
```

**Notes:**
- Need to implement stack trace extraction properly (TypeScript creates array of {filename, line, column})
- Python's traceback module can be used: `import traceback`
- Finally block executes in original environment (not catch environment)
- Finally block result is discarded (does not affect return value)

#### Task 3.2: Add TryCatch case to compilation dispatcher

**Location:** Main `_compile_ir` function

**Required change:**
Add to the dispatch logic (likely an if/elif chain or match statement):

```python
elif tag == "TryCatch":
    return _compile_trycatch(node, platform_fns, async_platform_fns, is_async_map)
```

**Files to check:**
- Find the main dispatch function in `compiler.py`
- Look for pattern like `if tag == "Block":`, `elif tag == "IfElse":`, etc.
- Add TryCatch case in alphabetical order

#### Task 3.3: Implement proper stack trace conversion

**Create helper function:**

```python
def _extract_stack_trace(exception: Exception) -> list[EastStruct]:
    """Extract stack trace from exception and convert to East format.

    Returns:
        List of structs with {filename: str, line: int, column: int}
    """
    import traceback
    from east.types.type_system import StructType, StringType, IntegerType
    from east.ir.builders import location

    # Stack entry type: {filename: String, line: Integer, column: Integer}
    # Note: Python doesn't track column numbers in tracebacks

    stack_frames = []
    tb = exception.__traceback__

    while tb is not None:
        frame = tb.tb_frame
        filename = frame.f_code.co_filename
        line = tb.tb_lineno
        column = 0  # Python doesn't track column numbers

        # Create location struct
        loc_struct = location(filename, line, column)
        stack_frames.append(loc_struct)

        tb = tb.tb_next

    return stack_frames
```

**Usage in _compile_trycatch:**
```python
from east.types.containers import EastArray
from east.types.type_system import StructType, StringType, IntegerType

# In catch block:
stack_trace = _extract_stack_trace(e)
stack_type = StructType([
    ("filename", StringType),
    ("line", IntegerType),
    ("column", IntegerType)
])
catch_env[stack_name] = EastArray(stack_type, stack_trace)
```

---

## 4. IR Builder Functions

### File: `east/ir/builders.py`

**Status:** ✅ Likely complete - no changes needed for finally

**Verification needed:**
- Check if there's a `ir_trycatch()` builder function
- If it exists, update signature to accept optional `finally_body` parameter
- If it doesn't exist, create it

#### Task 4.1: Create or update `ir_trycatch` builder function

**Example implementation:**

```python
def ir_trycatch(
    typ: EastType,
    loc: EastStruct,
    try_body: EastVariant,
    catch_body: EastVariant,
    message_var: EastVariant,
    stack_var: EastVariant,
    finally_body: EastVariant | None = None,
) -> EastVariant:
    """Create a TryCatch IR node.

    Args:
        typ: Return type (union of try and catch types)
        loc: Location
        try_body: IR for try block
        catch_body: IR for catch block
        message_var: Variable IR for error message (String)
        stack_var: Variable IR for stack trace (Array of location structs)
        finally_body: Optional IR for finally block

    Returns:
        TryCatch IR variant
    """
    # Get TryCatch struct type from IRType
    trycatch_cases = IRType.value  # Get variant cases
    trycatch_struct_type = None
    for case in trycatch_cases:
        if case.name == "TryCatch":
            trycatch_struct_type = case.type
            break

    if trycatch_struct_type is None:
        raise ValueError("TryCatch case not found in IRType")

    # Create struct class
    trycatch_class = _struct_class_from_type(trycatch_struct_type)

    # Build struct data
    struct_data = {
        "type": typ,
        "location": loc,
        "try_body": try_body,
        "catch_body": catch_body,
        "message": message_var,
        "stack": stack_var,
    }

    # Add finally_body if present
    if finally_body is not None:
        struct_data["finally_body"] = finally_body
    else:
        # Need to add null/placeholder value for optional field
        # Check how East handles optional struct fields in Python
        # May need to use null or omit the field entirely
        struct_data["finally_body"] = null  # Or might need to omit

    trycatch_struct = trycatch_class.create(**struct_data)
    return EastVariant(IRType, Case("TryCatch", trycatch_struct))
```

**Notes:**
- Need to verify how East Python handles optional struct fields
- May need to check TypeScript implementation for field presence checking
- The struct might need all fields present with null values, or might support missing fields

---

## 5. Block Builder / Expression API

### File: Search for block builder or expression builder module

**Status:** 🔍 **INVESTIGATION REQUIRED**

**Tasks:**
- Find where the Python equivalent of TypeScript's `BlockBuilder` and `TryCatchExpr` classes are
- Look for files with names like `expr.py`, `block.py`, `expression.py`, `builder.py`
- Check if there's a high-level API for building East programs in Python

#### Task 5.1: Locate Python expression/block builder

**Search patterns:**
```bash
# Look for class definitions similar to TypeScript's
grep -r "class.*Builder" east/
grep -r "def try_" east/
grep -r "def catch" east/
```

#### Task 5.2: Implement `.finally()` method on try-catch builder

**Reference:** TypeScript `src/expr/block.ts` lines 1604-1671

**Key changes needed:**
1. Make `.catch()` return `self` instead of `None`
2. Add flag to prevent calling `.catch()` twice: `self._catch_called = False`
3. Implement `.finally()` method that:
   - Takes a body function: `($: BlockBuilder) -> void | Expr`
   - Builds the finally block statements
   - Adds finally_body to the TryCatch AST/IR
   - Does NOT affect the return type
   - Returns `None` (ends the chain)

**Example Python implementation:**

```python
class TryCatchExpr:
    """Builder for try-catch-finally expressions."""

    def __init__(self, try_ast, message_var, stack_var, return_type):
        self._try_ast = try_ast
        self._message_var = message_var
        self._stack_var = stack_var
        self._return_type = return_type
        self._catch_called = False
        self._ast = None  # Will be set by catch()

    def catch(self, body_fn) -> 'TryCatchExpr':
        """Define the catch block.

        Args:
            body_fn: Function taking ($, message, stack) -> void | Expr

        Returns:
            self for chaining .finally()

        Raises:
            RuntimeError: If .catch() called more than once
        """
        if self._catch_called:
            raise RuntimeError("Cannot call .catch() more than once on the same try block")

        self._catch_called = True

        # Create block builder for catch body
        $ = BlockBuilder(self._return_type)

        # Execute user's catch body
        ret = body_fn($, self._message_var, self._stack_var)

        # Build catch statements
        stmts = $.statements
        if ret is not None:
            stmts.append(ret)

        # Ensure catch block ends with proper type
        if len(stmts) == 0 or not is_subtype(stmts[-1].type, NullType):
            stmts.append(null_value)

        # Build catch body IR/AST
        catch_body = build_block(stmts) if len(stmts) > 1 else stmts[0]

        # Create TryCatch AST/IR
        self._ast = ir_trycatch(
            typ=type_union(self._try_ast.type, catch_body.type),
            loc=get_location(),
            try_body=self._try_ast,
            catch_body=catch_body,
            message_var=self._message_var,
            stack_var=self._stack_var,
            finally_body=None,  # Not set yet
        )

        # Update return type if both try and catch return Never
        if is_type_equal(self._try_ast.type, NeverType) and \
           is_type_equal(catch_body.type, NeverType):
            self._ast.value.type = NeverType

        return self  # Return self for chaining

    def finally_(self, body_fn) -> None:
        """Define the finally block.

        Args:
            body_fn: Function taking ($) -> void | Expr

        Note:
            Finally block is for side effects only and does not affect return type.
        """
        if self._ast is None:
            raise RuntimeError("Must call .catch() before .finally()")

        # Create block builder for finally body
        $ = BlockBuilder(self._return_type)

        # Execute user's finally body
        ret = body_fn($)

        # Build finally statements
        stmts = $.statements
        if ret is not None:
            stmts.append(ret)

        # Ensure finally block ends with proper type
        if len(stmts) == 0 or not is_subtype(stmts[-1].type, NullType):
            stmts.append(null_value)

        # Build finally body IR/AST
        finally_body = build_block(stmts) if len(stmts) > 1 else stmts[0]

        # Add finally body to existing TryCatch AST/IR
        self._ast.value.finally_body = finally_body

        # Return type unchanged - finally doesn't affect type
```

**Notes:**
- Method named `finally_` (with underscore) because `finally` is a Python keyword
- Check if Python builder uses AST or IR directly
- May need to adapt based on actual implementation structure

---

## 6. Testing

### Directory: `tests/`

**Status:** 🔍 **INVESTIGATION + NEW TESTS REQUIRED**

**Investigation:**
- No TryCatch tests found in `tests/` directory
- Need to create comprehensive test suite

#### Task 6.1: Create test file for try-catch-finally

**File:** `tests/test_trycatch.py`

**Reference:** TypeScript `test/block.spec.ts` lines 582-759

**Required tests (18 total):**

1. ✅ **try-catch with no error** - Verify try block executes, catch skipped
2. ✅ **try-catch with error** - Verify error caught and catch executes
3. ✅ **try-catch returns correct type on success** - Type checking
4. ✅ **try-catch returns correct type on error** - Type checking
5. ✅ **try-finally executes finally on success** - Finally runs after successful try
6. ✅ **try-finally executes finally on error** - Finally runs after error
7. ✅ **try-catch-finally all execute correctly** - All three blocks work together (success path)
8. ✅ **try-catch-finally executes finally after catch** - All three blocks work (error path)
9. ✅ **finally executes on early return from try** - Control flow test
10. ✅ **finally executes on early return from catch** - Control flow test
11. ✅ **finally can modify variables but not affect return value** - Side effects only
12. ✅ **finally block with multiple statements** - Multiple operations in finally
13. ✅ **nested try-finally blocks** - Nested error handling
14. ⚠️ **error with message and stack** - Verify error info passed to catch
15. ⚠️ **finally with break in loop** - Control flow with break
16. ⚠️ **finally with continue in loop** - Control flow with continue
17. ⚠️ **cannot call catch twice** - Error handling
18. ⚠️ **try-finally without catch** - Optional catch block

**Test template:**

```python
import pytest
from east import (
    East,
    IntegerType,
    BooleanType,
    ArrayType,
    compile_function,
)


def test_try_catch_no_error():
    """Try-catch with no error should execute try block only."""

    def program($):
        result = $.let(0)

        $.try_($ => {
            $.assign(result, 42)
        }).catch(($, message, stack) => {
            $.assign(result, -1)
        })

        return result

    fn = compile_function([], IntegerType, program)
    assert fn() == 42


def test_try_catch_with_error():
    """Try-catch with error should execute catch block."""

    def program($):
        result = $.let(0)
        arr = $.const([1, 2, 3])

        $.try_($ => {
            arr.get(10)  # Out of bounds error
            $.assign(result, 42)
        }).catch(($, message, stack) => {
            $.assign(result, -1)
        })

        return result

    fn = compile_function([], IntegerType, program)
    assert fn() == -1


def test_try_finally_executes_finally_on_success():
    """Try-finally should execute finally block on success."""

    def program($):
        result = $.let(0)
        finally_executed = $.let(False)

        $.try_($ => {
            $.assign(result, 42)
        }).finally_($ => {
            $.assign(finally_executed, True)
        })

        return result, finally_executed

    fn = compile_function([], StructType([...]), program)
    result, executed = fn()
    assert result == 42
    assert executed == True


def test_try_catch_finally_all_execute():
    """Try-catch-finally should execute all three blocks correctly."""

    def program($):
        result = $.let(0)
        finally_executed = $.let(False)

        $.try_($ => {
            $.assign(result, 42)
        }).catch(($, message, stack) => {
            $.assign(result, -1)
        }).finally_($ => {
            $.assign(finally_executed, True)
        })

        return result, finally_executed

    fn = compile_function([], StructType([...]), program)
    result, executed = fn()
    assert result == 42
    assert executed == True


def test_finally_executes_on_early_return():
    """Finally block should execute even with early return from try."""

    def program($, x):
        finally_executed = $.let(False)

        $.try_($ => {
            $.if_(x == 0, $ => {
                $.return_(1)
            })
            $.return_(x * 2)
        }).finally_($ => {
            $.assign(finally_executed, True)
        })

        # Should not reach here
        return 999

    fn = compile_function([IntegerType], IntegerType, program)
    assert fn(0) == 1
    # Need to verify finally_executed is True somehow...


def test_cannot_call_catch_twice():
    """Should raise error when calling .catch() twice."""

    with pytest.raises(RuntimeError, match="Cannot call .catch\\(\\) more than once"):
        def program($):
            $.try_($ => {
                pass
            }).catch(($, msg, stack) => {
                pass
            }).catch(($, msg, stack) => {  # Should error here
                pass
            })

        compile_function([], NullType, program)

# ... more tests following the TypeScript examples
```

**Notes:**
- Need to determine Python API for East program construction
- Tests should match TypeScript functionality exactly
- May need helper functions for common patterns
- Consider async versions of tests if needed

#### Task 6.2: Add integration tests

**File:** `tests/test_integration_trycatch.py`

**Tests needed:**
- Complex nested try-catch-finally
- Error handling in recursive functions
- Try-catch with closures and captures
- Platform function errors in try-catch
- Async platform functions in try-catch-finally

---

## 7. Documentation

### File: Create `docs/ERROR_HANDLING.md`

**Reference:** TypeScript `USAGE.md` lines 267-389

**Required sections:**

1. **Overview**
   - Introduction to try-catch-finally
   - Error handling philosophy in East
   - When to use try-catch vs other patterns

2. **Basic Usage**
   - Simple try-catch example
   - Error message and stack trace access
   - Return type semantics

3. **Finally Blocks**
   - When to use finally
   - Cleanup operations
   - Side effects only (no return value influence)

4. **Control Flow**
   - Early returns from try/catch
   - Break and continue in loops
   - Nested try-catch blocks

5. **Examples**
   - Safe array access
   - Resource cleanup
   - Search with cleanup
   - Try-finally without catch

6. **API Reference**
   - `$.try_(body_fn)` - Create try block
   - `.catch(body_fn)` - Add catch block (chainable)
   - `.finally_(body_fn)` - Add finally block (terminal)

**Template:**

```markdown
# Error Handling in East Python

## Overview

East provides try-catch-finally for error handling with familiar Python-like semantics...

## Basic Try-Catch

```python
def safe_divide($, x, y):
    result = $.let(0)

    $.try_($ => {
        $.assign(result, x.divide(y))
    }).catch(($, message, stack) => {
        # message is a String
        # stack is an Array of {filename: String, line: Integer, column: Integer}
        $.assign(result, 0)
    })

    return result
```

## Finally Blocks

Finally blocks execute cleanup code regardless of control flow:

```python
def with_cleanup($, arr):
    resource_open = $.let(True)
    result = $.let(0)

    $.try_($ => {
        $.assign(result, arr.get(0))
    }).catch(($, message, stack) => {
        $.assign(result, -1)
    }).finally_($ => {
        # Always executes, even on early return or error
        $.assign(resource_open, False)
    })

    return result
```

[... rest of documentation ...]
```

---

## 8. Examples

### File: `examples/error_handling.py`

**Create comprehensive example demonstrating:**

1. Basic error handling
2. Resource cleanup with finally
3. Early returns with cleanup
4. Nested error handling
5. Error propagation

```python
"""Examples of error handling with try-catch-finally in East Python.

This module demonstrates various error handling patterns.
"""

from east import (
    East,
    IntegerType,
    StringType,
    ArrayType,
    BooleanType,
    compile_function,
)


# Example 1: Safe array access
def safe_array_get():
    """Safely access array element with error handling."""

    def program($, arr, index):
        result = $.let(0)

        $.try_($ => {
            $.assign(result, arr.get(index))
        }).catch(($, message, stack) => {
            # Return -1 on error
            $.assign(result, -1)
        })

        return result

    return compile_function(
        [ArrayType(IntegerType), IntegerType],
        IntegerType,
        program
    )


# Example 2: Resource cleanup
def with_resource_cleanup():
    """Demonstrate resource cleanup with finally block."""

    def program($, arr):
        resource_open = $.let(True)
        result = $.let(0)
        error_occurred = $.let(False)

        $.try_($ => {
            $.assign(result, arr.get(0))
        }).catch(($, message, stack) => {
            $.assign(error_occurred, True)
            $.assign(result, -1)
        }).finally_($ => {
            # Cleanup always happens
            $.assign(resource_open, False)
        })

        # Return struct with all values
        return $.struct({
            "result": result,
            "resource_open": resource_open,
            "error_occurred": error_occurred,
        })

    return compile_function([ArrayType(IntegerType)], StructType([...]), program)


# Example 3: Search with early return and cleanup
def search_with_cleanup():
    """Search array with early return, ensuring cleanup."""

    def program($, arr, target):
        search_started = $.let(False)

        $.try_($ => {
            $.assign(search_started, True)

            $.for_(arr, ($, value, i) => {
                $.if_(value == target, $ => {
                    $.return_(i)  # Early return
                })
            })

            $.return_(-1)  # Not found
        }).finally_($ => {
            # Log that search completed (even on early return)
            # In real code, might close file, release lock, etc.
            pass
        })

        return -1  # Unreachable if try returns

    return compile_function(
        [ArrayType(IntegerType), IntegerType],
        IntegerType,
        program
    )


if __name__ == "__main__":
    # Test examples
    safe_get = safe_array_get()
    print(safe_get([1, 2, 3], 1))  # 2
    print(safe_get([1, 2, 3], 10))  # -1

    cleanup = with_resource_cleanup()
    result = cleanup([42])
    print(f"Success: {result}")  # resource_open should be False

    result = cleanup([])
    print(f"Error: {result}")  # resource_open still False, error_occurred True

    search = search_with_cleanup()
    print(search([10, 20, 30], 20))  # 1
    print(search([10, 20, 30], 99))  # -1
```

---

## 9. Implementation Order & Dependencies

**Recommended implementation order:**

### Phase 1: Type System & IR (No runtime dependency)
1. ✅ Update `_TryCatchIR` in `type_system.py` (Task 1.1)
2. ✅ Create/update `ir_trycatch` builder in `builders.py` (Task 4.1)

### Phase 2: Analysis (Depends on Phase 1)
3. ✅ Update TryCatch analysis in `analyze.py` (Task 2.1)

### Phase 3: Runtime (Depends on Phases 1-2)
4. ⚠️ Implement `_compile_trycatch` in `compiler.py` (Task 3.1)
5. ⚠️ Add TryCatch dispatch case (Task 3.2)
6. ⚠️ Implement stack trace extraction (Task 3.3)

### Phase 4: High-Level API (Depends on Phases 1-3)
7. 🔍 Locate expression builder (Task 5.1)
8. 🔍 Implement `.finally_()` method (Task 5.2)
9. 🔍 Update `.catch()` to return self (Task 5.2)

### Phase 5: Testing (Can start after Phase 3)
10. 📝 Create basic tests (Task 6.1 - tests 1-8)
11. 📝 Create control flow tests (Task 6.1 - tests 9-13)
12. 📝 Create edge case tests (Task 6.1 - tests 14-18)
13. 📝 Create integration tests (Task 6.2)

### Phase 6: Documentation & Examples (Can parallelize with testing)
14. 📝 Write error handling docs (Task 7)
15. 📝 Create examples (Task 8)

---

## 10. Verification Checklist

After implementation, verify:

- [ ] All IR nodes include optional finally_body field
- [ ] Analyze properly handles finally_body in async detection
- [ ] Compiler generates correct try-catch-finally in both sync/async modes
- [ ] Stack traces properly extracted and formatted
- [ ] Finally blocks execute in all control flow scenarios:
  - [ ] Normal completion of try
  - [ ] Error caught in catch
  - [ ] Early return from try
  - [ ] Early return from catch
  - [ ] Break from try (in loop)
  - [ ] Continue from try (in loop)
- [ ] Finally blocks don't affect return type
- [ ] Cannot call `.catch()` twice
- [ ] `.catch()` returns self for chaining
- [ ] `.finally_()` accepts block builder function
- [ ] All 18 tests pass
- [ ] Examples run without errors
- [ ] Documentation is clear and accurate

---

## 11. Known Issues & Open Questions

### Questions to resolve:

1. **Optional struct fields in Python IR:**
   - How does East Python handle optional struct fields?
   - Should finally_body be null or omitted when not present?
   - Check TypeScript serialization: does it omit or include null?

2. **Error message format:**
   - TypeScript uses `String(e)` - is this adequate?
   - Should we extract more structured error info?
   - Do we need error type information?

3. **Stack trace format:**
   - Python doesn't track column numbers - use 0 or estimate?
   - Should we limit stack depth?
   - Filter internal East frames?

4. **High-level API location:**
   - Where is the Python equivalent of TypeScript's `BlockBuilder`?
   - Is there already a `.try_()` method somewhere?
   - What's the naming convention (finally_ with underscore)?

5. **Async error handling:**
   - Do Python async functions propagate exceptions correctly?
   - Need special handling for async finally blocks?

6. **Testing framework:**
   - What test framework does east-py use? (pytest assumed)
   - How to test finally execution with early returns?
   - How to capture side effects in tests?

### TypeScript-Python differences:

- **Keyword conflicts:** `finally` is a keyword in Python, need `finally_`
- **Lambda syntax:** Python uses different syntax than TypeScript arrows
- **Exception types:** Python has more specific exception types
- **Stack traces:** Different structure in Python vs JS
- **Optional parameters:** Python handles differently than TypeScript's `?`

---

## 12. Success Criteria

Implementation is complete when:

1. ✅ All type definitions updated with finally_body
2. ✅ IR analysis handles finally blocks
3. ✅ Compiler generates correct try-catch-finally code
4. ✅ All 18 tests pass
5. ✅ Documentation complete and accurate
6. ✅ Examples run successfully
7. ✅ Code review passed
8. ✅ No regressions in existing tests
9. ✅ Stack traces properly formatted
10. ✅ Control flow semantics match TypeScript exactly

---

## 13. Timeline Estimate

**Assuming 1 developer working full-time:**

- Phase 1 (Type System): 2-4 hours
- Phase 2 (Analysis): 1-2 hours
- Phase 3 (Runtime): 8-12 hours (includes stack trace work)
- Phase 4 (High-Level API): 4-6 hours (depends on finding existing code)
- Phase 5 (Testing): 8-10 hours
- Phase 6 (Docs/Examples): 4-6 hours

**Total: 27-40 hours (3-5 days)**

---

## References

- **East TypeScript Commit:** 792c7bd9de2df38698b6fa3b2d2d3a63fce39121
- **Key Files Changed:**
  - `src/ast.ts` - Added finally_body to TryCatchAST
  - `src/ast_to_ir.ts` - Process finally block in IR conversion
  - `src/compile.ts` - Compile finally block in both sync/async
  - `src/expr/block.ts` - Implement .finally() method, make .catch() chainable
  - `src/ir.ts` - Add finally_body to TryCatchIR type
  - `test/block.spec.ts` - 13 new tests for finally semantics
  - `USAGE.md` - 120+ lines of documentation and examples

---

**Document Version:** 1.0
**Created:** 2025-11-10
**Last Updated:** 2025-11-10
**Status:** Ready for implementation
