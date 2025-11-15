# TODO: Complete Python Compiler Implementation

**Based on:** ../East TypeScript compiler (src/compile.ts)
**Status:** Python compiler has 11/30 IR node types implemented
**Goal:** Achieve feature parity with TypeScript compiler

## Overview

The East Python runtime compiler (`east/runtime/compiler.py`) is **incomplete**. It only implements 11 out of 30 IR node types that exist in the TypeScript implementation. This blocks the ability to run TypeScript-exported test IR in Python.

**Current Status:**
- ✅ **11 implemented:** Value, Variable, Builtin, Block, IfElse, While, Let, Platform, TryCatch, Function, NewRef
- ❌ **19 missing:** Error, Assign, As, UnwrapRecursive, WrapRecursive, Call, Match, ForArray, ForSet, ForDict, GetField, Struct, Variant, NewArray, NewSet, NewDict, Return, Continue, Break

**Why this matters:**
- TypeScript-exported test IR cannot execute in Python
- Cross-implementation compatibility testing is blocked
- Python implementation is incomplete compared to TypeScript

**Impact:**
- **tests/test_typescript_exports.py** - All tests fail due to missing implementations
- **Cross-language IR compatibility** - Cannot verify Python/TypeScript consistency
- **Feature completeness** - Python is missing ~63% of compiler features

---

## Current State Analysis

### File: `east/runtime/compiler.py`

**Lines 136-158:** Compiler dispatch logic

```python
if tag == "Function":
    return _compile_function(ir, platform_list, is_async_map)
if tag == "Value":
    return _compile_value(ir)
if tag == "Variable":
    return _compile_variable(ir)
if tag == "Builtin":
    return _compile_builtin(ir, platform_list, is_async_map)
if tag == "Block":
    return _compile_block(ir, platform_list, is_async_map)
if tag == "IfElse":
    return _compile_ifelse(ir, platform_list, is_async_map)
if tag == "While":
    return _compile_while(ir, platform_list, is_async_map)
if tag == "Let":
    return _compile_let(ir, platform_list, is_async_map)
if tag == "Platform":
    return _compile_platform(ir, platform_list, is_async_map)
if tag == "TryCatch":
    return _compile_trycatch(ir, platform_list, is_async_map)
if tag == "NewRef":
    return _compile_newref(ir, platform_list, is_async_map)

# All other tags fall through to:
raise NotImplementedError(f"Compilation for {tag} not yet implemented")
```

**Problem:** 19 IR node types are not handled and raise NotImplementedError.

---

## Missing Implementations (Priority Order)

### **Priority 1: Core Language Features** (Required for basic programs)

These are essential for running simple East programs.

#### 1. **Call** - Function calls
**Priority:** CRITICAL
**Complexity:** Medium
**Reference:** ../East/src/compile.ts:352-411

**Why critical:** Cannot call functions without this. Blocks almost all real programs.

**TypeScript implementation:**
```typescript
} else if (ir.type === "Call") {
  const compiled_function = compile(ir.value.function, ctx, expected_return_type);
  const compiled_arguments = ir.value.arguments.map((arg) => compile(arg, ctx, expected_return_type));

  if (ir.value.isAsync) {
    return async (runtime_ctx: Record<string, any>) => {
      const f = await compiled_function(runtime_ctx);
      const args = [];
      for (const arg of compiled_arguments) {
        args.push(await arg(runtime_ctx));
      }
      return await f(...args);
    };
  } else {
    return (runtime_ctx: Record<string, any>) => {
      const f = compiled_function(runtime_ctx);
      const args = compiled_arguments.map((arg) => arg(runtime_ctx));
      return f(...args);
    };
  }
}
```

**Python implementation needed:**
```python
def _compile_call(
    ir: EastVariant,
    platform: list[PlatformFunction],
    is_async_map: dict[int, bool]
) -> Callable:
    """Compile function call IR node.

    Compiles: function(...arguments)
    """
    is_async = is_async_map[id(ir)]

    # Compile function expression and arguments
    func_compiled = _compile(ir.value.function, platform, is_async_map)
    args_compiled = [_compile(arg, platform, is_async_map) for arg in ir.value.arguments]

    if is_async:
        async def call_async(ctx):
            func = await func_compiled(ctx)
            args = []
            for arg_compiled in args_compiled:
                args.append(await arg_compiled(ctx))
            return await func(*args)
        return call_async
    else:
        def call_sync(ctx):
            func = func_compiled(ctx)
            args = [arg_compiled(ctx) for arg_compiled in args_compiled]
            return func(*args)
        return call_sync
```

---

#### 2. **Return** - Early return from function
**Priority:** CRITICAL
**Complexity:** Low
**Reference:** ../East/src/compile.ts:919-933

**Why critical:** Cannot exit functions early. Limits control flow.

**TypeScript implementation:**
```typescript
} else if (ir.type === "Return") {
  const compiled_value = compile(ir.value.value, ctx, expected_return_type);

  if (ir.value.isAsync) {
    return async (runtime_ctx: Record<string, any>) => {
      throw { type: "return", value: await compiled_value(runtime_ctx) };
    };
  } else {
    return (runtime_ctx: Record<string, any>) => {
      throw { type: "return", value: compiled_value(runtime_ctx) };
    };
  }
}
```

**Python implementation needed:**
```python
class ReturnException(Exception):
    """Exception used to implement early return from functions."""
    def __init__(self, value):
        self.value = value
        super().__init__()

def _compile_return(
    ir: EastVariant,
    platform: list[PlatformFunction],
    is_async_map: dict[int, bool]
) -> Callable:
    """Compile return statement.

    Throws ReturnException with the return value.
    """
    is_async = is_async_map[id(ir)]
    value_compiled = _compile(ir.value.value, platform, is_async_map)

    if is_async:
        async def return_async(ctx):
            value = await value_compiled(ctx)
            raise ReturnException(value)
        return return_async
    else:
        def return_sync(ctx):
            value = value_compiled(ctx)
            raise ReturnException(value)
        return return_sync
```

**Note:** Need to update `_compile_function` to catch `ReturnException`:
```python
# In _compile_function, wrap body execution:
try:
    result = body_compiled(ctx)
    return result
except ReturnException as e:
    return e.value
```

---

#### 3. **Assign** - Variable assignment
**Priority:** HIGH
**Complexity:** Low
**Reference:** ../East/src/compile.ts:226-270

**Why high priority:** Cannot update mutable variables without this.

**TypeScript implementation:**
```typescript
} else if (ir.type === "Assign") {
  const compiled_variable = compile(ir.value.variable, ctx, expected_return_type);
  const compiled_value = compile(ir.value.value, ctx, expected_return_type);

  if (ir.value.isAsync) {
    return async (runtime_ctx: Record<string, any>) => {
      const variable_name = (ir.value.variable.value as VariableIR).name;
      runtime_ctx[variable_name] = await compiled_value(runtime_ctx);
      return null;
    };
  } else {
    return (runtime_ctx: Record<string, any>) => {
      const variable_name = (ir.value.variable.value as VariableIR).name;
      runtime_ctx[variable_name] = compiled_value(runtime_ctx);
      return null;
    };
  }
}
```

**Python implementation needed:**
```python
def _compile_assign(
    ir: EastVariant,
    platform: list[PlatformFunction],
    is_async_map: dict[int, bool]
) -> Callable:
    """Compile variable assignment.

    Compiles: variable = value
    """
    from east.types.primitives import Null

    is_async = is_async_map[id(ir)]
    value_compiled = _compile(ir.value.value, platform, is_async_map)
    variable_name = ir.value.variable.value.name

    if is_async:
        async def assign_async(ctx):
            ctx[variable_name] = await value_compiled(ctx)
            return Null()
        return assign_async
    else:
        def assign_sync(ctx):
            ctx[variable_name] = value_compiled(ctx)
            return Null()
        return assign_sync
```

---

#### 4. **As** - Type casting
**Priority:** HIGH
**Complexity:** Low
**Reference:** ../East/src/compile.ts:271-278

**Why high priority:** Currently blocking Boolean test. Needed for type assertions.

**TypeScript implementation:**
```typescript
} else if (ir.type === "As") {
  const compiled_value = compile(ir.value.value, ctx, expected_return_type);
  // Type assertion - just pass through value at runtime
  return compiled_value;
}
```

**Python implementation needed:**
```python
def _compile_as(
    ir: EastVariant,
    platform: list[PlatformFunction],
    is_async_map: dict[int, bool]
) -> Callable:
    """Compile type assertion (As).

    Type assertions are compile-time only - runtime just passes value through.
    """
    # Simply compile the inner value and pass it through
    return _compile(ir.value.value, platform, is_async_map)
```

---

#### 5. **Struct** - Struct literal creation
**Priority:** HIGH
**Complexity:** Low
**Reference:** ../East/src/compile.ts:786-813

**Why high priority:** Cannot create structured data without this.

**TypeScript implementation:**
```typescript
} else if (ir.type === "Struct") {
  const compiled_fields = ir.value.fields.map((field) => compile(field, ctx, expected_return_type));
  const field_names = (ir.value.type.value as StructTypeValue).value.map((f) => f.name);

  if (ir.value.isAsync) {
    return async (runtime_ctx: Record<string, any>) => {
      const struct: Record<string, any> = {};
      for (let i = 0; i < field_names.length; i++) {
        struct[field_names[i]] = await compiled_fields[i](runtime_ctx);
      }
      return struct;
    };
  } else {
    return (runtime_ctx: Record<string, any>) => {
      const struct: Record<string, any> = {};
      for (let i = 0; i < field_names.length; i++) {
        struct[field_names[i]] = compiled_fields[i](runtime_ctx);
      }
      return struct;
    };
  }
}
```

**Python implementation needed:**
```python
def _compile_struct(
    ir: EastVariant,
    platform: list[PlatformFunction],
    is_async_map: dict[int, bool]
) -> Callable:
    """Compile struct literal.

    Compiles: { field1: value1, field2: value2, ... }
    """
    is_async = is_async_map[id(ir)]

    # Get field names from type
    struct_type = ir.value.type
    field_names = [field[0] for field in struct_type.value]

    # Compile field values
    fields_compiled = [_compile(field_val, platform, is_async_map)
                      for field_val in ir.value.fields]

    if is_async:
        async def struct_async(ctx):
            struct = {}
            for name, field_compiled in zip(field_names, fields_compiled):
                struct[name] = await field_compiled(ctx)
            return struct
        return struct_async
    else:
        def struct_sync(ctx):
            struct = {}
            for name, field_compiled in zip(field_names, fields_compiled):
                struct[name] = field_compiled(ctx)
            return struct
        return struct_sync
```

---

#### 6. **GetField** - Struct field access
**Priority:** HIGH
**Complexity:** Low
**Reference:** ../East/src/compile.ts:767-785

**Why high priority:** Cannot access struct fields without this.

**TypeScript implementation:**
```typescript
} else if (ir.type === "GetField") {
  const compiled_struct = compile(ir.value.struct, ctx, expected_return_type);
  const field_name = ir.value.field;

  if (ir.value.isAsync) {
    return async (runtime_ctx: Record<string, any>) => {
      const struct = await compiled_struct(runtime_ctx);
      return struct[field_name];
    };
  } else {
    return (runtime_ctx: Record<string, any>) => {
      const struct = compiled_struct(runtime_ctx);
      return struct[field_name];
    };
  }
}
```

**Python implementation needed:**
```python
def _compile_getfield(
    ir: EastVariant,
    platform: list[PlatformFunction],
    is_async_map: dict[int, bool]
) -> Callable:
    """Compile struct field access.

    Compiles: struct.field
    """
    is_async = is_async_map[id(ir)]
    struct_compiled = _compile(ir.value.struct, platform, is_async_map)
    field_name = ir.value.field

    if is_async:
        async def getfield_async(ctx):
            struct = await struct_compiled(ctx)
            return struct[field_name]
        return getfield_async
    else:
        def getfield_sync(ctx):
            struct = struct_compiled(ctx)
            return struct[field_name]
        return getfield_sync
```

---

#### 7. **Variant** - Variant constructor
**Priority:** HIGH
**Complexity:** Low
**Reference:** ../East/src/compile.ts:814-833

**Why high priority:** Cannot create variant values (Result, Option, etc.) without this.

**TypeScript implementation:**
```typescript
} else if (ir.type === "Variant") {
  const compiled_value = compile(ir.value.value, ctx, expected_return_type);
  const case_name = ir.value.case;

  if (ir.value.isAsync) {
    return async (runtime_ctx: Record<string, any>) => {
      return { type: case_name, value: await compiled_value(runtime_ctx) };
    };
  } else {
    return (runtime_ctx: Record<string, any>) => {
      return { type: case_name, value: compiled_value(runtime_ctx) };
    };
  }
}
```

**Python implementation needed:**
```python
def _compile_variant(
    ir: EastVariant,
    platform: list[PlatformFunction],
    is_async_map: dict[int, bool]
) -> Callable:
    """Compile variant constructor.

    Compiles: SomeCase(value)
    """
    is_async = is_async_map[id(ir)]
    value_compiled = _compile(ir.value.value, platform, is_async_map)
    case_name = ir.value.case

    if is_async:
        async def variant_async(ctx):
            value = await value_compiled(ctx)
            return {"type": case_name, "value": value}
        return variant_async
    else:
        def variant_sync(ctx):
            value = value_compiled(ctx)
            return {"type": case_name, "value": value}
        return variant_sync
```

---

### **Priority 2: Control Flow** (Loop constructs)

#### 8. **ForArray** - Array iteration
**Priority:** MEDIUM-HIGH
**Complexity:** Medium
**Reference:** ../East/src/compile.ts:515-578

**TypeScript implementation:** (64 lines)
- Compiles array expression
- Iterates with index (key) and element
- Supports break/continue via exceptions
- Handles async iteration

**Python implementation needed:**
```python
class BreakException(Exception):
    """Exception used to implement break in loops."""
    pass

class ContinueException(Exception):
    """Exception used to implement continue in loops."""
    pass

def _compile_forarray(
    ir: EastVariant,
    platform: list[PlatformFunction],
    is_async_map: dict[int, bool]
) -> Callable:
    """Compile for-array loop.

    Compiles: for (key, element) in array { body }
    """
    from east.types.primitives import Null

    is_async = is_async_map[id(ir)]
    array_compiled = _compile(ir.value.array, platform, is_async_map)
    body_compiled = _compile(ir.value.body, platform, is_async_map)

    key_var_name = ir.value.key.value.name
    element_var_name = ir.value.element.value.name

    if is_async:
        async def forarray_async(ctx):
            array = await array_compiled(ctx)
            for i, elem in enumerate(array):
                # Create child context with loop variables
                child_ctx = {**ctx, key_var_name: i, element_var_name: elem}
                try:
                    await body_compiled(child_ctx)
                except BreakException:
                    break
                except ContinueException:
                    continue
            return Null()
        return forarray_async
    else:
        def forarray_sync(ctx):
            array = array_compiled(ctx)
            for i, elem in enumerate(array):
                child_ctx = {**ctx, key_var_name: i, element_var_name: elem}
                try:
                    body_compiled(child_ctx)
                except BreakException:
                    break
                except ContinueException:
                    continue
            return Null()
        return forarray_sync
```

---

#### 9. **ForSet** - Set iteration
**Priority:** MEDIUM
**Complexity:** Medium
**Reference:** ../East/src/compile.ts:579-638

Similar to ForArray but iterates over sets (unordered).

---

#### 10. **ForDict** - Dictionary iteration
**Priority:** MEDIUM
**Complexity:** Medium
**Reference:** ../East/src/compile.ts:639-703

Iterates over dictionary key-value pairs.

---

#### 11. **Break** - Break from loop
**Priority:** MEDIUM
**Complexity:** Low
**Reference:** ../East/src/compile.ts:943-951

```python
def _compile_break(
    ir: EastVariant,
    platform: list[PlatformFunction],
    is_async_map: dict[int, bool]
) -> Callable:
    """Compile break statement."""
    is_async = is_async_map[id(ir)]

    if is_async:
        async def break_async(ctx):
            raise BreakException()
        return break_async
    else:
        def break_sync(ctx):
            raise BreakException()
        return break_sync
```

---

#### 12. **Continue** - Continue to next loop iteration
**Priority:** MEDIUM
**Complexity:** Low
**Reference:** ../East/src/compile.ts:934-942

```python
def _compile_continue(
    ir: EastVariant,
    platform: list[PlatformFunction],
    is_async_map: dict[int, bool]
) -> Callable:
    """Compile continue statement."""
    is_async = is_async_map[id(ir)]

    if is_async:
        async def continue_async(ctx):
            raise ContinueException()
        return continue_async
    else:
        def continue_sync(ctx):
            raise ContinueException()
        return continue_sync
```

---

#### 13. **Match** - Pattern matching on variants
**Priority:** MEDIUM
**Complexity:** High
**Reference:** ../East/src/compile.ts:444-471

**TypeScript implementation:**
```typescript
} else if (ir.type === "Match") {
  const compiled_variant = compile(ir.value.variant, ctx, expected_return_type);
  const compiled_cases: Record<string, (ctx: Record<string, any>) => any> = {};

  for (const { case: k, variable, body } of ir.value.cases) {
    const variable_name = variable.value.name;
    const compiled_body = compile(body, ctx, expected_return_type);
    compiled_cases[k] = (runtime_ctx: Record<string, any>) => {
      return compiled_body({ ...runtime_ctx, [variable_name]: runtime_ctx["$match_value"] });
    };
  }

  if (ir.value.isAsync) {
    return async (runtime_ctx: Record<string, any>) => {
      const variant = await compiled_variant(runtime_ctx);
      const case_handler = compiled_cases[variant.type];
      return await case_handler({ ...runtime_ctx, $match_value: variant.value });
    };
  } else {
    return (runtime_ctx: Record<string, any>) => {
      const variant = compiled_variant(runtime_ctx);
      const case_handler = compiled_cases[variant.type];
      return case_handler({ ...runtime_ctx, $match_value: variant.value });
    };
  }
}
```

**Python implementation needed:**
```python
def _compile_match(
    ir: EastVariant,
    platform: list[PlatformFunction],
    is_async_map: dict[int, bool]
) -> Callable:
    """Compile match (pattern matching on variants).

    Compiles: match variant { Case1(x) => body1, Case2(y) => body2, ... }
    """
    is_async = is_async_map[id(ir)]
    variant_compiled = _compile(ir.value.variant, platform, is_async_map)

    # Compile each case
    cases_compiled = {}
    for case in ir.value.cases:
        case_name = case.case
        variable_name = case.variable.value.name
        body_compiled = _compile(case.body, platform, is_async_map)
        cases_compiled[case_name] = (variable_name, body_compiled)

    if is_async:
        async def match_async(ctx):
            variant = await variant_compiled(ctx)
            case_name = variant["type"]
            case_value = variant["value"]

            variable_name, body_compiled = cases_compiled[case_name]
            child_ctx = {**ctx, variable_name: case_value}
            return await body_compiled(child_ctx)
        return match_async
    else:
        def match_sync(ctx):
            variant = variant_compiled(ctx)
            case_name = variant["type"]
            case_value = variant["value"]

            variable_name, body_compiled = cases_compiled[case_name]
            child_ctx = {**ctx, variable_name: case_value}
            return body_compiled(child_ctx)
        return match_sync
```

---

### **Priority 3: Container Creation**

#### 14. **NewArray** - Array literal
**Priority:** MEDIUM
**Complexity:** Low
**Reference:** ../East/src/compile.ts:848-869

```python
def _compile_newarray(
    ir: EastVariant,
    platform: list[PlatformFunction],
    is_async_map: dict[int, bool]
) -> Callable:
    """Compile array literal.

    Compiles: [element1, element2, ...]
    """
    is_async = is_async_map[id(ir)]
    elements_compiled = [_compile(elem, platform, is_async_map)
                        for elem in ir.value.values]

    if is_async:
        async def newarray_async(ctx):
            array = []
            for elem_compiled in elements_compiled:
                array.append(await elem_compiled(ctx))
            return array
        return newarray_async
    else:
        def newarray_sync(ctx):
            return [elem_compiled(ctx) for elem_compiled in elements_compiled]
        return newarray_sync
```

---

#### 15. **NewSet** - Set literal
**Priority:** LOW
**Complexity:** Low
**Reference:** ../East/src/compile.ts:870-892

Similar to NewArray but creates a set.

---

#### 16. **NewDict** - Dictionary literal
**Priority:** LOW
**Complexity:** Low
**Reference:** ../East/src/compile.ts:893-918

```python
def _compile_newdict(
    ir: EastVariant,
    platform: list[PlatformFunction],
    is_async_map: dict[int, bool]
) -> Callable:
    """Compile dictionary literal.

    Compiles: { key1: value1, key2: value2, ... }
    """
    is_async = is_async_map[id(ir)]
    entries_compiled = [(
        _compile(entry.key, platform, is_async_map),
        _compile(entry.value, platform, is_async_map)
    ) for entry in ir.value.entries]

    if is_async:
        async def newdict_async(ctx):
            result = {}
            for key_compiled, value_compiled in entries_compiled:
                key = await key_compiled(ctx)
                value = await value_compiled(ctx)
                result[key] = value
            return result
        return newdict_async
    else:
        def newdict_sync(ctx):
            result = {}
            for key_compiled, value_compiled in entries_compiled:
                key = key_compiled(ctx)
                value = value_compiled(ctx)
                result[key] = value
            return result
        return newdict_sync
```

---

### **Priority 4: Advanced Features**

#### 17. **Error** - Throw error
**Priority:** LOW
**Complexity:** Low
**Reference:** ../East/src/compile.ts:91-98

```python
def _compile_error(
    ir: EastVariant,
    platform: list[PlatformFunction],
    is_async_map: dict[int, bool]
) -> Callable:
    """Compile error (throw exception).

    Compiles: throw error_message
    """
    is_async = is_async_map[id(ir)]
    message_compiled = _compile(ir.value.message, platform, is_async_map)

    if is_async:
        async def error_async(ctx):
            message = await message_compiled(ctx)
            raise RuntimeError(message)
        return error_async
    else:
        def error_sync(ctx):
            message = message_compiled(ctx)
            raise RuntimeError(message)
        return error_sync
```

---

#### 18. **UnwrapRecursive** - Unwrap recursive type
**Priority:** LOW
**Complexity:** Low
**Reference:** ../East/src/compile.ts:279-287

```python
def _compile_unwraprecursive(
    ir: EastVariant,
    platform: list[PlatformFunction],
    is_async_map: dict[int, bool]
) -> Callable:
    """Compile unwrap of recursive type.

    Simply passes through the value (recursive types are transparent at runtime).
    """
    return _compile(ir.value.value, platform, is_async_map)
```

---

#### 19. **WrapRecursive** - Wrap in recursive type
**Priority:** LOW
**Complexity:** Low
**Reference:** ../East/src/compile.ts:288-296

```python
def _compile_wraprecursive(
    ir: EastVariant,
    platform: list[PlatformFunction],
    is_async_map: dict[int, bool]
) -> Callable:
    """Compile wrap in recursive type.

    Simply passes through the value (recursive types are transparent at runtime).
    """
    return _compile(ir.value.value, platform, is_async_map)
```

---

## Implementation Plan

### Phase 1: Critical Blockers (1-2 days)
**Goal:** Make basic TypeScript tests runnable

1. ✅ Fix IR analysis variable scoping (COMPLETED)
2. **Implement Call** - Cannot run any function calls
3. **Implement As** - Blocking Boolean test
4. **Implement Return** - Cannot exit functions early
5. **Implement Assign** - Cannot mutate variables

**After Phase 1:** Simple functional programs should work

---

### Phase 2: Data Structures (1 day)
**Goal:** Enable working with structured data

6. **Implement Struct** - Create struct literals
7. **Implement GetField** - Access struct fields
8. **Implement Variant** - Create variant values
9. **Implement Match** - Pattern match on variants
10. **Implement NewArray** - Create arrays
11. **Implement NewDict** - Create dictionaries

**After Phase 2:** Can work with data structures and pattern matching

---

### Phase 3: Iteration (1 day)
**Goal:** Enable loops

12. **Implement ForArray** - Iterate over arrays
13. **Implement ForSet** - Iterate over sets
14. **Implement ForDict** - Iterate over dicts
15. **Implement Break** - Exit loops
16. **Implement Continue** - Skip to next iteration

**After Phase 3:** Full loop support

---

### Phase 4: Remaining Features (0.5 days)
**Goal:** Complete feature parity

17. **Implement Error** - Throw errors
18. **Implement NewSet** - Create sets
19. **Implement UnwrapRecursive** - Recursive type unwrap
20. **Implement WrapRecursive** - Recursive type wrap

**After Phase 4:** 100% feature parity with TypeScript

---

## Testing Strategy

### Test Each Implementation

For each compiler implementation, add tests:

**File:** `tests/runtime/test_compiler.py`

Example for Call:
```python
def test_compile_call():
    """Test function call compilation."""
    loc = location("test", 1, 1)

    # Create function: (x) => x + 1
    param = ir_variable(IntegerType, "x", loc)
    add_one = ir_builtin(IntegerType, loc, "IntegerAdd", [], [
        param,
        ir_value(IntegerType, loc, 1)
    ])
    func = ir_function(
        FunctionType([IntegerType], IntegerType, []),
        loc, [], [param], add_one
    )

    # Call function: func(42)
    call = ir_call(IntegerType, loc, func, [ir_value(IntegerType, loc, 42)])

    # Compile and execute
    compiled = compile(call, [])
    result = compiled()

    assert result == 43
```

### Verify TypeScript Export Tests

After each phase, run:
```bash
.venv/bin/python -m pytest tests/test_typescript_exports.py -v
```

Track progress:
- Phase 1: Basic tests should pass
- Phase 2: Data structure tests should pass
- Phase 3: Loop tests should pass
- Phase 4: All tests should pass

---

## Success Criteria

Implementation is complete when:

1. ✅ All 19 missing IR node types are implemented
2. ✅ Each implementation has unit tests
3. ✅ All TypeScript export tests pass
4. ✅ No regressions in existing tests (1213 tests)
5. ✅ Code follows existing patterns in compiler.py
6. ✅ Both sync and async variants work correctly
7. ✅ Exception handling (Return, Break, Continue, Error) works
8. ✅ Cross-implementation compatibility verified

---

## Implementation Checklist

### Core Language (Phase 1)
- [ ] Call - Function calls
- [ ] As - Type casting
- [ ] Return - Early return
- [ ] Assign - Variable assignment

### Data Structures (Phase 2)
- [ ] Struct - Struct literals
- [ ] GetField - Struct field access
- [ ] Variant - Variant constructors
- [ ] Match - Pattern matching
- [ ] NewArray - Array literals
- [ ] NewDict - Dictionary literals

### Iteration (Phase 3)
- [ ] ForArray - Array loops
- [ ] ForSet - Set loops
- [ ] ForDict - Dictionary loops
- [ ] Break - Exit loops
- [ ] Continue - Skip iteration

### Remaining (Phase 4)
- [ ] Error - Throw exceptions
- [ ] NewSet - Set literals
- [ ] UnwrapRecursive - Unwrap recursive
- [ ] WrapRecursive - Wrap recursive

### Testing
- [ ] Unit tests for each implementation
- [ ] TypeScript export tests passing
- [ ] No test regressions
- [ ] Edge cases covered

---

## Notes

### Exception-Based Control Flow

The compiler uses exceptions for control flow:
- **ReturnException** - Early return from functions
- **BreakException** - Exit loops
- **ContinueException** - Skip to next iteration

These need to be caught in the appropriate places:
- Functions catch ReturnException
- Loops catch BreakException and ContinueException

### Async/Sync Variants

Every compilation function must handle both async and sync cases:
- Check `is_async_map[id(ir)]`
- Generate appropriate async/sync compiled functions
- Use `await` in async paths
- Propagate async correctly through nested expressions

### Context Handling

Runtime context (`ctx`) is a dictionary mapping variable names to values:
- Parent scopes pass context to children
- Loop variables add to child context
- Match case variables add to child context
- Let statements modify context

---

## Estimated Effort

**Total:** 3-4 days of focused work

- **Phase 1 (Critical):** 1-2 days
  - Call (4 hours)
  - As (0.5 hours)
  - Return (2 hours including function updates)
  - Assign (1 hour)

- **Phase 2 (Data):** 1 day
  - Struct, GetField, Variant (3 hours)
  - Match (4 hours - complex)
  - NewArray, NewDict (1 hour)

- **Phase 3 (Loops):** 1 day
  - ForArray, ForSet, ForDict (6 hours)
  - Break, Continue (1 hour)

- **Phase 4 (Remaining):** 0.5 days
  - Error, NewSet, UnwrapRecursive, WrapRecursive (3 hours)

**Testing:** Included in each phase

---

## References

- **TypeScript Compiler:** ../East/src/compile.ts
- **Python Compiler:** east/runtime/compiler.py
- **IR Types:** east/types/type_system.py (IRType definition)
- **IR Builders:** east/ir/builders.py
- **IR Analysis:** east/ir/analyze.py (already supports all IR types)

---

**Document Version:** 1.0
**Created:** 2025-11-12
**Status:** Ready for implementation
**Priority:** HIGH - Blocks cross-implementation testing
