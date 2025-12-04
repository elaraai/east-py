# Design Document: AsyncFunction Support for east-py

This document details the changes required to adapt east-py to support the new `AsyncFunction` type and related changes from the TypeScript East repository (commits `8c728172b97a9a420540b4f9ddb1dd4a98b981ad` to `26d32827efc9dbc501a7445a008d9f150cb063d2`).

## Summary of TypeScript Changes

### 1. New `AsyncFunctionType` Added

A new type variant `AsyncFunction` was added alongside the existing `Function` type:

```typescript
// TypeScript
type AsyncFunctionType<I, O> = { type: "AsyncFunction", inputs: I, output: O };
```

### 2. `FunctionType` Simplified (BREAKING CHANGE)

The `platforms` field was **removed** from `FunctionType`:

```typescript
// OLD
type FunctionType<I, O> = { type: "Function", inputs: I, output: O, platforms: string[] | null };

// NEW
type FunctionType<I, O> = { type: "Function", inputs: I, output: O };
```

**Rationale**: Async behavior is now explicitly encoded in the type system (AsyncFunction vs Function) rather than tracked via platform function references.

### 3. New IR Nodes

Two new IR nodes were added:

- **`AsyncFunction`**: Same structure as `Function` (`type`, `location`, `captures`, `parameters`, `body`)
- **`CallAsync`**: Same structure as `Call` (`type`, `location`, `function`, `arguments`) - explicitly awaits the function result

### 4. `Platform` IR Node Updated (BREAKING CHANGE)

The `Platform` IR node now includes an `async: boolean` field:

```typescript
// OLD
type PlatformIR = variant<"Platform", { type, location, name, arguments }>;

// NEW
type PlatformIR = variant<"Platform", { type, location, name, arguments, async: boolean }>;
```

### 5. Subtyping Rule: Function <: AsyncFunction

Synchronous functions are now subtypes of async functions with matching signatures:

```
Function<[A], B> <: AsyncFunction<[A], B>
```

This allows a sync function to be used anywhere an async function is expected.

### 6. Type Union/Intersection Updated

- **Union of Function + AsyncFunction = AsyncFunction**
- **Intersection of Function + AsyncFunction = Function** (more specific)

### 7. Data Type Validation Relaxed

The following type constructors no longer validate that inner types are data types at construction time:
- `RefType`
- `ArrayType`
- `DictType` (still validates keys are immutable)
- `StructType`
- `VariantType`
- `RecursiveType`

### 8. Platform Definition API Changes

- `East.platform()` now creates sync platforms
- New `East.asyncPlatform()` creates async platforms
- `.implementAsync()` is removed - both use `.implement()`

---

## Required Changes for east-py

### Phase 1: Type System (`east/types/types.py`)

#### 1.1 Add `AsyncFunctionTypeDef` and `AsyncFunctionTypeValue`

```python
class AsyncFunctionTypeDef(TypedDict):
    """Async Function type."""
    type: Literal["AsyncFunction"]
    value: AsyncFunctionTypeValue

class AsyncFunctionTypeValue(TypedDict):
    """Async Function type value (inputs, output)."""
    inputs: list[EastType]
    output: EastType
```

#### 1.2 Remove `platforms` from `FunctionTypeValue`

```python
# OLD
class FunctionTypeValue(TypedDict):
    inputs: list[EastType]
    output: EastType
    platforms: list[str]  # REMOVE THIS

# NEW
class FunctionTypeValue(TypedDict):
    inputs: list[EastType]
    output: EastType
```

#### 1.3 Update `FunctionType()` Constructor

```python
# OLD
def FunctionType(inputs: list[EastType], output: EastType, platforms: list[str]) -> EastVariant[FunctionTypeValue]:
    return EastVariant("Function", {"inputs": inputs, "output": output, "platforms": platforms})

# NEW
def FunctionType(inputs: list[EastType], output: EastType) -> EastVariant[FunctionTypeValue]:
    return EastVariant("Function", {"inputs": inputs, "output": output})
```

#### 1.4 Add `AsyncFunctionType()` Constructor

```python
def AsyncFunctionType(inputs: list[EastType], output: EastType) -> EastVariant[AsyncFunctionTypeValue]:
    """Create an async function type."""
    return EastVariant("AsyncFunction", {"inputs": inputs, "output": output})
```

#### 1.5 Add `is_async_function_type()` Type Guard

```python
def is_async_function_type(typ: EastType) -> TypeGuard[EastVariant[AsyncFunctionTypeValue]]:
    """Check if a type is an AsyncFunction type."""
    return typ.type == "AsyncFunction"
```

#### 1.6 Update `is_data_type()` to Handle AsyncFunction

```python
def is_data_type(typ: EastType, recursive_type: EastType | None = None) -> bool:
    # ... existing code ...
    if is_function_type(typ):
        return False
    if is_async_function_type(typ):  # ADD THIS
        return False
    # ... rest of function ...
```

#### 1.7 Update `is_immutable_type()` to Handle AsyncFunction

```python
def is_immutable_type(typ: EastType, recursive_type: EastType | None = None) -> bool:
    if (
        is_array_type(typ)
        or is_set_type(typ)
        or is_dict_type(typ)
        or is_ref_type(typ)
        or is_function_type(typ)
        or is_async_function_type(typ)  # ADD THIS
    ):
        return False
    # ... rest of function ...
```

#### 1.8 Update `is_subtype()` for Function <: AsyncFunction

```python
def is_subtype(t1: EastType, t2: EastType) -> bool:
    # ... existing code ...

    # Handle Function types
    if is_function_type(t1):
        # Function <: Function OR Function <: AsyncFunction
        if is_function_type(t2) or is_async_function_type(t2):
            inputs1 = t1.value["inputs"]
            inputs2 = t2.value["inputs"]
            if len(inputs1) != len(inputs2):
                return False
            # Contravariant inputs
            for i1, i2 in zip(inputs1, inputs2, strict=False):
                if not is_subtype(i2, i1):
                    return False
            # Covariant output
            return is_subtype(t1.value["output"], t2.value["output"])
        return False

    # Handle AsyncFunction types
    if is_async_function_type(t1):
        # AsyncFunction <: AsyncFunction only (not to sync Function)
        if is_async_function_type(t2):
            inputs1 = t1.value["inputs"]
            inputs2 = t2.value["inputs"]
            if len(inputs1) != len(inputs2):
                return False
            for i1, i2 in zip(inputs1, inputs2, strict=False):
                if not is_subtype(i2, i1):
                    return False
            return is_subtype(t1.value["output"], t2.value["output"])
        return False
    # ... rest of function ...
```

#### 1.9 Update `type_equal()` for AsyncFunction

Add a new case to handle `AsyncFunction` type equality checking.

#### 1.10 Update `type_union()` and `type_intersect()`

- **Union**: `Function + AsyncFunction = AsyncFunction`
- **Intersect**: `Function + AsyncFunction = Function`

#### 1.11 Update `recursive_type()` to Handle AsyncFunction

```python
# In replace_markers() function:
if is_async_function_type(t):
    new_inputs = [replace_markers(inp, stack_depth) for inp in t.value["inputs"]]
    new_output = replace_markers(t.value["output"], stack_depth)
    return EastVariant("AsyncFunction", {"inputs": new_inputs, "output": new_output})
```

#### 1.12 Relax Data Type Validation in Constructors

Remove the `is_data_type()` checks from:
- `ArrayType()` - Remove the validation (keep the function)
- `RefType()` - Remove the validation
- `DictType()` - Keep key immutability check, remove value data type check
- `StructType()` - Remove field data type checks
- `VariantType()` - Remove case data type checks

---

### Phase 2: IR Types (`east/types/ir.py` or `east/types/type_of_type.py`)

#### 2.1 Update `EastTypeType` to Include AsyncFunction

```python
EastTypeType = RecursiveType(type => VariantType({
    # ... existing cases ...
    "Function": StructType({ "inputs": ArrayType(type), "output": type }),
    "AsyncFunction": StructType({ "inputs": ArrayType(type), "output": type }),  # ADD
}))
```

#### 2.2 Update `IRType` to Include AsyncFunction and CallAsync

```python
IRType = RecursiveType(ir => VariantType({
    # ... existing cases ...
    "Function": StructType({ "type": EastTypeType, "location": LocationType, "captures": ArrayType(ir), "parameters": ArrayType(ir), "body": ir }),
    "AsyncFunction": StructType({ "type": EastTypeType, "location": LocationType, "captures": ArrayType(ir), "parameters": ArrayType(ir), "body": ir }),  # ADD
    "Call": StructType({ "type": EastTypeType, "location": LocationType, "function": ir, "arguments": ArrayType(ir) }),
    "CallAsync": StructType({ "type": EastTypeType, "location": LocationType, "function": ir, "arguments": ArrayType(ir) }),  # ADD
    "Platform": StructType({ "type": EastTypeType, "location": LocationType, "name": StringType, "arguments": ArrayType(ir), "async": BooleanType }),  # ADD async field
}))
```

#### 2.3 Add TypedDict Definitions for New IR Nodes

```python
class AsyncFunctionIRValue(TypedDict):
    """AsyncFunction IR node value."""
    type: EastTypeValue
    location: LocationValue
    captures: list[IR]
    parameters: list[IR]
    body: IR

class CallAsyncIRValue(TypedDict):
    """CallAsync IR node value."""
    type: EastTypeValue
    location: LocationValue
    function: IR
    arguments: list[IR]
```

#### 2.4 Update `PlatformIRValue` to Include `async` Field

```python
class PlatformIRValue(TypedDict):
    """Platform IR node value."""
    type: EastTypeValue
    location: LocationValue
    name: str
    arguments: list[IR]
    async_: bool  # Use async_ to avoid keyword conflict, or "async" with quotes
```

---

### Phase 3: IR Builders (`east/ir/builders.py`)

#### 3.1 Add `ir_async_function()` Builder

```python
def ir_async_function(
    typ: EastTypeValue,
    loc: LocationValue,
    captures: list[IR],
    parameters: list[IR],
    body: IR,
) -> IR:
    """Create an AsyncFunction IR node."""
    captures_array = EastArray(IRType, captures)
    params_array = EastArray(IRType, parameters)

    function_struct: AsyncFunctionIRValue = {
        "type": typ,
        "location": loc,
        "captures": captures_array,
        "parameters": params_array,
        "body": body,
    }
    return EastVariant("AsyncFunction", function_struct)
```

#### 3.2 Add `ir_call_async()` Builder

```python
def ir_call_async(
    typ: EastTypeValue,
    loc: LocationValue,
    function: IR,
    arguments: list[IR],
) -> IR:
    """Create a CallAsync IR node (awaits the function result)."""
    args_array = EastArray(IRType, arguments)

    call_struct: CallAsyncIRValue = {
        "type": typ,
        "location": loc,
        "function": function,
        "arguments": args_array,
    }
    return EastVariant("CallAsync", call_struct)
```

#### 3.3 Update `ir_platform()` to Include `async_` Parameter

```python
def ir_platform(
    typ: EastTypeValue,
    loc: LocationValue,
    platform_name: str,
    arguments: list[IR],
    async_: bool = False,  # ADD THIS PARAMETER
) -> IR:
    """Create a Platform IR node."""
    args_array = EastArray(IRType, arguments)

    platform_struct: PlatformIRValue = {
        "type": typ,
        "location": loc,
        "name": platform_name,
        "arguments": args_array,
        "async": async_,  # ADD THIS
    }
    return EastVariant("Platform", platform_struct)
```

---

### Phase 4: Compiler (`east/runtime/compiler.py`)

#### 4.1 Add Handler for `AsyncFunction` IR Node

```python
def _compile_ir(ir: IR, platform_fns, async_platform_fns) -> tuple[Callable, bool]:
    tag = ir["type"]

    # ... existing cases ...
    if tag == "AsyncFunction":
        return _compile_async_function(ir, platform_fns, async_platform_fns)
    if tag == "CallAsync":
        return _compile_call_async(ir, platform_fns, async_platform_fns)
    # ...
```

#### 4.2 Implement `_compile_async_function()`

```python
def _compile_async_function(
    node: IR,
    platform_fns: dict[str, Callable[..., Any]],
    async_platform_fns: set[str],
) -> tuple[Callable, bool]:
    """Compile an AsyncFunction IR node to a Python async callable."""
    func_struct = node["value"]

    # Compile the body - async functions always produce async bodies
    body_compiled, _ = _compile_ir(func_struct["body"], platform_fns, async_platform_fns)

    param_names = [param["value"]["name"] for param in func_struct["parameters"]]
    capture_names = [cap["value"]["name"] for cap in func_struct["captures"]]

    # AsyncFunction always creates an async callable
    def make_async_fn(parent_env):
        async def compiled_fn_async(*args):
            if len(args) != len(param_names):
                raise TypeError(f"Function expects {len(param_names)} arguments, got {len(args)}")

            local_env = dict(zip(param_names, args, strict=False))

            if capture_names:
                env = CaptureAwareEnv(local_env, parent_env, capture_names)
            else:
                env = local_env

            try:
                result = body_compiled(env)
                # Always await in async function - body may return a coroutine
                if hasattr(result, '__await__'):
                    result = await result
                return result
            except ReturnException as e:
                return e.value

        return compiled_fn_async

    # AsyncFunction creation itself is NOT async (isAsync: false in TS)
    return FunctionFactory(make_async_fn), False
```

#### 4.3 Implement `_compile_call_async()`

```python
def _compile_call_async(
    node: IR,
    platform_fns: dict[str, Callable[..., Any]],
    async_platform_fns: set[str],
) -> tuple[Callable, bool]:
    """Compile a CallAsync IR node - always awaits the function result."""
    call_struct = node["value"]

    fn_compiled, _ = _compile_ir(call_struct["function"], platform_fns, async_platform_fns)
    args_compiled = [_compile_ir(arg, platform_fns, async_platform_fns)[0]
                     for arg in call_struct["arguments"]]

    async def call_async_exec(env):
        # Get the function (may be a FunctionFactory)
        fn = fn_compiled(env)
        if isinstance(fn, FunctionFactory):
            fn = fn.make(env)

        # Evaluate and await all arguments
        args = []
        for arg_fn in args_compiled:
            arg_val = arg_fn(env)
            if hasattr(arg_val, '__await__'):
                arg_val = await arg_val
            args.append(arg_val)

        # Call the async function and await the result
        result = fn(*args)
        if hasattr(result, '__await__'):
            result = await result
        return result

    # CallAsync is always async
    return call_async_exec, True
```

#### 4.4 Update `_compile_platform()` to Use `async` Field

```python
def _compile_platform(
    node: IR,
    platform_fns: dict[str, Callable[..., Any]],
    async_platform_fns: set[str],
) -> tuple[Callable, bool]:
    """Compile a Platform IR node."""
    platform_struct = node["value"]
    platform_name = platform_struct["name"]
    is_async_platform = platform_struct.get("async", False)  # USE IR FIELD, not set lookup

    # ... rest of implementation ...

    return compiled_fn, is_async_platform
```

#### 4.5 Update `_compile_call()` for Sync-Only Behavior

With the new design, `Call` only calls sync functions:

```python
def _compile_call(
    node: IR,
    platform_fns: dict[str, Callable[..., Any]],
    async_platform_fns: set[str],
) -> tuple[Callable, bool]:
    """Compile a Call IR node - for sync function calls only."""
    # ... compile function and arguments ...

    # Determine if any argument is async (arguments may still be async even for sync function)
    any_arg_async = any(arg_async for _, arg_async in args_compiled_with_async)

    if any_arg_async:
        async def call_exec_async(env):
            fn = fn_compiled(env)
            if isinstance(fn, FunctionFactory):
                fn = fn.make(env)

            args = []
            for arg_fn, _ in args_compiled_with_async:
                arg_val = arg_fn(env)
                if hasattr(arg_val, '__await__'):
                    arg_val = await arg_val
                args.append(arg_val)

            # Call sync function (no await on result)
            return fn(*args)

        return call_exec_async, True
    else:
        def call_exec(env):
            fn = fn_compiled(env)
            if isinstance(fn, FunctionFactory):
                fn = fn.make(env)
            args = [arg_fn(env) for arg_fn, _ in args_compiled_with_async]
            return fn(*args)

        return call_exec, False
```

---

### Phase 5: Serialization (`east/serialization/`)

#### 5.1 Update `east_parser.py`

Add parsing support for:
- `AsyncFunction` type: `.AsyncFunction (inputs=[...], output=...)`
- `AsyncFunction` IR node
- `CallAsync` IR node
- `Platform` IR node with `async` field

#### 5.2 Update `east_printer.py`

Update `print_type()` to handle AsyncFunction:

```python
def print_type(typ: EastType) -> str:
    # ... existing code ...
    if is_async_function_type(typ):
        inputs_str = ", ".join(print_type(i) for i in typ.value["inputs"])
        output_str = print_type(typ.value["output"])
        return f".AsyncFunction (inputs=[{inputs_str}], output={output_str})"
    # ... rest of function ...
```

#### 5.3 Update `beast2.py` and `json.py`

Add serialization/deserialization for new type and IR variants.

---

### Phase 6: IR Analysis (`east/ir/analyze.py`)

#### 6.1 Add Handler for `AsyncFunction` Node

```python
def analyze_ir(ir: IR, platform_defs: list[PlatformDefinition], ctx: VariableContext) -> AnalyzedIR:
    # ... existing code ...

    if node_type == "AsyncFunction":
        # Validate type is AsyncFunction
        if node.value["type"].type != "AsyncFunction":
            raise AnalysisError(f"Expected AsyncFunction type")

        # Create new context for function body
        fn_ctx = {}

        # Add captured variables (validate they're in scope)
        for capture_var in node.value["captures"]:
            # ... same validation as Function ...

        # Add parameters to context
        for param in node.value["parameters"]:
            # ... same as Function ...

        # Visit function body
        body_analyzed = visit(node.value["body"], fn_ctx, expected_output)

        # Creating a function is sync (isAsync: false)
        return AnalyzedIR(node, body=body_analyzed, is_async=False)
```

#### 6.2 Add Handler for `CallAsync` Node

```python
if node_type == "CallAsync":
    # CallAsync is always async
    is_async = True

    # Visit function expression
    fn_analyzed = visit(node.value["function"], ctx)

    # Validate it's an AsyncFunction type
    if fn_analyzed.value["type"].type != "AsyncFunction":
        raise AnalysisError(f"CallAsync expects AsyncFunction type")

    # Analyze arguments
    args_analyzed = []
    for arg in node.value["arguments"]:
        arg_analyzed = visit(arg, ctx)
        args_analyzed.append(arg_analyzed)

    return AnalyzedIR(node, function=fn_analyzed, arguments=args_analyzed, is_async=True)
```

#### 6.3 Update `Call` Handler

Remove platform-based async detection. `Call` is now only async if its arguments are async:

```python
if node_type == "Call":
    fn_analyzed = visit(node.value["function"], ctx)

    # Validate it's a Function type (not AsyncFunction - use CallAsync for that)
    if fn_analyzed.value["type"].type != "Function":
        raise AnalysisError(f"Call expects Function type")

    is_async = False
    args_analyzed = []
    for arg in node.value["arguments"]:
        arg_analyzed = visit(arg, ctx)
        args_analyzed.append(arg_analyzed)
        if arg_analyzed.is_async:
            is_async = True

    return AnalyzedIR(node, function=fn_analyzed, arguments=args_analyzed, is_async=is_async)
```

#### 6.4 Update `Builtin` Handler

Remove platform-based async propagation for function arguments. Builtins now only accept sync functions:

```python
if node_type == "Builtin":
    # Builtins are synchronous and only accept sync function arguments
    is_async = False

    args_analyzed = []
    for arg in node.value["arguments"]:
        arg_analyzed = visit(arg, ctx)
        args_analyzed.append(arg_analyzed)
        if arg_analyzed.is_async:
            is_async = True
        # NOTE: Removed check for function argument async platform calls

    return AnalyzedIR(node, arguments=args_analyzed, is_async=is_async)
```

---

### Phase 7: Platform Functions (`east/runtime/platform.py`)

#### 7.1 Update `PlatformFunction` TypedDict

The current implementation may already support this via the `type` field:

```python
class PlatformFunction(TypedDict):
    name: str
    type: Literal["sync", "async"]
    fn: Callable[..., Any]
```

No changes needed if this is already the structure. If `platforms` was stored elsewhere (like in FunctionType), remove that.

---

### Phase 8: Platform Packages (`east-py-std`, `east-py-io`)

#### 8.1 Update Platform Definitions

Platform functions should now be explicitly marked as sync or async based on their behavior:

```python
# east-py-std example
log_platform = platform("log", [StringType], NullType, is_async=False)
fetch_platform = async_platform("fetch", [StringType], StringType)  # New helper

# east-py-io example (all async)
s3_read_platform = async_platform("s3.read", [StringType], BlobType)
postgres_query_platform = async_platform("postgres.query", [StringType], ArrayType(DictType(...)))
```

Consider adding helper functions:

```python
def platform(name: str, inputs: list[EastType], output: EastType) -> PlatformFunction:
    """Create a sync platform function definition."""
    return {"name": name, "type": "sync", "fn": None}  # fn set via .implement()

def async_platform(name: str, inputs: list[EastType], output: EastType) -> PlatformFunction:
    """Create an async platform function definition."""
    return {"name": name, "type": "async", "fn": None}
```

---

## Migration Guide

### Breaking Changes

1. **`FunctionType()` signature changed**: Remove the `platforms` parameter
   ```python
   # OLD
   FunctionType([IntegerType], IntegerType, ["log", "fetch"])

   # NEW
   FunctionType([IntegerType], IntegerType)
   ```

2. **`ir_platform()` requires `async_` parameter**: Add boolean for async behavior
   ```python
   # OLD
   ir_platform(typ, loc, "fetch", args)

   # NEW
   ir_platform(typ, loc, "fetch", args, async_=True)
   ```

3. **Platform IR deserialization**: Existing serialized IR with `Platform` nodes will fail to parse (missing `async` field). Provide a migration tool or version the format.

### Migration Steps

1. Update all `FunctionType()` calls to remove `platforms` argument
2. Update all `ir_platform()` calls to add `async_` parameter
3. Update serialization format version
4. Re-export any persisted IR with new format
5. Update east-py-std and east-py-io platform definitions

---

## Testing Strategy

### New Test Cases Required

1. **Type System**
   - `AsyncFunctionType` construction and equality
   - `is_async_function_type()` predicate
   - `Function <: AsyncFunction` subtyping
   - `Union(Function, AsyncFunction) == AsyncFunction`
   - `Intersect(Function, AsyncFunction) == Function`

2. **IR Builders**
   - `ir_async_function()` creates valid IR
   - `ir_call_async()` creates valid IR
   - `ir_platform()` with `async_=True/False`

3. **Compiler**
   - `AsyncFunction` compiles to async callable
   - `CallAsync` awaits function result
   - `Call` does NOT await function result
   - `Platform` with `async=true` is awaited

4. **Serialization**
   - Round-trip `AsyncFunctionType`
   - Round-trip `AsyncFunction` IR
   - Round-trip `CallAsync` IR
   - Round-trip `Platform` IR with `async` field

5. **Integration**
   - Sync function passed where AsyncFunction expected (subtyping)
   - Async platform function called from async context
   - Error when async platform called from sync context

---

## Implementation Order

Recommended implementation order to minimize conflicts:

1. **Phase 1**: Type System updates (most foundational)
2. **Phase 2**: IR Type definitions
3. **Phase 3**: IR Builders
4. **Phase 5**: Serialization (enables testing)
5. **Phase 6**: IR Analysis
6. **Phase 4**: Compiler
7. **Phase 7-8**: Platform packages

Each phase should be a separate commit with tests.

---

## Appendix: Full Diff Summary

### Files Changed in TypeScript

| File | Changes |
|------|---------|
| `src/types.ts` | Added AsyncFunctionType, removed platforms from FunctionType, updated subtyping |
| `src/ir.ts` | Added AsyncFunctionIR, CallAsyncIR, Platform.async field |
| `src/type_of_type.ts` | Added AsyncFunction variant to EastTypeType |
| `src/compile.ts` | Added compilation for AsyncFunction, CallAsync; simplified Function/Call |
| `src/analyze.ts` | Added analysis for AsyncFunction, CallAsync; removed platform tracking |
| `src/ast_to_ir.ts` | Added context.async, AsyncFunction conversion, CallAsync generation |
| `src/builtins.ts` | Removed platforms parameter from FunctionType calls |
| `src/expr/index.ts` | Added asyncFunction, asyncPlatform exports |
| `src/expr/asyncfunction.ts` | New file for AsyncFunctionExpr |
| `src/expr/block.ts` | Added async context tracking |

### Estimated Lines of Change in Python

| Module | Estimated LOC |
|--------|---------------|
| `east/types/types.py` | ~200 |
| `east/types/ir.py` or `type_of_type.py` | ~50 |
| `east/ir/builders.py` | ~60 |
| `east/runtime/compiler.py` | ~150 |
| `east/ir/analyze.py` | ~100 |
| `east/serialization/east_parser.py` | ~50 |
| `east/serialization/east_printer.py` | ~20 |
| `east/serialization/json.py` | ~30 |
| `east/serialization/beast2.py` | ~30 |
| Tests | ~300 |
| **Total** | **~990** |
