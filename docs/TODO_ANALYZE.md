# IR Analysis & Async Propagation

**Source Files (TypeScript)**:
- /home/crambelsoupy/src/East/src/analyze.ts (1,542 lines)
- /home/crambelsoupy/src/East/src/analyze.spec.ts (~500 lines)

**Target Files (Python)**:
- east/ir/analyze.py (TO BE CREATED)
- tests/ir/test_analyze.py (TO BE CREATED - directory needs creation)

---

## analyze.ts → east/ir/analyze.py (TO BE CREATED)

### Type Definitions (4 types)
- [ ] Port PlatformDefinition type/class
  - [ ] name: str
  - [ ] inputs: list[EastType]
  - [ ] output: EastType
  - [ ] is_async: bool
  - [ ] implementation: Callable
- [ ] Port AnalyzedIR generic type
  - [ ] Adds is_async: bool field to all IR node types
  - [ ] Creates analyzed versions: AnalyzedValue, AnalyzedVariable, etc.
- [ ] Port VariableMetadata type/class
  - [ ] type: EastType
  - [ ] mutable: bool
  - [ ] defined_by: IR node
  - [ ] captured: bool
- [ ] Port VariableContext type
  - [ ] dict[str, VariableMetadata]

### Main Function (1 function)
- [ ] Port analyze_ir(ir: IR, platform_defs: list[PlatformDefinition], ctx: VariableContext | None = None) -> AnalyzedIR
  - [ ] Validate IR tree structure
  - [ ] Propagate is_async metadata through all IR nodes:
    - [ ] Value: always sync (is_async=False)
    - [ ] Variable: always sync (is_async=False)
    - [ ] Block: async if any statement is async
    - [ ] IfElse: async if predicate, if_body, or else_body is async
    - [ ] While: async if predicate or body is async
    - [ ] Break/Continue/Return: propagate from child
    - [ ] Let: async if value expression is async
    - [ ] Assign: async if value expression is async
    - [ ] NewArray/NewSet/NewDict: async if any element expression is async
    - [ ] ForArray/ForSet/ForDict: async if collection or body is async
    - [ ] Struct: async if any field value is async
    - [ ] GetField: async if struct expression is async
    - [ ] Variant: async if value expression is async
    - [ ] Match: async if value or any case body is async
    - [ ] Function: track async in body, function itself is sync
    - [ ] Call: async if function resolves to async OR any argument is async
    - [ ] Platform: async if platform function is_async=True OR any argument is async
    - [ ] Builtin: always sync (is_async=False)
    - [ ] Error: async if message expression is async
    - [ ] TryCatch: async if try_body or catch_body is async
    - [ ] As: async if value expression is async
    - [ ] UnwrapRecursive/WrapRecursive: async if value expression is async
  - [ ] Build variable context tracking
    - [ ] Track variable definitions (Let nodes)
    - [ ] Track variable mutations (Assign nodes)
    - [ ] Track variable captures (in closures)
    - [ ] Validate variable references exist
  - [ ] Validate platform function existence
    - [ ] Check platform function names against platform_defs
    - [ ] Throw error for unknown platform functions
  - [ ] Return enriched IR with is_async metadata on all nodes

---

## analyze.spec.ts → tests/ir/test_analyze.py (TO BE CREATED)

**Note**: These are Python unit tests. East backend compliance tests will be executed as compiled IR.

### Test Suite: Basic validation (1 test)
- [ ] should accept valid Value IR node

### Test Suite: is_async propagation (17 tests)
- [ ] Value expressions should be synchronous (is_async=False)
- [ ] Async platform function call should be async (is_async=True)
- [ ] Sync platform function call should be synchronous (is_async=False)
- [ ] Platform function with async argument should be async
- [ ] Let with async value should be async
- [ ] Block with async statement should be async
- [ ] Block with only sync statements should be sync
- [ ] IfElse with async predicate should be async
- [ ] IfElse with async if_body should be async
- [ ] IfElse with async else_body should be async
- [ ] IfElse with all sync branches should be sync
- [ ] While with async predicate should be async
- [ ] While with async body should be async
- [ ] ForEach with async body should be async
- [ ] Call with async function body should be async
- [ ] Function node should track async in body
- [ ] Nested async propagation through multiple levels

### Test Suite: Error cases (1 test)
- [ ] should reject unknown platform function name

### Test Suite: compile/compileAsync integration (4 tests)
- [ ] compile() should throw when IR contains async operations
- [ ] compileAsync() should throw when IR contains no async operations
- [ ] compile() should succeed with only sync platform functions
- [ ] compileAsync() should succeed with async platform functions

---

## Integration Notes

The analyze_ir function is used to:
1. **Validate IR** before execution
2. **Propagate async metadata** to determine if sync or async interpreter needed
3. **Track variable context** for optimization and validation
4. **Validate platform functions** exist before execution

This should integrate with:
- **east/runtime/interpreter.py**: Use analyzed IR for execution
- **east/builtins/registry.py**: Validate builtin references
- Future compiler/optimizer that may use variable context metadata
