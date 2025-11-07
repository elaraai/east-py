# East Python Runtime - Implementation TODO

This TODO list tracks the porting of the TypeScript East runtime to Python with **precise equivalence**.

## Organization

Detailed task breakdowns are organized into separate files in `docs/`:

- **[docs/TODO_TYPES.md](docs/TODO_TYPES.md)** - Type system, comparison functions, and default values
  _6 TypeScript files (types, comparison, default) → Python implementation and tests_

- **[docs/TODO_SERIALIZATION.md](docs/TODO_SERIALIZATION.md)** - Serialization formats and datetime formatting
  _14 TypeScript files (beast, east, json, datetime_format) → Python implementation and tests_

- **[docs/TODO_BUILTINS.md](docs/TODO_BUILTINS.md)** - 220 builtin function implementations
  _Current status: 160/220 (73%) implemented, 60 remaining_

- **[docs/TODO_ANALYZE.md](docs/TODO_ANALYZE.md)** - IR analysis and async propagation
  _Validate IR, propagate async metadata, track variable context_

---

## Quick Status

### Completed (Phase 1-2)
- [x] **Core type system** - Primitives, containers, structural types implemented
- [x] **Basic serialization** - East text format (print/parse) partially working
- [x] **JSON serialization** - Core implementation with 17 tests passing
- [x] **Fuzz testing** - Property-based testing infrastructure
- [x] **Type-specific equality** - equal_for() with cycle detection
- [x] **Type printing** - print_type() matching TypeScript exactly
- [x] **Most builtins** - 160/220 builtins implemented (73%)

### In Progress (Phase 2.5)
- [ ] **Type system operations** - TypeUnion, TypeIntersect, TypeEqual, etc.
- [ ] **Comparison functions** - is_for, less_for, greater_for, compare_for, etc.
- [ ] **Default values** - default_value() and minimal_value() functions
- [ ] **Complete JSON** - Remaining ~53 tests (Never, Function, frozen, refs, errors)
- [ ] **Complete East format** - Aliasing support, error cases, full validation
- [ ] **Complete builtins** - Remaining 60 builtins (regex, generators, higher-order functions)

### Not Started (Phase 3+)
- [ ] **IR Analysis** - analyze_ir() with async propagation and variable tracking
- [ ] **Beast serialization** - Binary format with byte-ordering preservation
- [ ] **DateTime formatting** - Tokenize, parse, print, validate datetime formats
- [ ] **Remaining builtins** - Collection generators, binary search, higher-order operations

---

## Testing Strategy

### Python Unit Tests
Python unit tests validate the runtime implementation itself:
- `tests/types/` - Type system operations
- `tests/utils/` - Comparison and default functions
- `tests/serialization/` - All serialization formats
- `tests/builtins/` - Builtin function behavior

### East Compliance Tests (Future)
Once the IR interpreter is complete, East language tests will be compiled to IR and executed:
- `/home/crambelsoupy/src/East/test/*.spec.ts` - East language test suites
- These test the runtime's correctness from the language perspective
- **No porting required** - executed as compiled IR

---

## File Mappings

### TypeScript → Python

**Core Type System:**
- `East/src/types.ts` → `east/types/type_system.py`
- `East/src/comparison.ts` → `east/utils/ordering.py`
- `East/src/default.ts` → `east/utils/default.py` (TO BE CREATED)

**IR Analysis:**
- `East/src/analyze.ts` → `east/ir/analyze.py` (TO BE CREATED)

**Serialization:**
- `East/src/serialization/beast.ts` → `east/serialization/beast.py` (TO BE CREATED)
- `East/src/serialization/east.ts` → `east/serialization/east_printer.py` + `east_parser.py` (EXISTS - partial)
- `East/src/serialization/json.ts` → `east/serialization/json.py` (EXISTS - partial, 17/70 tests)

**DateTime Format:**
- `East/src/datetime_format/*.ts` → `east/serialization/datetime_format/*.py` (TO BE CREATED)

**Builtins:**
- `East/src/builtins.ts` → `east/builtins/*.py` (EXISTS - 160/220 implemented)

---

## Implementation Principles

1. **Precise Equivalence** - Python runtime must match TypeScript behavior exactly
2. **Test-Driven** - Port tests alongside implementation, validate with fuzz testing
3. **Type Safety** - Use Python type hints throughout
4. **Performance** - Optimize for common cases, use native Python structures where appropriate
5. **Maintainability** - Clear code organization, comprehensive documentation

---

## Progress Tracking

See individual TODO files in `docs/` for detailed checkbox tracking of:
- Functions to implement
- Test suites to port
- Edge cases to handle
- Integration points

**Total Scope**: ~13,823 lines of TypeScript across 22 files to port with precise equivalence.
