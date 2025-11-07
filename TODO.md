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
