# East.py

Python runtime for the [East programming language](https://github.com/elara-ai/East).

## Overview

East.py is a Python backend that enables East IR to be compiled and executed in Python environments. It provides:

- **Complete type system** - Full representation of all East types (primitives, containers, structs, variants, functions)
- **IR compiler** - Compiles East IR nodes to executable Python functions
- **212+ builtin functions** - Array, Set, Dict, String, DateTime, Blob, Integer, Float operations
- **Serialization** - East text format, JSON, and BEAST (Binary East) support
- **DateTime formatting** - Custom datetime parsing and printing with format strings
- **Type analysis** - IR validation and type inference
- **Platform integration** - Embed East functions in Python applications

## Current Status

✅ **Fully Implemented:**
- Type system (all East types)
- IR builders and compiler
- 212 builtin functions (100% coverage)
- Serialization (East text, JSON, BEAST)
- DateTime formatting (parse/print)
- Type comparison and equality
- Container types (Array, Set, Dict)
- Platform integration API

**Test Coverage:** 980 tests passing, 84% code coverage

## Installation

```bash
# Install from source
git clone https://github.com/elara-ai/east-py
cd east-py
pip install -e .
```

## Quick Start

### Working with East Types

```python
from east.types.type_system import IntegerType, StringType, StructType
from east.types.containers import EastArray
from east.serialization.east_parser import parse_east
from east.serialization.east_printer import print_for

# Create a struct type
PersonType = StructType([("name", StringType), ("age", IntegerType)])

# Parse East text format
person = parse_east(PersonType, '(name="Alice", age=30)')

# Print back to East text format
printer = print_for(PersonType)
text = printer(person)
print(text)  # (name="Alice", age=30)

# Work with containers
arr = EastArray(IntegerType, [1, 2, 3, 4, 5])
```

### Using Builtin Functions

```python
from east.builtins.registry import get_builtin
from east.types.type_system import IntegerType
from east.types.containers import EastArray

# Get a builtin function
array_map = get_builtin("ArrayMap")

# Use it (note: builtins require type parameters)
arr = EastArray(IntegerType, [1, 2, 3])
doubled = array_map(arr, lambda x: x * 2, IntegerType, IntegerType)
print(list(doubled))  # [2, 4, 6]
```

### Compiling and Executing IR

```python
from east.runtime.compiler import compile
from east.ir.builders import ir_value, location
from east.types.type_system import IntegerType

# Create a simple IR node
loc = location("example.east", 1, 1)
ir_node = ir_value(IntegerType, loc, 42)

# Compile to Python function
compiled_fn = compile(ir_node)

# Execute (with empty environment)
result = compiled_fn({})
print(result)  # 42
```

### JSON Serialization

```python
import json
from east.serialization.json import to_json_for, from_json_for
from east.types.type_system import IntegerType, StringType, StructType

# Create a struct type
PersonType = StructType([("name", StringType), ("age", IntegerType)])

# Encode to JSON-compatible dict
encoder = to_json_for(PersonType)
json_dict = encoder({"name": "Bob", "age": 25})
json_str = json.dumps(json_dict)
print(json_str)  # {"name": "Bob", "age": "25"}

# Decode from JSON string
decoder = from_json_for(PersonType)
person = decoder(json.loads(json_str))
print(person)  # (name='Bob', age=25)
```

## Development

```bash
# First-time setup (installs dependencies and pre-commit hooks)
make install

# Development workflow
make test          # Run test suite
make lint          # Run linter (ruff)
make format        # Format code
make typecheck     # Type check with mypy
make check         # Run all checks (lint + typecheck + test)

# Other useful commands
make repl          # Start Python REPL with east loaded
make coverage      # Generate HTML coverage report
make lint-fix      # Auto-fix linting issues
make clean         # Clean build artifacts

# Run specific test suites (using uv)
uv run pytest tests/builtins/test_builtins.py -v
uv run pytest tests/serialization/test_json.py -v
uv run pytest tests/types/test_types.py -v
```

## Architecture

### Module Structure

- `east/types/` - Type system implementation
  - `type_system.py` - Core type definitions and constructors
  - `primitives.py` - Null, Boolean, Integer, Float, String, Blob, DateTime
  - `containers.py` - Array, Set, Dict implementations
  - `structural.py` - Struct, Variant, Function types

- `east/builtins/` - Builtin function implementations
  - `array.py` - Array operations (map, filter, reduce, sort, search, etc.)
  - `set_ops.py` - Set operations (union, intersection, map, reduce, etc.)
  - `dict_ops.py` - Dict operations (map, filter, merge, etc.)
  - `string.py` - String operations (split, join, regex, JSON, etc.)
  - `datetime_ops.py` - DateTime operations (add, diff, compare, etc.)
  - `comparison.py` - Comparison and equality functions

- `east/serialization/` - Serialization formats
  - `east_parser.py` - Parse East text format
  - `east_printer.py` - Print East text format
  - `json.py` - JSON encoding/decoding
  - `beast.py` - Binary East format

- `east/ir/` - Intermediate representation
  - `builders.py` - Helper functions for building IR nodes
  - `analyze.py` - Type checking and validation

- `east/runtime/` - Execution engine
  - `compiler.py` - Compile IR to Python functions
  - `platform.py` - Platform integration API

- `east/datetime_format/` - DateTime formatting
  - `parse.py` - Parse datetime strings with format
  - `print.py` - Print datetime with format
  - `tokenize.py` - Format string tokenization


## License

Proprietary - See [LICENSE.md](LICENSE.md) for details.

## Related Projects

- [East](https://github.com/elara-ai/East) - TypeScript frontend and reference implementation
- [East.jl](https://github.com/elara-ai/East.jl) - Julia backend with native code compilation
- [Elara](https://elara.ai) - Real-time analytics platform using East
