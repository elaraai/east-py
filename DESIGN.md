# East.py Design and Requirements Document

## Executive Summary

This document specifies the design and requirements for **East.py**, a Python runtime backend for the East programming language. East.py will enable East IR to be executed efficiently in Python environments, complementing the existing Julia backend (East.jl) and TypeScript reference implementation.

## Background

### The East Language

East is an embedded, statically-typed programming language designed for the Elara analytics platform. Key characteristics:

- **Statically typed** with structural typing
- **Embedded language** designed to run on "platforms" (host environments)
- **Serializable IR** as the "narrow waist" enabling cross-platform execution
- **Controlled side-effects** - no I/O except through platform functions
- **Three serialization formats**: East text format, JSON (planned), BEAST binary format (planned)

### East.jl Reference

The Julia backend (East.jl) provides:
- Complete type system representation using nominal types
- Sophisticated handling of recursive types
- Type-directed serialization/deserialization
- IR compilation to native Julia code via metaprogramming
- ~195 builtin functions
- ~3,943 lines of well-architected code

## Goals and Requirements

### Primary Goals

1. **Runtime Execution**: Execute East IR in Python environments with correct semantics
2. **Type Representation**: Represent all East data types as Python objects
3. **Serialization**: Read and write East text format, JSON, and BEAST formats
4. **Platform Integration**: Provide convenient API for building platforms in Python
5. **Performance**: Achieve reasonable performance for analytics workloads

### Non-Goals

1. **Language Frontend**: No East parser/type-checker needed (use TypeScript frontend)
2. **Native Code Compilation**: Unlike Julia, Python won't compile to native machine code
3. **JIT Optimization**: Initial version focuses on correctness over optimization

## Python-Specific Considerations

### Advantages of Python

- **Rich ecosystem** for data science and analytics (NumPy, Pandas, etc.)
- **Easy integration** with existing Python analytics codebases
- **Ubiquitous deployment** - runs everywhere, easy to install
- **Dynamic nature** allows flexible type representation
- **Strong serialization support** (pickle, JSON, etc.)

### Challenges and Design Constraints

1. **No compile-time type generation**: Python lacks Julia's `eval`-at-compile-time metaprogramming
2. **Dynamic typing**: Type checking happens at runtime, not compile time
3. **Performance**: Interpreted execution will be slower than Julia's compiled code
4. **Immutability**: Python has limited immutability support (need to enforce carefully)
5. **Value semantics**: Python uses reference semantics by default

## Architecture Design

### Module Structure

```
east_py/
├── __init__.py              # Main package exports
├── types/
│   ├── __init__.py
│   ├── primitives.py        # Null, Bool, Int, Float, String, Blob, DateTime
│   ├── containers.py        # Array, Set, Dict (mutable)
│   ├── struct.py            # Struct type and instances
│   ├── variant.py           # Variant type and Case instances
│   ├── type_system.py       # EastType representation
│   └── recursive.py         # Recursive type handling
├── serialization/
│   ├── __init__.py
│   ├── tokenizer.py         # Stream-based tokenizer
│   ├── parser.py            # Type-directed parser
│   ├── printer.py           # Value serializer
│   ├── json_format.py       # JSON serialization
│   └── beast.py             # Binary BEAST format
├── ir/
│   ├── __init__.py
│   ├── definitions.py       # IR node definitions
│   └── location.py          # Source location tracking
├── runtime/
│   ├── __init__.py
│   ├── interpreter.py       # IR interpreter
│   ├── builtins.py          # ~195 builtin functions
│   ├── error.py             # Error handling and stack traces
│   └── platform.py          # Platform integration API
├── utils/
│   ├── __init__.py
│   ├── ordering.py          # Total ordering implementation
│   └── matching.py          # Pattern matching utilities
└── tests/
    ├── __init__.py
    ├── test_types.py
    ├── test_serialization.py
    ├── test_interpreter.py
    └── test_builtins.py
```

### Core Design Decisions

#### 1. Type Representation Strategy

**Decision**: Use **dataclasses with runtime type tracking** instead of nominal type generation.

**Rationale**:
- Python cannot generate types at runtime as efficiently as Julia
- Dataclasses provide immutability via `frozen=True`
- Runtime type objects track structure (field names/types)
- More Pythonic approach using existing language features

**Implementation**:

```python
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class EastStruct:
    """Base class for all East struct instances."""
    _east_type: 'StructType'  # Runtime type information
    _values: tuple[Any, ...]  # Field values

    def __getattr__(self, name: str) -> Any:
        # Dynamic field access
        idx = self._east_type.field_index(name)
        return self._values[idx]

    def __eq__(self, other) -> bool:
        # Structural equality
        return (isinstance(other, EastStruct) and
                self._east_type == other._east_type and
                self._values == other._values)

@dataclass(frozen=True)
class StructType:
    """Runtime representation of a struct type."""
    fields: tuple[tuple[str, 'EastType'], ...]

    def create(self, **kwargs) -> EastStruct:
        # Validate and create instance
        values = tuple(kwargs[name] for name, _ in self.fields)
        return EastStruct(self, values)
```

#### 2. Variant Representation

**Decision**: Use **tagged union with Case wrapper**.

```python
@dataclass(frozen=True)
class Case:
    """A single case in a variant."""
    tag: str
    value: Any

@dataclass(frozen=True)
class EastVariant:
    """Base class for all East variant instances."""
    _east_type: 'VariantType'
    _case: Case

    @property
    def tag(self) -> str:
        return self._case.tag

    @property
    def value(self) -> Any:
        return self._case.value

# Pattern matching helper
def match(variant: EastVariant) -> dict:
    """Returns a dict for pattern matching."""
    return {variant.tag: variant.value}
```

#### 3. Recursive Types

**Decision**: Use **forward references and lazy type resolution**.

**Rationale**:
- Python's type system supports forward references
- Recursive structures created at runtime, not compile time
- Type objects can reference themselves through string names

```python
@dataclass
class RecursiveTypeRef:
    """Placeholder for recursive type reference."""
    depth: int  # Levels up to reference

class TypeBuilder:
    """Context for building recursive types."""
    def __init__(self):
        self.stack: list[EastType] = []

    def recursive_type(self, builder_fn):
        """Build a recursive type with placeholders."""
        placeholder = RecursiveTypeRef(0)
        self.stack.append(placeholder)
        result = builder_fn(placeholder)
        self.stack.pop()
        # Resolve placeholder to actual type
        return self._resolve(result)
```

#### 4. Container Types

**Decision**: Use **Python collections with East ordering semantics**.

- **Array**: Python `list` with runtime type annotation
- **Set**: Custom `SortedSet` class using East's total ordering
- **Dict**: Custom `SortedDict` class using East's total ordering

```python
from sortedcontainers import SortedSet, SortedDict

class EastArray(list):
    """East array with type tracking."""
    def __init__(self, element_type: EastType, items=None):
        super().__init__(items or [])
        self.element_type = element_type

class EastSet:
    """East set with sorted semantics."""
    def __init__(self, element_type: EastType):
        self.element_type = element_type
        self._data = SortedSet(key=east_compare_key)

class EastDict:
    """East dict with sorted key semantics."""
    def __init__(self, key_type: EastType, value_type: EastType):
        self.key_type = key_type
        self.value_type = value_type
        self._data = SortedDict(east_compare_key)
```

#### 5. Runtime Execution: Interpreter vs. Compilation

**Decision**: Implement an **tree-walking interpreter** initially, with optional bytecode compilation later.

**Rationale**:
- Python lacks Julia's metaprogramming for generating efficient native code
- Tree-walking interpreter is simpler to implement and maintain
- Good enough performance for many analytics workloads
- Can optimize hot paths with bytecode compilation in v2

```python
class Interpreter:
    """Interprets East IR."""

    def __init__(self, platform: Platform):
        self.platform = platform
        self.builtins = get_builtins()

    def eval(self, ir: IR, env: Environment) -> Any:
        """Evaluate IR node in environment."""
        match ir.kind:
            case 'Value':
                return ir.value
            case 'Variable':
                return env.get(ir.name)
            case 'Let':
                value = self.eval(ir.value, env)
                new_env = env.extend(ir.name, value, ir.mutable)
                return self.eval(ir.body, new_env)
            case 'Block':
                result = None
                for stmt in ir.statements:
                    result = self.eval(stmt, env)
                return result
            # ... handle all IR cases
```

#### 6. Builtin Functions

**Decision**: Implement all ~195 builtins as **Python functions in a registry**.

```python
BUILTINS: dict[str, Callable] = {}

def builtin(name: str, input_types: list[EastType], output_type: EastType):
    """Decorator to register a builtin function."""
    def decorator(fn: Callable) -> Callable:
        BUILTINS[name] = {
            'fn': fn,
            'inputs': input_types,
            'output': output_type
        }
        return fn
    return decorator

@builtin('IntegerAdd', [IntegerType, IntegerType], IntegerType)
def integer_add(a: int, b: int) -> int:
    return a + b

@builtin('ArrayLength', [ArrayType(TypeVar('T'))], IntegerType)
def array_length(arr: EastArray) -> int:
    return len(arr)
```

#### 7. Serialization Strategy

**Decision**: Use **type-directed parsing and printing** similar to Julia implementation.

- **Tokenizer**: Stream-based with one-token lookahead
- **Parser**: Dispatch on target type, validate structure
- **Printer**: Emit East text format with proper escaping

```python
class TokenStream:
    """Stream-based tokenizer for East text format."""
    def __init__(self, text: str):
        self.text = text
        self.pos = 0
        self.line = 1
        self.column = 1
        self._peeked: Optional[Token] = None

    def peek(self) -> Optional[Token]:
        if self._peeked is None:
            self._peeked = self._next_token()
        return self._peeked

    def next(self) -> Optional[Token]:
        if self._peeked is not None:
            token = self._peeked
            self._peeked = None
            return token
        return self._next_token()

def parse_east(target_type: EastType, text: str) -> Any:
    """Parse East text format into typed value."""
    stream = TokenStream(text)
    return _parse_value(stream, target_type)

def print_east(value: Any, value_type: EastType) -> str:
    """Print value in East text format."""
    buffer = StringIO()
    _print_value(buffer, value, value_type)
    return buffer.getvalue()
```

## Detailed Requirements

### 1. Type System Requirements

#### Primitive Types

| East Type | Python Representation | Notes |
|-----------|----------------------|-------|
| Null | `None` | Singleton |
| Boolean | `bool` | True/False |
| Integer | `int` | Arbitrary precision (64-bit in practice) |
| Float | `float` | IEEE 754 double (64-bit) |
| String | `str` | UTF-8 |
| Blob | `bytes` | Immutable byte array |
| DateTime | `datetime.datetime` | Timezone-aware UTC |

#### Container Types

- **Array**: Mutable, ordered, 0-indexed
- **Set**: Mutable, sorted by East total ordering, unique elements
- **Dict**: Mutable, sorted by key using East total ordering

#### Structural Types

- **Struct**: Immutable product type with named fields
  - Structural equality and ordering
  - Hashable (for use in sets/dicts)
  - Field access by name
  - Display as `(field1=value1, field2=value2)`

- **Variant**: Immutable sum type with tagged cases
  - Each case has a tag (string) and value (any type)
  - Pattern matching support
  - Display as `.Tag value` or `.Tag` for null values

#### Recursive Types

- Support arbitrary recursion (trees, DAGs, cycles)
- Reference by depth: `:Recursive(n)` means n levels up
- Efficient representation avoiding infinite structures

#### EastType

- Self-hosted representation (EastType is itself an East variant)
- Cases for all type constructors
- Conversion to/from Python type annotations (where possible)

### 2. Serialization Requirements

#### East Text Format

**Must support**:
- All primitive literals: `null`, `true`, `42`, `3.14`, `"hello"`, `0xdeadbeef`, `2025-01-01T00:00:00.000Z`
- Arrays: `[1, 2, 3]`
- Sets: `{1, 2, 3}` (empty: `{}`)
- Dicts: `{"a": 1, "b": 2}` (empty: `{:}`)
- Structs: `(name="Alice", age=30)`
- Variants: `.Some 42`, `.None`
- Identifier escaping with backticks: `` `field with spaces` ``

**Parser requirements**:
- Type-directed: parse according to expected type
- Strict validation: correct field order, no duplicates
- Good error messages with line/column numbers
- Handle edge cases: empty collections, special float values (NaN, Infinity)

**Printer requirements**:
- Deterministic output (sorted keys/fields)
- Proper escaping (strings, identifiers)
- Compact format (minimal whitespace)

#### JSON Format

**Must support**:
- Bidirectional conversion between East values and JSON
- Type annotations embedded in JSON where needed
- Compatible with standard JSON parsers

#### BEAST Binary Format

**Must support**:
- Efficient binary encoding
- Forward/backward compatibility
- Streaming read/write for large datasets

### 3. Runtime Execution Requirements

#### IR Interpretation

**Must handle all IR node types** (~30 cases):
- Values and variables
- Control flow: Block, IfElse, While, Break, Continue, Return
- Bindings: Let, Assign
- Collections: NewArray, NewSet, NewDict, ForArray, ForSet, ForDict
- Structures: Struct, GetField, Variant, Match
- Functions: Function, Call, Platform, Builtin
- Errors: Error, TryCatch
- Types: As, UnwrapRecursive, WrapRecursive

#### Variable Scoping

- Proper lexical scoping
- Support for mutable and immutable variables
- Closure capture (functions can reference outer variables)

#### Error Handling

- Exceptions with East stack traces
- Source location tracking (file, line, column)
- Accumulate stack as error propagates
- Convert Python exceptions to East errors

#### Performance Targets

- **Simple operations**: <10μs overhead per operation
- **Collections**: O(1) for array access, O(log n) for set/dict
- **Function calls**: <100μs overhead
- **Pattern matching**: <50μs per case

### 4. Builtin Functions Requirements

Implement all ~195 builtin functions across categories:

1. **Boolean**: `and`, `or`, `not`
2. **Comparison**: `equals`, `notEquals`, `lessThan`, etc.
3. **Integer arithmetic**: `add`, `subtract`, `multiply`, `divide`, `modulo`, `power`, `negate`, `abs`
4. **Float arithmetic**: Same as integer plus `floor`, `ceil`, `round`, `sqrt`, `log`, `exp`, `sin`, `cos`, etc.
5. **String operations**: `concat`, `length`, `substring`, `indexOf`, `split`, `join`, `trim`, etc.
6. **Array operations**: `length`, `get`, `set`, `pushLast`, `popLast`, `slice`, `concat`, etc.
7. **Set operations**: `size`, `has`, `add`, `remove`, `union`, `intersection`, `difference`
8. **Dict operations**: `size`, `has`, `get`, `set`, `remove`, `keys`, `values`, `entries`
9. **Blob operations**: `length`, `get`, `slice`, `concat`
10. **DateTime operations**: `now`, `parse`, `format`, `add`, `subtract`, `components`
11. **Type system**: `typeOf`, `stringPrintEast`, `stringParseEast`

### 5. Platform Integration Requirements

#### Platform API

```python
class Platform:
    """Platform function provider."""

    def get_function(self, name: str) -> Optional[Callable]:
        """Get platform function by name."""
        raise NotImplementedError

    def list_functions(self) -> dict[str, FunctionSignature]:
        """List all available platform functions."""
        raise NotImplementedError

@dataclass
class FunctionSignature:
    """Type signature for a function."""
    inputs: list[EastType]
    output: EastType
    platforms: set[str]
```

#### Platform Example

```python
class MyPlatform(Platform):
    def __init__(self):
        self.functions = {
            'log': self.log_message,
            'loadData': self.load_data,
        }

    def get_function(self, name: str) -> Optional[Callable]:
        return self.functions.get(name)

    def log_message(self, msg: str) -> None:
        print(f"[Platform] {msg}")

    def load_data(self, path: str) -> EastArray:
        # Load data from file system
        ...
```

#### Execution API

```python
def execute(ir: IR, platform: Platform, **inputs) -> Any:
    """Execute East IR with platform and inputs."""
    interpreter = Interpreter(platform)
    env = Environment(inputs)
    try:
        return interpreter.eval(ir, env)
    except EastError as e:
        # Add source location to stack trace
        raise
```

### 6. Testing Requirements

#### Unit Tests

- Test each primitive type
- Test each container type
- Test struct and variant creation/access
- Test recursive type handling
- Test serialization round-trips
- Test all builtin functions
- Test IR interpretation for each node type
- Test error handling and stack traces

#### Integration Tests

- Parse and execute East programs
- Test platform integration
- Test closure capture
- Test complex nested data structures

#### Compliance Tests

- Run East language compliance suite (from `../east/test`)
- All tests must pass (same as Julia backend)

### 7. Documentation Requirements

- **README**: Installation, quick start, basic usage
- **API Reference**: All public classes and functions
- **Type System Guide**: How East types map to Python
- **Platform Guide**: How to build platforms
- **Serialization Guide**: Format specifications
- **Performance Guide**: Optimization tips

### 8. Packaging and Distribution

- **Package name**: `east-lang` (or `eastpy`)
- **Minimum Python version**: 3.11 (for match statements)
- **Dependencies**:
  - `sortedcontainers` (for sorted sets/dicts)
  - `typing_extensions` (for better type hints)
- **Optional dependencies**:
  - `numpy` (for numerical operations)
  - `pandas` (for data frame integration)
- **Distribution**: PyPI via `pip install east-lang`

### 9. Development Tooling

#### Dependency Management: uv

The project will use **[uv](https://github.com/astral-sh/uv)** for dependency management and project tooling. uv is a modern, fast Python package manager written in Rust that replaces pip, pip-tools, virtualenv, poetry, and more.

**Rationale**:
- **Fast**: 10-100x faster than pip
- **Reliable**: Proper dependency resolution like poetry but much faster
- **Simple**: Single tool replaces multiple tools
- **Modern**: Built for Python 3.11+ workflows
- **Compatible**: Works with standard `pyproject.toml`

**Project configuration** (`pyproject.toml`):

```toml
[project]
name = "east-lang"
version = "0.1.0"
description = "Python runtime for the East programming language"
readme = "README.md"
requires-python = ">=3.11"
license = {text = "MIT"}
authors = [
    {name = "Your Name", email = "your.email@example.com"}
]
keywords = ["east", "embedded", "language", "runtime", "interpreter"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Topic :: Software Development :: Interpreters",
]

dependencies = [
    "sortedcontainers>=2.4.0",
    "typing-extensions>=4.12.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-cov>=5.0.0",
    "mypy>=1.11.0",
    "ruff>=0.6.0",
    "pre-commit>=3.8.0",
    "pydocstyle>=6.3.0",
    "bandit[toml]>=1.7.9",
]
numeric = [
    "numpy>=1.26.0",
]
data = [
    "pandas>=2.0.0",
]

[project.urls]
Homepage = "https://github.com/yourusername/east-py"
Documentation = "https://east-py.readthedocs.io"
Repository = "https://github.com/yourusername/east-py"
Issues = "https://github.com/yourusername/east-py/issues"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.uv]
dev-dependencies = [
    "pytest>=8.0.0",
    "pytest-cov>=5.0.0",
    "mypy>=1.11.0",
    "ruff>=0.6.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
python_classes = "Test*"
python_functions = "test_*"
addopts = "--verbose --cov=east_py --cov-report=term-missing"

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = false  # Gradual typing
check_untyped_defs = true

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = [
    "E",    # pycodestyle errors
    "W",    # pycodestyle warnings
    "F",    # pyflakes
    "I",    # isort (import sorting)
    "N",    # pep8-naming
    "UP",   # pyupgrade
    "B",    # flake8-bugbear
    "C4",   # flake8-comprehensions
    "SIM",  # flake8-simplify
    "RET",  # flake8-return
    "ARG",  # flake8-unused-arguments
]
ignore = [
    "E501",   # Line too long (handled by formatter)
    "N812",   # Lowercase imported as non-lowercase (East types)
]

[tool.ruff.lint.per-file-ignores]
"tests/**/*.py" = ["ARG"]  # Allow unused arguments in tests

[tool.coverage.run]
source = ["east_py"]
omit = ["*/tests/*", "*/tmp/*"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
    "if TYPE_CHECKING:",
]
```

#### Pre-commit Hooks

**`.pre-commit-config.yaml`** - Automatically runs checks before each commit:

```yaml
# See https://pre-commit.com for more information
repos:
  # Ruff - Fast Python linter and formatter
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.9
    hooks:
      # Run the linter
      - id: ruff
        args: [--fix]
      # Run the formatter
      - id: ruff-format

  # Type checking with mypy
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.11.2
    hooks:
      - id: mypy
        additional_dependencies:
          - sortedcontainers
          - typing-extensions
        args: [--ignore-missing-imports]

  # Standard pre-commit hooks
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-toml
      - id: check-added-large-files
        args: [--maxkb=1000]
      - id: check-merge-conflict
      - id: check-case-conflict
      - id: mixed-line-ending
        args: [--fix=lf]
      - id: name-tests-test
        args: [--pytest-test-first]

  # Check docstrings
  - repo: https://github.com/PyCQA/pydocstyle
    rev: 6.3.0
    hooks:
      - id: pydocstyle
        additional_dependencies: [tomli]
        args: [--convention=google]
        files: ^east_py/

  # Security checks
  - repo: https://github.com/PyCQA/bandit
    rev: 1.7.9
    hooks:
      - id: bandit
        args: [-c, pyproject.toml]
        additional_dependencies: ["bandit[toml]"]
```

**Additional pyproject.toml configuration**:

```toml
[tool.pydocstyle]
convention = "google"
add-ignore = ["D100", "D104"]  # Missing docstring in public module/package

[tool.bandit]
exclude_dirs = ["tests", "tmp"]
skips = ["B101"]  # Allow assert statements
```

**Setup pre-commit hooks**:

```bash
# After installing dependencies
make install  # Automatically installs hooks

# Or manually
uv run pre-commit install

# Run hooks on all files manually
make pre-commit

# Update hook versions
make pre-commit-update
```

#### Static Analysis Tools Overview

The project uses a comprehensive suite of static analysis tools:

| Tool | Purpose | Auto-fixes | Speed |
|------|---------|-----------|-------|
| **ruff** | Linter + formatter (replaces flake8, isort, black) | ✅ | Very fast |
| **mypy** | Static type checking | ❌ | Moderate |
| **pydocstyle** | Docstring conventions (Google style) | ❌ | Fast |
| **bandit** | Security vulnerability detection | ❌ | Fast |
| **pre-commit hooks** | File hygiene (whitespace, YAML, etc.) | ✅ | Very fast |

**Ruff checks** (selected lints):
- **E/W**: PEP 8 style errors and warnings
- **F**: Pyflakes (undefined names, unused imports)
- **I**: Import sorting (replaces isort)
- **N**: PEP 8 naming conventions
- **UP**: Python syntax upgrades (use modern syntax)
- **B**: Bugbear (common bugs and design problems)
- **C4**: Comprehension improvements
- **SIM**: Code simplification suggestions
- **RET**: Return statement improvements
- **ARG**: Unused arguments detection

**Mypy** checks:
- Type annotations consistency
- Return type correctness
- Argument type matching
- Optional gradual typing (not required everywhere initially)

**Pydocstyle** checks:
- Google-style docstrings for public APIs
- Missing docstrings in modules and classes
- Docstring format consistency

**Bandit** checks:
- Use of `assert` in non-test code
- Hardcoded passwords/secrets
- Use of `eval()` or `exec()`
- SQL injection vulnerabilities
- Insecure randomness

**Pre-commit automatic checks**:
- Remove trailing whitespace
- Add newline at end of files
- Check YAML/TOML syntax
- Detect merge conflicts
- Check for large files (>1MB)
- Enforce LF line endings

#### IDE Integration

**VS Code** (`.vscode/settings.json`):

```json
{
  "python.defaultInterpreterPath": ".venv/bin/python",
  "python.analysis.typeCheckingMode": "basic",
  "python.testing.pytestEnabled": true,
  "python.testing.unittestEnabled": false,
  "[python]": {
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.organizeImports": true,
      "source.fixAll": true
    },
    "editor.defaultFormatter": "charliermarsh.ruff"
  },
  "ruff.enable": true,
  "ruff.lint.run": "onSave",
  "mypy-type-checker.importStrategy": "fromEnvironment"
}
```

**PyCharm/IntelliJ**:
- Enable mypy plugin
- Configure ruff as external tool
- Set up file watchers for auto-formatting

**Required VS Code extensions**:
- `charliermarsh.ruff` - Ruff linter and formatter
- `ms-python.mypy-type-checker` - Mypy integration
- `ms-python.python` - Python language support

#### CI/CD Configuration (GitHub Actions)

**`.github/workflows/ci.yml`**:

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v3

      - name: Set up Python
        run: uv python install 3.11

      - name: Install dependencies
        run: uv sync --all-extras

      - name: Run pre-commit hooks
        run: uv run pre-commit run --all-files

      - name: Type check with mypy
        run: uv run mypy east_py

  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python-version: ["3.11", "3.12", "3.13"]

    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v3

      - name: Set up Python ${{ matrix.python-version }}
        run: uv python install ${{ matrix.python-version }}

      - name: Install dependencies
        run: uv sync

      - name: Run tests
        run: uv run pytest --cov=east_py --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v4
        with:
          file: ./coverage.xml
          fail_ci_if_error: true

  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v3

      - name: Set up Python
        run: uv python install 3.11

      - name: Build package
        run: uv build

      - name: Check package
        run: uv run twine check dist/*
```

#### Makefile

Similar to East.jl, the project includes a `Makefile` for common development tasks:

```makefile
.PHONY: install test repl lint format typecheck clean build publish pre-commit

# Install dependencies and pre-commit hooks
install:
	uv sync
	uv run pre-commit install

# Run test suite
test:
	uv run pytest

# Start Python REPL with east_py loaded
repl:
	uv run python -i -c "from east_py import *; print('East.py REPL ready')"

# Run linter (ruff)
lint:
	uv run ruff check east_py tests

# Auto-fix linting issues
lint-fix:
	uv run ruff check --fix east_py tests

# Format code
format:
	uv run ruff format east_py tests

# Type check with mypy
typecheck:
	uv run mypy east_py

# Run all quality checks (lint + typecheck + test)
check: lint typecheck test

# Clean build artifacts and cache
clean:
	rm -rf build/ dist/ *.egg-info
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Build distribution packages
build: clean
	uv build

# Publish to PyPI (requires authentication)
publish: build
	uv publish

# Install in development mode (editable install)
dev:
	uv pip install -e ".[dev]"

# Run benchmarks (if implemented)
bench:
	uv run python -m pytest tests/bench --benchmark-only

# Generate coverage report
coverage:
	uv run pytest --cov=east_py --cov-report=html
	@echo "Coverage report generated in htmlcov/index.html"

# Run pre-commit hooks on all files
pre-commit:
	uv run pre-commit run --all-files

# Update pre-commit hook versions
pre-commit-update:
	uv run pre-commit autoupdate
```

**Usage examples**:

```bash
# First-time setup
make install

# Development workflow
make test          # Run tests
make lint          # Check code quality
make format        # Format code
make typecheck     # Check types
make check         # Run all checks (lint + typecheck + test)

# Pre-commit hooks (auto-run on git commit)
make pre-commit    # Manually run all hooks
git commit -m "..."  # Hooks run automatically

# Interactive development
make repl          # Start REPL with east_py loaded

# Before committing (if not using pre-commit hooks)
make check         # Ensure everything passes

# Release workflow
make build         # Build distribution
make publish       # Publish to PyPI
```

#### Development Workflow

1. **Initial setup**:
   ```bash
   # Install uv (one-time)
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Clone and setup project
   git clone https://github.com/yourusername/east-py
   cd east-py
   make install
   ```

2. **Daily development**:
   ```bash
   # Make changes to code
   make test          # Run tests
   make lint          # Check style
   make format        # Auto-format
   ```

3. **Before committing**:
   ```bash
   make check         # Run all checks
   ```

4. **Interactive testing**:
   ```bash
   make repl          # Quick experimentation

   # Or run experiments in tmp/
   uv run python tmp/experiment.py
   ```

#### Project Structure

```
east-py/
├── Makefile                    # Development commands
├── pyproject.toml             # Project metadata and dependencies
├── uv.lock                    # Locked dependencies (auto-generated)
├── .pre-commit-config.yaml    # Pre-commit hooks configuration
├── .python-version            # Python version (3.11+)
├── .gitignore                 # Git ignore patterns
├── README.md                  # Project overview
├── LICENSE                    # MIT license
├── .vscode/                   # VS Code configuration
│   └── settings.json
├── .github/                   # GitHub Actions CI/CD
│   └── workflows/
│       └── ci.yml
├── east_py/                   # Main package
│   ├── __init__.py
│   ├── types/
│   ├── serialization/
│   ├── ir/
│   ├── runtime/
│   └── utils/
├── tests/                 # Test suite
│   ├── __init__.py
│   ├── test_types.py
│   ├── test_serialization.py
│   ├── test_interpreter.py
│   └── test_builtins.py
├── tmp/                   # Temporary experiments (gitignored)
│   └── experiment.py
├── docs/                  # Documentation
│   ├── index.md
│   ├── api.md
│   └── guide.md
└── examples/              # Example usage
    ├── basic.py
    ├── platform.py
    └── advanced.py
```

#### Comparison with East.jl

| Command | East.jl | East.py | Notes |
|---------|---------|---------|-------|
| Install deps | `make install` | `make install` | East.py also installs pre-commit hooks |
| Run tests | `make test` | `make test` | ✅ Parity |
| Start REPL | `make repl` | `make repl` | ✅ Parity |
| Lint code | N/A | `make lint` | Python: ruff linter |
| Format code | N/A | `make format` | Python: ruff formatter |
| Type check | N/A | `make typecheck` | Python: mypy |
| Check all | N/A | `make check` | Python: lint + typecheck + test |
| Pre-commit | N/A | `make pre-commit` | Python: run all hooks |
| Build package | N/A | `make build` | Python: build for PyPI |
| Publish | N/A | `make publish` | Python: publish to PyPI |
| Coverage | N/A | `make coverage` | Python: generate HTML coverage |
| Clean | N/A | `make clean` | Python: remove build artifacts |

**Key differences**:
- **East.jl**: Focuses on core development (install, test, repl)
- **East.py**: Adds comprehensive static analysis and packaging tools
- **Both**: Share same core workflow (install → test → repl)

## Implementation Phases

### Phase 1: Core Types (Weeks 1-2)

- Implement primitive types
- Implement container types (Array, Set, Dict)
- Implement Struct and Variant
- Implement EastType representation
- Basic recursive type support
- Unit tests for all types

### Phase 2: Serialization (Weeks 3-4)

- Implement tokenizer
- Implement type-directed parser
- Implement printer
- Test serialization round-trips
- Handle edge cases

### Phase 3: IR and Interpreter (Weeks 5-7)

- Define IR node types
- Implement interpreter for all node types
- Implement variable scoping
- Implement error handling
- Test execution

### Phase 4: Builtins (Week 8)

- Implement all ~195 builtin functions
- Test each builtin
- Verify semantics match East specification

### Phase 5: Platform Integration (Week 9)

- Design platform API
- Implement example platforms
- Test platform function calls
- Document platform guide

### Phase 6: Polish and Compliance (Week 10)

- Run East compliance test suite
- Fix bugs and edge cases
- Performance profiling and optimization
- Documentation
- Packaging for PyPI

## Performance Considerations

### Expected Performance

- **Slower than Julia**: 10-100x slower due to interpretation
- **Faster than pure Python**: Optimized data structures help
- **Acceptable for analytics**: Most time spent in NumPy/Pandas

### Optimization Strategies

1. **Cache type checks**: Avoid repeated type validation
2. **Optimize hot paths**: Common operations (array access, arithmetic)
3. **Use NumPy**: For array operations when possible
4. **JIT compilation**: Consider PyPy or Numba for hot functions (future)
5. **Bytecode compilation**: Compile IR to Python bytecode (future)

## Differences from Julia Backend

### Structural Differences

| Aspect | Julia (East.jl) | Python (East.py) |
|--------|-----------------|------------------|
| Type system | Compile-time nominal types | Runtime type objects |
| Execution | Compiled to native code | Interpreted |
| Type generation | `eval` at compile time | Runtime dataclass instances |
| Performance | Very fast (near C) | Moderate (interpreted) |
| Metaprogramming | Extensive (macros, `eval`) | Limited (decorators) |
| Immutability | Language-level | Enforced by dataclasses |

### API Differences

- Julia uses multiple dispatch; Python uses methods/functions
- Julia uses `Symbol` for identifiers; Python uses strings
- Julia's `NamedTuple` vs Python's dataclass with `__getattr__`
- Julia's `@match` macro vs Python's `match` statement (Python 3.10+)

### Feature Parity

Both backends must:
- ✅ Support all East types
- ✅ Support all serialization formats
- ✅ Execute all IR node types correctly
- ✅ Implement all builtin functions
- ✅ Pass compliance test suite
- ✅ Provide platform integration API

## Security Considerations

### Sandboxing

- No file system access (except through platform)
- No network access (except through platform)
- No subprocess execution
- No `eval()` or code generation (beyond East IR)

### Resource Limits

- Maximum recursion depth (configurable)
- Maximum array/set/dict size (configurable)
- Execution timeout (platform responsibility)

### Data Isolation

- Platforms can freeze input data to prevent mutation
- No shared mutable state between executions
- Each execution gets fresh environment

## Compatibility

### Cross-Backend Compatibility

- **IR Format**: Must be identical across all backends
- **Serialization**: Must produce/consume identical output
- **Semantics**: Behavior must match exactly
- **Builtins**: Same results for all builtin functions

### Versioning

- Follow semantic versioning (SemVer)
- Major version must match East language version
- Breaking changes require major version bump

## Success Criteria

1. ✅ All primitive and container types work correctly
2. ✅ Struct and Variant types support full semantics
3. ✅ Recursive types handle arbitrary recursion
4. ✅ Serialization round-trips correctly for all types
5. ✅ All IR node types execute correctly
6. ✅ All ~195 builtin functions pass tests
7. ✅ East compliance test suite passes 100%
8. ✅ Platform integration API is intuitive and well-documented
9. ✅ Performance is acceptable for analytics workloads
10. ✅ Package is published to PyPI and easy to install

## Open Questions

1. **NumPy integration**: Should we optimize Array operations with NumPy?
   - Pro: Much faster for numerical operations
   - Con: Adds heavyweight dependency
   - Decision: Make NumPy optional, detect and use when available

2. **Type hints**: Should we use Python type hints extensively?
   - Pro: Better IDE support, static analysis
   - Con: East's type system doesn't map cleanly to Python's
   - Decision: Use hints where helpful, but don't force it

3. **Async support**: Should we support async platform functions?
   - Pro: Better integration with async Python code
   - Con: East semantics are synchronous
   - Decision: Platform can use async internally, but East sees blocking

4. **Bytecode compilation**: Worth the complexity?
   - Pro: Significant performance boost
   - Con: Complex to implement and maintain
   - Decision: Phase 2 feature, start with interpreter

## Conclusion

East.py will provide a robust, Pythonic runtime for the East language, enabling East programs to execute in Python environments with full semantic compatibility. The design prioritizes correctness, maintainability, and ease of integration over raw performance, making it ideal for analytics platforms, data science workflows, and embedded scripting scenarios where Python is the natural choice.
