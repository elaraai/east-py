# east-py-cli Design

A command-line interface for running East IR programs with Python platform functions.

## Overview

`east-py-cli` is a thin CLI that:
1. Loads East IR from files (.beast2, .east, .json)
2. Imports runtime packages for platform functions
3. Compiles and executes the IR

It does **not** handle dependency installation - that's the caller's responsibility (user via uv/pip, or e3 via task init).

## Usage

```bash
# Basic usage
east-py run program.beast2

# With runtime packages (must be installed in current env)
east-py run --runtime east-py-std --runtime east-py-io program.beast2

# With input/output
east-py run --runtime east-py-std program.beast2 \
  --input data.beast2 \
  --output result.beast2

# Shorthand for common runtimes
east-py run --std --io program.beast2
```

## Commands

### `east-py run`

Run an East IR program.

```
east-py run [OPTIONS] <IR_FILE>

Arguments:
  IR_FILE                    Path to IR file (.beast2, .east, or .json)

Options:
  -r, --runtime <PACKAGE>    Python package providing platform functions
                             Can be specified multiple times
  --std                      Shorthand for --runtime east-py-std
  --io                       Shorthand for --runtime east-py-io
  -i, --input <FILE>         Input data file (passed to program)
  -o, --output <FILE>        Output file path (program result written here)
  -v, --verbose              Enable verbose output
  --help                     Show help
```

### `east-py version`

Show version information.

```
east-py version

# Output:
# east-py-cli 0.1.0
# east-py 0.1.0
# Runtimes available:
#   east-py-std 0.1.0 (if installed)
#   east-py-io 0.1.0 (if installed)
```

## Runtime Package Convention

Runtime packages must export a `platform` attribute containing a list of `PlatformFunction` objects:

```python
# east_py_std/__init__.py
from east.runtime.platform import PlatformFunction

platform = [
    PlatformFunction(name="console_log", ...),
    PlatformFunction(name="time_now", ...),
    # ...
]
```

The CLI imports the package and accesses this attribute:

```python
import importlib

def load_runtime(package_name: str) -> list[PlatformFunction]:
    """Import a runtime package and return its platform functions."""
    module_name = package_name.replace("-", "_")
    mod = importlib.import_module(module_name)

    # Try known attribute names
    for attr in ["platform", "python_platform", "python_io_platform"]:
        if hasattr(mod, attr):
            return getattr(mod, attr)

    raise ValueError(f"Runtime package {package_name} has no 'platform' export")
```

## File Format Detection

IR format is auto-detected by extension:

| Extension | Format | Parser |
|-----------|--------|--------|
| `.beast2` | Binary East v2 | `decode_beast2_for(IRType)` |
| `.east` | East text format | `parse_east(IRType, text)` |
| `.json` | JSON | `decode_json_for(IRType)` |

## Execution Flow

```
1. Parse CLI arguments
2. Load IR file (format auto-detected, type is IRType)
3. Validate IR is FunctionType or AsyncFunctionType
4. Extract parameter types and return type from function signature
5. Validate --input count matches function arity
6. For each --runtime:
   a. Import package
   b. Get platform functions
   c. Add to platform list
7. Parse each --input file using its corresponding parameter type
8. Compile IR with platform functions
9. Execute compiled function with parsed inputs
10. Serialize result using return type, write to --output
```

## Input/Output Handling

### Multiple Inputs

Functions can have multiple parameters. Inputs are provided positionally:

```bash
# Function signature: (sales: Array<Sale>, config: Config) -> Result
east-py run program.beast2 \
  --input sales.beast2 \
  --input config.json \
  --output result.beast2
```

The order of `--input` flags matches the function's parameter order.

### Type-Directed Parsing

**Important:** East parsing is type-directed. The CLI must know the target type to parse correctly.

The flow is:
1. Parse IR file (type is always `IRType`)
2. Validate IR is `FunctionType` or `AsyncFunctionType`
3. Extract parameter types from `function.inputs`
4. Parse each `--input` file using its corresponding parameter type
5. Execute the function with parsed arguments
6. Serialize result using `function.output` type

```python
from east.types.types import is_function_type, is_async_function_type

def run_ir(ir, input_files, output_file, ...):
    # Validate IR is a function
    if not (is_function_type(ir) or is_async_function_type(ir)):
        raise ValueError(f"IR must be a function, got: {ir.type}")

    # Get function signature
    if is_function_type(ir):
        param_types = ir.value["inputs"]
        return_type = ir.value["output"]
    else:  # AsyncFunction
        param_types = ir.value["inputs"]
        return_type = ir.value["output"]

    # Validate input count
    if len(input_files) != len(param_types):
        raise ValueError(
            f"Function expects {len(param_types)} inputs, got {len(input_files)}"
        )

    # Parse each input with its declared type
    inputs = []
    for file_path, param_type in zip(input_files, param_types):
        value = load_value(file_path, param_type)
        inputs.append(value)

    # Compile and execute
    compiled = compile_ir(ir, platform_fns)
    result = compiled(*inputs)

    # Serialize output with return type
    if output_file:
        save_value(output_file, result, return_type)
```

### Format Detection

File format is determined by extension:

| Extension | Format | Notes |
|-----------|--------|-------|
| `.beast2` | Binary East v2 | Efficient, recommended for large data |
| `.beast` | Binary East v2 | Alias for .beast2 |
| `.east` | East text format | Human-readable, good for debugging |
| `.json` | JSON | Interop with other tools |

### Validation Checks

The CLI validates:

1. **IR is a function:** Must be `FunctionType` or `AsyncFunctionType`
2. **Input count matches:** Number of `--input` flags equals function arity
3. **Input types parse:** Each input file must successfully parse as its declared type
4. **Output path writable:** If `--output` specified, path must be writable

### Error Messages

```bash
# Wrong number of inputs
$ east-py run program.beast2 --input a.beast2
Error: Function expects 2 inputs, got 1
  Parameters: (sales: Array<Sale>, config: Config)

# Type mismatch
$ east-py run program.beast2 --input bad.json --input config.json
Error: Failed to parse input 0 (sales.json) as Array<Sale>
  Expected array, got object at root

# IR is not a function
$ east-py run value.beast2
Error: IR must be a function type, got: Struct<{name: String, age: Integer}>
```

## Package Structure

```
packages/east-py-cli/
├── pyproject.toml
├── east_py_cli/
│   ├── __init__.py
│   ├── __main__.py      # Entry point: python -m east_py_cli
│   ├── cli.py           # CLI argument parsing
│   ├── loader.py        # IR loading and format detection
│   └── runner.py        # Compilation and execution
```

## Dependencies

Minimal dependencies to keep the CLI lightweight:

```toml
[project]
dependencies = [
    "east-py",      # Core runtime (required)
]

[project.optional-dependencies]
dev = [
    "pytest",
    "mypy",
    "ruff",
]
```

Runtime packages (east-py-std, east-py-io) are **not** dependencies - they're installed separately by the user or e3 task.

## Entry Points

```toml
[project.scripts]
east-py = "east_py_cli:main"
```

This allows running via:
- `east-py run ...` (after pip install)
- `python -m east_py_cli run ...`
- `uv run east-py run ...`

## Error Handling

```
# Missing runtime package
$ east-py run --runtime east-py-io program.beast2
Error: Runtime package 'east-py-io' not found.
Install it with: uv add east-py-io

# Invalid IR file
$ east-py run bad.beast2
Error: Failed to parse IR from bad.beast2: Invalid magic bytes

# Missing platform function
$ east-py run program.beast2
Error: Platform function 'sqlite_connect' not found.
Required runtimes may not be loaded. Try: --runtime east-py-io

# Runtime error
$ east-py run program.beast2
Error: Execution failed at line 42: Division by zero
```

## Example Workflows

### Standalone Development

```bash
# Create project
mkdir my-east-project && cd my-east-project
uv init

# Add CLI and runtimes
uv add east-py-cli east-py-std

# Run program
uv run east-py run --std my-program.json
```

### With e3 (Future)

The e3 task system bundles dependencies and runs in isolated environments:

```east
# Task object
(
    init_tree = {
        "pyproject.toml": "...",  # Contains: east-py-cli, east-py-std, east-py-io
        "uv.lock": "...",
    },
    init = .some [.literal "uv", .literal "sync", .literal "--project", .bin],
    run = [
        .literal "uv", .literal "run", .literal "--project", .bin,
        .literal "east-py", .literal "run",
        .literal "--runtime", .literal "east-py-std",
        .literal "--runtime", .literal "east-py-io",
        .object "abc123...",  # IR file
        .literal "--input", .input 0,
        .literal "--output", .output,
    ],
)
```

## Implementation Notes

### CLI Library Choice

Using `argparse` (stdlib) for zero additional dependencies:

```python
import argparse

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="east-py",
        description="Run East IR programs with Python platform functions",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # run command
    run_parser = subparsers.add_parser("run", help="Run an IR program")
    run_parser.add_argument("ir_file", help="Path to IR file")
    run_parser.add_argument("-r", "--runtime", action="append", default=[])
    run_parser.add_argument("--std", action="store_true")
    run_parser.add_argument("--io", action="store_true")
    run_parser.add_argument("-i", "--input")
    run_parser.add_argument("-o", "--output")
    run_parser.add_argument("-v", "--verbose", action="store_true")

    return parser
```

### Async Support

The compiled function may be sync or async. The runner handles both:

```python
import asyncio

def run_compiled(compiled_fn, input_value=None):
    if asyncio.iscoroutinefunction(compiled_fn):
        return asyncio.run(compiled_fn(input_value) if input_value else compiled_fn())
    else:
        return compiled_fn(input_value) if input_value else compiled_fn()
```

## Migration Path

1. Add `platform` alias to existing packages:
   ```python
   # east_py_std/__init__.py
   platform = python_platform  # Add alias
   ```

2. Create east-py-cli package with minimal implementation

3. Test with existing IR files from compliance tests

4. Publish to PyPI alongside other east-py packages
