# East.py

Python runtime for the [East programming language](https://github.com/elara-ai/East).

## Overview

East.py is a Python backend that enables East IR to be executed in Python environments. It provides:

- Complete type system representation for all East types
- Serialization and deserialization (East text format, JSON, BEAST)
- Tree-walking interpreter for East IR
- ~195 builtin functions
- Platform integration API for embedding East in Python applications

## Installation

```bash
# Install from PyPI
pip install east-py

# Or install from source
git clone https://github.com/elara-ai/east-py
cd east-py
make install
```

## Quick Start

```python
import east

# Parse East text format
struct_type = east.struct_type([("name", east.StringType), ("age", east.IntegerType)])
value = east.parse("(name=\"Alice\", age=30)", struct_type)

# Execute East IR
platform = east.Platform()  # Custom platform with your functions
result = east.execute(ir, platform, **inputs)

# Serialize to East text format
text = east.print_east(value, struct_type)
```

## Development

```bash
# First-time setup
make install

# Development workflow
make test          # Run tests
make lint          # Check code quality
make format        # Format code
make typecheck     # Check types
make check         # Run all checks

# Interactive development
make repl          # Start REPL with east loaded
```

## Documentation

See [DESIGN.md](DESIGN.md) for architecture and design decisions.

See [TODO.md](TODO.md) for implementation progress.

## Project Status

**Alpha** - Under active development. The API may change.

See [TODO.md](TODO.md) for current development status.

## License

Proprietary - See [LICENSE.md](LICENSE.md) for details.

## Related Projects

- [East](https://github.com/elara-ai/East) - TypeScript frontend and reference implementation
- [East.jl](https://github.com/elara-ai/East.jl) - Julia backend with native code compilation
- [Elara](https://elara.ai) - Real-time analytics platform using East
