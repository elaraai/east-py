# East Python

> Python runtime and platform functions for the East language

[![Python Version](https://img.shields.io/badge/python-%3E%3D3.11-brightgreen.svg)](https://python.org)
[![uv](https://img.shields.io/badge/uv-package%20manager-blueviolet.svg)](https://docs.astral.sh/uv/)

**East Python** provides the Python runtime for executing [East language](https://github.com/elaraai/East) programs, including the core compiler, 200+ builtins, and platform functions for I/O, data science, and machine learning.

## Packages

### Python Runtime

| Package | Description |
|---------|-------------|
| [east-py](packages/east-py/) | Core runtime - type system, IR compiler, 200+ builtins, serialization |
| [east-py-std](packages/east-py-std/) | Standard platform functions - console, crypto, fetch, fs, path, random, time |
| [east-py-io](packages/east-py-io/) | I/O platform functions - S3, databases, file formats, compression |
| [east-py-datascience](packages/east-py-datascience/) | Data science & ML - optimization, gradient boosting, neural networks, explainability |
| [east-py-cli](packages/east-py-cli/) | CLI for running East IR programs |

### TypeScript Type Definitions

| Package | Description | npm | License |
|---------|-------------|-----|---------|
| [@elaraai/east-py-datascience](packages/east-py-datascience/) | TypeScript types for data science platform functions | [![npm](https://img.shields.io/npm/v/@elaraai/east-py-datascience)](https://www.npmjs.com/package/@elaraai/east-py-datascience) | [![License](https://img.shields.io/badge/license-AGPL--3.0-blue.svg)](packages/east-py-datascience/LICENSE.md) |

## Features

### Core Runtime (east-py)
- **Type System** - Full East type support including primitives, structs, variants, arrays, sets, maps
- **IR Compiler** - Compiles and executes East IR with platform function dispatch
- **200+ Builtins** - Math, strings, collections, dates, JSON, regex, and more
- **Serialization** - MessagePack-based binary format for efficient data transfer

### Platform Functions
- **Standard** - Console I/O, cryptography, HTTP fetch, filesystem, paths, random, time
- **I/O** - S3, PostgreSQL, MongoDB, Redis, Parquet, CSV, Excel, compression
- **Data Science** - MADS, Optuna, SimAnneal, XGBoost, LightGBM, NGBoost, PyTorch, GP, SHAP

## Quick Start

```bash
# Create a new project
uv init myproject && cd myproject

# Install packages
uv add git+https://github.com/elaraai/east-py#subdirectory=packages/east-py
uv add git+https://github.com/elaraai/east-py#subdirectory=packages/east-py-std
uv add git+https://github.com/elaraai/east-py#subdirectory=packages/east-py-io
uv add git+https://github.com/elaraai/east-py#subdirectory=packages/east-py-datascience
```

## Development

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- Docker (for east-py-io integration tests)

### Setup

```bash
make install      # Install dependencies
make install-cli  # Install east-py command globally
```

### Commands

```bash
make test         # Run all tests
make lint         # Run linter
make typecheck    # Run type checker
make check        # Run lint + typecheck + test
make help         # Show all available commands
```

### Docker Services

east-py-io requires Docker for integration tests:

```bash
make services-up    # Start Docker services
make services-down  # Stop Docker services
```

## License

- **Python Runtime Packages** (east-py, east-py-std, east-py-io, east-py-datascience, east-py-cli): Commercial license - contact support@elara.ai
- **TypeScript Type Definitions** (@elaraai/east-py-datascience npm package): Dual-licensed under [AGPL-3.0](packages/east-py-datascience/LICENSE.md) (open source) or commercial license

## Links

- **Website**: [https://elaraai.com/](https://elaraai.com/)
- **East Repository**: [https://github.com/elaraai/East](https://github.com/elaraai/East)
- **Issues**: [https://github.com/elaraai/east-py/issues](https://github.com/elaraai/east-py/issues)
- **Email**: support@elara.ai

---

*Developed by [Elara AI Pty Ltd](https://elaraai.com/)*
