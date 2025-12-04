# East Python Monorepo

This monorepo contains Python implementations for the East programming language.

## Packages

| Package | Description |
|---------|-------------|
| [east-py](packages/east-py/) | Core runtime - type system, IR compiler, builtins, serialization |
| [east-py-std](packages/east-py-std/) | Standard platform functions - console, crypto, fetch, fs, path, random, time |
| [east-py-io](packages/east-py-io/) | I/O platform functions - S3, databases, file formats, compression |
| [east-py-cli](packages/east-py-cli/) | CLI for running East IR programs |

## Setup

```bash
uv init myproject && cd myproject
```

## Installation

Install packages from GitHub using uv:

```bash
uv add git+https://github.com/elaraai/east-py#subdirectory=packages/east-py
uv add git+https://github.com/elaraai/east-py#subdirectory=packages/east-py-std
uv add git+https://github.com/elaraai/east-py#subdirectory=packages/east-py-io
uv add git+https://github.com/elaraai/east-py#subdirectory=packages/east-py-cli
```

## Development

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager

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

### Docker Services (for east-py-io)

east-py-io requires Docker for integration tests:

```bash
make services-up    # Start Docker services
make services-down  # Stop Docker services
```

## License

See individual package directories for license information.
