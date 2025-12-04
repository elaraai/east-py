# East Python Monorepo

This monorepo contains Python implementations for the East programming language.

## Packages

| Package | Description |
|---------|-------------|
| [east-py](packages/east-py/) | Core Python runtime for East - type system, IR compiler, builtins, serialization |
| [east-py-io](packages/east-py-io/) | I/O platform functions - S3, databases, file formats, compression |

## Development

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager

### Setup

```bash
# Install dependencies for all packages
make install
```

### Commands

```bash
make test              # Run all tests
make test-east-py      # Run east-py tests only
make test-east-py-io   # Run east-py-io tests only
make lint              # Run linter
make typecheck         # Run type checker
make check             # Run all quality checks
make help              # Show all available commands
```

### Docker Services (for east-py-io)

east-py-io requires Docker for integration tests:

```bash
make services          # Start Docker services
make test-integration  # Run integration tests
make services-down     # Stop Docker services
```

## Publishing

Each package can be published independently:

```bash
# Build packages
make build

# Or individually
uv build --package east-py
uv build --package east-py-io

# Publish
uv publish --package east-py
uv publish --package east-py-io
```

## License

See individual package directories for license information.
