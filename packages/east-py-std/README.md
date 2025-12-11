# east-py-std

[![License: BSL 1.1](https://img.shields.io/badge/License-BSL%201.1-orange.svg)](LICENSE.md)

Standard platform functions for the [East programming language](https://github.com/elaraai/East) in Python.

Python equivalent of [@elaraai/east-node](https://github.com/elaraai/east-node) - provides platform functions for console I/O, filesystem, HTTP, crypto, time, path manipulation, and random number generation.

## Installation

```bash
pip install east-py-std
```

## Quick Start

```python
import asyncio
from east.runtime.compiler import compile_async
from east_py_std import python_platform

# Assuming you have East IR from the TypeScript compiler
# compiled_fn = compile_async(ir, python_platform)
# await compiled_fn()
```

## Platform Functions

### Console I/O (`console_impl`)

- `console_log(message: String) -> Null` - Write to stdout with newline
- `console_error(message: String) -> Null` - Write to stderr with newline
- `console_write(message: String) -> Null` - Write to stdout without newline

### Cryptography (`crypto_impl`)

- `crypto_random_bytes(length: Integer) -> Blob` - Generate cryptographically secure random bytes
- `crypto_hash_sha256(data: String) -> String` - SHA-256 hash of UTF-8 string (hex)
- `crypto_hash_sha256_bytes(data: Blob) -> Blob` - SHA-256 hash of binary data
- `crypto_uuid() -> String` - Generate UUID v4

### HTTP Fetch (`fetch_impl`)

- `fetch_get(url: String) -> String` - HTTP GET request (async)
- `fetch_post(url: String, body: String) -> String` - HTTP POST request (async)
- `fetch_request(config: FetchRequestConfig) -> FetchResponse` - Custom HTTP request (async)

### Filesystem (`fs_impl`)

- `fs_read_file(path: String) -> String` - Read file as UTF-8 text
- `fs_write_file(path: String, content: String) -> Null` - Write UTF-8 text to file
- `fs_append_file(path: String, content: String) -> Null` - Append text to file
- `fs_delete_file(path: String) -> Null` - Delete file
- `fs_exists(path: String) -> Boolean` - Check if path exists
- `fs_is_file(path: String) -> Boolean` - Check if path is a file
- `fs_is_directory(path: String) -> Boolean` - Check if path is a directory
- `fs_create_directory(path: String) -> Null` - Create directory (recursive)
- `fs_read_directory(path: String) -> Array<String>` - List directory contents
- `fs_read_file_bytes(path: String) -> Blob` - Read file as binary
- `fs_write_file_bytes(path: String, content: Blob) -> Null` - Write binary to file

### Path Manipulation (`path_impl`)

- `path_join(segments: Array<String>) -> String` - Join path segments
- `path_resolve(path: String) -> String` - Resolve to absolute path
- `path_dirname(path: String) -> String` - Get directory name
- `path_basename(path: String) -> String` - Get base name (filename)
- `path_extname(path: String) -> String` - Get file extension

### Random Numbers (`random_impl`)

14 random number generation functions using cryptographically secure RNG:

- `random_uniform() -> Float` - Uniform [0, 1)
- `random_normal() -> Float` - Standard normal N(0, 1)
- `random_range(min: Integer, max: Integer) -> Integer` - Uniform integer [min, max]
- `random_exponential(lambda: Float) -> Float` - Exponential distribution
- `random_weibull(k: Float) -> Float` - Weibull distribution
- `random_bernoulli(p: Float) -> Integer` - Bernoulli trial (0 or 1)
- `random_binomial(n: Integer, p: Float) -> Integer` - Binomial distribution
- `random_geometric(p: Float) -> Integer` - Geometric distribution
- `random_poisson(lambda: Float) -> Integer` - Poisson distribution
- `random_pareto(alpha: Float) -> Float` - Pareto (power law) distribution
- `random_log_normal(mu: Float, sigma: Float) -> Float` - Log-normal distribution
- `random_irwin_hall(n: Integer) -> Float` - Sum of n uniform variables
- `random_bates(n: Integer) -> Float` - Average of n uniform variables
- `random_seed(seed: Integer) -> Null` - Seed RNG (no-op in Python)

### Time (`time_impl`)

- `time_now() -> Integer` - Current Unix timestamp in milliseconds (sync)
- `time_sleep(ms: Integer) -> Null` - Sleep for milliseconds (async)

## Usage

### Full Platform (Async)

```python
from east_py_std import python_platform
from east.runtime.compiler import compile_async

# Use all platform functions (includes async operations)
compiled_fn = compile_async(ir, python_platform)
await compiled_fn()
```

### Sync-Only Subset

```python
from east_py_std import python_platform_sync
from east.runtime.compiler import compile

# Use only synchronous platform functions
# Excludes: fetch_*, time_sleep
compiled_fn = compile(ir, python_platform_sync)
compiled_fn()
```

### Individual Modules

```python
from east_py_std import console_impl, fs_impl, crypto_impl

# Use specific platform function groups
platform = [*console_impl, *fs_impl, *crypto_impl]
compiled_fn = compile(ir, platform)
```

## Development

```bash
# First-time setup (installs dependencies)
make install

# Development workflow
make test          # Run test suite
make lint          # Run linter (ruff)
make format        # Format code
make typecheck     # Type check with mypy
make check         # Run all checks (lint + typecheck + test)

# Other useful commands
make repl          # Start Python REPL with east_py_std loaded
make coverage      # Generate HTML coverage report
make lint-fix      # Auto-fix linting issues
make clean         # Clean build artifacts
```

## License

**BSL 1.1 (Business Source License):**
- Non-production use (evaluation, testing, development) is free
- Production use by or on behalf of for-profit entities requires a commercial license
- Code becomes AGPL-3.0 four years after each release

See [LICENSE.md](LICENSE.md) for full details.

**Commercial licensing:** support@elara.ai

### Related Repositories

- **[east](https://github.com/elaraai/east)** - East language TypeScript frontend and reference implementation
- **[east-node](https://github.com/elaraai/east-node)** - Node.js runtime and platform functions for East
- **[e3](https://github.com/elaraai/e3)** - TypeScript SDK for authoring Elara solutions

## About Elara

East is developed by [Elara AI Pty Ltd](https://elaraai.com/), an AI-powered platform that creates economic digital twins of businesses that optimize performance. Elara combines business objectives, decisions and data to help organizations make data-driven decisions across work management, purchasing, customer engagement, and investment planning. East powers the computational layer of Elara solutions, enabling the expression of complex business logic and data in a simple, type-safe and portable language.

---
