# east-py-io

I/O platform functions for the [East programming language](https://github.com/elaraai/East) in Python.

Python equivalent of [@elaraai/east-node-io](https://github.com/elaraai/east-node-io) - provides platform functions for S3 object storage and SQLite database operations.

## Installation

```bash
pip install east-py-io
```

## Quick Start

```python
import asyncio
from east.runtime.compiler import compile_async
from east_py_io import python_io_platform

# Assuming you have East IR from the TypeScript compiler
# compiled_fn = compile_async(ir, python_io_platform)
# await compiled_fn()
```

## Platform Functions

### S3 Operations (`s3_impl`)

6 functions for AWS S3 and S3-compatible object storage (MinIO, Backblaze, etc.):

- `s3_put_object(config: S3Config, key: String, data: Blob) -> Null` - Upload object (async)
- `s3_get_object(config: S3Config, key: String) -> Blob` - Download object (async)
- `s3_head_object(config: S3Config, key: String) -> S3ObjectMetadata` - Get metadata without downloading (async)
- `s3_delete_object(config: S3Config, key: String) -> Null` - Delete object, idempotent (async)
- `s3_list_objects(config: S3Config, prefix: String, maxKeys: Integer) -> S3ListResult` - List with pagination (async)
- `s3_presign_url(config: S3Config, key: String, expiresIn: Integer) -> String` - Generate presigned URL (async)

**S3Config structure:**
```typescript
{
  region: String,
  bucket: String,
  accessKeyId: Option<String>,
  secretAccessKey: Option<String>,
  endpoint: Option<String>  // For S3-compatible services
}
```

**S3ObjectMetadata structure:**
```typescript
{
  key: String,
  size: Integer,
  lastModified: DateTime,
  contentType: Option<String>,
  etag: Option<String>
}
```

**S3ListResult structure:**
```typescript
{
  objects: Array<S3ObjectMetadata>,
  isTruncated: Boolean,
  continuationToken: Option<String>
}
```

### SQLite Database (`sqlite_impl`)

4 functions for SQLite database operations with connection pooling:

- `sqlite_connect(config: SqliteConfig) -> ConnectionHandle` - Connect to database, returns handle (async)
- `sqlite_query(handle: ConnectionHandle, sql: String, params: Array<SqlParameter>) -> SqlResult` - Execute parameterized query (async)
- `sqlite_close(handle: ConnectionHandle) -> Null` - Close connection (async)
- `sqlite_close_all() -> Null` - Close all connections, useful for cleanup (async)

**SqliteConfig structure:**
```typescript
{
  path: String,
  readOnly: Option<Boolean>,
  memory: Option<Boolean>
}
```

**SqlParameter variant:**
```typescript
String(String) | Integer(Integer) | Float(Float) | Boolean(Boolean) |
Null(Null) | Blob(Blob) | DateTime(DateTime)
```

**SqlResult variant:**
```typescript
select({ rows: Array<Dict<String, SqlParameter>> }) |
insert({ rowsAffected: Integer, lastInsertId: Option<Integer> }) |
update({ rowsAffected: Integer }) |
delete({ rowsAffected: Integer })
```

## Usage

### Full Platform (Async)

```python
from east_py_io import python_io_platform
from east.runtime.compiler import compile_async

# Use all I/O platform functions (all are async)
compiled_fn = compile_async(ir, python_io_platform)
await compiled_fn()
```

### Individual Modules

```python
from east_py_io import s3_impl, sqlite_impl

# Use specific platform function groups
platform = [*s3_impl, *sqlite_impl]
compiled_fn = compile_async(ir, platform)

# Or just one module
platform = [*sqlite_impl]
compiled_fn = compile_async(ir, platform)
```

### Type Definitions

```python
from east_py_io import (
    S3ConfigType,
    S3ObjectMetadataType,
    S3ListResultType,
    SqliteConfigType,
    SqlParameterType,
    SqlResultType,
)

# Use type definitions in your platform functions
```

## Development

```bash
# First-time setup (installs dependencies)
make install

# Development workflow
make test          # Run test suite (17 tests)
make lint          # Run linter (ruff)
make lint-fix      # Auto-fix linting issues
make typecheck     # Type check with mypy
make check         # Run all checks (lint + typecheck + test)

# Other useful commands
make coverage      # Generate HTML coverage report
make clean         # Clean build artifacts
```

## Related Projects

- [East](https://github.com/elaraai/East) - TypeScript frontend and reference implementation
- [east-py](https://github.com/elaraai/east-py) - Python runtime for East
- [east-node-io](https://github.com/elaraai/east-node-io) - Node.js I/O platform functions
- [east-py-std](https://github.com/elaraai/east-py-std) - Python standard platform functions
- [Elara](https://elara.ai) - Real-time analytics platform using East

## License

AGPL-3.0 - See [LICENSE](LICENSE) for details.
