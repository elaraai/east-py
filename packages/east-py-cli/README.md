# east-py-cli

Command-line interface for running East IR programs with Python platform functions.

## Installation

```bash
uv add east-py-cli

# Add runtimes as needed
uv add east-py-std   # console, crypto, fetch, fs, path, random, time
uv add east-py-io    # s3, sqlite, postgres, mysql, redis, mongodb, csv, xlsx, xml, gzip, tar, zip, ftp, sftp
```

## Usage

```bash
# Run a program with standard platform functions
east-py run --std program.beast2

# Run with multiple runtimes
east-py run --runtime east-py-std --runtime east-py-io program.json

# Shorthand flags
east-py run --std --io program.east

# With input and output files
east-py run --std program.beast2 \
  --input data.beast2 \
  --input config.json \
  --output result.beast2

# Verbose output
east-py run --std -v program.beast2

# Show version and available runtimes
east-py version
```

## File Formats

IR and data files are auto-detected by extension:

| Extension | Format |
|-----------|--------|
| `.beast2`, `.beast` | Binary East v2 |
| `.east` | East text format |
| `.json` | JSON |

## Creating Runtime Packages

Runtime packages must export a `platform` attribute containing platform functions:

```python
from east.runtime.platform import PlatformFunction

platform = [
    PlatformFunction(name="my_function", ...),
    # ...
]
```
