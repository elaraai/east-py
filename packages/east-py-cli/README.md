# east-py-cli

[![License: Commercial](https://img.shields.io/badge/License-Commercial-orange.svg)](LICENSE.md)

Command-line interface for running East IR programs with Python platform functions.

## Installation

```bash
uv add east-py-cli

# Platform packages are installed separately as needed
uv add east-py-std          # console, crypto, fetch, fs, path, random, time
uv add east-py-io           # s3, sqlite, postgres, mysql, redis, mongodb, xlsx, xml, gzip, tar, zip, ftp, sftp
uv add east-py-datascience  # sklearn, scipy, xgboost, lightgbm, ngboost, torch, shap, optuna, simanneal, mads
```

## Usage

### Running Programs

```bash
# Run with platform packages
east-py run program.beast2 -p east-py-std

# Run with multiple platforms
east-py run program.json -p east-py-std -p east-py-io -p east-py-datascience

# With input and output files
east-py run program.beast2 \
  -p east-py-std \
  --input data.beast2 \
  --input config.json \
  --output result.beast2

# Verbose output
east-py run program.beast2 -p east-py-std -v
```

### Version and Platform Info

```bash
# Show CLI version
east-py version

# Show version with platform info
east-py version -p east-py-std -p east-py-io
```

Example output:
```
east-py-cli 0.1.0
east-py 0.1.0

Platforms:
  east-py-std 0.1.0 (47 platform functions)
  east-py-io 0.1.0 (59 platform functions)
```

## File Formats

IR and data files are auto-detected by extension:

| Extension | Format |
|-----------|--------|
| `.beast2`, `.beast` | Binary East v2 |
| `.east` | East text format |
| `.json` | JSON |

## Creating Platform Packages

Platform packages must export a `platform` attribute containing a list of platform functions:

```python
# my_platform/__init__.py
from east.runtime.platform import platform_function

@platform_function("my_func", inputs=[StringType], output=IntegerType)
def my_func_impl(s):
    return len(s)

platform = [my_func_impl]
```

## License

Commercial - See [LICENSE.md](LICENSE.md) for details.
