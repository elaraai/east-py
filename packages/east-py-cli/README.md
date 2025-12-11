# east-py-cli

[![License: BSL 1.1](https://img.shields.io/badge/License-BSL%201.1-orange.svg)](LICENSE.md)

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
