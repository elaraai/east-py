# East Data Science

East Data Science provides data science and ML platform functions for the East language.

## Purpose

East Data Science enables East programs to use ML and optimization algorithms by providing:

- **Platform Functions**: TypeScript type definitions that compile to East IR
- **Python Runtime**: Python implementations that execute the platform functions
- **Testing Infrastructure**: Tests written in East (TypeScript) that export IR and run on Python

## Structure

This is a hybrid TypeScript + Python package:

- `/src` - TypeScript source code (platform function type definitions)
- `/src/east_py_datascience` - Python source code (platform function implementations)
- `/tests` - Python tests that run exported IR from TypeScript tests

## Development

### TypeScript (Type Definitions)

```bash
npm run build      # Compile TypeScript to JavaScript
npm run test       # Run tests (runs compiled .js - requires build first)
npm run lint       # Check code quality with ESLint
npm run test:export # Export test IR to /tmp/east-py-datascience
```

### Python (Runtime Implementations)

```bash
uv run pytest      # Run Python tests
uv run pytest -v   # Run with verbose output
```

## Standards

**All development MUST follow the mandatory standards defined in [STANDARDS.md](./STANDARDS.md).**

## Modules

- **MADS** (`mads.ts` / `mads.py`): Derivative-free blackbox optimization using PyNomadBBO
  - Single-objective optimization with constraints
  - Multi-objective optimization (Pareto front)
