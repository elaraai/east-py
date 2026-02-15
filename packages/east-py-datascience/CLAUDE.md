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

### Optimization
- **MADS** (`mads/` — `mads.ts` / `mads.py`): Derivative-free blackbox optimization using PyNomadBBO
- **Optuna** (`optuna/` — `optuna.ts` / `optuna.py`): Bayesian optimization with TPE sampler
- **SimAnneal** (`simanneal/` — `simanneal.ts` / `simanneal.py`): Simulated annealing for discrete optimization
- **ALNS** (`alns/` — `alns.ts` / `alns.py`): Adaptive large neighborhood search (generic over solution type)
- **Scipy** (`scipy/` — `scipy.ts` / `scipy_impl.py`): Scientific optimization, curve fitting, interpolation, statistics
- **Optimization** (`optimization/` — `optimization.ts` / `optimization.py`): Iterative coordinate descent optimization
- **GoogleOr** (`google_or/` — `google_or.ts` / `cpsat.py`, `routing.py`, `linear.py`, `graph.py`): Google OR-Tools — CP-SAT, vehicle routing, LP/MIP, min-cost flow, max flow, assignment

### Machine Learning
- **XGBoost** (`xgboost/` — `xgboost.ts` / `xgboost_impl.py`): Gradient boosting (regression, classification, quantile)
- **LightGBM** (`lightgbm/` — `lightgbm.ts` / `lightgbm_impl.py`): Fast gradient boosting with leaf-wise growth
- **NGBoost** (`ngboost/` — `ngboost.ts` / `ngboost_impl.py`): Probabilistic gradient boosting with uncertainty
- **Torch** (`torch/` — `torch.ts` / `torch_impl.py`): Neural networks with PyTorch (MLP)
- **Lightning** (`lightning/` — `lightning.ts` / `lightning_impl.py`): PyTorch Lightning neural networks (MLP, autoencoder, conv1d, sequential, transformer)
- **GP** (`gp/` — `gp.ts` / `gp_impl.py`): Gaussian Process regression

### ML Utilities
- **Sklearn** (`sklearn/` — `sklearn.ts` / `sklearn.py`): Preprocessing, metrics, data splitting, regressor chains
- **MAPIE** (`mapie/` — `mapie.ts` / `mapie_impl.py`): Conformal prediction intervals

### Explainability
- **Shap** (`shap/` — `shap.ts` / `shap_impl.py`): SHAP values for model interpretation
