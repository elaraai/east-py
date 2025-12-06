# East Data Science

> Data science and ML platform functions for the East language

[![TypeScript: AGPL-3.0](https://img.shields.io/badge/TypeScript-AGPL--3.0-blue.svg)](LICENSE.md)
[![Python: Commercial](https://img.shields.io/badge/Python-Commercial-orange.svg)](LICENSE.md)
[![Node Version](https://img.shields.io/badge/node-%3E%3D22.0.0-brightgreen.svg)](https://nodejs.org)

**East Data Science** provides machine learning and optimization platform functions for the [East language](https://github.com/elaraai/East).

## Installation

```bash
npm install @elaraai/east-py-datascience @elaraai/east
```

## Quick Start

```typescript
import { East, FloatType, variant } from "@elaraai/east";
import { MADS } from "@elaraai/east-py-datascience";

// Define objective function: minimize sum of squares
const objective = East.function([MADS.Types.VectorType], FloatType, ($, x) => {
    const x0 = $.let(x.get(0n));
    const x1 = $.let(x.get(1n));
    return $.return(x0.multiply(x0).add(x1.multiply(x1)));
});

// Optimize
const optimize = East.function([], MADS.Types.ResultType, $ => {
    const x0 = $.let([0.5, 0.5]);
    const bounds = $.let({
        lower: [-1.0, -1.0],
        upper: [1.0, 1.0],
    });
    const config = $.let({
        max_bb_eval: variant('some', 100n),
        display_degree: variant('some', 0n),
        direction_type: variant('none', null),
        initial_mesh_size: variant('none', null),
        min_mesh_size: variant('none', null),
        seed: variant('some', 42n),
    });

    return $.return(MADS.optimize(objective, x0, bounds, variant('none', null), config));
});
```

## Modules

### Optimization

| Module | Description | Use Cases |
|--------|-------------|-----------|
| **MADS** | Derivative-free blackbox optimization using NOMAD algorithm | Functions without derivatives, expensive evaluations, noisy/discontinuous objectives |
| **Optuna** | Bayesian optimization with TPE sampler | Hyperparameter tuning, mixed-type parameters, efficient search with few evaluations |
| **SimAnneal** | Simulated annealing for discrete optimization | TSP, scheduling, subset selection, knapsack, assignment problems |
| **Scipy** | Scientific optimization and curve fitting | Gradient-based minimization, curve fitting, interpolation, statistics |

### Machine Learning

| Module | Description | Use Cases |
|--------|-------------|-----------|
| **Sklearn** | Core ML utilities from scikit-learn | Train/test split, preprocessing (StandardScaler, MinMaxScaler), metrics, multi-target regression |
| **XGBoost** | Gradient boosting with XGBoost | Regression, classification, feature importance, fast training |
| **LightGBM** | Fast gradient boosting with leaf-wise growth | Large datasets, high cardinality features, faster than XGBoost on big data |
| **NGBoost** | Natural gradient boosting with uncertainty | Probabilistic predictions, confidence intervals, uncertainty quantification |
| **Torch** | Neural networks with PyTorch | MLP regression/classification, deep learning, custom architectures |
| **GP** | Gaussian Process regression | Small datasets, uncertainty quantification, Bayesian optimization surrogate |

### Explainability

| Module | Description | Use Cases |
|--------|-------------|-----------|
| **Shap** | SHAP values for model interpretation | Feature importance, model explanations, debugging predictions |

## Documentation

See [USAGE.md](./USAGE.md) for detailed API reference with examples.

## Development

```bash
npm run build     # Compile TypeScript
npm run test      # Run test suite
npm run lint      # Check code quality
```

## License

Dual-licensed:
- **Open Source**: [AGPL-3.0](LICENSE.md) - Free for open source use
- **Commercial**: Available for proprietary use - contact support@elara.ai

## Links

- **Website**: [https://elaraai.com/](https://elaraai.com/)
- **East Repository**: [https://github.com/elaraai/East](https://github.com/elaraai/East)
- **Issues**: [https://github.com/elaraai/east-py/issues](https://github.com/elaraai/east-py/issues)
- **Email**: support@elara.ai

---

*Developed by [Elara AI Pty Ltd](https://elaraai.com/)*
