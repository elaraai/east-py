# East Data Science

> Data science and ML platform functions for the East language

[![License](https://img.shields.io/badge/license-AGPL--3.0-blue.svg)](LICENSE.md)
[![Node Version](https://img.shields.io/badge/node-%3E%3D22.0.0-brightgreen.svg)](https://nodejs.org)

**East Data Science** provides machine learning and optimization platform functions for the [East language](https://github.com/elaraai/East).

## Features

- **MADS** - Derivative-free blackbox optimization (NOMAD algorithm)
- **Optuna** - Bayesian optimization with TPE sampler
- **SimAnneal** - Simulated annealing for discrete/combinatorial problems

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

### MADS (Derivative-Free Optimization)

MADS (Mesh Adaptive Direct Search) provides derivative-free blackbox optimization using the NOMAD algorithm. Ideal for:
- Functions with no exploitable derivatives
- Computationally expensive evaluations
- Noisy or discontinuous objective functions

```typescript
import { MADS, MADSConstraintType } from "@elaraai/east-py-datascience";

// Access types
MADS.Types.VectorType      // ArrayType(FloatType)
MADS.Types.BoundsType      // StructType({ lower, upper })
MADS.Types.ConfigType      // StructType({ max_bb_eval, seed, ... })
MADS.Types.ResultType      // StructType({ x_best, f_best, bb_eval, success })

// Optimize
MADS.optimize(objective, x0, bounds, constraints, config)
```

### Optuna (Bayesian Optimization)

Optuna provides Bayesian optimization using the TPE (Tree-structured Parzen Estimator) sampler. Ideal for:
- Parameter tuning with mixed types (int, float, categorical)
- Efficient search with few evaluations
- Automatic early stopping with pruners

```typescript
import { Optuna, ParamSpaceType, NamedParamType } from "@elaraai/east-py-datascience";

// Access types
Optuna.Types.ParamValueType     // VariantType({ int, float, string, bool })
Optuna.Types.ParamSpaceType     // StructType({ name, kind, low, high, choices })
Optuna.Types.StudyConfigType    // StructType({ direction, n_trials, ... })
Optuna.Types.StudyResultType    // StructType({ best_params, best_score, trials })

// Optimize
Optuna.optimize(search_space, objective, config)
```

### SimAnneal (Simulated Annealing)

SimAnneal provides simulated annealing for discrete and combinatorial optimization. Ideal for:
- Permutation problems (TSP, scheduling)
- Subset selection problems
- General discrete state optimization

```typescript
import { SimAnneal } from "@elaraai/east-py-datascience";

// Access types
SimAnneal.Types.DiscreteStateType   // VariantType({ int_array, bool_array })
SimAnneal.Types.AnnealConfigType    // StructType({ t_max, t_min, steps, ... })
SimAnneal.Types.AnnealResultType    // StructType({ best_state, best_energy, ... })

// Optimize permutation (e.g., TSP)
SimAnneal.optimizePermutation(initial_perm, energy_fn, config)

// Optimize subset selection
SimAnneal.optimizeSubset(initial_selection, energy_fn, config)

// General optimization with custom move function
SimAnneal.optimize(initial_state, energy_fn, move_fn, config)
```

## Planned Modules

The following modules are planned for implementation in order:

1. **scikit-learn** - Classification, regression, clustering (ONNX export)
2. **scipy** - Statistical functions and utilities
3. **xgboost** - Gradient boosting (ONNX export)
4. **lightgbm** - Light gradient boosting (ONNX export)
5. **ngboost** - Natural gradient boosting with uncertainty
6. **shap** - Model explainability and feature importance
7. **torch** - PyTorch neural networks (ONNX export) *[optional]*
8. **gp** - Gaussian processes *[optional]*

## Documentation

See [USAGE.md](./USAGE.md) for detailed usage guide with examples.

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
