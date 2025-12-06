# East Data Science Usage Guide

Usage guide for East Data Science platform functions.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Platform Functions](#platform-functions)
  - [MADS (Derivative-Free Optimization)](#mads-derivative-free-optimization)
  - [Optuna (Bayesian Optimization)](#optuna-bayesian-optimization)
- [Error Handling](#error-handling)

---

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

---

## Accessing Types

All module types are accessible via a nested `Types` property:

```typescript
import { MADS, Optuna } from "@elaraai/east-py-datascience";

// Access MADS types
MADS.Types.VectorType          // ArrayType(FloatType)
MADS.Types.BoundsType          // StructType({ lower, upper })
MADS.Types.ConfigType          // StructType({ max_bb_eval, ... })
MADS.Types.ResultType          // StructType({ x_best, f_best, ... })

// Access Optuna types
Optuna.Types.ParamValueType    // VariantType({ int, float, string, bool })
Optuna.Types.ParamSpaceType    // StructType({ name, kind, low, high, choices })
Optuna.Types.StudyResultType   // StructType({ best_params, best_score, trials })
```

**Pattern:**
- `Module.Types.TypeName` - Access types through the module namespace
- Flat exports (e.g., `MADSResultType`) are also available

---

## Platform Functions

### MADS (Derivative-Free Optimization)

MADS (Mesh Adaptive Direct Search) provides derivative-free blackbox optimization using the NOMAD algorithm. Ideal for:
- Functions with no exploitable derivatives
- Computationally expensive evaluations
- Noisy or discontinuous objective functions

**Import:**
```typescript
import { MADS, MADSConstraintType } from "@elaraai/east-py-datascience";
```

**Functions:**
| Signature | Description |
|-----------|-------------|
| `MADS.optimize(objective, x0, bounds, constraints, config)` | Run MADS optimization |

**Types:**

| Type | Description |
|------|-------------|
| `MADS.Types.VectorType` | `ArrayType(FloatType)` - Input/output vector |
| `MADS.Types.ScalarObjectiveType` | `FunctionType([VectorType], FloatType)` - Objective function |
| `MADS.Types.BoundsType` | `StructType({ lower: VectorType, upper: VectorType })` |
| `MADS.Types.ConfigType` | Configuration with `max_bb_eval`, `seed`, etc. |
| `MADS.Types.ResultType` | Result with `x_best`, `f_best`, `bb_eval`, `success` |
| `MADSConstraintType` | `VariantType({ eb: ObjectiveType, pb: ObjectiveType })` |

**Config Options:**

| Field | Type | Description |
|-------|------|-------------|
| `max_bb_eval` | `Option<Integer>` | Maximum blackbox evaluations |
| `display_degree` | `Option<Integer>` | Output verbosity (0=silent) |
| `direction_type` | `Option<DirectionType>` | Search direction strategy |
| `initial_mesh_size` | `Option<Float>` | Initial mesh granularity |
| `min_mesh_size` | `Option<Float>` | Minimum mesh size (stopping criterion) |
| `seed` | `Option<Integer>` | Random seed for reproducibility |

**Example - Unconstrained Optimization:**
```typescript
import { East, FloatType, variant } from "@elaraai/east";
import { MADS } from "@elaraai/east-py-datascience";

// Minimize Rosenbrock function
const rosenbrock = East.function([MADS.Types.VectorType], FloatType, ($, x) => {
    const x0 = $.let(x.get(0n));
    const x1 = $.let(x.get(1n));
    const term1 = $.let(East.value(100.0).multiply(x1.subtract(x0.multiply(x0)).pow(2.0)));
    const term2 = $.let(East.value(1.0).subtract(x0).pow(2.0));
    return $.return(term1.add(term2));
});

const optimize = East.function([], MADS.Types.ResultType, $ => {
    const x0 = $.let([-0.5, 0.5]);
    const bounds = $.let({
        lower: [-2.0, -2.0],
        upper: [2.0, 2.0],
    });
    const config = $.let({
        max_bb_eval: variant('some', 500n),
        display_degree: variant('some', 0n),
        direction_type: variant('none', null),
        initial_mesh_size: variant('none', null),
        min_mesh_size: variant('none', null),
        seed: variant('some', 42n),
    });

    return $.return(MADS.optimize(rosenbrock, x0, bounds, variant('none', null), config));
});
```

**Example - Constrained Optimization:**
```typescript
import { East, FloatType, ArrayType, variant } from "@elaraai/east";
import { MADS, MADSConstraintType } from "@elaraai/east-py-datascience";

// Objective: minimize x^2 + y^2
const objective = East.function([MADS.Types.VectorType], FloatType, ($, x) => {
    const x0 = $.let(x.get(0n));
    const x1 = $.let(x.get(1n));
    return $.return(x0.multiply(x0).add(x1.multiply(x1)));
});

// Constraint: x + y >= 1 (reformulated as 1 - x - y <= 0)
const constraint = East.function([MADS.Types.VectorType], FloatType, ($, x) => {
    const x0 = $.let(x.get(0n));
    const x1 = $.let(x.get(1n));
    return $.return(East.value(1.0).subtract(x0).subtract(x1));
});

const optimize = East.function([], MADS.Types.ResultType, $ => {
    const x0 = $.let([0.5, 0.5]);
    const bounds = $.let({
        lower: [0.0, 0.0],
        upper: [2.0, 2.0],
    });
    const constraints = $.let([
        variant('pb', constraint),  // Progressive barrier constraint
    ], ArrayType(MADSConstraintType));
    const config = $.let({
        max_bb_eval: variant('some', 200n),
        display_degree: variant('some', 0n),
        direction_type: variant('none', null),
        initial_mesh_size: variant('none', null),
        min_mesh_size: variant('none', null),
        seed: variant('some', 42n),
    });

    return $.return(MADS.optimize(objective, x0, bounds, variant('some', constraints), config));
});
```

---

### Optuna (Bayesian Optimization)

Optuna provides Bayesian optimization using the TPE (Tree-structured Parzen Estimator) sampler. Ideal for:
- Parameter tuning with mixed types (int, float, categorical)
- Efficient search with few evaluations
- Automatic early stopping with pruners

**Import:**
```typescript
import { Optuna, ParamSpaceType, NamedParamType } from "@elaraai/east-py-datascience";
```

**Functions:**
| Signature | Description |
|-----------|-------------|
| `Optuna.optimize(search_space, objective, config)` | Run Bayesian optimization |

**Types:**

| Type | Description |
|------|-------------|
| `Optuna.Types.ParamValueType` | `VariantType({ int, float, string, bool })` |
| `Optuna.Types.ParamSpaceKindType` | `VariantType({ int, float, categorical, log_uniform })` |
| `Optuna.Types.ParamSpaceType` | Parameter definition with `name`, `kind`, `low`, `high`, `choices` |
| `Optuna.Types.NamedParamType` | `StructType({ name: String, value: ParamValueType })` |
| `Optuna.Types.StudyConfigType` | Config with `direction`, `n_trials`, `random_state`, `pruner` |
| `Optuna.Types.StudyResultType` | Result with `best_params`, `best_score`, `trials` |

**Config Options:**

| Field | Type | Description |
|-------|------|-------------|
| `direction` | `Option<Direction>` | `minimize` or `maximize` |
| `n_trials` | `Integer` | Number of optimization trials |
| `random_state` | `Option<Integer>` | Random seed for reproducibility |
| `pruner` | `Option<Pruner>` | Early stopping (`none`, `median`, `hyperband`) |

**Example - Float Parameter:**
```typescript
import { East, FloatType, ArrayType, variant } from "@elaraai/east";
import { Optuna, ParamSpaceType, NamedParamType } from "@elaraai/east-py-datascience";

// Objective: minimize (x - 2)^2
const objective = East.function(
    [ArrayType(NamedParamType)],
    FloatType,
    ($, params) => {
        const xParam = $.let(params.get(0n));
        const xValue = $.let(xParam.value);
        const x = $.let(0.0);
        $.match(xValue, {
            int: ($, v) => $.assign(x, v.toFloat()),
            float: ($, v) => $.assign(x, v),
            string: $ => $.assign(x, 0.0),
            bool: $ => $.assign(x, 0.0),
        });
        const diff = $.let(x.subtract(2.0));
        return $.return(diff.multiply(diff));
    }
);

const optimize = East.function([], Optuna.Types.StudyResultType, $ => {
    const search_space = $.let([
        {
            name: "x",
            kind: variant("float", null),
            low: variant("some", 0.0),
            high: variant("some", 5.0),
            choices: variant("none", null),
        },
    ], ArrayType(ParamSpaceType));

    const config = $.let({
        direction: variant("some", variant("minimize", null)),
        n_trials: 30n,
        random_state: variant("some", 42n),
        pruner: variant("none", null),
    });

    return $.return(Optuna.optimize(search_space, objective, config));
});
```

**Example - Categorical Parameter:**
```typescript
import { East, FloatType, ArrayType, variant } from "@elaraai/east";
import { Optuna, ParamSpaceType, NamedParamType } from "@elaraai/east-py-datascience";

// Objective: score based on category selection
const objective = East.function(
    [ArrayType(NamedParamType)],
    FloatType,
    ($, params) => {
        const catParam = $.let(params.get(0n));
        const catValue = $.let(catParam.value);
        const score = $.let(10.0);
        $.match(catValue, {
            int: $ => $.assign(score, 10.0),
            float: $ => $.assign(score, 10.0),
            string: ($, s) => {
                $.if(East.equal(s, "optimal"), $ => {
                    $.assign(score, 0.0);
                }).elseIf(East.equal(s, "good"), $ => {
                    $.assign(score, 1.0);
                }).elseIf(East.equal(s, "bad"), $ => {
                    $.assign(score, 5.0);
                });
            },
            bool: $ => $.assign(score, 10.0),
        });
        return $.return(score);
    }
);

const optimize = East.function([], Optuna.Types.StudyResultType, $ => {
    const search_space = $.let([
        {
            name: "strategy",
            kind: variant("categorical", null),
            low: variant("none", null),
            high: variant("none", null),
            choices: variant("some", [
                variant("string", "optimal"),
                variant("string", "good"),
                variant("string", "bad"),
            ]),
        },
    ], ArrayType(ParamSpaceType));

    const config = $.let({
        direction: variant("some", variant("minimize", null)),
        n_trials: 15n,
        random_state: variant("some", 42n),
        pruner: variant("none", null),
    });

    return $.return(Optuna.optimize(search_space, objective, config));
});
```

**Example - Mixed Parameters:**
```typescript
import { East, FloatType, ArrayType, variant } from "@elaraai/east";
import { Optuna, ParamSpaceType, NamedParamType } from "@elaraai/east-py-datascience";

const objective = East.function(
    [ArrayType(NamedParamType)],
    FloatType,
    ($, params) => {
        // Extract learning_rate (float), n_estimators (int), booster (categorical)
        // ... process params and return score
        return $.return(0.0);
    }
);

const optimize = East.function([], Optuna.Types.StudyResultType, $ => {
    const search_space = $.let([
        {
            name: "learning_rate",
            kind: variant("log_uniform", null),  // Log-uniform sampling
            low: variant("some", 0.001),
            high: variant("some", 0.1),
            choices: variant("none", null),
        },
        {
            name: "n_estimators",
            kind: variant("int", null),
            low: variant("some", 50.0),
            high: variant("some", 500.0),
            choices: variant("none", null),
        },
        {
            name: "booster",
            kind: variant("categorical", null),
            low: variant("none", null),
            high: variant("none", null),
            choices: variant("some", [
                variant("string", "gbtree"),
                variant("string", "dart"),
            ]),
        },
    ], ArrayType(ParamSpaceType));

    const config = $.let({
        direction: variant("some", variant("minimize", null)),
        n_trials: 50n,
        random_state: variant("some", 123n),
        pruner: variant("some", variant("median", null)),  // Enable median pruning
    });

    return $.return(Optuna.optimize(search_space, objective, config));
});
```

---

## Error Handling

Platform functions throw errors on failure. Common scenarios:

- **Invalid bounds**: Lower bound exceeds upper bound
- **Dimension mismatch**: x0 size doesn't match bounds dimensions
- **Invalid parameter space**: Missing required fields
- **Optimization failure**: No feasible solution found

---

## License

Dual-licensed under AGPL-3.0 (open source) and commercial license. See [LICENSE.md](LICENSE.md).
