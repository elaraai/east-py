# East Data Science Usage Guide

Usage guide for East Data Science platform functions.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Accessing Types](#accessing-types)
- [Platform Functions](#platform-functions)
  - [MADS (Derivative-Free Optimization)](#mads-derivative-free-optimization)
  - [Optuna (Bayesian Optimization)](#optuna-bayesian-optimization)
  - [SimAnneal (Simulated Annealing)](#simanneal-simulated-annealing)
  - [Sklearn (Machine Learning Utilities)](#sklearn-machine-learning-utilities)
  - [Scipy (Scientific Computing)](#scipy-scientific-computing)
  - [XGBoost (Gradient Boosting)](#xgboost-gradient-boosting)
  - [LightGBM (Fast Gradient Boosting)](#lightgbm-fast-gradient-boosting)
  - [NGBoost (Probabilistic Gradient Boosting)](#ngboost-probabilistic-gradient-boosting)
  - [Torch (Neural Networks)](#torch-neural-networks)
  - [GP (Gaussian Process)](#gp-gaussian-process)
  - [Shap (Model Explainability)](#shap-model-explainability)
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
import { MADS, Optuna, Sklearn, XGBoost } from "@elaraai/east-py-datascience";

// Access MADS types
MADS.Types.VectorType          // ArrayType(FloatType)
MADS.Types.BoundsType          // StructType({ lower, upper })
MADS.Types.ConfigType          // StructType({ max_bb_eval, ... })
MADS.Types.ResultType          // StructType({ x_best, f_best, ... })

// Access Optuna types
Optuna.Types.ParamValueType    // VariantType({ int, float, string, bool })
Optuna.Types.ParamSpaceType    // StructType({ name, kind, low, high, choices })
Optuna.Types.StudyResultType   // StructType({ best_params, best_score, trials })

// Access Sklearn types
Sklearn.Types.SplitConfigType  // StructType({ test_size, random_state, shuffle })
Sklearn.Types.ModelBlobType    // VariantType({ standard_scaler, min_max_scaler, ... })

// Access XGBoost types
XGBoost.Types.XGBoostConfigType // StructType({ n_estimators, max_depth, ... })
XGBoost.Types.ModelBlobType     // VariantType({ xgboost_regressor, xgboost_classifier })
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
| `MADS.optimize(objective: ScalarObjectiveType, x0: VectorType, bounds: BoundsType, constraints: Option<Array<ConstraintType>>, config: ConfigType): ResultType` | Run MADS optimization |

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
| `Optuna.optimize(search_space: Array<ParamSpaceType>, objective: FunctionType<[Array<NamedParamType>], Float>, config: StudyConfigType): StudyResultType` | Run Bayesian optimization |

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
        const x = $.let(xParam.value.unwrap('float'));
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
        const category = $.let(catParam.value.unwrap('string'));
        const score = $.let(10.0);
        $.if(East.equal(category, "optimal"), $ => {
            $.assign(score, 0.0);
        }).elseIf(East.equal(category, "good"), $ => {
            $.assign(score, 1.0);
        }).elseIf(East.equal(category, "bad"), $ => {
            $.assign(score, 5.0);
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

---

### SimAnneal (Simulated Annealing)

Simulated Annealing provides discrete/combinatorial optimization. Ideal for:
- Permutation problems (TSP, scheduling)
- Subset selection (feature selection, knapsack)
- Assignment problems
- Any discrete optimization with many local minima

**Import:**
```typescript
import { SimAnneal } from "@elaraai/east-py-datascience";
```

**Functions:**
| Signature | Description |
|-----------|-------------|
| `SimAnneal.optimize(initial_state: DiscreteStateType, energy_fn: EnergyFunctionType, move_fn: MoveFunctionType, config: ConfigType): ResultType` | Run with custom move function |
| `SimAnneal.optimizePermutation(initial_perm: Array<Integer>, energy_fn: PermutationEnergyType, config: ConfigType): ResultType` | Optimize permutation with swap moves |
| `SimAnneal.optimizeSubset(initial_selection: Array<Boolean>, energy_fn: SubsetEnergyType, config: ConfigType): ResultType` | Optimize subset with bit-flip moves |

**Types:**

| Type | Description |
|------|-------------|
| `SimAnneal.Types.DiscreteStateType` | `VariantType({ int_array, bool_array })` |
| `SimAnneal.Types.EnergyFunctionType` | `FunctionType([DiscreteStateType], FloatType)` |
| `SimAnneal.Types.MoveFunctionType` | `FunctionType([DiscreteStateType], DiscreteStateType)` |
| `SimAnneal.Types.ConfigType` | Config with `t_max`, `t_min`, `steps`, etc. |
| `SimAnneal.Types.ResultType` | Result with `best_state`, `best_energy`, `success` |

**Config Options:**

| Field | Type | Description |
|-------|------|-------------|
| `t_max` | `Option<Float>` | Starting temperature (default 25000.0) |
| `t_min` | `Option<Float>` | Ending temperature (default 2.5) |
| `steps` | `Option<Integer>` | Total iterations (default 50000) |
| `updates` | `Option<Integer>` | Progress report frequency (0=silent) |
| `auto_schedule` | `Option<Float>` | Minutes for auto-calibration |
| `random_state` | `Option<Integer>` | Random seed for reproducibility |

**Example - Permutation Optimization (TSP-like):**
```typescript
import { East, FloatType, ArrayType, IntegerType, variant } from "@elaraai/east";
import { SimAnneal } from "@elaraai/east-py-datascience";

// Energy: sum of absolute differences between adjacent elements
const energy = East.function([ArrayType(IntegerType)], FloatType, ($, perm) => {
    const total = $.let(0.0);
    $.forArray(perm, ($, i, val) => {
        $.if(i.lessThan(perm.length().subtract(1n)), $ => {
            const next = $.let(perm.get(i.add(1n)));
            const diff = $.let(val.subtract(next).toFloat().abs());
            $.assign(total, total.add(diff));
        });
    });
    return $.return(total);
});

const optimize = East.function([], SimAnneal.Types.ResultType, $ => {
    const initial = $.let([0n, 3n, 1n, 4n, 2n]);
    const config = $.let({
        t_max: variant("some", 1000.0),
        t_min: variant("some", 1.0),
        steps: variant("some", 10000n),
        updates: variant("none", null),
        auto_schedule: variant("none", null),
        random_state: variant("some", 42n),
    });

    return $.return(SimAnneal.optimizePermutation(initial, energy, config));
});
```

**Example - Subset Selection:**
```typescript
import { East, FloatType, ArrayType, BooleanType, variant } from "@elaraai/east";
import { SimAnneal } from "@elaraai/east-py-datascience";

// Energy: prefer selecting fewer items while maximizing value
const energy = East.function([ArrayType(BooleanType)], FloatType, ($, selection) => {
    const values = $.let([10.0, 20.0, 15.0, 25.0, 5.0]);
    const total = $.let(0.0);
    $.forArray(selection, ($, i, selected) => {
        $.if(selected, $ => {
            $.assign(total, total.subtract(values.get(i)));
        });
    });
    return $.return(total);
});

const optimize = East.function([], SimAnneal.Types.ResultType, $ => {
    const initial = $.let([false, false, false, false, false]);
    const config = $.let({
        t_max: variant("some", 100.0),
        t_min: variant("some", 0.1),
        steps: variant("some", 5000n),
        updates: variant("none", null),
        auto_schedule: variant("none", null),
        random_state: variant("some", 42n),
    });

    return $.return(SimAnneal.optimizeSubset(initial, energy, config));
});
```

---

### Sklearn (Machine Learning Utilities)

Sklearn provides core ML utilities: preprocessing, data splitting, flexible metrics, and multi-target regression.

**Import:**
```typescript
import { Sklearn } from "@elaraai/east-py-datascience";
```

**Functions:**
| Signature | Description |
|-----------|-------------|
| `Sklearn.trainTestSplit(X: MatrixType, y: VectorType, config: SplitConfigType): SplitResultType` | Split data into train/test sets |
| `Sklearn.trainValTestSplit(X: MatrixType, Y: MatrixType, config: ThreeWaySplitConfigType): ThreeWaySplitResultType` | Split data into train/val/test sets |
| `Sklearn.standardScalerFit(X: MatrixType): ModelBlobType` | Fit StandardScaler to data |
| `Sklearn.standardScalerTransform(model: ModelBlobType, X: MatrixType): MatrixType` | Transform data with fitted scaler |
| `Sklearn.minMaxScalerFit(X: MatrixType): ModelBlobType` | Fit MinMaxScaler to data |
| `Sklearn.minMaxScalerTransform(model: ModelBlobType, X: MatrixType): MatrixType` | Transform data with fitted scaler |
| `Sklearn.computeMetrics(y_true: VectorType, y_pred: VectorType, metrics: Array<RegressionMetricType>): MetricsResultType` | Compute selected regression metrics |
| `Sklearn.computeMetricsMulti(Y_true: MatrixType, Y_pred: MatrixType, metrics: Array<RegressionMetricType>, config: MultiMetricsConfigType): MultiMetricsResultType` | Compute multi-target regression metrics |
| `Sklearn.computeClassificationMetrics(y_true: LabelVectorType, y_pred: LabelVectorType, metrics: Array<ClassificationMetricType>, config: ClassificationMetricsConfigType): ClassificationMetricResultsType` | Compute selected classification metrics |
| `Sklearn.computeClassificationMetricsMulti(Y_true: MatrixType, Y_pred: MatrixType, metrics: Array<ClassificationMetricType>, config: MultiClassificationConfigType): MultiClassificationMetricResultsType` | Compute multi-target classification metrics |
| `Sklearn.regressorChainTrain(X: MatrixType, Y: MatrixType, config: RegressorChainConfigType): ModelBlobType` | Train multi-target regressor chain |
| `Sklearn.regressorChainPredict(model: ModelBlobType, X: MatrixType): MatrixType` | Predict with regressor chain |

**Types:**

| Type | Description |
|------|-------------|
| `Sklearn.Types.SplitConfigType` | Config with `test_size`, `random_state`, `shuffle` |
| `Sklearn.Types.SplitResultType` | Result with `X_train`, `X_test`, `y_train`, `y_test` |
| `Sklearn.Types.ThreeWaySplitConfigType` | Config with `val_size`, `test_size`, `random_state`, `shuffle` |
| `Sklearn.Types.ThreeWaySplitResultType` | Result with `X_train`, `X_val`, `X_test`, `Y_train`, `Y_val`, `Y_test` |
| `Sklearn.Types.RegressionMetricType` | Variant: `mse`, `rmse`, `mae`, `r2`, `mape`, `explained_variance`, `max_error`, `median_ae` |
| `Sklearn.Types.ClassificationMetricType` | Variant: `accuracy`, `balanced_accuracy`, `precision`, `recall`, `f1`, `matthews_corrcoef`, `cohen_kappa`, `jaccard` |
| `Sklearn.Types.MetricAggregationType` | Variant: `per_target`, `uniform_average` |
| `Sklearn.Types.RegressorChainConfigType` | Config with `base_estimator`, `order`, `random_state` |

**Example - Train/Test Split:**
```typescript
import { East, variant } from "@elaraai/east";
import { Sklearn } from "@elaraai/east-py-datascience";

const split = East.function([], Sklearn.Types.SplitResultType, $ => {
    const X = $.let([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]);
    const y = $.let([1.0, 2.0, 3.0, 4.0, 5.0]);
    const config = $.let({
        test_size: variant('some', 0.2),
        random_state: variant('some', 42n),
        shuffle: variant('some', true),
    });
    return $.return(Sklearn.trainTestSplit(X, y, config));
});
```

**Example - Train/Val/Test Split (3-way):**
```typescript
import { East, variant } from "@elaraai/east";
import { Sklearn } from "@elaraai/east-py-datascience";

const split = East.function([], Sklearn.Types.ThreeWaySplitResultType, $ => {
    const X = $.let([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0],
                     [11.0, 12.0], [13.0, 14.0], [15.0, 16.0], [17.0, 18.0], [19.0, 20.0]]);
    const Y = $.let([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0],
                     [11.0, 12.0], [13.0, 14.0], [15.0, 16.0], [17.0, 18.0], [19.0, 20.0]]);
    const config = $.let({
        val_size: variant('some', 0.15),
        test_size: variant('some', 0.15),
        random_state: variant('some', 42n),
        shuffle: variant('some', true),
    });
    return $.return(Sklearn.trainValTestSplit(X, Y, config));
    // Returns: X_train (70%), X_val (15%), X_test (15%), Y_train, Y_val, Y_test
});
```

**Example - Flexible Regression Metrics:**
```typescript
import { East, variant } from "@elaraai/east";
import { Sklearn } from "@elaraai/east-py-datascience";

const compute = East.function([], Sklearn.Types.MetricsResultType, $ => {
    const y_true = $.let([1.0, 2.0, 3.0, 4.0, 5.0]);
    const y_pred = $.let([1.1, 2.0, 2.9, 4.1, 5.0]);

    // Select only the metrics you need
    const results = $.let(Sklearn.computeMetrics(
        y_true,
        y_pred,
        [variant('mse', null), variant('r2', null), variant('mae', null)]
    ));
    return $.return(results);
    // Returns: [{ metric: #mse, value: 0.006 }, { metric: #r2, value: 0.997 }, ...]
});
```

**Example - Multi-Target Metrics:**
```typescript
import { East, variant } from "@elaraai/east";
import { Sklearn } from "@elaraai/east-py-datascience";

const compute = East.function([], Sklearn.Types.MultiMetricsResultType, $ => {
    const Y_true = $.let([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]);
    const Y_pred = $.let([[1.1, 10.5], [2.0, 20.0], [2.9, 29.5]]);

    const config = $.let({
        aggregation: variant('some', variant('per_target', null)),
    });

    const results = $.let(Sklearn.computeMetricsMulti(
        Y_true,
        Y_pred,
        [variant('mse', null), variant('r2', null)],
        config
    ));
    return $.return(results);
    // Returns: [{ metric: #mse, value: #per_target([0.006, 0.166]) }, ...]
});
```

**Example - Classification Metrics:**
```typescript
import { East, variant } from "@elaraai/east";
import { Sklearn } from "@elaraai/east-py-datascience";

const compute = East.function([], Sklearn.Types.ClassificationMetricResultsType, $ => {
    const y_true = $.let([0n, 1n, 1n, 0n, 1n]);
    const y_pred = $.let([0n, 1n, 0n, 0n, 1n]);

    const config = $.let({
        average: variant('some', variant('binary', null)),
    });

    const results = $.let(Sklearn.computeClassificationMetrics(
        y_true,
        y_pred,
        [variant('accuracy', null), variant('f1', null), variant('precision', null)],
        config
    ));
    return $.return(results);
});
```

**Example - StandardScaler:**
```typescript
import { East, variant } from "@elaraai/east";
import { Sklearn } from "@elaraai/east-py-datascience";

const scale = East.function([], Sklearn.Types.MatrixType, $ => {
    const X_train = $.let([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
    const X_test = $.let([[2.0, 3.0], [4.0, 5.0]]);

    const scaler = $.let(Sklearn.standardScalerFit(X_train));
    const X_scaled = $.let(Sklearn.standardScalerTransform(scaler, X_test));

    return $.return(X_scaled);
});
```

---

### Scipy (Scientific Computing)

Scipy provides statistics, optimization, interpolation, and curve fitting.

**Import:**
```typescript
import { Scipy } from "@elaraai/east-py-datascience";
```

**Functions:**
| Signature | Description |
|-----------|-------------|
| `Scipy.curveFit(curve_fn: CurveFunctionType, x: VectorType, y: VectorType, config: CurveFitConfigType): CurveFitResultType` | Fit parametric curve to data |
| `Scipy.statsDescribe(data: VectorType): StatsDescribeResultType` | Compute descriptive statistics |
| `Scipy.statsPearsonr(x: VectorType, y: VectorType): CorrelationResultType` | Compute Pearson correlation |
| `Scipy.statsSpearmanr(x: VectorType, y: VectorType): CorrelationResultType` | Compute Spearman correlation |
| `Scipy.interpolate1dFit(x: VectorType, y: VectorType, config: InterpolateConfigType): ModelBlobType` | Fit 1D interpolator |
| `Scipy.interpolate1dPredict(model: ModelBlobType, x: VectorType): VectorType` | Evaluate interpolator |
| `Scipy.optimizeMinimize(objective: ScalarObjectiveType, x0: VectorType, config: OptimizeConfigType): OptimizeResultType` | Minimize scalar function |
| `Scipy.optimizeMinimizeQuadratic(x0: VectorType, quadratic_config: QuadraticConfigType, opt_config: OptimizeConfigType): OptimizeResultType` | Minimize quadratic function |
| `Scipy.optimizeDualAnnealing(objective: ScalarObjectiveType, x0: Option<VectorType>, bounds: DualAnnealBoundsType, config: DualAnnealConfigType): DualAnnealResultType` | Global optimization using dual annealing |

**Types:**

| Type | Description |
|------|-------------|
| `Scipy.Types.OptimizeMethodType` | `bfgs`, `l_bfgs_b`, `nelder_mead`, `powell`, `cg` |
| `Scipy.Types.InterpolationKindType` | `linear`, `cubic`, `quadratic` |
| `Scipy.Types.CurveFunctionType` | Built-in curves or custom function |
| `Scipy.Types.StatsDescribeResultType` | Statistics: `count`, `mean`, `variance`, etc. |
| `Scipy.Types.CorrelationResultType` | `correlation`, `pvalue` |
| `Scipy.Types.CurveFitResultType` | `params`, `success`, `r_squared` |
| `Scipy.Types.OptimizeResultType` | `x`, `fun`, `success`, `nit` |
| `Scipy.Types.DualAnnealBoundsType` | `StructType({ lower: VectorType, upper: VectorType })` |
| `Scipy.Types.DualAnnealConfigType` | Config with `maxfun`, `maxiter`, `initial_temp`, `seed`, etc. |
| `Scipy.Types.DualAnnealResultType` | `x`, `fun`, `nfev`, `nit`, `success`, `message` |

**Example - Curve Fitting:**
```typescript
import { East, variant } from "@elaraai/east";
import { Scipy } from "@elaraai/east-py-datascience";

const fit = East.function([], Scipy.Types.CurveFitResultType, $ => {
    const x = $.let([0.0, 1.0, 2.0, 3.0, 4.0]);
    const y = $.let([1.0, 2.7, 7.4, 20.1, 54.6]);  // Exponential growth

    const curve_fn = $.let(variant('exponential_growth', null));
    const config = $.let({
        max_iter: variant('some', 5000n),
        initial_guess: variant('none', null),
    });

    return $.return(Scipy.curveFit(curve_fn, x, y, config));
});
```

**Example - Optimization:**
```typescript
import { East, FloatType, variant } from "@elaraai/east";
import { Scipy } from "@elaraai/east-py-datascience";

// Minimize Rosenbrock
const objective = East.function([Scipy.Types.VectorType], FloatType, ($, x) => {
    const x0 = $.let(x.get(0n));
    const x1 = $.let(x.get(1n));
    const term1 = $.let(East.value(100.0).multiply(x1.subtract(x0.multiply(x0)).pow(2.0)));
    const term2 = $.let(East.value(1.0).subtract(x0).pow(2.0));
    return $.return(term1.add(term2));
});

const minimize = East.function([], Scipy.Types.OptimizeResultType, $ => {
    const x0 = $.let([0.0, 0.0]);
    const config = $.let({
        method: variant('some', variant('l_bfgs_b', null)),
        max_iter: variant('some', 1000n),
        tol: variant('some', 1e-8),
    });

    return $.return(Scipy.optimizeMinimize(objective, x0, config));
});
```

**Example - Dual Annealing (Global Optimization):**
```typescript
import { East, FloatType, variant } from "@elaraai/east";
import { Scipy } from "@elaraai/east-py-datascience";

// Minimize Rastrigin function (has many local minima)
const objective = East.function([Scipy.Types.VectorType], FloatType, ($, x) => {
    const x0 = $.let(x.get(0n));
    const x1 = $.let(x.get(1n));
    const A = $.let(10.0);
    // f(x) = 10n + sum(x_i^2 - 10*cos(2*pi*x_i))
    const term0 = $.let(x0.multiply(x0).subtract(A.multiply(x0.multiply(6.283185).cos())));
    const term1 = $.let(x1.multiply(x1).subtract(A.multiply(x1.multiply(6.283185).cos())));
    return $.return(A.multiply(2.0).add(term0).add(term1));
});

const minimize = East.function([], Scipy.Types.DualAnnealResultType, $ => {
    const bounds = $.let({
        lower: [-5.12, -5.12],
        upper: [5.12, 5.12],
    });
    const config = $.let({
        maxfun: variant('some', 1000n),
        maxiter: variant('some', 1000n),
        initial_temp: variant('none', null),
        restart_temp_ratio: variant('none', null),
        visit: variant('none', null),
        accept: variant('none', null),
        seed: variant('some', 42n),
        no_local_search: variant('none', null),
    });

    return $.return(Scipy.optimizeDualAnnealing(objective, variant('none', null), bounds, config));
});
```

---

### XGBoost (Gradient Boosting)

XGBoost provides gradient boosting for regression and classification.

**Import:**
```typescript
import { XGBoost } from "@elaraai/east-py-datascience";
```

**Functions:**
| Signature | Description |
|-----------|-------------|
| `XGBoost.trainRegressor(X: MatrixType, y: VectorType, config: XGBoostConfigType): ModelBlobType` | Train XGBoost regressor |
| `XGBoost.trainClassifier(X: MatrixType, y: LabelVectorType, config: XGBoostConfigType): ModelBlobType` | Train XGBoost classifier |
| `XGBoost.predict(model: ModelBlobType, X: MatrixType): VectorType` | Predict with regressor |
| `XGBoost.predictClass(model: ModelBlobType, X: MatrixType): LabelVectorType` | Predict class labels |
| `XGBoost.predictProba(model: ModelBlobType, X: MatrixType): MatrixType` | Get class probabilities |

**Types:**

| Type | Description |
|------|-------------|
| `XGBoost.Types.XGBoostConfigType` | Config with `n_estimators`, `max_depth`, `learning_rate`, etc. |
| `XGBoost.Types.ModelBlobType` | `xgboost_regressor` or `xgboost_classifier` |

**Config Options:**

| Field | Type | Description |
|-------|------|-------------|
| `n_estimators` | `Option<Integer>` | Number of boosting rounds (default 100) |
| `max_depth` | `Option<Integer>` | Maximum tree depth (default 6) |
| `learning_rate` | `Option<Float>` | Step size shrinkage (default 0.3) |
| `min_child_weight` | `Option<Integer>` | Minimum child weight (default 1) |
| `subsample` | `Option<Float>` | Subsample ratio (default 1.0) |
| `colsample_bytree` | `Option<Float>` | Column subsample ratio (default 1.0) |
| `reg_alpha` | `Option<Float>` | L1 regularization (default 0) |
| `reg_lambda` | `Option<Float>` | L2 regularization (default 1) |
| `random_state` | `Option<Integer>` | Random seed |
| `n_jobs` | `Option<Integer>` | Parallel threads (default -1) |

**Example - Regression:**
```typescript
import { East, variant } from "@elaraai/east";
import { XGBoost } from "@elaraai/east-py-datascience";

const train = East.function([], XGBoost.Types.ModelBlobType, $ => {
    const X = $.let([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]);
    const y = $.let([1.0, 2.0, 3.0, 4.0]);
    const config = $.let({
        n_estimators: variant('some', 100n),
        max_depth: variant('some', 3n),
        learning_rate: variant('some', 0.1),
        min_child_weight: variant('none', null),
        subsample: variant('none', null),
        colsample_bytree: variant('none', null),
        reg_alpha: variant('none', null),
        reg_lambda: variant('none', null),
        random_state: variant('some', 42n),
        n_jobs: variant('none', null),
    });
    return $.return(XGBoost.trainRegressor(X, y, config));
});
```

---

### LightGBM (Fast Gradient Boosting)

LightGBM provides fast gradient boosting with leaf-wise tree growth.

**Import:**
```typescript
import { LightGBM } from "@elaraai/east-py-datascience";
```

**Functions:**
| Signature | Description |
|-----------|-------------|
| `LightGBM.trainRegressor(X: MatrixType, y: VectorType, config: LightGBMConfigType): ModelBlobType` | Train LightGBM regressor |
| `LightGBM.trainClassifier(X: MatrixType, y: LabelVectorType, config: LightGBMConfigType): ModelBlobType` | Train LightGBM classifier |
| `LightGBM.predict(model: ModelBlobType, X: MatrixType): VectorType` | Predict with regressor |
| `LightGBM.predictClass(model: ModelBlobType, X: MatrixType): LabelVectorType` | Predict class labels |
| `LightGBM.predictProba(model: ModelBlobType, X: MatrixType): MatrixType` | Get class probabilities |

**Types:**

| Type | Description |
|------|-------------|
| `LightGBM.Types.LightGBMConfigType` | Config with `n_estimators`, `num_leaves`, etc. |
| `LightGBM.Types.ModelBlobType` | `lightgbm_regressor` or `lightgbm_classifier` |

**Config Options:**

| Field | Type | Description |
|-------|------|-------------|
| `n_estimators` | `Option<Integer>` | Number of boosting rounds (default 100) |
| `max_depth` | `Option<Integer>` | Maximum depth, -1 unlimited (default -1) |
| `learning_rate` | `Option<Float>` | Step size shrinkage (default 0.1) |
| `num_leaves` | `Option<Integer>` | Maximum leaves per tree (default 31) |
| `min_child_samples` | `Option<Integer>` | Minimum samples in leaf (default 20) |
| `subsample` | `Option<Float>` | Subsample ratio (default 1.0) |
| `colsample_bytree` | `Option<Float>` | Column subsample ratio (default 1.0) |
| `reg_alpha` | `Option<Float>` | L1 regularization (default 0) |
| `reg_lambda` | `Option<Float>` | L2 regularization (default 0) |
| `random_state` | `Option<Integer>` | Random seed |
| `n_jobs` | `Option<Integer>` | Parallel threads (default -1) |

**Example - Classification:**
```typescript
import { East, variant } from "@elaraai/east";
import { LightGBM } from "@elaraai/east-py-datascience";

const train = East.function([], LightGBM.Types.ModelBlobType, $ => {
    const X = $.let([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]);
    const y = $.let([0n, 0n, 1n, 1n]);
    const config = $.let({
        n_estimators: variant('some', 50n),
        max_depth: variant('some', 5n),
        learning_rate: variant('some', 0.1),
        num_leaves: variant('some', 31n),
        min_child_samples: variant('none', null),
        subsample: variant('none', null),
        colsample_bytree: variant('none', null),
        reg_alpha: variant('none', null),
        reg_lambda: variant('none', null),
        random_state: variant('some', 42n),
        n_jobs: variant('none', null),
    });
    return $.return(LightGBM.trainClassifier(X, y, config));
});
```

---

### NGBoost (Probabilistic Gradient Boosting)

NGBoost provides probabilistic predictions with uncertainty quantification using natural gradient boosting.

**Import:**
```typescript
import { NGBoost } from "@elaraai/east-py-datascience";
```

**Functions:**
| Signature | Description |
|-----------|-------------|
| `NGBoost.trainRegressor(X: MatrixType, y: VectorType, config: NGBoostConfigType): ModelBlobType` | Train NGBoost regressor |
| `NGBoost.predict(model: ModelBlobType, X: MatrixType): VectorType` | Point predictions (mean) |
| `NGBoost.predictDist(model: ModelBlobType, X: MatrixType, config: NGBoostPredictConfigType): NGBoostPredictResultType` | Predictions with uncertainty |

**Types:**

| Type | Description |
|------|-------------|
| `NGBoost.Types.NGBoostDistributionType` | `normal` or `lognormal` |
| `NGBoost.Types.NGBoostConfigType` | Config with `n_estimators`, `learning_rate`, etc. |
| `NGBoost.Types.NGBoostPredictConfigType` | Config with `confidence_level` |
| `NGBoost.Types.NGBoostPredictResultType` | `predictions`, `std`, `lower`, `upper` |
| `NGBoost.Types.ModelBlobType` | `ngboost_regressor` |

**Example - Probabilistic Predictions:**
```typescript
import { East, variant } from "@elaraai/east";
import { NGBoost } from "@elaraai/east-py-datascience";

const train = East.function([], NGBoost.Types.ModelBlobType, $ => {
    const X = $.let([[1.0], [2.0], [3.0], [4.0], [5.0]]);
    const y = $.let([2.1, 3.9, 6.2, 7.8, 10.1]);
    const config = $.let({
        n_estimators: variant('some', 100n),
        learning_rate: variant('some', 0.01),
        minibatch_frac: variant('none', null),
        col_sample: variant('none', null),
        random_state: variant('some', 42n),
        distribution: variant('some', variant('normal', {})),
    });
    return $.return(NGBoost.trainRegressor(X, y, config));
});

const predictWithUncertainty = East.function(
    [NGBoost.Types.ModelBlobType],
    NGBoost.Types.NGBoostPredictResultType,
    ($, model) => {
        const X_test = $.let([[1.5], [2.5], [3.5]]);
        const config = $.let({
            confidence_level: variant('some', 0.95),
        });
        return $.return(NGBoost.predictDist(model, X_test, config));
    }
);
```

---

### Torch (Neural Networks)

Torch provides neural network models (MLP) using PyTorch.

**Import:**
```typescript
import { Torch } from "@elaraai/east-py-datascience";
```

**Functions:**
| Signature | Description |
|-----------|-------------|
| `Torch.mlpTrain(X: MatrixType, y: VectorType, mlp_config: TorchMLPConfigType, train_config: TorchTrainConfigType): TorchTrainOutputType` | Train MLP model (single output) |
| `Torch.mlpPredict(model: ModelBlobType, X: MatrixType): VectorType` | Make predictions (single output) |
| `Torch.mlpTrainMulti(X: MatrixType, Y: MatrixType, mlp_config: TorchMLPConfigType, train_config: TorchTrainConfigType): TorchTrainOutputType` | Train MLP model (multi-output) |
| `Torch.mlpPredictMulti(model: ModelBlobType, X: MatrixType): MatrixType` | Make predictions (multi-output) |
| `Torch.mlpEncode(model: ModelBlobType, X: MatrixType, layer_index: Integer): MatrixType` | Extract intermediate layer activations (embeddings) |
| `Torch.mlpDecode(model: ModelBlobType, embeddings: MatrixType, layer_index: Integer): MatrixType` | Decode embeddings back through decoder portion |

**Types:**

| Type | Description |
|------|-------------|
| `Torch.Types.TorchActivationType` | `relu`, `tanh`, `sigmoid`, `leaky_relu` |
| `Torch.Types.TorchLossType` | `mse`, `mae`, `cross_entropy` |
| `Torch.Types.TorchOptimizerType` | `adam`, `sgd`, `adamw`, `rmsprop` |
| `Torch.Types.TorchMLPConfigType` | MLP architecture config |
| `Torch.Types.TorchTrainConfigType` | Training config |
| `Torch.Types.TorchTrainOutputType` | `model` + `result` (losses) |
| `Torch.Types.ModelBlobType` | Serialized PyTorch MLP model (contains `data`, `n_features`, `hidden_layers`, `output_dim`) |

**MLP Config Options:**

| Field | Type | Description |
|-------|------|-------------|
| `hidden_layers` | `Array<Integer>` | Hidden layer sizes, e.g., [64, 32] |
| `activation` | `Option<Activation>` | Activation function (default relu) |
| `dropout` | `Option<Float>` | Dropout rate (default 0.0) |
| `output_dim` | `Option<Integer>` | Output dimension (default 1) |

**Train Config Options:**

| Field | Type | Description |
|-------|------|-------------|
| `epochs` | `Option<Integer>` | Number of epochs (default 100) |
| `batch_size` | `Option<Integer>` | Batch size (default 32) |
| `learning_rate` | `Option<Float>` | Learning rate (default 0.001) |
| `loss` | `Option<Loss>` | Loss function (default mse) |
| `optimizer` | `Option<Optimizer>` | Optimizer (default adam) |
| `early_stopping` | `Option<Integer>` | Patience, 0=disabled |
| `validation_split` | `Option<Float>` | Validation fraction (default 0.2) |
| `random_state` | `Option<Integer>` | Random seed |

**Example - MLP Training:**
```typescript
import { East, variant } from "@elaraai/east";
import { Torch } from "@elaraai/east-py-datascience";

const train = East.function([], Torch.Types.TorchTrainOutputType, $ => {
    const X = $.let([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]);
    const y = $.let([3.0, 7.0, 11.0, 15.0]);

    const mlp_config = $.let({
        hidden_layers: [32n, 16n],
        activation: variant('some', variant('relu', {})),
        dropout: variant('some', 0.1),
        output_dim: variant('none', null),
    });

    const train_config = $.let({
        epochs: variant('some', 100n),
        batch_size: variant('some', 4n),
        learning_rate: variant('some', 0.01),
        loss: variant('none', null),
        optimizer: variant('none', null),
        early_stopping: variant('some', 10n),
        validation_split: variant('some', 0.2),
        random_state: variant('some', 42n),
    });

    return $.return(Torch.mlpTrain(X, y, mlp_config, train_config));
});
```

**Example - Multi-Output MLP Training:**
```typescript
import { East, variant } from "@elaraai/east";
import { Torch } from "@elaraai/east-py-datascience";

// Train multi-output regression (e.g., predicting 2 targets)
const train = East.function([], Torch.Types.TorchTrainOutputType, $ => {
    const X = $.let([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]);
    const Y = $.let([[3.0, 1.0], [7.0, 2.0], [11.0, 3.0], [15.0, 4.0]]);  // 2 output columns

    const mlp_config = $.let({
        hidden_layers: [32n, 16n],
        activation: variant('some', variant('relu', {})),
        dropout: variant('some', 0.1),
        output_dim: variant('none', null),  // Inferred from Y.shape[1]
    });

    const train_config = $.let({
        epochs: variant('some', 100n),
        batch_size: variant('some', 4n),
        learning_rate: variant('some', 0.01),
        loss: variant('none', null),
        optimizer: variant('none', null),
        early_stopping: variant('some', 10n),
        validation_split: variant('some', 0.2),
        random_state: variant('some', 42n),
    });

    return $.return(Torch.mlpTrainMulti(X, Y, mlp_config, train_config));
});

// Predict with multi-output model
const predict = East.function([Torch.Types.ModelBlobType], Torch.Types.MatrixType, ($, model) => {
    const X_test = $.let([[2.0, 3.0], [4.0, 5.0]]);
    return $.return(Torch.mlpPredictMulti(model, X_test));
    // Returns matrix with shape [2, 2] (2 samples, 2 outputs)
});
```

**Example - Autoencoder Encode/Decode:**
```typescript
import { East, variant } from "@elaraai/east";
import { Torch } from "@elaraai/east-py-datascience";

// Train autoencoder: 4 features -> 8 -> 2 (bottleneck) -> 8 -> 4 features
const trainAutoencoder = East.function([], Torch.Types.TorchTrainOutputType, $ => {
    const X = $.let([
        [1.0, 0.0, 0.0, 0.0],  // One-hot encoded origins
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]);

    const mlp_config = $.let({
        hidden_layers: [8n, 2n, 8n],  // Bottleneck at index 1 (2 dimensions)
        activation: variant('some', variant('relu', {})),
        dropout: variant('none', null),
        output_dim: variant('none', null),
    });

    const train_config = $.let({
        epochs: variant('some', 100n),
        batch_size: variant('some', 2n),
        learning_rate: variant('some', 0.01),
        loss: variant('some', variant('mse', {})),
        optimizer: variant('some', variant('adam', {})),
        early_stopping: variant('some', 20n),
        validation_split: variant('some', 0.2),
        random_state: variant('some', 42n),
    });

    // Train as autoencoder (input = output)
    return $.return(Torch.mlpTrainMulti(X, X, mlp_config, train_config));
});

// Extract embeddings and blend them
const blendOrigins = East.function([Torch.Types.ModelBlobType], Torch.Types.MatrixType, ($, model) => {
    const X_origins = $.let([
        [1.0, 0.0, 0.0, 0.0],  // Origin A
        [0.0, 1.0, 0.0, 0.0],  // Origin B
    ]);

    // Extract bottleneck embeddings (layer_index=1 for the 2-dim bottleneck)
    const embeddings = $.let(Torch.mlpEncode(model, X_origins, 1n));
    // embeddings: [[emb_A_0, emb_A_1], [emb_B_0, emb_B_1]]

    // Compute 50/50 blend embedding
    const emb_A = $.let(embeddings.get(0n));
    const emb_B = $.let(embeddings.get(1n));
    const blend_emb = $.let([
        emb_A.get(0n).multiply(0.5).add(emb_B.get(0n).multiply(0.5)),
        emb_A.get(1n).multiply(0.5).add(emb_B.get(1n).multiply(0.5)),
    ]);

    // Decode blended embedding back to output space
    const blend_matrix = $.let([blend_emb]);
    const reconstructed = $.let(Torch.mlpDecode(model, blend_matrix, 1n));
    // reconstructed: [[0.5, 0.5, 0.0, 0.0]] (approx blended weights)

    return $.return(reconstructed);
});
```

---

### GP (Gaussian Process)

GP provides Gaussian Process regression with uncertainty quantification.

**Import:**
```typescript
import { GP } from "@elaraai/east-py-datascience";
```

**Functions:**
| Signature | Description |
|-----------|-------------|
| `GP.train(X: MatrixType, y: VectorType, config: GPConfigType): ModelBlobType` | Train GP regressor |
| `GP.predict(model: ModelBlobType, X: MatrixType): VectorType` | Point predictions (mean) |
| `GP.predictStd(model: ModelBlobType, X: MatrixType): GPPredictResultType` | Predictions with uncertainty |

**Types:**

| Type | Description |
|------|-------------|
| `GP.Types.GPKernelType` | `rbf`, `matern_1_2`, `matern_3_2`, `matern_5_2`, `rational_quadratic`, `dot_product` |
| `GP.Types.GPConfigType` | Config with `kernel`, `alpha`, `normalize_y`, etc. |
| `GP.Types.GPPredictResultType` | `mean` and `std` vectors |
| `GP.Types.ModelBlobType` | `gp_regressor` |

**Config Options:**

| Field | Type | Description |
|-------|------|-------------|
| `kernel` | `Option<Kernel>` | Kernel type (default rbf) |
| `alpha` | `Option<Float>` | Noise level (default 1e-10) |
| `n_restarts_optimizer` | `Option<Integer>` | Optimizer restarts (default 0) |
| `normalize_y` | `Option<Boolean>` | Normalize targets (default false) |
| `random_state` | `Option<Integer>` | Random seed |

**Example - GP with Uncertainty:**
```typescript
import { East, variant } from "@elaraai/east";
import { GP } from "@elaraai/east-py-datascience";

const train = East.function([], GP.Types.ModelBlobType, $ => {
    const X = $.let([[1.0], [2.0], [3.0], [4.0], [5.0]]);
    const y = $.let([1.0, 4.0, 9.0, 16.0, 25.0]);  // y = x^2
    const config = $.let({
        kernel: variant('some', variant('rbf', {})),
        alpha: variant('some', 1e-10),
        n_restarts_optimizer: variant('some', 5n),
        normalize_y: variant('some', true),
        random_state: variant('some', 42n),
    });
    return $.return(GP.train(X, y, config));
});

const predictWithStd = East.function(
    [GP.Types.ModelBlobType],
    GP.Types.GPPredictResultType,
    ($, model) => {
        const X_test = $.let([[1.5], [2.5], [3.5]]);
        return $.return(GP.predictStd(model, X_test));
    }
);
```

---

### Shap (Model Explainability)

Shap provides model-agnostic feature importance using SHAP values.

**Import:**
```typescript
import { Shap } from "@elaraai/east-py-datascience";
```

**Functions:**
| Signature | Description |
|-----------|-------------|
| `Shap.treeExplainerCreate(model: TreeModelBlobType): ShapModelBlobType` | Create TreeExplainer for tree models |
| `Shap.kernelExplainerCreate(model: AnyModelBlobType, X_background: MatrixType): ShapModelBlobType` | Create KernelExplainer for any model |
| `Shap.computeValues(explainer: ShapModelBlobType, X: MatrixType, feature_names: Array<String>): ShapResultType` | Compute SHAP values |
| `Shap.featureImportance(shap_values: MatrixType, feature_names: Array<String>): FeatureImportanceType` | Get global feature importance |

**Types:**

| Type | Description |
|------|-------------|
| `Shap.Types.ShapModelBlobType` | `shap_tree_explainer` or `shap_kernel_explainer` |
| `Shap.Types.ShapResultType` | `shap_values`, `base_value`, `feature_names` |
| `Shap.Types.FeatureImportanceType` | `feature_names`, `importances`, `std` |
| `Shap.Types.TreeModelBlobType` | XGBoost or LightGBM models |
| `Shap.Types.AnyModelBlobType` | Any supported model |

**Example - Feature Importance:**
```typescript
import { East, variant } from "@elaraai/east";
import { Shap, XGBoost } from "@elaraai/east-py-datascience";

const explain = East.function(
    [XGBoost.Types.ModelBlobType, Shap.Types.MatrixType],
    Shap.Types.FeatureImportanceType,
    ($, model, X) => {
        const explainer = $.let(Shap.treeExplainerCreate(model));
        const feature_names = $.let(["feature1", "feature2"]);
        const shap_result = $.let(Shap.computeValues(explainer, X, feature_names));
        const importance = $.let(Shap.featureImportance(shap_result.shap_values, feature_names));
        return $.return(importance);
    }
);
```

---

## Error Handling

Platform functions throw errors on failure. Common scenarios:

- **Invalid bounds**: Lower bound exceeds upper bound
- **Dimension mismatch**: X and y have different sample counts
- **Invalid parameter space**: Missing required fields
- **Optimization failure**: No feasible solution found
- **Model mismatch**: Wrong model type passed to predict function

---

## License

- **TypeScript (npm)**: Dual-licensed under AGPL-3.0 and commercial license
- **Python runtime**: Business Source License 1.1 (BSL 1.1)

See [LICENSE.md](LICENSE.md) for details.
