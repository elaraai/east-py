# East Data Science API Reference

Complete function signatures, types, and arguments for all data science platform modules.

---

## Table of Contents

- [MADS (Derivative-Free Optimization)](#mads-derivative-free-optimization)
- [Optuna (Bayesian Optimization)](#optuna-bayesian-optimization)
- [SimAnneal (Simulated Annealing)](#simanneal-simulated-annealing)
- [ALNS (Adaptive Large Neighborhood Search)](#alns-adaptive-large-neighborhood-search)
- [Sklearn (Machine Learning Utilities)](#sklearn-machine-learning-utilities)
- [Scipy (Scientific Computing)](#scipy-scientific-computing)
- [XGBoost (Gradient Boosting)](#xgboost-gradient-boosting)
- [LightGBM (Fast Gradient Boosting)](#lightgbm-fast-gradient-boosting)
- [NGBoost (Probabilistic Gradient Boosting)](#ngboost-probabilistic-gradient-boosting)
- [Torch (Neural Networks)](#torch-neural-networks)
- [Lightning (PyTorch Lightning)](#lightning-pytorch-lightning)
- [GP (Gaussian Process)](#gp-gaussian-process)
- [MAPIE (Conformal Prediction)](#mapie-conformal-prediction)
- [Shap (Model Explainability)](#shap-model-explainability)

---

## MADS (Derivative-Free Optimization)

MADS (Mesh Adaptive Direct Search) provides derivative-free blackbox optimization using the NOMAD algorithm.

**Import:**
```typescript
import { MADS, MADSConstraintType } from "@elaraai/east-py-datascience";
```

**Functions:**
| Signature | Description |
|-----------|-------------|
| `MADS.optimize(objective: ScalarObjectiveType, x0: VectorType, bounds: BoundsType, constraints: OptionType<Array<ConstraintType>>, config: ConfigType): ResultType` | Run MADS optimization |

**Types:**

| Type | Description |
|------|-------------|
| `MADS.Types.VectorType` | `ArrayType(FloatType)` |
| `MADS.Types.ScalarObjectiveType` | `FunctionType([VectorType], FloatType)` |
| `MADS.Types.BoundsType` | `StructType({ lower: VectorType, upper: VectorType })` |
| `MADS.Types.DirectionType` | `VariantType({ ortho_2n, ortho_n_plus_1, lt_2n, single })` |
| `MADS.Types.ConfigType` | `StructType({ max_bb_eval: OptionType<Integer>, display_degree: OptionType<Integer>, direction_type: OptionType<DirectionType>, initial_mesh_size: OptionType<Float>, min_mesh_size: OptionType<Float>, seed: OptionType<Integer> })` |
| `MADS.Types.ResultType` | `StructType({ x_best: VectorType, f_best: Float, bb_eval: Integer, success: Boolean })` |
| `MADSConstraintType` | `VariantType({ eb: ScalarObjectiveType, pb: ScalarObjectiveType })` |

**Config Options:**

| Field | Type | Description |
|-------|------|-------------|
| `max_bb_eval` | `OptionType<Integer>` | Maximum blackbox evaluations |
| `display_degree` | `OptionType<Integer>` | Output verbosity (0=silent) |
| `direction_type` | `OptionType<DirectionType>` | Search direction strategy |
| `initial_mesh_size` | `OptionType<Float>` | Initial mesh granularity |
| `min_mesh_size` | `OptionType<Float>` | Minimum mesh size (stopping criterion) |
| `seed` | `OptionType<Integer>` | Random seed for reproducibility |

---

## Optuna (Bayesian Optimization)

Optuna provides Bayesian optimization using the TPE (Tree-structured Parzen Estimator) sampler.

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
| `Optuna.Types.ParamValueType` | `VariantType({ int: Integer, float: Float, string: String, bool: Boolean })` |
| `Optuna.Types.ParamSpaceKindType` | `VariantType({ int, float, categorical, log_uniform })` |
| `Optuna.Types.ParamSpaceType` | `StructType({ name: String, kind: ParamSpaceKindType, low: OptionType<Float>, high: OptionType<Float>, choices: OptionType<Array<ParamValueType>> })` |
| `Optuna.Types.NamedParamType` | `StructType({ name: String, value: ParamValueType })` |
| `Optuna.Types.OptimizationDirectionType` | `VariantType({ minimize, maximize })` |
| `Optuna.Types.PrunerType` | `VariantType({ none, median, hyperband })` |
| `Optuna.Types.StudyConfigType` | `StructType({ direction: OptionType<OptimizationDirectionType>, n_trials: Integer, random_state: OptionType<Integer>, pruner: OptionType<PrunerType> })` |
| `Optuna.Types.TrialResultType` | `StructType({ trial_id: Integer, params: Array<NamedParamType>, score: Float })` |
| `Optuna.Types.StudyResultType` | `StructType({ best_params: Array<NamedParamType>, best_score: Float, trials: Array<TrialResultType> })` |

**Config Options:**

| Field | Type | Description |
|-------|------|-------------|
| `direction` | `OptionType<Direction>` | `minimize` or `maximize` |
| `n_trials` | `Integer` | Number of optimization trials |
| `random_state` | `OptionType<Integer>` | Random seed for reproducibility |
| `pruner` | `OptionType<Pruner>` | Early stopping (`none`, `median`, `hyperband`) |

---

## SimAnneal (Simulated Annealing)

Simulated Annealing provides discrete/combinatorial optimization for permutation and subset problems.

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
| `SimAnneal.Types.DiscreteStateType` | `VariantType({ int_array: Array<Integer>, bool_array: Array<Boolean> })` |
| `SimAnneal.Types.EnergyFunctionType` | `FunctionType([DiscreteStateType], FloatType)` |
| `SimAnneal.Types.MoveFunctionType` | `FunctionType([DiscreteStateType], DiscreteStateType)` |
| `SimAnneal.Types.PermutationEnergyType` | `FunctionType([Array<Integer>], FloatType)` |
| `SimAnneal.Types.SubsetEnergyType` | `FunctionType([Array<Boolean>], FloatType)` |
| `SimAnneal.Types.ConfigType` | `StructType({ t_max: OptionType<Float>, t_min: OptionType<Float>, steps: OptionType<Integer>, updates: OptionType<Integer>, auto_schedule: OptionType<Float>, random_state: OptionType<Integer> })` |
| `SimAnneal.Types.ResultType` | `StructType({ best_state: DiscreteStateType, best_energy: Float, success: Boolean })` |

**Config Options:**

| Field | Type | Description |
|-------|------|-------------|
| `t_max` | `OptionType<Float>` | Starting temperature (default 25000.0) |
| `t_min` | `OptionType<Float>` | Ending temperature (default 2.5) |
| `steps` | `OptionType<Integer>` | Total iterations (default 50000) |
| `updates` | `OptionType<Integer>` | Progress report frequency (0=silent) |
| `auto_schedule` | `OptionType<Float>` | Minutes for auto-calibration |
| `random_state` | `OptionType<Integer>` | Random seed for reproducibility |

---

## ALNS (Adaptive Large Neighborhood Search)

ALNS provides combinatorial optimization using destroy-repair operators. Designed for problems where:
- Solutions are combinatorial (assignments, schedules, routes)
- Domain-specific destroy/repair operators can be defined
- The objective function may be complex or black-box

**Import:**
```typescript
import { ALNS } from "@elaraai/east-py-datascience";
```

**Functions:**
| Signature | Description |
|-----------|-------------|
| `ALNS.optimize([SolutionType], initial: S, objective: S -> Float, destroy_operators: Array<S -> S>, repair_operators: Array<S -> S>, config: ConfigType): ResultType` | Run ALNS optimization (generic over solution type S) |

**Types:**

| Type | Description |
|------|-------------|
| `ALNS.Types.SimulatedAnnealingConfigType` | `StructType({ start_temperature: OptionType<Float>, end_temperature: OptionType<Float>, step: OptionType<Float> })` |
| `ALNS.Types.RecordToRecordConfigType` | `StructType({ threshold: OptionType<Float> })` |
| `ALNS.Types.AcceptanceCriterionType` | `VariantType({ simulated_annealing, hill_climbing, record_to_record })` |
| `ALNS.Types.RouletteWheelConfigType` | `StructType({ scores: OptionType<Array<Integer>>, decay: OptionType<Float> })` |
| `ALNS.Types.OperatorSelectionType` | `VariantType({ roulette_wheel })` |
| `ALNS.Types.StopCriterionType` | `VariantType({ max_iterations: Integer, max_runtime: Float, no_improvement: Integer })` |
| `ALNS.Types.ConfigType` | `StructType({ stop: OptionType<StopCriterionType>, acceptance: OptionType<AcceptanceCriterionType>, operator_selection: OptionType<OperatorSelectionType>, seed: OptionType<Integer> })` |
| `ALNS.Types.ResultType` | `StructType({ best_solution: "S", best_objective: Float, iterations: Integer, runtime: Float, success: Boolean })` |

**Config Options:**

| Field | Type | Description |
|-------|------|-------------|
| `stop` | `OptionType<StopCriterionType>` | Stopping criterion (default: max_iterations 1000) |
| `acceptance` | `OptionType<AcceptanceCriterionType>` | Acceptance criterion (default: simulated_annealing) |
| `operator_selection` | `OptionType<OperatorSelectionType>` | Operator selection strategy (default: roulette_wheel) |
| `seed` | `OptionType<Integer>` | Random seed for reproducibility |

**Stop Criteria:**

| Variant | Description |
|---------|-------------|
| `max_iterations` | Stop after N iterations |
| `max_runtime` | Stop after N seconds |
| `no_improvement` | Stop after N iterations without improvement |

**Acceptance Criteria:**

| Variant | Description |
|---------|-------------|
| `simulated_annealing` | Probabilistic acceptance (start_temp, end_temp, step) |
| `hill_climbing` | Only accept improvements |
| `record_to_record` | Accept if within threshold of best |

**Operator Selection:**

| Variant | Description |
|---------|-------------|
| `roulette_wheel` | Weighted random selection (scores: [new_best, better, accepted, rejected], decay) |

**Generic Platform Function:**

ALNS uses a generic platform function where the solution type `S` is user-defined. Pass the type parameter first:

```typescript
// Call with type parameter array first
ALNS.optimize([MySolutionType], initial, objective, destroyOps, repairOps, config)
```

---

## Sklearn (Machine Learning Utilities)

Sklearn provides core ML utilities: preprocessing, data splitting, metrics, and multi-target regression.

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
| `Sklearn.Types.SplitConfigType` | `StructType({ test_size: OptionType<Float>, random_state: OptionType<Integer>, shuffle: OptionType<Boolean>, stratify: OptionType<LabelVectorType>, min_stratify_samples: OptionType<Integer> })` |
| `Sklearn.Types.SplitResultType` | `StructType({ X_train: MatrixType, X_test: MatrixType, y_train: VectorType, y_test: VectorType, rejected_indices: LabelVectorType })` |
| `Sklearn.Types.ThreeWaySplitConfigType` | `StructType({ val_size: OptionType<Float>, test_size: OptionType<Float>, random_state: OptionType<Integer>, shuffle: OptionType<Boolean>, stratify: OptionType<LabelVectorType>, min_stratify_samples: OptionType<Integer> })` |
| `Sklearn.Types.ThreeWaySplitResultType` | `StructType({ X_train, X_val, X_test: MatrixType, Y_train, Y_val, Y_test: MatrixType, rejected_indices: LabelVectorType })` |
| `Sklearn.Types.RegressionMetricType` | `VariantType({ mse, rmse, mae, r2, mape, explained_variance, max_error, median_ae, mean_error, pinball_loss: Float, huber: Float, mean_tweedie_deviance: Float })` |
| `Sklearn.Types.ClassificationMetricType` | `VariantType({ accuracy, balanced_accuracy, precision, recall, f1, matthews_corrcoef, cohen_kappa, jaccard })` |
| `Sklearn.Types.MetricAggregationType` | `VariantType({ per_target, uniform_average })` |
| `Sklearn.Types.ClassificationAverageType` | `VariantType({ macro, micro, weighted, binary })` |
| `Sklearn.Types.MetricsResultType` | `ArrayType(StructType({ metric: RegressionMetricType, value: Float }))` |
| `Sklearn.Types.MultiMetricsConfigType` | `StructType({ aggregation: OptionType<MetricAggregationType> })` |
| `Sklearn.Types.MultiMetricsResultType` | `ArrayType(StructType({ metric: RegressionMetricType, value: VariantType({ scalar: Float, per_target: VectorType }) }))` |
| `Sklearn.Types.ClassificationMetricsConfigType` | `StructType({ average: OptionType<ClassificationAverageType> })` |
| `Sklearn.Types.ClassificationMetricResultsType` | `ArrayType(StructType({ metric: ClassificationMetricType, value: Float }))` |
| `Sklearn.Types.MultiClassificationConfigType` | `StructType({ average: OptionType<ClassificationAverageType>, aggregation: OptionType<MetricAggregationType> })` |
| `Sklearn.Types.MultiClassificationMetricResultsType` | `ArrayType(StructType({ metric: ClassificationMetricType, value: VariantType({ scalar: Float, per_target: VectorType }) }))` |
| `Sklearn.Types.RegressorChainBaseConfigType` | `VariantType({ xgboost: XGBoostConfigType, lightgbm: LightGBMConfigType, ngboost: NGBoostConfigType, gp: GPConfigType })` |
| `Sklearn.Types.RegressorChainConfigType` | `StructType({ base_estimator: RegressorChainBaseConfigType, order: OptionType<Array<Integer>>, random_state: OptionType<Integer> })` |

---

## Scipy (Scientific Computing)

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
| `Scipy.statsPercentile(data: VectorType, percentiles: VectorType): VectorType` | Compute percentiles (0-100) |
| `Scipy.statsIqr(data: VectorType): Float` | Compute interquartile range (Q3 - Q1) |
| `Scipy.statsMedian(data: VectorType): Float` | Compute median value |
| `Scipy.statsMad(data: VectorType): Float` | Compute median absolute deviation |
| `Scipy.statsRobust(data: VectorType): RobustStatsResultType` | Compute robust statistics (median, iqr, mad, q1, q3) |
| `Scipy.interpolate1dFit(x: VectorType, y: VectorType, config: InterpolateConfigType): ModelBlobType` | Fit 1D interpolator |
| `Scipy.interpolate1dPredict(model: ModelBlobType, x: VectorType): VectorType` | Evaluate interpolator |
| `Scipy.optimizeMinimize(objective: ScalarObjectiveType, x0: VectorType, config: OptimizeConfigType): OptimizeResultType` | Minimize scalar function |
| `Scipy.optimizeMinimizeQuadratic(x0: VectorType, quadratic_config: QuadraticConfigType, opt_config: OptimizeConfigType): OptimizeResultType` | Minimize quadratic function |
| `Scipy.optimizeDualAnnealing(objective: ScalarObjectiveType, x0: OptionType<VectorType>, bounds: DualAnnealBoundsType, config: DualAnnealConfigType): DualAnnealResultType` | Global optimization using dual annealing |

**Types:**

| Type | Description |
|------|-------------|
| `Scipy.Types.OptimizeMethodType` | `VariantType({ bfgs, l_bfgs_b, nelder_mead, powell, cg })` |
| `Scipy.Types.InterpolationKindType` | `VariantType({ linear, cubic, quadratic })` |
| `Scipy.Types.OptimizeConfigType` | `StructType({ method: OptionType<OptimizeMethodType>, max_iter: OptionType<Integer>, tol: OptionType<Float> })` |
| `Scipy.Types.InterpolateConfigType` | `StructType({ kind: OptionType<InterpolationKindType> })` |
| `Scipy.Types.CurveFitConfigType` | `StructType({ max_iter: OptionType<Integer>, initial_guess: OptionType<VectorType> })` |
| `Scipy.Types.QuadraticConfigType` | `StructType({ A: MatrixType, b: VectorType, c: Float })` - Quadratic f(x) = 0.5*x'Ax + b'x + c |
| `Scipy.Types.CurveFunctionType` | `VariantType({ exponential_decay, exponential_growth, logistic, power_law, linear, quadratic, cubic, custom: { fn, n_params, param_bounds } })` |
| `Scipy.Types.StatsDescribeResultType` | `StructType({ count: Integer, mean: Float, variance: Float, skewness: Float, kurtosis: Float, min: Float, max: Float })` |
| `Scipy.Types.RobustStatsResultType` | `StructType({ median: Float, iqr: Float, mad: Float, q1: Float, q3: Float })` |
| `Scipy.Types.CorrelationResultType` | `StructType({ correlation: Float, pvalue: Float })` |
| `Scipy.Types.CurveFitResultType` | `StructType({ params: VectorType, success: Boolean, r_squared: Float })` |
| `Scipy.Types.OptimizeResultType` | `StructType({ x: VectorType, fun: Float, success: Boolean, nit: Integer })` |
| `Scipy.Types.DualAnnealBoundsType` | `StructType({ lower: VectorType, upper: VectorType })` |
| `Scipy.Types.DualAnnealConfigType` | `StructType({ maxfun: OptionType<Integer>, maxiter: OptionType<Integer>, initial_temp: OptionType<Float>, restart_temp_ratio: OptionType<Float>, visit: OptionType<Float>, accept: OptionType<Float>, seed: OptionType<Integer>, no_local_search: OptionType<Boolean> })` |
| `Scipy.Types.DualAnnealResultType` | `StructType({ x: VectorType, fun: Float, nfev: Integer, nit: Integer, success: Boolean, message: String })` |

---

## XGBoost (Gradient Boosting)

XGBoost provides gradient boosting for regression, classification, and quantile regression.

**Import:**
```typescript
import { XGBoost } from "@elaraai/east-py-datascience";
```

**Functions:**
| Signature | Description |
|-----------|-------------|
| `XGBoost.trainRegressor(X: MatrixType, y: VectorType, config: XGBoostConfigType): ModelBlobType` | Train XGBoost regressor |
| `XGBoost.trainClassifier(X: MatrixType, y: LabelVectorType, config: XGBoostConfigType): ModelBlobType` | Train XGBoost classifier |
| `XGBoost.trainQuantile(X: MatrixType, y: VectorType, config: XGBoostQuantileConfigType): ModelBlobType` | Train XGBoost quantile regressor |
| `XGBoost.predict(model: ModelBlobType, X: MatrixType): VectorType` | Predict with regressor |
| `XGBoost.predictClass(model: ModelBlobType, X: MatrixType): LabelVectorType` | Predict class labels |
| `XGBoost.predictProba(model: ModelBlobType, X: MatrixType): MatrixType` | Get class probabilities |
| `XGBoost.predictQuantile(model: ModelBlobType, X: MatrixType): XGBoostQuantilePredictResultType` | Predict quantiles |

**Types:**

| Type | Description |
|------|-------------|
| `XGBoost.Types.XGBoostConfigType` | `StructType({ n_estimators: OptionType<Integer>, max_depth: OptionType<Integer>, learning_rate: OptionType<Float>, min_child_weight: OptionType<Integer>, subsample: OptionType<Float>, colsample_bytree: OptionType<Float>, reg_alpha: OptionType<Float>, reg_lambda: OptionType<Float>, gamma: OptionType<Float>, random_state: OptionType<Integer>, n_jobs: OptionType<Integer>, sample_weight: OptionType<VectorType>, categorical_features: OptionType<LabelVectorType>, max_cat_to_onehot: OptionType<Integer>, max_cat_threshold: OptionType<Integer> })` |
| `XGBoost.Types.XGBoostQuantileConfigType` | `StructType({ quantiles: VectorType, n_estimators: OptionType<Integer>, max_depth: OptionType<Integer>, learning_rate: OptionType<Float>, min_child_weight: OptionType<Integer>, subsample: OptionType<Float>, colsample_bytree: OptionType<Float>, reg_alpha: OptionType<Float>, reg_lambda: OptionType<Float>, gamma: OptionType<Float>, random_state: OptionType<Integer>, n_jobs: OptionType<Integer>, sample_weight: OptionType<VectorType>, categorical_features: OptionType<LabelVectorType>, max_cat_to_onehot: OptionType<Integer>, max_cat_threshold: OptionType<Integer> })` |
| `XGBoost.Types.XGBoostQuantilePredictResultType` | `StructType({ quantiles: VectorType, predictions: MatrixType })` |
| `XGBoost.Types.ModelBlobType` | `xgboost_regressor`, `xgboost_classifier`, or `xgboost_quantile` |

**Config Options:**

| Field | Type | Description |
|-------|------|-------------|
| `n_estimators` | `OptionType<Integer>` | Number of boosting rounds (default 100) |
| `max_depth` | `OptionType<Integer>` | Maximum tree depth (default 6) |
| `learning_rate` | `OptionType<Float>` | Step size shrinkage (default 0.3) |
| `min_child_weight` | `OptionType<Integer>` | Minimum child weight (default 1) |
| `subsample` | `OptionType<Float>` | Subsample ratio (default 1.0) |
| `colsample_bytree` | `OptionType<Float>` | Column subsample ratio (default 1.0) |
| `reg_alpha` | `OptionType<Float>` | L1 regularization (default 0) |
| `reg_lambda` | `OptionType<Float>` | L2 regularization (default 1) |
| `random_state` | `OptionType<Integer>` | Random seed |
| `n_jobs` | `OptionType<Integer>` | Parallel threads (default -1) |
| `sample_weight` | `OptionType<Vector>` | Sample weights for training |

---

## LightGBM (Fast Gradient Boosting)

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
| `LightGBM.Types.LightGBMConfigType` | `StructType({ n_estimators: OptionType<Integer>, max_depth: OptionType<Integer>, learning_rate: OptionType<Float>, num_leaves: OptionType<Integer>, min_child_samples: OptionType<Integer>, subsample: OptionType<Float>, colsample_bytree: OptionType<Float>, reg_alpha: OptionType<Float>, reg_lambda: OptionType<Float>, random_state: OptionType<Integer>, n_jobs: OptionType<Integer> })` |
| `LightGBM.Types.ModelBlobType` | `lightgbm_regressor` or `lightgbm_classifier` |

**Config Options:**

| Field | Type | Description |
|-------|------|-------------|
| `n_estimators` | `OptionType<Integer>` | Number of boosting rounds (default 100) |
| `max_depth` | `OptionType<Integer>` | Maximum depth, -1 unlimited (default -1) |
| `learning_rate` | `OptionType<Float>` | Step size shrinkage (default 0.1) |
| `num_leaves` | `OptionType<Integer>` | Maximum leaves per tree (default 31) |
| `min_child_samples` | `OptionType<Integer>` | Minimum samples in leaf (default 20) |
| `subsample` | `OptionType<Float>` | Subsample ratio (default 1.0) |
| `colsample_bytree` | `OptionType<Float>` | Column subsample ratio (default 1.0) |
| `reg_alpha` | `OptionType<Float>` | L1 regularization (default 0) |
| `reg_lambda` | `OptionType<Float>` | L2 regularization (default 0) |
| `random_state` | `OptionType<Integer>` | Random seed |
| `n_jobs` | `OptionType<Integer>` | Parallel threads (default -1) |

---

## NGBoost (Probabilistic Gradient Boosting)

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
| `NGBoost.Types.NGBoostDistributionType` | `VariantType({ normal, lognormal })` |
| `NGBoost.Types.NGBoostConfigType` | `StructType({ n_estimators: OptionType<Integer>, learning_rate: OptionType<Float>, minibatch_frac: OptionType<Float>, col_sample: OptionType<Float>, random_state: OptionType<Integer>, distribution: OptionType<NGBoostDistributionType> })` |
| `NGBoost.Types.NGBoostPredictConfigType` | `StructType({ confidence_level: OptionType<Float> })` |
| `NGBoost.Types.NGBoostPredictResultType` | `StructType({ predictions: VectorType, std: OptionType<VectorType>, lower: OptionType<VectorType>, upper: OptionType<VectorType> })` |
| `NGBoost.Types.ModelBlobType` | `ngboost_regressor` |

---

## Torch (Neural Networks)

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
| `Torch.Types.TorchActivationType` | `VariantType({ relu, tanh, sigmoid, leaky_relu })` |
| `Torch.Types.TorchOutputActivationType` | `VariantType({ none, softmax, sigmoid })` |
| `Torch.Types.TorchLossType` | `VariantType({ mse, mae, cross_entropy, kl_div, bce, bce_with_logits })` |
| `Torch.Types.TorchOptimizerType` | `VariantType({ adam, sgd, adamw, rmsprop })` |
| `Torch.Types.TorchMLPConfigType` | `StructType({ hidden_layers: ArrayType<Integer>, activation: OptionType<TorchActivationType>, output_activation: OptionType<TorchOutputActivationType>, dropout: OptionType<Float>, output_dim: OptionType<Integer> })` |
| `Torch.Types.TorchTrainConfigType` | `StructType({ epochs: OptionType<Integer>, batch_size: OptionType<Integer>, learning_rate: OptionType<Float>, loss: OptionType<TorchLossType>, optimizer: OptionType<TorchOptimizerType>, early_stopping: OptionType<Integer>, validation_split: OptionType<Float>, random_state: OptionType<Integer> })` |
| `Torch.Types.TorchTrainResultType` | `StructType({ train_losses: VectorType, val_losses: VectorType, best_epoch: Integer })` |
| `Torch.Types.TorchTrainOutputType` | `StructType({ model: ModelBlobType, result: TorchTrainResultType })` |
| `Torch.Types.ModelBlobType` | Serialized PyTorch MLP model |

**MLP Config Options:**

| Field | Type | Description |
|-------|------|-------------|
| `hidden_layers` | `Array<Integer>` | Hidden layer sizes, e.g., [64, 32] |
| `activation` | `OptionType<Activation>` | Hidden layer activation (default relu) |
| `output_activation` | `OptionType<OutputActivation>` | Output layer activation (default none/linear) |
| `dropout` | `OptionType<Float>` | Dropout rate (default 0.0) |
| `output_dim` | `OptionType<Integer>` | Output dimension (default 1) |

**Train Config Options:**

| Field | Type | Description |
|-------|------|-------------|
| `epochs` | `OptionType<Integer>` | Number of epochs (default 100) |
| `batch_size` | `OptionType<Integer>` | Batch size (default 32) |
| `learning_rate` | `OptionType<Float>` | Learning rate (default 0.001) |
| `loss` | `OptionType<Loss>` | Loss function (default mse) |
| `optimizer` | `OptionType<Optimizer>` | Optimizer (default adam) |
| `early_stopping` | `OptionType<Integer>` | Patience, 0=disabled |
| `validation_split` | `OptionType<Float>` | Validation fraction (default 0.2) |
| `random_state` | `OptionType<Integer>` | Random seed |

---

## Lightning (PyTorch Lightning)

Lightning provides production-grade neural network training using PyTorch Lightning with early stopping, gradient clipping, multiple architectures (including temporal), and conditional generation.

**Import:**
```typescript
import { Lightning } from "@elaraai/east-py-datascience";
```

**Functions:**
| Signature | Description |
|-----------|-------------|
| `Lightning.train(X: MatrixType, y: MatrixType, config: ConfigType, masks: OptionType<Tensor3DBoolType>, group_weights: OptionType<GroupWeightsType>, conditions: OptionType<MatrixType>): ResultType` | Train a Lightning model |
| `Lightning.predict(model: ModelBlobType, X: MatrixType, masks: OptionType<Tensor3DBoolType>, conditions: OptionType<MatrixType>): MatrixType` | Predict using trained model |
| `Lightning.encode(model: ModelBlobType, X: MatrixType): MatrixType` | Encode to latent space (autoencoder architectures only) |
| `Lightning.decode(model: ModelBlobType, z: MatrixType): MatrixType` | Decode from latent space (non-conditional autoencoders) |
| `Lightning.decodeConditional(model: ModelBlobType, z: MatrixType, conditions: MatrixType): MatrixType` | Decode with condition vectors (conditional autoencoders) |
| `Lightning.generateSequence(model: ModelBlobType, prefix: MatrixType, condition: OptionType<MatrixType>, config: GenerateConfigType): MatrixType` | Generate sequence autoregressively from sequential model |

**Types:**

| Type | Description |
|------|-------------|
| `Lightning.Types.OutputType` | `VariantType({ regression: Null, binary: { pos_weight: OptionType<Vector> }, multiclass: { n_classes: Integer, class_weights: OptionType<Vector> }, multi_head: { n_heads: Integer, n_classes_per_head: Integer, class_weights: OptionType<Matrix> } })` |
| `Lightning.Types.CellType` | `VariantType({ lstm: Null, gru: Null })` - RNN cell type for sequential architecture |
| `Lightning.Types.GenerateConfigType` | `StructType({ n_steps: Integer, temperature: Float, return_probs: Boolean })` - Config for sequence generation |
| `Lightning.Types.ArchitectureType` | See Architecture Types below |
| `Lightning.Types.EpochCallbackType` | `FunctionType([Integer, Float, Float], Null)` - Callback: (epoch, train_loss, val_loss) -> void |
| `Lightning.Types.ConfigType` | `StructType({ architecture: ArchitectureType, output: OutputType, learning_rate: OptionType<Float>, max_epochs: OptionType<Integer>, patience: OptionType<Integer>, batch_size: OptionType<Integer>, dropout: OptionType<Float>, gradient_clip: OptionType<Float>, weight_decay: OptionType<Float>, random_state: OptionType<Integer>, epoch_callback: OptionType<EpochCallbackType> })` |
| `Lightning.Types.ResultType` | `StructType({ model: ModelBlobType, train_loss: Float, val_loss: Float, best_epoch: Integer })` |
| `Lightning.Types.ModelBlobType` | `VariantType({ lightning: { data: Blob, n_features: Integer, output_dim: Integer, architecture_type: String, output_type: String, latent_dim: OptionType<Integer> } })` |
| `Lightning.Types.Tensor3DBoolType` | `ArrayType(ArrayType(ArrayType(Boolean)))` - 3D boolean masks (n_samples, n_heads, n_classes) |
| `Lightning.Types.GroupWeightsType` | `StructType({ weights: VariantType({ binary: Array<Array<Float>>, multi_head: Array<Array<Array<Float>>> }), sample_groups: Array<Integer> })` |

**Architecture Types:**

| Variant | Fields | Description |
|---------|--------|-------------|
| `mlp` | `{ hidden_layers: Array<Integer> }` | Simple feedforward MLP |
| `autoencoder` | `{ encoder_layers: Array<Integer>, latent_dim: Integer, decoder_layers: Array<Integer> }` | Autoencoder with bottleneck |
| `conv1d` | `{ n_channels: Integer, sequence_length: Integer, conv_channels: Array<Integer>, kernel_size: Integer, latent_dim: Integer, condition_dim: OptionType<Integer> }` | 1D convolutional autoencoder for temporal patterns |
| `sequential` | `{ n_channels: Integer, sequence_length: Integer, hidden_size: Integer, n_layers: Integer, cell_type: CellType, latent_dim: Integer, bidirectional: Boolean, condition_dim: OptionType<Integer> }` | LSTM/GRU autoencoder for long-range dependencies |
| `transformer` | `{ n_channels: Integer, sequence_length: Integer, d_model: Integer, n_attention_heads: Integer, n_layers: Integer, d_ff: OptionType<Integer>, latent_dim: Integer, condition_dim: OptionType<Integer> }` | Attention-based autoencoder |

**Config Options:**

| Field | Type | Description |
|-------|------|-------------|
| `architecture` | `ArchitectureType` | Model architecture |
| `output` | `OutputType` | Output mode (regression, binary, multiclass, multi_head) |
| `learning_rate` | `OptionType<Float>` | Learning rate (default 1e-3) |
| `max_epochs` | `OptionType<Integer>` | Maximum training epochs (default 100) |
| `patience` | `OptionType<Integer>` | Early stopping patience (default 10) |
| `batch_size` | `OptionType<Integer>` | Batch size (default 32) |
| `dropout` | `OptionType<Float>` | Dropout rate (default 0.1) |
| `gradient_clip` | `OptionType<Float>` | Gradient clipping value (default 1.0) |
| `weight_decay` | `OptionType<Float>` | L2 regularization (default 0) |
| `random_state` | `OptionType<Integer>` | Random seed for reproducibility |
| `epoch_callback` | `OptionType<EpochCallbackType>` | Optional callback called each epoch |

**Output Modes:**

| Mode | Loss Function | Use Case |
|------|---------------|----------|
| `regression` | MSE | Continuous targets |
| `binary` | BCE with per-position pos_weights, masks | Binary classification with optional masking |
| `multiclass` | CrossEntropy with optional class_weights | Single-label multi-class |
| `multi_head` | N independent CrossEntropy heads, masks | Multi-label with mutex per head, optional masking |

**Masks:**
- Binary: Shape `(n_samples, 1, output_dim)` - masked positions excluded from loss and set to 0 in predictions
- Multi-head: Shape `(n_samples, n_heads, n_classes)` - masked classes get -inf logits (0 probability after softmax)

**Group Weights:**
- Per-sample class weighting via discrete groups (e.g., by grade/category)
- `sample_groups`: Maps each sample to a group index
- `weights.binary`: `[n_groups][output_dim]` pos_weights per group
- `weights.multi_head`: `[n_groups][n_heads][n_classes]` class_weights per group

**Conditional Generation:**
- Temporal architectures (conv1d, sequential, transformer) support `condition_dim`
- Pass `conditions` matrix `(n_samples, condition_dim)` to train, predict, and decodeConditional
- Use case: Stage 1 embeddings as conditioning input for Stage 2 models

---

## GP (Gaussian Process)

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
| `GP.Types.GPKernelType` | `VariantType({ rbf, matern_1_2, matern_3_2, matern_5_2, rational_quadratic, dot_product })` |
| `GP.Types.GPConfigType` | `StructType({ kernel: OptionType<GPKernelType>, alpha: OptionType<Float>, n_restarts_optimizer: OptionType<Integer>, normalize_y: OptionType<Boolean>, random_state: OptionType<Integer> })` |
| `GP.Types.GPPredictResultType` | `StructType({ mean: VectorType, std: VectorType })` |
| `GP.Types.ModelBlobType` | `gp_regressor` |

**Config Options:**

| Field | Type | Description |
|-------|------|-------------|
| `kernel` | `OptionType<Kernel>` | Kernel type (default rbf) |
| `alpha` | `OptionType<Float>` | Noise level (default 1e-10) |
| `n_restarts_optimizer` | `OptionType<Integer>` | Optimizer restarts (default 0) |
| `normalize_y` | `OptionType<Boolean>` | Normalize targets (default false) |
| `random_state` | `OptionType<Integer>` | Random seed |

---

## MAPIE (Conformal Prediction)

MAPIE provides conformal prediction intervals with coverage guarantees using the MAPIE 1.2.0 API.

**Import:**
```typescript
import { MAPIE } from "@elaraai/east-py-datascience";
```

**Functions:**
| Signature | Description |
|-----------|-------------|
| `MAPIE.trainConformalRegressor(X_train: MatrixType, y_train: VectorType, X_calib: MatrixType, y_calib: VectorType, config: MAPIEConfigType): MAPIERegressorBlobType` | Train split or cross conformal regressor |
| `MAPIE.trainCQR(X_train: MatrixType, y_train: VectorType, X_calib: MatrixType, y_calib: VectorType, config: MAPIECQRConfigType): MAPIERegressorBlobType` | Train Conformalized Quantile Regression |
| `MAPIE.predictInterval(model: MAPIERegressorBlobType, X: MatrixType): IntervalResultType` | Predict with intervals |
| `MAPIE.trainConformalClassifier(X_train: MatrixType, y_train: LabelVectorType, X_calib: MatrixType, y_calib: LabelVectorType, config: MAPIEClassifierConfigType): MAPIEClassifierBlobType` | Train split conformal classifier |
| `MAPIE.predictSet(model: MAPIEClassifierBlobType, X: MatrixType): PredictionSetResultType` | Predict with prediction sets |
| `MAPIE.uncertaintyPredictorRegressor(model: MAPIERegressorBlobType): UncertaintyPredictorType` | Create uncertainty predictor for SHAP (predicts interval width) |
| `MAPIE.uncertaintyPredictorClassifier(model: MAPIEClassifierBlobType): UncertaintyPredictorType` | Create uncertainty predictor for SHAP (predicts set size) |

**Types:**

| Type | Description |
|------|-------------|
| `MAPIE.Types.ConformalMethodType` | `VariantType({ split, cross })` |
| `MAPIE.Types.MAPIEXGBoostConfigType` | `StructType({ n_estimators, max_depth, learning_rate, min_child_weight, subsample, colsample_bytree, reg_alpha, reg_lambda, gamma, random_state: all OptionType })` |
| `MAPIE.Types.MAPIELightGBMConfigType` | `StructType({ n_estimators, max_depth, learning_rate, num_leaves, min_child_samples, subsample, colsample_bytree, reg_alpha, reg_lambda, random_state: all OptionType })` |
| `MAPIE.Types.BaseModelType` | `VariantType({ xgboost: MAPIEXGBoostConfigType, lightgbm: MAPIELightGBMConfigType })` |
| `MAPIE.Types.MAPIEConfigType` | `StructType({ base_model: BaseModelType, method: OptionType<ConformalMethodType>, confidence_level: OptionType<Float>, cv_folds: OptionType<Integer>, random_state: OptionType<Integer> })` |
| `MAPIE.Types.MAPIECQRConfigType` | `StructType({ xgboost_config: MAPIEXGBoostConfigType, confidence_level: OptionType<Float>, random_state: OptionType<Integer> })` |
| `MAPIE.Types.ClassificationMethodType` | `VariantType({ lac, aps })` |
| `MAPIE.Types.BaseClassifierType` | `VariantType({ xgboost: MAPIEXGBoostConfigType, lightgbm: MAPIELightGBMConfigType })` |
| `MAPIE.Types.MAPIEClassifierConfigType` | `StructType({ base_model: BaseClassifierType, method: OptionType<ClassificationMethodType>, confidence_level: OptionType<Float>, random_state: OptionType<Integer> })` |
| `MAPIE.Types.MAPIEBaseModelDataType` | `VariantType({ xgboost: BlobType, lightgbm: BlobType, histogram: BlobType })` |
| `MAPIE.Types.MAPIERegressorBlobType` | `VariantType({ mapie_split, mapie_cross, mapie_cqr })` - each with tagged data |
| `MAPIE.Types.MAPIEClassifierBlobType` | `VariantType({ mapie_classifier: StructType({ data, n_features, n_classes, classes, confidence_level }) })` |
| `MAPIE.Types.UncertaintyPredictorType` | `VariantType({ mapie_interval_width, mapie_set_size })` - for SHAP integration |
| `MAPIE.Types.IntervalResultType` | `StructType({ lower: VectorType, pred: VectorType, upper: VectorType })` |
| `MAPIE.Types.PredictionSetResultType` | `StructType({ pred: Array<Integer>, sets: Array<Array<Integer>>, probabilities: MatrixType, set_sizes: Array<Integer> })` |

**Config Options (MAPIEConfigType):**

| Field | Type | Description |
|-------|------|-------------|
| `base_model` | `BaseModelType` | XGBoost or LightGBM base model config |
| `method` | `OptionType<ConformalMethod>` | `split` or `cross` (default split) |
| `confidence_level` | `OptionType<Float>` | Coverage probability (default 0.9 = 90% intervals) |
| `cv_folds` | `OptionType<Integer>` | CV folds for cross method (default 5) |
| `random_state` | `OptionType<Integer>` | Random seed |

**Config Options (MAPIEClassifierConfigType):**

| Field | Type | Description |
|-------|------|-------------|
| `base_model` | `BaseClassifierType` | XGBoost or LightGBM base classifier config |
| `method` | `OptionType<ClassificationMethod>` | `lac` or `aps` (default lac) |
| `confidence_level` | `OptionType<Float>` | Coverage probability (default 0.9 = 90% coverage) |
| `random_state` | `OptionType<Integer>` | Random seed |

**Conformal Methods:**

| Method | Description |
|--------|-------------|
| `split` | Split conformal - trains on train set, calibrates on calibration set |
| `cross` | Cross conformal - combines train and calib, uses CV for calibration |

**Classification Conformity Scores:**

| Score | Description |
|-------|-------------|
| `lac` | Least Ambiguous set-valued Classifier - produces smallest prediction sets |
| `aps` | Adaptive Prediction Sets - adapts to probabilities (multiclass only) |

---

## Shap (Model Explainability)

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
| `Shap.Types.ShapModelBlobType` | `VariantType({ shap_tree_explainer: BlobType, shap_kernel_explainer: BlobType })` |
| `Shap.Types.ShapValuesType` | `VariantType({ matrix: MatrixType, tensor: ArrayType<MatrixType> })` |
| `Shap.Types.ShapBaseValueType` | `VariantType({ single: Float, per_class: VectorType })` |
| `Shap.Types.ShapResultType` | `StructType({ shap_values: ShapValuesType, base_value: ShapBaseValueType, feature_names: ArrayType<String> })` |
| `Shap.Types.FeatureImportanceType` | `StructType({ feature_names: ArrayType<String>, importances: VectorType, std: OptionType<VectorType> })` |
| `Shap.Types.MapieRegressorShapResultType` | `StructType({ point_prediction: ShapResultType, interval_width: ShapResultType })` |
| `Shap.Types.MapieClassifierShapResultType` | `StructType({ class_probabilities: ShapResultType, prediction_set_size: ShapResultType })` |
| `Shap.Types.TreeModelBlobType` | XGBoost models only (regressor/classifier/quantile). Note: LightGBM is not supported for TreeExplainer due to SHAP compatibility issues - use KernelExplainer instead. |
| `Shap.Types.AnyModelBlobType` | Any supported model (XGBoost, LightGBM, NGBoost, GP, Torch MLP, RegressorChain, MAPIE regressors/classifiers, MAPIE uncertainty predictors) |

**Supported Models:**

| Explainer | Supported Models |
|-----------|------------------|
| `treeExplainerCreate` | `xgboost_regressor`, `xgboost_classifier`, `xgboost_quantile` (XGBoost only) |
| `kernelExplainerCreate` | All models: XGBoost, LightGBM, `ngboost_regressor`, `gp_regressor`, `torch_mlp`, `regressor_chain`, MAPIE models (`mapie_split`, `mapie_cross`, `mapie_cqr`, `mapie_classifier`), MAPIE uncertainty predictors (`mapie_interval_width`, `mapie_set_size`) |

Note: LightGBM models are not supported for TreeExplainer due to SHAP compatibility issues. Use KernelExplainer for LightGBM models.

**MAPIE Uncertainty Explanation:**

To explain what drives prediction uncertainty in MAPIE models:
1. Create uncertainty predictor: `MAPIE.uncertaintyPredictorRegressor(model)` or `MAPIE.uncertaintyPredictorClassifier(model)`
2. Create KernelExplainer with the uncertainty predictor
3. Compute SHAP values to see which features increase/decrease uncertainty

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
Sklearn.Types.SplitConfigType  // StructType({ test_size, random_state, shuffle, stratify, min_stratify_samples })
Sklearn.Types.ModelBlobType    // VariantType({ standard_scaler, min_max_scaler, ... })

// Access XGBoost types
XGBoost.Types.XGBoostConfigType // StructType({ n_estimators, max_depth, ... })
XGBoost.Types.ModelBlobType     // VariantType({ xgboost_regressor, xgboost_classifier })
```

**Pattern:**
- `Module.Types.TypeName` - Access types through the module namespace
- Flat exports (e.g., `MADSResultType`) are also available
