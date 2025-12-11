# Extension: Flexible Multi-Target Metrics for Sklearn

## Problem

### 1. Single-target only
Current `sklearn_metrics_regression` only supports 1D vectors:

```typescript
// Current - single target only
Sklearn.metricsRegression(
    y_true: VectorType,   // [1.0, 2.0, 3.0, 4.0]
    y_pred: VectorType    // [1.1, 2.1, 2.9, 4.2]
): RegressionMetricsType
```

But multi-target regression (e.g., RegressorChain) produces matrices:

```typescript
// Multi-target data shape
Y_true: MatrixType  // 2D: samples × targets
// [
//   [B_f₁, B_0₁, k₁, lam₁],   ← sample 1, 4 targets
//   [B_f₂, B_0₂, k₂, lam₂],   ← sample 2, 4 targets
//   ...
// ]
```

### 2. Fixed metric struct is inflexible
Current design returns a fixed struct with 5 metrics:

```typescript
export const RegressionMetricsType = StructType({
    mse: FloatType,
    rmse: FloatType,
    mae: FloatType,
    r2: FloatType,
    mape: FloatType,
});
```

Problems:
- Can't request only specific metrics (compute all even if only need R²)
- Can't add new metrics without breaking API
- Missing useful sklearn metrics (explained_variance, max_error, etc.)

---

## Proposed Design

Two separate functions with clean type signatures:
- `sklearn_compute_metrics` - single-target (Vector → scalar results)
- `sklearn_compute_metrics_multi` - multi-target (Matrix → per-target or aggregated results)

### Shared Types

```typescript
// ============================================================================
// Regression Metric Variants
// ============================================================================

/**
 * Available regression metrics from sklearn.metrics.
 */
export const RegressionMetricType = VariantType({
    /** Mean Squared Error - sklearn.metrics.mean_squared_error */
    mse: NullType,
    /** Root Mean Squared Error - sqrt(MSE) */
    rmse: NullType,
    /** Mean Absolute Error - sklearn.metrics.mean_absolute_error */
    mae: NullType,
    /** R² (coefficient of determination) - sklearn.metrics.r2_score */
    r2: NullType,
    /** Mean Absolute Percentage Error - sklearn.metrics.mean_absolute_percentage_error */
    mape: NullType,
    /** Explained Variance Score - sklearn.metrics.explained_variance_score */
    explained_variance: NullType,
    /** Max Error - sklearn.metrics.max_error */
    max_error: NullType,
    /** Median Absolute Error - sklearn.metrics.median_absolute_error */
    median_ae: NullType,
    /** Mean Squared Log Error - sklearn.metrics.mean_squared_log_error (requires positive values) */
    msle: NullType,
    /** Mean Pinball Loss - sklearn.metrics.mean_pinball_loss */
    pinball_loss: NullType,
    /** D² Pinball Score - sklearn.metrics.d2_pinball_score */
    d2_pinball: NullType,
    /** D² Absolute Error Score - sklearn.metrics.d2_absolute_error_score */
    d2_absolute_error: NullType,
});

// ============================================================================
// Classification Metric Variants
// ============================================================================

/**
 * Available classification metrics from sklearn.metrics.
 */
export const ClassificationMetricType = VariantType({
    /** Accuracy - sklearn.metrics.accuracy_score */
    accuracy: NullType,
    /** Balanced Accuracy - sklearn.metrics.balanced_accuracy_score */
    balanced_accuracy: NullType,
    /** Precision - sklearn.metrics.precision_score */
    precision: NullType,
    /** Recall - sklearn.metrics.recall_score */
    recall: NullType,
    /** F1 Score - sklearn.metrics.f1_score */
    f1: NullType,
    /** Matthews Correlation Coefficient - sklearn.metrics.matthews_corrcoef */
    matthews_corrcoef: NullType,
    /** Cohen's Kappa - sklearn.metrics.cohen_kappa_score */
    cohen_kappa: NullType,
    /** Jaccard Score - sklearn.metrics.jaccard_score */
    jaccard: NullType,
    /** Hamming Loss - sklearn.metrics.hamming_loss */
    hamming_loss: NullType,
    /** Zero-One Loss - sklearn.metrics.zero_one_loss */
    zero_one_loss: NullType,
    /** Log Loss (requires probabilities) - sklearn.metrics.log_loss */
    log_loss: NullType,
    /** ROC AUC (requires probabilities) - sklearn.metrics.roc_auc_score */
    roc_auc: NullType,
});

/**
 * Averaging strategy for multi-class classification metrics.
 */
export const ClassificationAverageType = VariantType({
    /** Calculate metrics for each label, return unweighted mean */
    macro: NullType,
    /** Calculate metrics globally by counting total TP, FP, FN */
    micro: NullType,
    /** Calculate metrics for each label, return weighted mean by support */
    weighted: NullType,
    /** Only for binary classification */
    binary: NullType,
});
```

### Single-Target Types & Function

```typescript
/**
 * Single metric result (scalar value).
 */
export const MetricResultType = StructType({
    /** Which metric was computed */
    metric: RegressionMetricType,
    /** Scalar metric value */
    value: FloatType,
});

/**
 * Result containing multiple computed metrics (single-target).
 */
export const MetricsResultType = ArrayType(MetricResultType);

/**
 * Compute regression metrics for single-target predictions.
 *
 * @param y_true - True target values (1D vector)
 * @param y_pred - Predicted target values (1D vector)
 * @param metrics - Array of metrics to compute
 * @returns Array of metric results with scalar values
 *
 * @example
 * ```typescript
 * const results = Sklearn.computeMetrics(
 *     y_true,
 *     y_pred,
 *     [variant('r2', null), variant('rmse', null), variant('mae', null)]
 * );
 * // Returns: [
 * //   { metric: #r2, value: 0.95 },
 * //   { metric: #rmse, value: 0.15 },
 * //   { metric: #mae, value: 0.12 }
 * // ]
 * ```
 */
export const sklearn_compute_metrics = East.platform(
    "sklearn_compute_metrics",
    [
        VectorType,                      // y_true
        VectorType,                      // y_pred
        ArrayType(RegressionMetricType), // metrics to compute
    ],
    MetricsResultType
);
```

### Multi-Target Types & Function

```typescript
/**
 * Aggregation strategy for multi-target metrics.
 */
export const MetricAggregationType = VariantType({
    /** Return metric for each target separately (default) */
    per_target: NullType,
    /** Average across all targets (uniform weights) */
    uniform_average: NullType,
    /** Variance-weighted average */
    variance_weighted: NullType,
});

/**
 * Configuration for multi-target metrics computation.
 */
export const MultiMetricsConfigType = StructType({
    /** How to aggregate metrics across targets (default: per_target) */
    aggregation: OptionType(MetricAggregationType),
});

/**
 * Multi-target metric result.
 * Value is either per-target vector or aggregated scalar.
 */
export const MultiMetricResultType = StructType({
    /** Which metric was computed */
    metric: RegressionMetricType,
    /** Metric value(s) */
    value: VariantType({
        /** Aggregated scalar value */
        scalar: FloatType,
        /** Per-target values [target_0, target_1, ...] */
        per_target: VectorType,
    }),
});

/**
 * Result containing multiple computed metrics (multi-target).
 */
export const MultiMetricsResultType = ArrayType(MultiMetricResultType);

/**
 * Compute regression metrics for multi-target predictions.
 *
 * @param Y_true - True target matrix [n_samples, n_targets]
 * @param Y_pred - Predicted target matrix [n_samples, n_targets]
 * @param metrics - Array of metrics to compute
 * @param config - Aggregation configuration
 * @returns Array of metric results with per-target or aggregated values
 *
 * @example Per-target breakdown
 * ```typescript
 * const results = Sklearn.computeMetricsMulti(
 *     Y_true,  // [n_samples, 4 targets]
 *     Y_pred,
 *     [variant('r2', null), variant('mse', null)],
 *     { aggregation: variant('some', variant('per_target', null)) }
 * );
 * // Returns: [
 * //   { metric: #r2, value: #per_target([0.98, 0.95, 0.97, 0.99]) },
 * //   { metric: #mse, value: #per_target([0.01, 0.02, 0.015, 0.008]) }
 * // ]
 * ```
 *
 * @example Aggregated
 * ```typescript
 * const results = Sklearn.computeMetricsMulti(
 *     Y_true,
 *     Y_pred,
 *     [variant('r2', null)],
 *     { aggregation: variant('some', variant('uniform_average', null)) }
 * );
 * // Returns: [{ metric: #r2, value: #scalar(0.9725) }]
 * ```
 */
export const sklearn_compute_metrics_multi = East.platform(
    "sklearn_compute_metrics_multi",
    [
        MatrixType,                      // Y_true [n_samples, n_targets]
        MatrixType,                      // Y_pred [n_samples, n_targets]
        ArrayType(RegressionMetricType), // metrics to compute
        MultiMetricsConfigType,          // aggregation config
    ],
    MultiMetricsResultType
);
```

### Classification Single-Target Types & Function

```typescript
/**
 * Configuration for classification metrics.
 */
export const ClassificationConfigType = StructType({
    /** Averaging strategy for multi-class (default: macro) */
    average: OptionType(ClassificationAverageType),
    /** Class probabilities for metrics that need them (log_loss, roc_auc) */
    y_score: OptionType(MatrixType),
});

/**
 * Single classification metric result.
 */
export const ClassificationMetricResultType = StructType({
    /** Which metric was computed */
    metric: ClassificationMetricType,
    /** Scalar metric value */
    value: FloatType,
});

/**
 * Result containing multiple computed classification metrics.
 */
export const ClassificationMetricsResultType = ArrayType(ClassificationMetricResultType);

/**
 * Compute classification metrics for single-target predictions.
 *
 * @param y_true - True class labels (1D integer array)
 * @param y_pred - Predicted class labels (1D integer array)
 * @param metrics - Array of metrics to compute
 * @param config - Configuration (averaging strategy, probabilities for roc_auc/log_loss)
 * @returns Array of metric results with scalar values
 *
 * @example Binary classification
 * ```typescript
 * const results = Sklearn.computeClassificationMetrics(
 *     y_true,  // [0, 1, 1, 0, 1]
 *     y_pred,  // [0, 1, 0, 0, 1]
 *     [variant('accuracy', null), variant('f1', null), variant('precision', null)],
 *     { average: variant('some', variant('binary', null)), y_score: variant('none', null) }
 * );
 * // Returns: [
 * //   { metric: #accuracy, value: 0.8 },
 * //   { metric: #f1, value: 0.8 },
 * //   { metric: #precision, value: 1.0 }
 * // ]
 * ```
 *
 * @example Multi-class with ROC AUC
 * ```typescript
 * const results = Sklearn.computeClassificationMetrics(
 *     y_true,
 *     y_pred,
 *     [variant('accuracy', null), variant('roc_auc', null)],
 *     {
 *         average: variant('some', variant('macro', null)),
 *         y_score: variant('some', probabilities)  // [n_samples, n_classes]
 *     }
 * );
 * ```
 */
export const sklearn_compute_classification_metrics = East.platform(
    "sklearn_compute_classification_metrics",
    [
        VectorType,                            // y_true (integer labels)
        VectorType,                            // y_pred (integer labels)
        ArrayType(ClassificationMetricType),   // metrics to compute
        ClassificationConfigType,              // config
    ],
    ClassificationMetricsResultType
);
```

### Classification Multi-Target Types & Function

```typescript
/**
 * Configuration for multi-target classification metrics.
 */
export const MultiClassificationConfigType = StructType({
    /** Averaging strategy for multi-class (default: macro) */
    average: OptionType(ClassificationAverageType),
    /** How to aggregate across targets (default: per_target) */
    aggregation: OptionType(MetricAggregationType),
    /** Class probabilities for metrics that need them - shape [n_samples, n_targets, n_classes] or [n_samples, n_targets] for binary */
    y_score: OptionType(MatrixType),
});

/**
 * Multi-target classification metric result.
 */
export const MultiClassificationMetricResultType = StructType({
    /** Which metric was computed */
    metric: ClassificationMetricType,
    /** Metric value(s) */
    value: VariantType({
        /** Aggregated scalar value */
        scalar: FloatType,
        /** Per-target values [target_0, target_1, ...] */
        per_target: VectorType,
    }),
});

/**
 * Result containing multiple computed classification metrics (multi-target).
 */
export const MultiClassificationMetricsResultType = ArrayType(MultiClassificationMetricResultType);

/**
 * Compute classification metrics for multi-target predictions.
 *
 * @param Y_true - True class labels matrix [n_samples, n_targets]
 * @param Y_pred - Predicted class labels matrix [n_samples, n_targets]
 * @param metrics - Array of metrics to compute
 * @param config - Configuration (averaging, aggregation, probabilities)
 * @returns Array of metric results with per-target or aggregated values
 *
 * @example Multi-label classification
 * ```typescript
 * const results = Sklearn.computeClassificationMetricsMulti(
 *     Y_true,  // [n_samples, n_labels]
 *     Y_pred,
 *     [variant('accuracy', null), variant('f1', null)],
 *     {
 *         average: variant('some', variant('binary', null)),
 *         aggregation: variant('some', variant('per_target', null)),
 *         y_score: variant('none', null)
 *     }
 * );
 * // Returns: [
 * //   { metric: #accuracy, value: #per_target([0.9, 0.85, 0.92]) },
 * //   { metric: #f1, value: #per_target([0.88, 0.82, 0.90]) }
 * // ]
 * ```
 */
export const sklearn_compute_classification_metrics_multi = East.platform(
    "sklearn_compute_classification_metrics_multi",
    [
        MatrixType,                            // Y_true [n_samples, n_targets]
        MatrixType,                            // Y_pred [n_samples, n_targets]
        ArrayType(ClassificationMetricType),   // metrics to compute
        MultiClassificationConfigType,         // config
    ],
    MultiClassificationMetricsResultType
);
```

### Update Grouped Exports

```typescript
export const SklearnTypes = {
    // ... existing ...
    // Regression
    RegressionMetricType,
    MetricResultType,
    MetricsResultType,
    MetricAggregationType,
    MultiMetricsConfigType,
    MultiMetricResultType,
    MultiMetricsResultType,
    // Classification
    ClassificationMetricType,
    ClassificationAverageType,
    ClassificationConfigType,
    ClassificationMetricResultType,
    ClassificationMetricsResultType,
    MultiClassificationConfigType,
    MultiClassificationMetricResultType,
    MultiClassificationMetricsResultType,
} as const;

export const Sklearn = {
    // ... existing ...
    /** Compute regression metrics (single-target) */
    computeMetrics: sklearn_compute_metrics,
    /** Compute regression metrics (multi-target) */
    computeMetricsMulti: sklearn_compute_metrics_multi,
    /** Compute classification metrics (single-target) */
    computeClassificationMetrics: sklearn_compute_classification_metrics,
    /** Compute classification metrics (multi-target) */
    computeClassificationMetricsMulti: sklearn_compute_classification_metrics_multi,
    // Keep old for backwards compatibility (deprecated)
    /** @deprecated Use computeMetrics instead */
    metricsRegression: sklearn_metrics_regression,
} as const;
```

---

## Python Implementation

### New Types (`types.py`)

```python
from east.types.types import NullType

# Regression metric enum (shared)
RegressionMetricType = VariantType([
    ("mse", NullType),
    ("rmse", NullType),
    ("mae", NullType),
    ("r2", NullType),
    ("mape", NullType),
    ("explained_variance", NullType),
    ("max_error", NullType),
    ("median_ae", NullType),
    ("msle", NullType),
    ("pinball_loss", NullType),
    ("d2_pinball", NullType),
    ("d2_absolute_error", NullType),
])

# Single-target result
MetricResultType = StructType([
    ("metric", RegressionMetricType),
    ("value", FloatType),
])

MetricsResultType = ArrayType(MetricResultType)

# Multi-target types
MetricAggregationType = VariantType([
    ("per_target", NullType),
    ("uniform_average", NullType),
    ("variance_weighted", NullType),
])

MultiMetricsConfigType = StructType([
    ("aggregation", OptionType(MetricAggregationType)),
])

MultiMetricValueType = VariantType([
    ("scalar", FloatType),
    ("per_target", VectorType),
])

MultiMetricResultType = StructType([
    ("metric", RegressionMetricType),
    ("value", MultiMetricValueType),
])

MultiMetricsResultType = ArrayType(MultiMetricResultType)

# Classification types
ClassificationMetricType = VariantType([
    ("accuracy", NullType),
    ("balanced_accuracy", NullType),
    ("precision", NullType),
    ("recall", NullType),
    ("f1", NullType),
    ("matthews_corrcoef", NullType),
    ("cohen_kappa", NullType),
    ("jaccard", NullType),
    ("hamming_loss", NullType),
    ("zero_one_loss", NullType),
    ("log_loss", NullType),
    ("roc_auc", NullType),
])

ClassificationAverageType = VariantType([
    ("macro", NullType),
    ("micro", NullType),
    ("weighted", NullType),
    ("binary", NullType),
])

ClassificationConfigType = StructType([
    ("average", OptionType(ClassificationAverageType)),
    ("y_score", OptionType(MatrixType)),
])

ClassificationMetricResultType = StructType([
    ("metric", ClassificationMetricType),
    ("value", FloatType),
])

ClassificationMetricsResultType = ArrayType(ClassificationMetricResultType)

MultiClassificationConfigType = StructType([
    ("average", OptionType(ClassificationAverageType)),
    ("aggregation", OptionType(MetricAggregationType)),
    ("y_score", OptionType(MatrixType)),
])

MultiClassificationMetricResultType = StructType([
    ("metric", ClassificationMetricType),
    ("value", MultiMetricValueType),
])

MultiClassificationMetricsResultType = ArrayType(MultiClassificationMetricResultType)
```

### Implementation (`sklearn_impl.py`)

```python
from sklearn import metrics as sklearn_metrics

# Metric function mapping
METRIC_FUNCTIONS = {
    "mse": lambda y_true, y_pred, **kw: sklearn_metrics.mean_squared_error(y_true, y_pred, **kw),
    "rmse": lambda y_true, y_pred, **kw: np.sqrt(sklearn_metrics.mean_squared_error(y_true, y_pred, **kw)),
    "mae": lambda y_true, y_pred, **kw: sklearn_metrics.mean_absolute_error(y_true, y_pred, **kw),
    "r2": lambda y_true, y_pred, **kw: sklearn_metrics.r2_score(y_true, y_pred, **kw),
    "mape": lambda y_true, y_pred, **kw: sklearn_metrics.mean_absolute_percentage_error(y_true, y_pred, **kw),
    "explained_variance": lambda y_true, y_pred, **kw: sklearn_metrics.explained_variance_score(y_true, y_pred, **kw),
    "max_error": lambda y_true, y_pred, **kw: sklearn_metrics.max_error(y_true, y_pred),
    "median_ae": lambda y_true, y_pred, **kw: sklearn_metrics.median_absolute_error(y_true, y_pred, **kw),
    "msle": lambda y_true, y_pred, **kw: sklearn_metrics.mean_squared_log_error(y_true, y_pred, **kw),
    "pinball_loss": lambda y_true, y_pred, **kw: sklearn_metrics.mean_pinball_loss(y_true, y_pred),
    "d2_pinball": lambda y_true, y_pred, **kw: sklearn_metrics.d2_pinball_score(y_true, y_pred),
    "d2_absolute_error": lambda y_true, y_pred, **kw: sklearn_metrics.d2_absolute_error_score(y_true, y_pred, **kw),
}

# Metrics that support multioutput parameter
MULTIOUTPUT_METRICS = {"mse", "rmse", "mae", "r2", "mape", "explained_variance", "median_ae", "msle", "d2_absolute_error"}


def sklearn_compute_metrics_impl(
    y_true: EastArray,
    y_pred: EastArray,
    metrics: EastArray,
) -> EastArray:
    """Compute regression metrics for single-target predictions."""
    y_true_np = east_vector_to_numpy(y_true)
    y_pred_np = east_vector_to_numpy(y_pred)

    results = []
    for metric_variant in metrics:
        metric_name = metric_variant.type
        metric_fn = METRIC_FUNCTIONS.get(metric_name)

        if metric_fn is None:
            raise ValueError(f"Unknown metric: {metric_name}")

        try:
            value = float(metric_fn(y_true_np, y_pred_np))
            results.append(EastStruct({
                "metric": EastVariant(metric_name, None),
                "value": value,
            }))
        except Exception:
            # Skip metrics that fail (e.g., MSLE with negative values)
            pass

    return EastArray(MetricResultType, results)


def sklearn_compute_metrics_multi_impl(
    Y_true: EastArray,
    Y_pred: EastArray,
    metrics: EastArray,
    config: EastStruct,
) -> EastArray:
    """Compute regression metrics for multi-target predictions."""
    Y_true_np = east_matrix_to_numpy(Y_true)
    Y_pred_np = east_matrix_to_numpy(Y_pred)

    # Get aggregation strategy
    agg_opt = _get_option(config.get("aggregation"), None)
    aggregation = agg_opt.type if agg_opt else "per_target"

    # Map to sklearn multioutput parameter
    multioutput_param = {
        "per_target": "raw_values",
        "uniform_average": "uniform_average",
        "variance_weighted": "variance_weighted",
    }.get(aggregation, "raw_values")

    results = []
    for metric_variant in metrics:
        metric_name = metric_variant.type
        metric_fn = METRIC_FUNCTIONS.get(metric_name)

        if metric_fn is None:
            raise ValueError(f"Unknown metric: {metric_name}")

        try:
            if metric_name in MULTIOUTPUT_METRICS:
                value = metric_fn(Y_true_np, Y_pred_np, multioutput=multioutput_param)
            else:
                # Metrics without multioutput support - compute per column
                if aggregation == "per_target":
                    value = np.array([
                        metric_fn(Y_true_np[:, i], Y_pred_np[:, i])
                        for i in range(Y_true_np.shape[1])
                    ])
                else:
                    # Average manually
                    per_target = [
                        metric_fn(Y_true_np[:, i], Y_pred_np[:, i])
                        for i in range(Y_true_np.shape[1])
                    ]
                    value = float(np.mean(per_target))

            # Format result
            if isinstance(value, np.ndarray):
                result_value = EastVariant("per_target", numpy_to_east_vector(value))
            else:
                result_value = EastVariant("scalar", float(value))

            results.append(EastStruct({
                "metric": EastVariant(metric_name, None),
                "value": result_value,
            }))
        except Exception:
            # Skip metrics that fail
            pass

    return EastArray(MultiMetricResultType, results)


# Classification metric function mapping
CLASSIFICATION_METRIC_FUNCTIONS = {
    "accuracy": lambda y_true, y_pred, **kw: sklearn_metrics.accuracy_score(y_true, y_pred),
    "balanced_accuracy": lambda y_true, y_pred, **kw: sklearn_metrics.balanced_accuracy_score(y_true, y_pred),
    "precision": lambda y_true, y_pred, **kw: sklearn_metrics.precision_score(y_true, y_pred, **kw),
    "recall": lambda y_true, y_pred, **kw: sklearn_metrics.recall_score(y_true, y_pred, **kw),
    "f1": lambda y_true, y_pred, **kw: sklearn_metrics.f1_score(y_true, y_pred, **kw),
    "matthews_corrcoef": lambda y_true, y_pred, **kw: sklearn_metrics.matthews_corrcoef(y_true, y_pred),
    "cohen_kappa": lambda y_true, y_pred, **kw: sklearn_metrics.cohen_kappa_score(y_true, y_pred),
    "jaccard": lambda y_true, y_pred, **kw: sklearn_metrics.jaccard_score(y_true, y_pred, **kw),
    "hamming_loss": lambda y_true, y_pred, **kw: sklearn_metrics.hamming_loss(y_true, y_pred),
    "zero_one_loss": lambda y_true, y_pred, **kw: sklearn_metrics.zero_one_loss(y_true, y_pred),
    "log_loss": lambda y_true, y_score, **kw: sklearn_metrics.log_loss(y_true, y_score),
    "roc_auc": lambda y_true, y_score, **kw: sklearn_metrics.roc_auc_score(y_true, y_score, **kw),
}

# Metrics that need 'average' parameter for multi-class
AVERAGE_METRICS = {"precision", "recall", "f1", "jaccard"}

# Metrics that need y_score (probabilities) instead of y_pred
PROBA_METRICS = {"log_loss", "roc_auc"}


def sklearn_compute_classification_metrics_impl(
    y_true: EastArray,
    y_pred: EastArray,
    metrics: EastArray,
    config: EastStruct,
) -> EastArray:
    """Compute classification metrics for single-target predictions."""
    y_true_np = east_vector_to_numpy(y_true).astype(int)
    y_pred_np = east_vector_to_numpy(y_pred).astype(int)

    # Get config options
    avg_opt = _get_option(config.get("average"), None)
    average = avg_opt.type if avg_opt else "macro"
    y_score_opt = _get_option(config.get("y_score"), None)
    y_score_np = east_matrix_to_numpy(y_score_opt) if y_score_opt else None

    results = []
    for metric_variant in metrics:
        metric_name = metric_variant.type
        metric_fn = CLASSIFICATION_METRIC_FUNCTIONS.get(metric_name)

        if metric_fn is None:
            raise ValueError(f"Unknown classification metric: {metric_name}")

        try:
            kwargs = {}
            if metric_name in AVERAGE_METRICS:
                kwargs["average"] = average

            if metric_name in PROBA_METRICS:
                if y_score_np is None:
                    continue  # Skip - needs probabilities
                value = float(metric_fn(y_true_np, y_score_np, **kwargs))
            else:
                value = float(metric_fn(y_true_np, y_pred_np, **kwargs))

            results.append(EastStruct({
                "metric": EastVariant(metric_name, None),
                "value": value,
            }))
        except Exception:
            pass  # Skip metrics that fail

    return EastArray(ClassificationMetricResultType, results)


def sklearn_compute_classification_metrics_multi_impl(
    Y_true: EastArray,
    Y_pred: EastArray,
    metrics: EastArray,
    config: EastStruct,
) -> EastArray:
    """Compute classification metrics for multi-target predictions."""
    Y_true_np = east_matrix_to_numpy(Y_true).astype(int)
    Y_pred_np = east_matrix_to_numpy(Y_pred).astype(int)
    n_targets = Y_true_np.shape[1]

    # Get config options
    avg_opt = _get_option(config.get("average"), None)
    average = avg_opt.type if avg_opt else "macro"
    agg_opt = _get_option(config.get("aggregation"), None)
    aggregation = agg_opt.type if agg_opt else "per_target"
    y_score_opt = _get_option(config.get("y_score"), None)
    y_score_np = east_matrix_to_numpy(y_score_opt) if y_score_opt else None

    results = []
    for metric_variant in metrics:
        metric_name = metric_variant.type
        metric_fn = CLASSIFICATION_METRIC_FUNCTIONS.get(metric_name)

        if metric_fn is None:
            raise ValueError(f"Unknown classification metric: {metric_name}")

        try:
            kwargs = {}
            if metric_name in AVERAGE_METRICS:
                kwargs["average"] = average

            # Compute per target
            per_target_values = []
            for i in range(n_targets):
                if metric_name in PROBA_METRICS:
                    if y_score_np is None:
                        break
                    # For binary multi-label: y_score is [n_samples, n_targets]
                    val = metric_fn(Y_true_np[:, i], y_score_np[:, i], **kwargs)
                else:
                    val = metric_fn(Y_true_np[:, i], Y_pred_np[:, i], **kwargs)
                per_target_values.append(val)

            if len(per_target_values) != n_targets:
                continue  # Skip if proba metrics without y_score

            # Format based on aggregation
            if aggregation == "per_target":
                result_value = EastVariant("per_target", numpy_to_east_vector(np.array(per_target_values)))
            else:
                # uniform_average or variance_weighted (treat same for classification)
                result_value = EastVariant("scalar", float(np.mean(per_target_values)))

            results.append(EastStruct({
                "metric": EastVariant(metric_name, None),
                "value": result_value,
            }))
        except Exception:
            pass

    return EastArray(MultiClassificationMetricResultType, results)
```

### Platform Registration

```python
sklearn_impl = [
    # ... existing ...
    PlatformFunction(
        name="sklearn_compute_metrics",
        inputs=[VectorType, VectorType, ArrayType(RegressionMetricType)],
        output=MetricsResultType,
        type="sync",
        fn=sklearn_compute_metrics_impl,
    ),
    PlatformFunction(
        name="sklearn_compute_metrics_multi",
        inputs=[MatrixType, MatrixType, ArrayType(RegressionMetricType), MultiMetricsConfigType],
        output=MultiMetricsResultType,
        type="sync",
        fn=sklearn_compute_metrics_multi_impl,
    ),
    PlatformFunction(
        name="sklearn_compute_classification_metrics",
        inputs=[VectorType, VectorType, ArrayType(ClassificationMetricType), ClassificationConfigType],
        output=ClassificationMetricsResultType,
        type="sync",
        fn=sklearn_compute_classification_metrics_impl,
    ),
    PlatformFunction(
        name="sklearn_compute_classification_metrics_multi",
        inputs=[MatrixType, MatrixType, ArrayType(ClassificationMetricType), MultiClassificationConfigType],
        output=MultiClassificationMetricsResultType,
        type="sync",
        fn=sklearn_compute_classification_metrics_multi_impl,
    ),
]
```

---

## Usage Examples

### Single-target

```typescript
const results = Sklearn.computeMetrics(
    y_true,
    y_pred,
    [variant('r2', null), variant('rmse', null), variant('mae', null)]
);
// Returns: [
//   { metric: #r2, value: 0.95 },
//   { metric: #rmse, value: 0.15 },
//   { metric: #mae, value: 0.12 }
// ]
```

### Multi-target with per-target breakdown

```typescript
const results = Sklearn.computeMetricsMulti(
    Y_true,  // [n_samples, 4 targets]
    Y_pred,
    [variant('r2', null), variant('mse', null)],
    { aggregation: variant('some', variant('per_target', null)) }
);
// Returns: [
//   { metric: #r2, value: #per_target([0.98, 0.95, 0.97, 0.99]) },
//   { metric: #mse, value: #per_target([0.01, 0.02, 0.015, 0.008]) }
// ]
```

### Multi-target with aggregation

```typescript
const results = Sklearn.computeMetricsMulti(
    Y_true,
    Y_pred,
    [variant('r2', null)],
    { aggregation: variant('some', variant('uniform_average', null)) }
);
// Returns: [{ metric: #r2, value: #scalar(0.9725) }]
```

### Classification (binary)

```typescript
const results = Sklearn.computeClassificationMetrics(
    y_true,  // [0, 1, 1, 0, 1]
    y_pred,  // [0, 1, 0, 0, 1]
    [variant('accuracy', null), variant('f1', null), variant('precision', null)],
    { average: variant('some', variant('binary', null)), y_score: variant('none', null) }
);
// Returns: [
//   { metric: #accuracy, value: 0.8 },
//   { metric: #f1, value: 0.8 },
//   { metric: #precision, value: 1.0 }
// ]
```

### Classification with ROC AUC

```typescript
const results = Sklearn.computeClassificationMetrics(
    y_true,
    y_pred,
    [variant('accuracy', null), variant('roc_auc', null)],
    {
        average: variant('some', variant('macro', null)),
        y_score: variant('some', probabilities)  // [n_samples, n_classes]
    }
);
```

### Multi-label classification

```typescript
const results = Sklearn.computeClassificationMetricsMulti(
    Y_true,  // [n_samples, n_labels]
    Y_pred,
    [variant('accuracy', null), variant('f1', null)],
    {
        average: variant('some', variant('binary', null)),
        aggregation: variant('some', variant('per_target', null)),
        y_score: variant('none', null)
    }
);
// Returns: [
//   { metric: #accuracy, value: #per_target([0.9, 0.85, 0.92]) },
//   { metric: #f1, value: #per_target([0.88, 0.82, 0.90]) }
// ]
```

---

## Available Metrics Reference

| Variant | sklearn function | Multi-output support |
|---------|-----------------|---------------------|
| `mse` | `mean_squared_error` | ✓ |
| `rmse` | `sqrt(mean_squared_error)` | ✓ |
| `mae` | `mean_absolute_error` | ✓ |
| `r2` | `r2_score` | ✓ |
| `mape` | `mean_absolute_percentage_error` | ✓ |
| `explained_variance` | `explained_variance_score` | ✓ |
| `max_error` | `max_error` | ✗ (computed per-column) |
| `median_ae` | `median_absolute_error` | ✓ |
| `msle` | `mean_squared_log_error` | ✓ (requires positive) |
| `pinball_loss` | `mean_pinball_loss` | ✗ (computed per-column) |
| `d2_pinball` | `d2_pinball_score` | ✗ (computed per-column) |
| `d2_absolute_error` | `d2_absolute_error_score` | ✓ |

### Classification Metrics

| Variant | sklearn function | Needs `average` | Needs `y_score` |
|---------|-----------------|-----------------|-----------------|
| `accuracy` | `accuracy_score` | ✗ | ✗ |
| `balanced_accuracy` | `balanced_accuracy_score` | ✗ | ✗ |
| `precision` | `precision_score` | ✓ | ✗ |
| `recall` | `recall_score` | ✓ | ✗ |
| `f1` | `f1_score` | ✓ | ✗ |
| `matthews_corrcoef` | `matthews_corrcoef` | ✗ | ✗ |
| `cohen_kappa` | `cohen_kappa_score` | ✗ | ✗ |
| `jaccard` | `jaccard_score` | ✓ | ✗ |
| `hamming_loss` | `hamming_loss` | ✗ | ✗ |
| `zero_one_loss` | `zero_one_loss` | ✗ | ✗ |
| `log_loss` | `log_loss` | ✗ | ✓ (probabilities) |
| `roc_auc` | `roc_auc_score` | ✗ | ✓ (probabilities) |

---

## Migration Guide

### Before (deprecated)
```typescript
const result = Sklearn.metricsRegression(y_true, y_pred);
// result.mse, result.rmse, result.mae, result.r2, result.mape
```

### After (single-target)
```typescript
const results = Sklearn.computeMetrics(
    y_true,
    y_pred,
    [
        variant('mse', null),
        variant('rmse', null),
        variant('mae', null),
        variant('r2', null),
        variant('mape', null),
    ]
);
// Loop through results array to get each metric
```

---

## Train/Val/Test Split

The types already exist in `sklearn.ts` (`ThreeWaySplitConfigType`, `ThreeWaySplitResultType`). Just need to add the platform function and implementation.

### TypeScript (`sklearn.ts`)

```typescript
/**
 * Split arrays into train, validation, and test subsets.
 *
 * Uses sklearn.model_selection.train_test_split twice to create a 3-way split.
 *
 * @param X - Feature matrix
 * @param Y - Target matrix (multi-target)
 * @param config - Split configuration with val_size and test_size
 * @returns Split result with X_train, X_val, X_test, Y_train, Y_val, Y_test
 *
 * @example
 * ```typescript
 * const result = Sklearn.trainValTestSplit(X, Y, {
 *     val_size: variant('some', 0.15),
 *     test_size: variant('some', 0.15),
 *     random_state: variant('some', 42n),
 *     shuffle: variant('some', true),
 * });
 * // result.X_train (70%), result.X_val (15%), result.X_test (15%)
 * ```
 */
export const sklearn_train_val_test_split = East.platform(
    "sklearn_train_val_test_split",
    [MatrixType, MultiTargetType, ThreeWaySplitConfigType],
    ThreeWaySplitResultType
);

// Add to Sklearn export:
export const Sklearn = {
    // ... existing ...
    /** Split arrays into train, validation, and test subsets */
    trainValTestSplit: sklearn_train_val_test_split,
} as const;
```

### Python Implementation (`sklearn_impl.py`)

```python
from sklearn.model_selection import train_test_split


def sklearn_train_val_test_split_impl(
    X: EastArray,
    Y: EastArray,
    config: EastStruct,
) -> EastStruct:
    """Split arrays into train, validation, and test subsets."""
    X_np = east_matrix_to_numpy(X)
    Y_np = east_matrix_to_numpy(Y)

    # Get config
    val_size = _get_option(config.get("val_size"), 0.15)
    test_size = _get_option(config.get("test_size"), 0.15)
    random_state = _get_option(config.get("random_state"), None)
    shuffle = _get_option(config.get("shuffle"), True)

    # First split: separate test set
    # test_size is proportion of total
    X_temp, X_test, Y_temp, Y_test = train_test_split(
        X_np, Y_np,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
    )

    # Second split: separate validation from training
    # val_size is proportion of total, need to adjust for remaining data
    # If total=100, test=15, remaining=85, val=15 -> val_ratio = 15/85 ≈ 0.176
    val_ratio = val_size / (1.0 - test_size)
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_temp, Y_temp,
        test_size=val_ratio,
        random_state=random_state,
        shuffle=shuffle,
    )

    return EastStruct({
        "X_train": numpy_to_east_matrix(X_train),
        "X_val": numpy_to_east_matrix(X_val),
        "X_test": numpy_to_east_matrix(X_test),
        "Y_train": numpy_to_east_matrix(Y_train),
        "Y_val": numpy_to_east_matrix(Y_val),
        "Y_test": numpy_to_east_matrix(Y_test),
    })


# Platform registration
PlatformFunction(
    name="sklearn_train_val_test_split",
    inputs=[MatrixType, MultiTargetType, ThreeWaySplitConfigType],
    output=ThreeWaySplitResultType,
    type="sync",
    fn=sklearn_train_val_test_split_impl,
),
```

### Test (`sklearn.spec.ts`)

```typescript
test("train_val_test_split creates 3-way split", $ => {
    // 10 samples, 3 features
    const X = $.let([
        [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 18.0],
        [19.0, 20.0, 21.0], [22.0, 23.0, 24.0], [25.0, 26.0, 27.0],
        [28.0, 29.0, 30.0],
    ]);
    // 10 samples, 2 targets
    const Y = $.let([
        [1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0],
        [11.0, 12.0], [13.0, 14.0], [15.0, 16.0], [17.0, 18.0], [19.0, 20.0],
    ]);

    const config = $.let({
        val_size: variant('some', 0.2),
        test_size: variant('some', 0.2),
        random_state: variant('some', 42n),
        shuffle: variant('some', true),
    });

    const result = $.let(Sklearn.trainValTestSplit(X, Y, config));

    // 60% train, 20% val, 20% test
    $(Assert.equal(result.X_train.size(), 6n));
    $(Assert.equal(result.X_val.size(), 2n));
    $(Assert.equal(result.X_test.size(), 2n));
    $(Assert.equal(result.Y_train.size(), 6n));
    $(Assert.equal(result.Y_val.size(), 2n));
    $(Assert.equal(result.Y_test.size(), 2n));
});
```

---

## Implementation Checklist

### TypeScript (`sklearn.ts`)

#### Regression
- [ ] Add `NullType` to imports
- [ ] Add `RegressionMetricType` variant
- [ ] Add `MetricResultType` struct
- [ ] Add `MetricsResultType` array
- [ ] Add `sklearn_compute_metrics` platform function
- [ ] Add `MetricAggregationType` variant
- [ ] Add `MultiMetricsConfigType` struct
- [ ] Add `MultiMetricResultType` struct
- [ ] Add `MultiMetricsResultType` array
- [ ] Add `sklearn_compute_metrics_multi` platform function

#### Classification
- [ ] Add `ClassificationMetricType` variant
- [ ] Add `ClassificationAverageType` variant
- [ ] Add `ClassificationConfigType` struct
- [ ] Add `ClassificationMetricResultType` struct
- [ ] Add `ClassificationMetricsResultType` array
- [ ] Add `sklearn_compute_classification_metrics` platform function
- [ ] Add `MultiClassificationConfigType` struct
- [ ] Add `MultiClassificationMetricResultType` struct
- [ ] Add `MultiClassificationMetricsResultType` array
- [ ] Add `sklearn_compute_classification_metrics_multi` platform function

#### Train/Val/Test Split
- [ ] Add `sklearn_train_val_test_split` platform function (types already exist)

#### Exports & Tests
- [ ] Update `SklearnTypes` exports
- [ ] Update `Sklearn` grouped export
- [ ] Add regression tests in `sklearn.spec.ts`
- [ ] Add classification tests in `sklearn.spec.ts`
- [ ] Add train_val_test_split test in `sklearn.spec.ts`

### Python

#### Types (`types.py`)
- [ ] Add regression type definitions
- [ ] Add classification type definitions

#### Implementation (`sklearn_impl.py`)
- [ ] Add `METRIC_FUNCTIONS` mapping (regression)
- [ ] Add `CLASSIFICATION_METRIC_FUNCTIONS` mapping
- [ ] Implement `sklearn_compute_metrics_impl`
- [ ] Implement `sklearn_compute_metrics_multi_impl`
- [ ] Implement `sklearn_compute_classification_metrics_impl`
- [ ] Implement `sklearn_compute_classification_metrics_multi_impl`
- [ ] Implement `sklearn_train_val_test_split_impl`
- [ ] Register all platform functions
- [ ] Update `__all__` exports
- [ ] Run tests
