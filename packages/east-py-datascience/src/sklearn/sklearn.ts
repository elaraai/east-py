/**
 * Copyright (c) 2025 Elara AI Pty Ltd
 * Dual-licensed under AGPL-3.0 and commercial license. See LICENSE for details.
 */

/**
 * Scikit-learn platform functions for East.
 *
 * Provides core machine learning utilities: preprocessing, model selection, and metrics.
 * Uses ONNX for model serialization to enable portable inference.
 *
 * @packageDocumentation
 */

import {
    East,
    StructType,
    VariantType,
    OptionType,
    IntegerType,
    BooleanType,
    FloatType,
    BlobType,
    ArrayType,
    StringType,
    NullType,
} from "@elaraai/east";
import { VectorType, MatrixType, LabelVectorType } from "../types.js";
import { XGBoostConfigType } from "../xgboost/xgboost.js";
import { LightGBMConfigType } from "../lightgbm/lightgbm.js";
import { NGBoostConfigType } from "../ngboost/ngboost.js";
import { GPConfigType } from "../gp/gp.js";

// Re-export shared types for convenience
export { VectorType, MatrixType, LabelVectorType } from "../types.js";
// Re-export config types used in RegressorChain
export { XGBoostConfigType } from "../xgboost/xgboost.js";
export { LightGBMConfigType } from "../lightgbm/lightgbm.js";
export { NGBoostConfigType } from "../ngboost/ngboost.js";
export { GPConfigType } from "../gp/gp.js";

// ============================================================================
// Config Types
// ============================================================================

/**
 * Configuration for train/test split.
 */
export const SplitConfigType = StructType({
    /** Proportion of data to include in test split (default 0.2) */
    test_size: OptionType(FloatType),
    /** Random seed for reproducibility */
    random_state: OptionType(IntegerType),
    /** Whether to shuffle data before splitting (default true) */
    shuffle: OptionType(BooleanType),
    /** Stratification labels (same length as X). Ensures proportional representation in each split. */
    stratify: OptionType(ArrayType(IntegerType)),
    /** Minimum samples per stratify class. Classes with fewer samples are rejected. (default 2) */
    min_stratify_samples: OptionType(IntegerType),
});

/**
 * Configuration for 3-way train/val/test split.
 */
export const ThreeWaySplitConfigType = StructType({
    /** Proportion of data for validation (default 0.15) */
    val_size: OptionType(FloatType),
    /** Proportion of data for test/holdout (default 0.15) */
    test_size: OptionType(FloatType),
    /** Random seed for reproducibility */
    random_state: OptionType(IntegerType),
    /** Whether to shuffle data before splitting (default true) */
    shuffle: OptionType(BooleanType),
    /** Stratification labels (same length as X). Ensures proportional representation in each split. */
    stratify: OptionType(ArrayType(IntegerType)),
    /** Minimum samples per stratify class. Classes with fewer samples are rejected. (default 3) */
    min_stratify_samples: OptionType(IntegerType),
});

// ============================================================================
// Result Types
// ============================================================================

/**
 * Result of train/test split.
 */
export const SplitResultType = StructType({
    /** Training features */
    X_train: MatrixType,
    /** Test features */
    X_test: MatrixType,
    /** Training labels */
    y_train: VectorType,
    /** Test labels */
    y_test: VectorType,
    /** Indices of rows rejected due to rare stratify classes (empty if no stratify or no rejections) */
    rejected_indices: ArrayType(IntegerType),
});

/**
 * Result of 3-way train/val/test split.
 */
export const ThreeWaySplitResultType = StructType({
    /** Training features */
    X_train: MatrixType,
    /** Validation features */
    X_val: MatrixType,
    /** Test/holdout features */
    X_test: MatrixType,
    /** Training targets (matrix) */
    Y_train: MatrixType,
    /** Validation targets (matrix) */
    Y_val: MatrixType,
    /** Test/holdout targets (matrix) */
    Y_test: MatrixType,
    /** Indices of rows rejected due to rare stratify classes (empty if no stratify or no rejections) */
    rejected_indices: ArrayType(IntegerType),
});

// ============================================================================
// Flexible Metrics Types
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
    /** Mean Error (bias) - mean(pred - true), should be ~0 for unbiased predictions */
    mean_error: NullType,
    /** Pinball Loss - proper scoring rule for quantile regression (requires alpha parameter) */
    pinball_loss: FloatType,
    /** Huber Loss - robust to outliers (requires delta parameter, default 1.0) */
    huber: FloatType,
    /** Mean Tweedie Deviance - for skewed distributions (requires power parameter) */
    mean_tweedie_deviance: FloatType,
});

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
 * Result containing multiple computed metrics.
 */
export const MetricsResultType = ArrayType(MetricResultType);

/**
 * Aggregation strategy for multi-target metrics.
 */
export const MetricAggregationType = VariantType({
    /** Return metric for each target separately (default) */
    per_target: NullType,
    /** Average across all targets (uniform weights) */
    uniform_average: NullType,
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

/**
 * Configuration for classification metrics.
 */
export const ClassificationMetricsConfigType = StructType({
    /** Averaging strategy for multi-class (default: macro) */
    average: OptionType(ClassificationAverageType),
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
export const ClassificationMetricResultsType = ArrayType(ClassificationMetricResultType);

/**
 * Configuration for multi-target classification metrics.
 */
export const MultiClassificationConfigType = StructType({
    /** Averaging strategy for multi-class (default: macro) */
    average: OptionType(ClassificationAverageType),
    /** How to aggregate across targets (default: per_target) */
    aggregation: OptionType(MetricAggregationType),
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
        /** Per-target values */
        per_target: VectorType,
    }),
});

/**
 * Result containing multiple computed classification metrics (multi-target).
 */
export const MultiClassificationMetricResultsType = ArrayType(MultiClassificationMetricResultType);

// ============================================================================
// Model Blob Types
// ============================================================================

/**
 * Model blob type for serialized sklearn models.
 *
 * Each model type has its own variant case containing ONNX bytes and metadata.
 */
export const SklearnModelBlobType = VariantType({
    /** StandardScaler model */
    standard_scaler: StructType({
        /** ONNX model bytes */
        onnx: BlobType,
        /** Number of input features */
        n_features: IntegerType,
    }),
    /** MinMaxScaler model */
    min_max_scaler: StructType({
        /** ONNX model bytes */
        onnx: BlobType,
        /** Number of input features */
        n_features: IntegerType,
    }),
    /** RegressorChain model */
    regressor_chain: StructType({
        /** Cloudpickle serialized chain */
        data: BlobType,
        /** Number of input features */
        n_features: IntegerType,
        /** Number of target outputs */
        n_targets: IntegerType,
        /** Base estimator type name */
        base_estimator_type: StringType,
    }),
});

// ============================================================================
// RegressorChain Types
// ============================================================================

/**
 * Base estimator configuration for RegressorChain.
 * Variant carries both the estimator type AND its configuration.
 */
export const RegressorChainBaseConfigType = VariantType({
    /** XGBoost regressor */
    xgboost: XGBoostConfigType,
    /** LightGBM regressor */
    lightgbm: LightGBMConfigType,
    /** NGBoost regressor */
    ngboost: NGBoostConfigType,
    /** Gaussian Process regressor */
    gp: GPConfigType,
});

/**
 * Configuration for RegressorChain.
 */
export const RegressorChainConfigType = StructType({
    /** Base estimator with its configuration */
    base_estimator: RegressorChainBaseConfigType,
    /** Chain order (indices of targets). None = natural order [0,1,2,...] */
    order: OptionType(ArrayType(IntegerType)),
    /** Random seed for reproducibility */
    random_state: OptionType(IntegerType),
});

// ============================================================================
// Platform Functions
// ============================================================================

/**
 * Split arrays into train and test subsets.
 *
 * @param X - Feature matrix
 * @param y - Target vector
 * @param config - Split configuration
 * @returns Split result with X_train, X_test, y_train, y_test
 */
export const sklearn_train_test_split = East.platform(
    "sklearn_train_test_split",
    [MatrixType, VectorType, SplitConfigType],
    SplitResultType
);

/**
 * Fit a StandardScaler to training data.
 *
 * Standardizes features by removing the mean and scaling to unit variance.
 *
 * @param X - Training feature matrix
 * @returns Model blob containing fitted scaler
 */
export const sklearn_standard_scaler_fit = East.platform(
    "sklearn_standard_scaler_fit",
    [MatrixType],
    SklearnModelBlobType
);

/**
 * Transform data using a fitted StandardScaler.
 *
 * @param model - Fitted scaler model blob
 * @param X - Feature matrix to transform
 * @returns Transformed feature matrix
 */
export const sklearn_standard_scaler_transform = East.platform(
    "sklearn_standard_scaler_transform",
    [SklearnModelBlobType, MatrixType],
    MatrixType
);

/**
 * Fit a MinMaxScaler to training data.
 *
 * Scales features to a given range (default [0, 1]).
 *
 * @param X - Training feature matrix
 * @returns Model blob containing fitted scaler
 */
export const sklearn_min_max_scaler_fit = East.platform(
    "sklearn_min_max_scaler_fit",
    [MatrixType],
    SklearnModelBlobType
);

/**
 * Transform data using a fitted MinMaxScaler.
 *
 * @param model - Fitted scaler model blob
 * @param X - Feature matrix to transform
 * @returns Transformed feature matrix
 */
export const sklearn_min_max_scaler_transform = East.platform(
    "sklearn_min_max_scaler_transform",
    [SklearnModelBlobType, MatrixType],
    MatrixType
);

/**
 * Train a RegressorChain for multi-target regression.
 *
 * Each model in the chain uses previous targets as additional features,
 * enabling modeling of dependencies between targets.
 *
 * @param X - Feature matrix
 * @param Y - Target matrix (rows=samples, cols=targets)
 * @param config - Chain configuration
 * @returns Model blob containing fitted chain
 */
export const sklearn_regressor_chain_train = East.platform(
    "sklearn_regressor_chain_train",
    [MatrixType, MatrixType, RegressorChainConfigType],
    SklearnModelBlobType
);

/**
 * Predict using a fitted RegressorChain.
 *
 * @param model - Fitted chain model blob
 * @param X - Feature matrix to predict
 * @returns Predicted target matrix
 */
export const sklearn_regressor_chain_predict = East.platform(
    "sklearn_regressor_chain_predict",
    [SklearnModelBlobType, MatrixType],
    MatrixType
);

/**
 * Split arrays into train, validation, and test subsets.
 *
 * @param X - Feature matrix
 * @param Y - Target matrix (multi-target)
 * @param config - Split configuration with val_size and test_size
 * @returns Split result with X_train, X_val, X_test, Y_train, Y_val, Y_test
 */
export const sklearn_train_val_test_split = East.platform(
    "sklearn_train_val_test_split",
    [MatrixType, MatrixType, ThreeWaySplitConfigType],
    ThreeWaySplitResultType
);

/**
 * Compute regression metrics for single-target predictions.
 *
 * @param y_true - True target values (1D vector)
 * @param y_pred - Predicted target values (1D vector)
 * @param metrics - Array of metrics to compute
 * @returns Array of metric results with scalar values
 */
export const sklearn_compute_metrics = East.platform(
    "sklearn_compute_metrics",
    [VectorType, VectorType, ArrayType(RegressionMetricType)],
    MetricsResultType
);

/**
 * Compute regression metrics for multi-target predictions.
 *
 * @param Y_true - True target matrix [n_samples, n_targets]
 * @param Y_pred - Predicted target matrix [n_samples, n_targets]
 * @param metrics - Array of metrics to compute
 * @param config - Aggregation configuration
 * @returns Array of metric results with per-target or aggregated values
 */
export const sklearn_compute_metrics_multi = East.platform(
    "sklearn_compute_metrics_multi",
    [MatrixType, MatrixType, ArrayType(RegressionMetricType), MultiMetricsConfigType],
    MultiMetricsResultType
);

/**
 * Compute classification metrics for single-target predictions.
 *
 * @param y_true - True class labels (1D integer array)
 * @param y_pred - Predicted class labels (1D integer array)
 * @param metrics - Array of metrics to compute
 * @param config - Configuration (averaging strategy)
 * @returns Array of metric results with scalar values
 */
export const sklearn_compute_classification_metrics = East.platform(
    "sklearn_compute_classification_metrics",
    [LabelVectorType, LabelVectorType, ArrayType(ClassificationMetricType), ClassificationMetricsConfigType],
    ClassificationMetricResultsType
);

/**
 * Compute classification metrics for multi-target predictions.
 *
 * @param Y_true - True class labels matrix [n_samples, n_targets]
 * @param Y_pred - Predicted class labels matrix [n_samples, n_targets]
 * @param metrics - Array of metrics to compute
 * @param config - Configuration (averaging, aggregation)
 * @returns Array of metric results with per-target or aggregated values
 */
export const sklearn_compute_classification_metrics_multi = East.platform(
    "sklearn_compute_classification_metrics_multi",
    [MatrixType, MatrixType, ArrayType(ClassificationMetricType), MultiClassificationConfigType],
    MultiClassificationMetricResultsType
);

// ============================================================================
// Grouped Export
// ============================================================================

/**
 * Type definitions for sklearn functions.
 */
export const SklearnTypes = {
    /** Vector type (array of floats) */
    VectorType,
    /** Matrix type (2D array of floats) */
    MatrixType,
    /** Label vector type (array of integers) */
    LabelVectorType,
    /** Split configuration type */
    SplitConfigType,
    /** Split result type */
    SplitResultType,
    /** 3-way split configuration type */
    ThreeWaySplitConfigType,
    /** 3-way split result type */
    ThreeWaySplitResultType,
    /** Model blob type for sklearn models */
    ModelBlobType: SklearnModelBlobType,
    /** RegressorChain base estimator config type */
    RegressorChainBaseConfigType,
    /** RegressorChain config type */
    RegressorChainConfigType,
    // Flexible metrics types
    /** Regression metric variant */
    RegressionMetricType,
    /** Single metric result */
    MetricResultType,
    /** Multiple metrics result */
    MetricsResultType,
    /** Metric aggregation type */
    MetricAggregationType,
    /** Multi-target metrics config */
    MultiMetricsConfigType,
    /** Multi-target metric result */
    MultiMetricResultType,
    /** Multi-target metrics result */
    MultiMetricsResultType,
    /** Classification metric variant */
    ClassificationMetricType,
    /** Classification averaging type */
    ClassificationAverageType,
    /** Classification metrics config */
    ClassificationMetricsConfigType,
    /** Classification metric result */
    ClassificationMetricResultType,
    /** Classification metrics result */
    ClassificationMetricResultsType,
    /** Multi-target classification config */
    MultiClassificationConfigType,
    /** Multi-target classification metric result */
    MultiClassificationMetricResultType,
    /** Multi-target classification metrics result */
    MultiClassificationMetricResultsType,
} as const;

/**
 * Scikit-learn machine learning utilities.
 *
 * Provides preprocessing, model selection, and metrics for ML workflows.
 *
 * @example
 * ```ts
 * import { East, variant } from "@elaraai/east";
 * import { Sklearn } from "@elaraai/east-py-datascience";
 *
 * const pipeline = East.function([], Sklearn.Types.SplitResultType, $ => {
 *     const X = $.let([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]);
 *     const y = $.let([1.0, 2.0, 3.0, 4.0]);
 *     const config = $.let({
 *         test_size: variant('some', 0.25),
 *         random_state: variant('some', 42n),
 *         shuffle: variant('some', true),
 *     });
 *     return $.return(Sklearn.trainTestSplit(X, y, config));
 * });
 * ```
 */
export const Sklearn = {
    /** Split arrays into train and test subsets */
    trainTestSplit: sklearn_train_test_split,
    /** Split arrays into train, validation, and test subsets */
    trainValTestSplit: sklearn_train_val_test_split,
    /** Fit a StandardScaler to data */
    standardScalerFit: sklearn_standard_scaler_fit,
    /** Transform data using fitted StandardScaler */
    standardScalerTransform: sklearn_standard_scaler_transform,
    /** Fit a MinMaxScaler to data */
    minMaxScalerFit: sklearn_min_max_scaler_fit,
    /** Transform data using fitted MinMaxScaler */
    minMaxScalerTransform: sklearn_min_max_scaler_transform,
    /** Compute regression metrics (single-target) */
    computeMetrics: sklearn_compute_metrics,
    /** Compute regression metrics (multi-target) */
    computeMetricsMulti: sklearn_compute_metrics_multi,
    /** Compute classification metrics (single-target) */
    computeClassificationMetrics: sklearn_compute_classification_metrics,
    /** Compute classification metrics (multi-target) */
    computeClassificationMetricsMulti: sklearn_compute_classification_metrics_multi,
    /** Train a RegressorChain for multi-target regression */
    regressorChainTrain: sklearn_regressor_chain_train,
    /** Predict using a fitted RegressorChain */
    regressorChainPredict: sklearn_regressor_chain_predict,
    /** Type definitions */
    Types: SklearnTypes,
} as const;
