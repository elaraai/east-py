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

// ============================================================================
// Class Weight Types
// ============================================================================

/**
 * Mode for computing class weights.
 */
export const ClassWeightModeType = VariantType({
    /** Weights are inversely proportional to class frequencies */
    balanced: NullType,
});

// ============================================================================
// Confusion Matrix Types
// ============================================================================

/**
 * Result type for confusion matrix.
 */
export const ConfusionMatrixResultType = StructType({
    /** Confusion matrix (n_classes x n_classes) */
    matrix: MatrixType,
    /** Class labels in order */
    classes: LabelVectorType,
});

// Re-export config types used in RegressorChain
export { XGBoostConfigType } from "../xgboost/xgboost.js";
export { LightGBMConfigType } from "../lightgbm/lightgbm.js";
export { NGBoostConfigType } from "../ngboost/ngboost.js";
export { GPConfigType } from "../gp/gp.js";

// ============================================================================
// Config Types
// ============================================================================

/**
 * Configuration for data splitting.
 *
 * Examples:
 * - 2-way: split_sizes: [0.8, 0.2] -> train/test
 * - 3-way: split_sizes: [0.7, 0.15, 0.15] -> train/val/test
 * - 4-way: split_sizes: [0.6, 0.1, 0.15, 0.15] -> train/val/calib/test
 */
export const SplitConfigType = StructType({
    /** Array of split proportions (must sum to 1.0). */
    split_sizes: ArrayType(FloatType),
    /** Random seed for reproducibility */
    random_state: OptionType(IntegerType),
    /** Whether to shuffle data before splitting (default true) */
    shuffle: OptionType(BooleanType),
    /**
     * Multiple stratification columns - combined into compound strata.
     * Each inner array is one column of labels (same length as X).
     * E.g., [[origin1, origin2, ...], [mpf1, mpf2, ...]] stratifies on origin × mpf.
     */
    stratify: OptionType(ArrayType(ArrayType(IntegerType))),
    /** Minimum samples per stratify class. Classes with fewer samples are rejected. (default = n_splits) */
    min_stratify_samples: OptionType(IntegerType),
    /**
     * Columns that must have overlapping representation in all splits (but not used for stratification).
     * Samples with values that don't appear in all splits are rejected.
     * Each inner array is one column of labels (same length as X).
     */
    overlap: OptionType(ArrayType(ArrayType(IntegerType))),
});

// ============================================================================
// Result Types
// ============================================================================

/**
 * Result of data splitting.
 */
export const SplitResultType = StructType({
    /** Array of feature matrices, one per split (in order of split_sizes) */
    X_splits: ArrayType(MatrixType),
    /** Array of target matrices, one per split (in order of split_sizes) */
    Y_splits: ArrayType(MatrixType),
    /** Indices of rows rejected due to rare stratify classes or missing overlap values */
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
 * Weights type for Cohen's Kappa score.
 */
export const CohenKappaWeightsType = VariantType({
    /** No weighting (default) */
    none: NullType,
    /** Linear weighting - penalizes disagreements linearly */
    linear: NullType,
    /** Quadratic weighting - penalizes disagreements quadratically */
    quadratic: NullType,
});

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
    /** Cohen's Kappa - sklearn.metrics.cohen_kappa_score (with optional weights) */
    cohen_kappa: CohenKappaWeightsType,
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
 * Multi-class strategy for ROC AUC.
 */
export const RocAucMultiClassType = VariantType({
    /** One-vs-rest (OvR) - computes AUC of each class against all others */
    ovr: NullType,
    /** One-vs-one (OvO) - computes pairwise AUC and averages */
    ovo: NullType,
});

/**
 * Configuration for ROC AUC score.
 */
export const RocAucConfigType = StructType({
    /** Multi-class strategy (default: ovr) */
    multi_class: OptionType(RocAucMultiClassType),
    /** Averaging strategy for multi-class: 'macro' or 'weighted' (default: macro) */
    average: OptionType(ClassificationAverageType),
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
    /** RobustScaler model */
    robust_scaler: StructType({
        /** ONNX model bytes */
        onnx: BlobType,
        /** Number of input features */
        n_features: IntegerType,
    }),
    /** LabelEncoder model */
    label_encoder: StructType({
        /** Cloudpickle serialized encoder */
        data: BlobType,
        /** Number of unique classes */
        n_classes: IntegerType,
    }),
    /** OrdinalEncoder model */
    ordinal_encoder: StructType({
        /** Cloudpickle serialized encoder */
        data: BlobType,
        /** Number of features */
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
 * Split arrays into N subsets (train/test, train/val/test, etc.).
 *
 * @param X - Feature matrix
 * @param Y - Target matrix
 * @param config - Split configuration with split_sizes, stratify, overlap
 * @returns Split result with X_splits, Y_splits arrays
 *
 * @example
 * ```ts
 * // 2-way split (train/test)
 * const result = Sklearn.split(X, Y, { split_sizes: [0.8, 0.2], ... });
 * const [X_train, X_test] = [result.X_splits.get(0n), result.X_splits.get(1n)];
 *
 * // 3-way split (train/val/test)
 * const result = Sklearn.split(X, Y, { split_sizes: [0.7, 0.15, 0.15], ... });
 *
 * // With multi-column stratification
 * const result = Sklearn.split(X, Y, {
 *     split_sizes: [0.7, 0.15, 0.15],
 *     stratify: variant('some', [origin_labels, mpf_labels]),
 *     overlap: variant('some', [class_labels]),
 * });
 * ```
 */
export const sklearn_split = East.platform(
    "sklearn_split",
    [MatrixType, MatrixType, SplitConfigType],
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
 * Fit a RobustScaler to training data.
 *
 * Scales features using statistics that are robust to outliers.
 * Centers data using the median and scales using the interquartile range (IQR).
 *
 * @param X - Training feature matrix
 * @returns Model blob containing fitted scaler
 */
export const sklearn_robust_scaler_fit = East.platform(
    "sklearn_robust_scaler_fit",
    [MatrixType],
    SklearnModelBlobType
);

/**
 * Transform data using a fitted RobustScaler.
 *
 * @param model - Fitted scaler model blob
 * @param X - Feature matrix to transform
 * @returns Transformed feature matrix
 */
export const sklearn_robust_scaler_transform = East.platform(
    "sklearn_robust_scaler_transform",
    [SklearnModelBlobType, MatrixType],
    MatrixType
);

/**
 * Fit a LabelEncoder to encode target labels.
 *
 * Encodes labels with values between 0 and n_classes-1.
 *
 * @param y - Target labels (1D integer array)
 * @returns Model blob containing fitted encoder
 */
export const sklearn_label_encoder_fit = East.platform(
    "sklearn_label_encoder_fit",
    [LabelVectorType],
    SklearnModelBlobType
);

/**
 * Transform labels using a fitted LabelEncoder.
 *
 * @param model - Fitted encoder model blob
 * @param y - Labels to transform
 * @returns Encoded labels (0 to n_classes-1)
 */
export const sklearn_label_encoder_transform = East.platform(
    "sklearn_label_encoder_transform",
    [SklearnModelBlobType, LabelVectorType],
    LabelVectorType
);

/**
 * Inverse transform encoded labels back to original values.
 *
 * @param model - Fitted encoder model blob
 * @param y - Encoded labels to inverse transform
 * @returns Original label values
 */
export const sklearn_label_encoder_inverse_transform = East.platform(
    "sklearn_label_encoder_inverse_transform",
    [SklearnModelBlobType, LabelVectorType],
    LabelVectorType
);

/**
 * Fit an OrdinalEncoder to encode categorical features.
 *
 * Encodes categorical features as ordinal integers.
 *
 * @param X - Feature matrix with categorical values
 * @returns Model blob containing fitted encoder
 */
export const sklearn_ordinal_encoder_fit = East.platform(
    "sklearn_ordinal_encoder_fit",
    [MatrixType],
    SklearnModelBlobType
);

/**
 * Transform features using a fitted OrdinalEncoder.
 *
 * @param model - Fitted encoder model blob
 * @param X - Feature matrix to transform
 * @returns Encoded feature matrix
 */
export const sklearn_ordinal_encoder_transform = East.platform(
    "sklearn_ordinal_encoder_transform",
    [SklearnModelBlobType, MatrixType],
    MatrixType
);

/**
 * Compute class weights for balanced training.
 *
 * Calculates weights inversely proportional to class frequencies,
 * useful for handling class imbalance in classification tasks.
 *
 * @param mode - How to compute weights (balanced)
 * @param y - Class labels (1D integer array)
 * @returns Weights for each class (ordered by class index)
 */
export const sklearn_compute_class_weight = East.platform(
    "sklearn_compute_class_weight",
    [ClassWeightModeType, LabelVectorType],
    VectorType
);

/**
 * Compute confusion matrix for classification results.
 *
 * Returns a matrix where entry [i,j] is the number of samples
 * with true label i that were predicted as label j.
 *
 * @param y_true - True class labels (1D integer array)
 * @param y_pred - Predicted class labels (1D integer array)
 * @returns Confusion matrix result with matrix and class labels
 */
export const sklearn_confusion_matrix = East.platform(
    "sklearn_confusion_matrix",
    [LabelVectorType, LabelVectorType],
    ConfusionMatrixResultType
);

/**
 * Compute ROC AUC score for classification results.
 *
 * For binary classification, pass probabilities for the positive class.
 * For multi-class, pass probability matrix (n_samples x n_classes).
 *
 * @param y_true - True class labels (1D integer array)
 * @param y_proba - Predicted probabilities (matrix: n_samples x n_classes)
 * @param config - Configuration for multi-class handling
 * @returns ROC AUC score
 */
export const sklearn_roc_auc_score = East.platform(
    "sklearn_roc_auc_score",
    [LabelVectorType, MatrixType, RocAucConfigType],
    FloatType
);

/**
 * Compute log loss (cross-entropy loss) for classification results.
 *
 * @param y_true - True class labels (1D integer array)
 * @param y_proba - Predicted probabilities (matrix: n_samples x n_classes)
 * @returns Log loss value
 */
export const sklearn_log_loss = East.platform(
    "sklearn_log_loss",
    [LabelVectorType, MatrixType],
    FloatType
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
    /** Class weight mode type */
    ClassWeightModeType,
    /** Confusion matrix result type */
    ConfusionMatrixResultType,
    /** ROC AUC multi-class strategy type */
    RocAucMultiClassType,
    /** ROC AUC configuration type */
    RocAucConfigType,
    /** Split configuration type */
    SplitConfigType,
    /** Split result type */
    SplitResultType,
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
    /** Cohen's Kappa weights type */
    CohenKappaWeightsType,
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
    /** Split arrays into N subsets (train/test, train/val/test, etc.) */
    split: sklearn_split,
    /** Fit a StandardScaler to data */
    standardScalerFit: sklearn_standard_scaler_fit,
    /** Transform data using fitted StandardScaler */
    standardScalerTransform: sklearn_standard_scaler_transform,
    /** Fit a MinMaxScaler to data */
    minMaxScalerFit: sklearn_min_max_scaler_fit,
    /** Transform data using fitted MinMaxScaler */
    minMaxScalerTransform: sklearn_min_max_scaler_transform,
    /** Fit a RobustScaler to data */
    robustScalerFit: sklearn_robust_scaler_fit,
    /** Transform data using fitted RobustScaler */
    robustScalerTransform: sklearn_robust_scaler_transform,
    /** Fit a LabelEncoder to labels */
    labelEncoderFit: sklearn_label_encoder_fit,
    /** Transform labels using fitted LabelEncoder */
    labelEncoderTransform: sklearn_label_encoder_transform,
    /** Inverse transform encoded labels */
    labelEncoderInverseTransform: sklearn_label_encoder_inverse_transform,
    /** Fit an OrdinalEncoder to features */
    ordinalEncoderFit: sklearn_ordinal_encoder_fit,
    /** Transform features using fitted OrdinalEncoder */
    ordinalEncoderTransform: sklearn_ordinal_encoder_transform,
    /** Compute regression metrics (single-target) */
    computeMetrics: sklearn_compute_metrics,
    /** Compute regression metrics (multi-target) */
    computeMetricsMulti: sklearn_compute_metrics_multi,
    /** Compute classification metrics (single-target) */
    computeClassificationMetrics: sklearn_compute_classification_metrics,
    /** Compute classification metrics (multi-target) */
    computeClassificationMetricsMulti: sklearn_compute_classification_metrics_multi,
    /** Compute class weights for balanced training */
    computeClassWeight: sklearn_compute_class_weight,
    /** Compute confusion matrix */
    confusionMatrix: sklearn_confusion_matrix,
    /** Compute ROC AUC score */
    rocAucScore: sklearn_roc_auc_score,
    /** Compute log loss (cross-entropy) */
    logLoss: sklearn_log_loss,
    /** Train a RegressorChain for multi-target regression */
    regressorChainTrain: sklearn_regressor_chain_train,
    /** Predict using a fitted RegressorChain */
    regressorChainPredict: sklearn_regressor_chain_predict,
    /** Type definitions */
    Types: SklearnTypes,
} as const;
