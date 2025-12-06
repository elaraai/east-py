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
});

/**
 * Regression metrics result.
 */
export const RegressionMetricsType = StructType({
    /** Mean Squared Error */
    mse: FloatType,
    /** Root Mean Squared Error */
    rmse: FloatType,
    /** Mean Absolute Error */
    mae: FloatType,
    /** R-squared (coefficient of determination) */
    r2: FloatType,
    /** Mean Absolute Percentage Error */
    mape: FloatType,
});

/**
 * Classification metrics result.
 */
export const ClassificationMetricsType = StructType({
    /** Accuracy score */
    accuracy: FloatType,
    /** Precision (weighted average) */
    precision: FloatType,
    /** Recall (weighted average) */
    recall: FloatType,
    /** F1 score (weighted average) */
    f1: FloatType,
});

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
 * Compute regression metrics.
 *
 * @param y_true - True target values
 * @param y_pred - Predicted target values
 * @returns Regression metrics (MSE, RMSE, MAE, R2, MAPE)
 */
export const sklearn_metrics_regression = East.platform(
    "sklearn_metrics_regression",
    [VectorType, VectorType],
    RegressionMetricsType
);

/**
 * Compute classification metrics.
 *
 * @param y_true - True class labels
 * @param y_pred - Predicted class labels
 * @returns Classification metrics (accuracy, precision, recall, F1)
 */
export const sklearn_metrics_classification = East.platform(
    "sklearn_metrics_classification",
    [LabelVectorType, LabelVectorType],
    ClassificationMetricsType
);

/**
 * Multi-target output type for RegressorChain.
 */
export const MultiTargetType = ArrayType(ArrayType(FloatType));

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
    [MatrixType, MultiTargetType, RegressorChainConfigType],
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
    MultiTargetType
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
    /** Regression metrics type */
    RegressionMetricsType,
    /** Classification metrics type */
    ClassificationMetricsType,
    /** Model blob type for sklearn models */
    ModelBlobType: SklearnModelBlobType,
    /** Multi-target output type */
    MultiTargetType,
    /** RegressorChain base estimator config type */
    RegressorChainBaseConfigType,
    /** RegressorChain config type */
    RegressorChainConfigType,
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
    /** Fit a StandardScaler to data */
    standardScalerFit: sklearn_standard_scaler_fit,
    /** Transform data using fitted StandardScaler */
    standardScalerTransform: sklearn_standard_scaler_transform,
    /** Fit a MinMaxScaler to data */
    minMaxScalerFit: sklearn_min_max_scaler_fit,
    /** Transform data using fitted MinMaxScaler */
    minMaxScalerTransform: sklearn_min_max_scaler_transform,
    /** Compute regression metrics */
    metricsRegression: sklearn_metrics_regression,
    /** Compute classification metrics */
    metricsClassification: sklearn_metrics_classification,
    /** Train a RegressorChain for multi-target regression */
    regressorChainTrain: sklearn_regressor_chain_train,
    /** Predict using a fitted RegressorChain */
    regressorChainPredict: sklearn_regressor_chain_predict,
    /** Type definitions */
    Types: SklearnTypes,
} as const;
