/**
 * Copyright (c) 2025 Elara AI Pty Ltd
 * Dual-licensed under AGPL-3.0 and commercial license. See LICENSE for details.
 */

/**
 * MAPIE conformal prediction intervals for East.
 *
 * Provides prediction intervals with coverage guarantees using
 * conformal prediction methods (MAPIE 1.2.0 API).
 *
 * @packageDocumentation
 */

import {
    East,
    StructType,
    VariantType,
    OptionType,
    ArrayType,
    IntegerType,
    FloatType,
    BlobType,
    NullType,
} from "@elaraai/east";
import { VectorType, MatrixType, LabelVectorType } from "../types.js";

// Re-export shared types for convenience
export { VectorType, MatrixType, LabelVectorType } from "../types.js";

// ============================================================================
// Config Types
// ============================================================================

/**
 * Conformal prediction method for regression.
 */
export const ConformalMethodType = VariantType({
    /** Split conformal - requires separate calibration set */
    split: NullType,
    /** Cross conformal - uses CV for calibration (combines train + calib) */
    cross: NullType,
});

/**
 * Configuration for XGBoost base model (subset for MAPIE).
 */
export const MAPIEXGBoostConfigType = StructType({
    /** Number of boosting rounds (default 100) */
    n_estimators: OptionType(IntegerType),
    /** Maximum tree depth (default 6) */
    max_depth: OptionType(IntegerType),
    /** Learning rate / step size shrinkage (default 0.3) */
    learning_rate: OptionType(FloatType),
    /** Minimum sum of instance weight needed in a child (default 1) */
    min_child_weight: OptionType(IntegerType),
    /** Subsample ratio of training instances (default 1.0) */
    subsample: OptionType(FloatType),
    /** Subsample ratio of columns when constructing trees (default 1.0) */
    colsample_bytree: OptionType(FloatType),
    /** L1 regularization term (default 0) */
    reg_alpha: OptionType(FloatType),
    /** L2 regularization term (default 1) */
    reg_lambda: OptionType(FloatType),
    /** Minimum loss reduction required to make a further partition (default 0) */
    gamma: OptionType(FloatType),
    /** Random seed for reproducibility */
    random_state: OptionType(IntegerType),
});

/**
 * Configuration for LightGBM base model (subset for MAPIE).
 */
export const MAPIELightGBMConfigType = StructType({
    /** Number of boosting rounds (default 100) */
    n_estimators: OptionType(IntegerType),
    /** Maximum tree depth, -1 for unlimited (default -1) */
    max_depth: OptionType(IntegerType),
    /** Learning rate / step size shrinkage (default 0.1) */
    learning_rate: OptionType(FloatType),
    /** Maximum number of leaves in one tree (default 31) */
    num_leaves: OptionType(IntegerType),
    /** Minimum number of samples required in a leaf (default 20) */
    min_child_samples: OptionType(IntegerType),
    /** Subsample ratio of training instances (default 1.0) */
    subsample: OptionType(FloatType),
    /** Subsample ratio of columns when constructing trees (default 1.0) */
    colsample_bytree: OptionType(FloatType),
    /** L1 regularization term (default 0) */
    reg_alpha: OptionType(FloatType),
    /** L2 regularization term (default 0) */
    reg_lambda: OptionType(FloatType),
    /** Random seed for reproducibility */
    random_state: OptionType(IntegerType),
});

/**
 * Base model type for MAPIE regression.
 */
export const BaseModelType = VariantType({
    /** XGBoost regressor as base model */
    xgboost: MAPIEXGBoostConfigType,
    /** LightGBM regressor as base model */
    lightgbm: MAPIELightGBMConfigType,
});

/**
 * Configuration for MAPIE conformal prediction.
 */
export const MAPIEConfigType = StructType({
    /** Base model configuration */
    base_model: BaseModelType,
    /** Conformal method (default: split) */
    method: OptionType(ConformalMethodType),
    /** Confidence level: coverage probability (default 0.9 = 90% intervals) */
    confidence_level: OptionType(FloatType),
    /** Number of CV folds for cross method (default 5) */
    cv_folds: OptionType(IntegerType),
    /** Random seed for reproducibility */
    random_state: OptionType(IntegerType),
});

/**
 * Configuration for CQR (Conformalized Quantile Regression).
 * Requires a base model that supports quantile regression (XGBoost).
 */
export const MAPIECQRConfigType = StructType({
    /** XGBoost config for the base quantile model */
    xgboost_config: MAPIEXGBoostConfigType,
    /** Confidence level: coverage probability (default 0.9 = 90% intervals) */
    confidence_level: OptionType(FloatType),
    /** Random seed for reproducibility */
    random_state: OptionType(IntegerType),
});

// ============================================================================
// Classification Config Types
// ============================================================================

/**
 * Classification conformal method (conformity score).
 */
export const ClassificationMethodType = VariantType({
    /** Least Ambiguous set-valued Classifier - smallest sets */
    lac: NullType,
    /** Adaptive Prediction Sets - adapts to probabilities */
    aps: NullType,
});

/**
 * Base classifier type for MAPIE classification.
 */
export const BaseClassifierType = VariantType({
    /** XGBoost classifier as base model */
    xgboost: MAPIEXGBoostConfigType,
    /** LightGBM classifier as base model */
    lightgbm: MAPIELightGBMConfigType,
});

/**
 * Configuration for MAPIE conformal classification.
 */
export const MAPIEClassifierConfigType = StructType({
    /** Base classifier configuration */
    base_model: BaseClassifierType,
    /** Classification conformity score method (default: lac) */
    method: OptionType(ClassificationMethodType),
    /** Confidence level: coverage probability (default 0.9 = 90% coverage) */
    confidence_level: OptionType(FloatType),
    /** Random seed for reproducibility */
    random_state: OptionType(IntegerType),
});

// ============================================================================
// Model Blob Types
// ============================================================================

/** Base model type indicator */
const BaseModelTypeIndicator = VariantType({
    xgboost: NullType,
    lightgbm: NullType,
});

/**
 * Model blob for MAPIE conformal regressor.
 */
export const MAPIERegressorBlobType = VariantType({
    /** MAPIE regressor with split conformal */
    mapie_split: StructType({
        /** Cloudpickle serialized SplitConformalRegressor */
        data: BlobType,
        /** Number of input features */
        n_features: IntegerType,
        /** Confidence level used during calibration */
        confidence_level: FloatType,
        /** Base model type ('xgboost' or 'lightgbm') */
        base_model_type: BaseModelTypeIndicator,
    }),
    /** MAPIE regressor with cross conformal */
    mapie_cross: StructType({
        /** Cloudpickle serialized CrossConformalRegressor */
        data: BlobType,
        /** Number of input features */
        n_features: IntegerType,
        /** Confidence level used during calibration */
        confidence_level: FloatType,
        /** Base model type ('xgboost' or 'lightgbm') */
        base_model_type: BaseModelTypeIndicator,
    }),
    /** MAPIE CQR regressor */
    mapie_cqr: StructType({
        /** Cloudpickle serialized ConformalizedQuantileRegressor */
        data: BlobType,
        /** Number of input features */
        n_features: IntegerType,
        /** Confidence level used during calibration */
        confidence_level: FloatType,
    }),
});

/**
 * Model blob for MAPIE conformal classifier.
 */
export const MAPIEClassifierBlobType = StructType({
    /** Cloudpickle serialized SplitConformalClassifier */
    data: BlobType,
    /** Number of input features */
    n_features: IntegerType,
    /** Number of classes */
    n_classes: IntegerType,
    /** Class labels */
    classes: ArrayType(IntegerType),
    /** Confidence level used during calibration */
    confidence_level: FloatType,
    /** Base model type ('xgboost' or 'lightgbm') */
    base_model_type: BaseModelTypeIndicator,
});

// ============================================================================
// Result Types
// ============================================================================

/**
 * Prediction interval result (regression).
 */
export const IntervalResultType = StructType({
    /** Lower bound of prediction interval */
    lower: VectorType,
    /** Point prediction (median/mean) */
    pred: VectorType,
    /** Upper bound of prediction interval */
    upper: VectorType,
});

/**
 * Prediction set result (classification).
 * For each sample, contains the set of classes included in the prediction set.
 */
export const PredictionSetResultType = StructType({
    /** Predicted class (argmax of probabilities) */
    pred: ArrayType(IntegerType),
    /** Prediction set membership matrix (n_samples x n_classes, 1 if class in set) */
    sets: ArrayType(ArrayType(IntegerType)),
    /** Class probabilities (n_samples x n_classes) */
    probabilities: MatrixType,
    /** Size of each prediction set */
    set_sizes: ArrayType(IntegerType),
});

// ============================================================================
// Platform Functions
// ============================================================================

// --------------------------------
// Regression Functions
// --------------------------------

/**
 * Train a MAPIE conformal regressor.
 *
 * For split conformal, uses X_calib/y_calib for calibration.
 * For cross conformal, combines train and calib data, uses CV for calibration.
 *
 * @param X_train - Training feature matrix
 * @param y_train - Training target vector
 * @param X_calib - Calibration feature matrix
 * @param y_calib - Calibration target vector
 * @param config - MAPIE configuration
 * @returns Model blob containing calibrated MAPIE regressor
 */
export const mapie_train_conformal_regressor = East.platform(
    "mapie_train_conformal_regressor",
    [MatrixType, VectorType, MatrixType, VectorType, MAPIEConfigType],
    MAPIERegressorBlobType
);

/**
 * Train a MAPIE CQR (Conformalized Quantile Regression) model.
 *
 * CQR combines quantile regression with conformal prediction for
 * adaptive intervals that are wider where uncertainty is higher.
 *
 * @param X_train - Training feature matrix
 * @param y_train - Training target vector
 * @param X_calib - Calibration feature matrix
 * @param y_calib - Calibration target vector
 * @param config - CQR configuration
 * @returns Model blob containing calibrated CQR model
 */
export const mapie_train_cqr = East.platform(
    "mapie_train_cqr",
    [MatrixType, VectorType, MatrixType, VectorType, MAPIECQRConfigType],
    MAPIERegressorBlobType
);

/**
 * Predict with intervals using a MAPIE regressor.
 *
 * Returns intervals at the confidence level specified during training.
 *
 * @param model - Trained MAPIE regressor blob
 * @param X - Feature matrix to predict
 * @returns Prediction intervals (lower, pred, upper)
 */
export const mapie_predict_interval = East.platform(
    "mapie_predict_interval",
    [MAPIERegressorBlobType, MatrixType],
    IntervalResultType
);

// --------------------------------
// Classification Functions
// --------------------------------

/**
 * Train a MAPIE conformal classifier.
 *
 * Uses split conformal prediction with calibration set for classification.
 *
 * @param X_train - Training feature matrix
 * @param y_train - Training labels (integers)
 * @param X_calib - Calibration feature matrix
 * @param y_calib - Calibration labels
 * @param config - Classifier configuration
 * @returns Model blob containing calibrated MAPIE classifier
 */
export const mapie_train_conformal_classifier = East.platform(
    "mapie_train_conformal_classifier",
    [MatrixType, LabelVectorType, MatrixType, LabelVectorType, MAPIEClassifierConfigType],
    MAPIEClassifierBlobType
);

/**
 * Predict with prediction sets using a MAPIE classifier.
 *
 * Returns prediction sets at the confidence level specified during training.
 *
 * @param model - Trained MAPIE classifier blob
 * @param X - Feature matrix to predict
 * @returns Prediction sets (pred, sets, probabilities, set_sizes)
 */
export const mapie_predict_set = East.platform(
    "mapie_predict_set",
    [MAPIEClassifierBlobType, MatrixType],
    PredictionSetResultType
);

// ============================================================================
// Grouped Export
// ============================================================================

/**
 * Type definitions for MAPIE functions.
 */
export const MAPIETypes = {
    // Config types
    ConformalMethodType,
    MAPIEXGBoostConfigType,
    MAPIELightGBMConfigType,
    BaseModelType,
    MAPIEConfigType,
    MAPIECQRConfigType,
    ClassificationMethodType,
    BaseClassifierType,
    MAPIEClassifierConfigType,
    // Model blob types
    MAPIERegressorBlobType,
    MAPIEClassifierBlobType,
    // Result types
    IntervalResultType,
    PredictionSetResultType,
    // Shared types
    VectorType,
    MatrixType,
    LabelVectorType,
} as const;

/**
 * MAPIE conformal prediction.
 *
 * Provides prediction intervals with coverage guarantees using
 * conformal prediction methods (MAPIE 1.2.0 API).
 *
 * @example
 * ```ts
 * import { East, variant } from "@elaraai/east";
 * import { MAPIE } from "@elaraai/east-py-datascience";
 *
 * const train = East.function([], MAPIE.Types.MAPIERegressorBlobType, $ => {
 *     const X_train = $.let([[1.0], [2.0], [3.0], [4.0], [5.0]]);
 *     const y_train = $.let([1.5, 2.5, 3.5, 4.5, 5.5]);
 *     const X_calib = $.let([[2.5], [4.5]]);
 *     const y_calib = $.let([3.0, 5.0]);
 *     const config = $.let({
 *         base_model: variant('xgboost', {
 *             n_estimators: variant('some', 50n),
 *             max_depth: variant('some', 3n),
 *             learning_rate: variant('some', 0.1),
 *             min_child_weight: variant('none', null),
 *             subsample: variant('none', null),
 *             colsample_bytree: variant('none', null),
 *             reg_alpha: variant('none', null),
 *             reg_lambda: variant('none', null),
 *             gamma: variant('none', null),
 *             random_state: variant('some', 42n),
 *         }),
 *         method: variant('some', variant('split', null)),
 *         confidence_level: variant('some', 0.9),
 *         cv_folds: variant('none', null),
 *         random_state: variant('some', 42n),
 *     });
 *     return $.return(MAPIE.trainConformalRegressor(X_train, y_train, X_calib, y_calib, config));
 * });
 * ```
 */
export const MAPIE = {
    // Regression
    /** Train MAPIE conformal regressor */
    trainConformalRegressor: mapie_train_conformal_regressor,
    /** Train MAPIE CQR (Conformalized Quantile Regression) */
    trainCQR: mapie_train_cqr,
    /** Predict with intervals */
    predictInterval: mapie_predict_interval,
    // Classification
    /** Train MAPIE conformal classifier */
    trainConformalClassifier: mapie_train_conformal_classifier,
    /** Predict with prediction sets */
    predictSet: mapie_predict_set,
    /** Type definitions */
    Types: MAPIETypes,
} as const;
