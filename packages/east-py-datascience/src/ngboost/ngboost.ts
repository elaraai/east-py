/**
 * Copyright (c) 2025 Elara AI Pty Ltd
 * Dual-licensed under AGPL-3.0 and commercial license. See LICENSE for details.
 */

/**
 * NGBoost platform functions for East.
 *
 * Provides probabilistic predictions with natural gradient boosting.
 * Returns mean, standard deviation, and confidence intervals.
 * Uses cloudpickle for model serialization.
 *
 * @packageDocumentation
 */

import {
    East,
    StructType,
    VariantType,
    OptionType,
    IntegerType,
    FloatType,
    BlobType,
    NullType,
} from "@elaraai/east";
import { VectorType, MatrixType } from "../types.js";

// Re-export shared types for convenience
export { VectorType, MatrixType } from "../types.js";

// ============================================================================
// Enum Types
// ============================================================================

/**
 * Distribution type for NGBoost.
 */
export const NGBoostDistributionType = VariantType({
    /** Normal (Gaussian) distribution */
    normal: NullType,
    /** Log-normal distribution (for positive targets) */
    lognormal: NullType,
});

// ============================================================================
// Config Types
// ============================================================================

/**
 * Configuration for NGBoost models.
 */
export const NGBoostConfigType = StructType({
    /** Number of boosting rounds (default 500) */
    n_estimators: OptionType(IntegerType),
    /** Learning rate / step size shrinkage (default 0.01) */
    learning_rate: OptionType(FloatType),
    /** Fraction of samples to use in each iteration (default 1.0) */
    minibatch_frac: OptionType(FloatType),
    /** Fraction of features to use in each iteration (default 1.0) */
    col_sample: OptionType(FloatType),
    /** Random seed for reproducibility */
    random_state: OptionType(IntegerType),
    /** Distribution type (default normal) */
    distribution: OptionType(NGBoostDistributionType),
});

/**
 * Configuration for NGBoost predictions with uncertainty.
 */
export const NGBoostPredictConfigType = StructType({
    /** Confidence level for intervals (default 0.95) */
    confidence_level: OptionType(FloatType),
});

// ============================================================================
// Result Types
// ============================================================================

/**
 * Result type for probabilistic predictions.
 */
export const NGBoostPredictResultType = StructType({
    /** Point predictions (mean) */
    predictions: VectorType,
    /** Standard deviation */
    std: OptionType(VectorType),
    /** Lower confidence interval */
    lower: OptionType(VectorType),
    /** Upper confidence interval */
    upper: OptionType(VectorType),
});

// ============================================================================
// Model Blob Types
// ============================================================================

/**
 * Model blob type for serialized NGBoost models.
 */
export const NGBoostModelBlobType = VariantType({
    /** NGBoost regressor model */
    ngboost_regressor: StructType({
        /** Cloudpickle serialized model */
        data: BlobType,
        /** Distribution type used */
        distribution: NGBoostDistributionType,
        /** Number of input features */
        n_features: IntegerType,
    }),
});

// ============================================================================
// Platform Functions
// ============================================================================

/**
 * Train an NGBoost regression model with probabilistic output.
 *
 * @param X - Feature matrix
 * @param y - Target vector
 * @param config - NGBoost configuration
 * @returns Model blob containing trained regressor
 */
export const ngboost_train_regressor = East.platform(
    "ngboost_train_regressor",
    [MatrixType, VectorType, NGBoostConfigType],
    NGBoostModelBlobType
);

/**
 * Make point predictions (mean) with a trained NGBoost regressor.
 *
 * @param model - Trained regressor model blob
 * @param X - Feature matrix
 * @returns Predicted values (mean of predictive distribution)
 */
export const ngboost_predict = East.platform(
    "ngboost_predict",
    [NGBoostModelBlobType, MatrixType],
    VectorType
);

/**
 * Get predictions with full uncertainty from NGBoost regressor.
 *
 * Returns mean, standard deviation, and confidence intervals.
 *
 * @param model - Trained regressor model blob
 * @param X - Feature matrix
 * @param config - Prediction configuration (confidence level)
 * @returns Predictions with uncertainty estimates
 */
export const ngboost_predict_dist = East.platform(
    "ngboost_predict_dist",
    [NGBoostModelBlobType, MatrixType, NGBoostPredictConfigType],
    NGBoostPredictResultType
);

// ============================================================================
// Grouped Export
// ============================================================================

/**
 * Type definitions for NGBoost functions.
 */
export const NGBoostTypes = {
    /** Vector type (array of floats) */
    VectorType,
    /** Matrix type (2D array of floats) */
    MatrixType,
    /** Distribution type for NGBoost */
    NGBoostDistributionType,
    /** NGBoost configuration type */
    NGBoostConfigType,
    /** Prediction configuration type */
    NGBoostPredictConfigType,
    /** Prediction result type */
    NGBoostPredictResultType,
    /** Model blob type for NGBoost models */
    ModelBlobType: NGBoostModelBlobType,
} as const;

/**
 * NGBoost probabilistic gradient boosting.
 *
 * Provides regression with uncertainty quantification using natural gradient boosting.
 *
 * @example
 * ```ts
 * import { East, variant } from "@elaraai/east";
 * import { NGBoost } from "@elaraai/east-py-datascience";
 *
 * const train = East.function([], NGBoost.Types.ModelBlobType, $ => {
 *     const X = $.let([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]);
 *     const y = $.let([1.0, 2.0, 3.0, 4.0]);
 *     const config = $.let({
 *         n_estimators: variant('some', 100n),
 *         learning_rate: variant('some', 0.01),
 *         minibatch_frac: variant('none', null),
 *         col_sample: variant('none', null),
 *         random_state: variant('some', 42n),
 *         distribution: variant('none', null),
 *     });
 *     return $.return(NGBoost.trainRegressor(X, y, config));
 * });
 * ```
 */
export const NGBoost = {
    /** Train NGBoost regressor */
    trainRegressor: ngboost_train_regressor,
    /** Make point predictions with regressor */
    predict: ngboost_predict,
    /** Get predictions with uncertainty */
    predictDist: ngboost_predict_dist,
    /** Type definitions */
    Types: NGBoostTypes,
} as const;
