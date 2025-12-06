/**
 * Copyright (c) 2025 Elara AI Pty Ltd
 * Dual-licensed under AGPL-3.0 and commercial license. See LICENSE for details.
 */

/**
 * Gaussian Process platform functions for East.
 *
 * Provides Gaussian Process regression using scikit-learn.
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
    BooleanType,
    BlobType,
    StringType,
} from "@elaraai/east";
import { VectorType, MatrixType } from "../types.js";

// Re-export shared types for convenience
export { VectorType, MatrixType } from "../types.js";

// ============================================================================
// Enum Types
// ============================================================================

/**
 * Kernel type for Gaussian Process.
 */
export const GPKernelType = VariantType({
    /** Radial Basis Function (squared exponential) */
    rbf: StructType({}),
    /** Matern with nu=1/2 (exponential) */
    matern_1_2: StructType({}),
    /** Matern with nu=3/2 */
    matern_3_2: StructType({}),
    /** Matern with nu=5/2 */
    matern_5_2: StructType({}),
    /** Rational Quadratic */
    rational_quadratic: StructType({}),
    /** Dot Product (linear) */
    dot_product: StructType({}),
});

// ============================================================================
// Config Types
// ============================================================================

/**
 * Configuration for Gaussian Process Regressor.
 */
export const GPConfigType = StructType({
    /** Kernel type (default rbf) */
    kernel: OptionType(GPKernelType),
    /** Noise level added to diagonal (default 1e-10) */
    alpha: OptionType(FloatType),
    /** Number of restarts for optimizer (default 0) */
    n_restarts_optimizer: OptionType(IntegerType),
    /** Whether to normalize target values (default false) */
    normalize_y: OptionType(BooleanType),
    /** Random seed for reproducibility */
    random_state: OptionType(IntegerType),
});

// ============================================================================
// Result Types
// ============================================================================

/**
 * Result type for GP prediction with uncertainty.
 */
export const GPPredictResultType = StructType({
    /** Predicted mean values */
    mean: VectorType,
    /** Predicted standard deviation (uncertainty) */
    std: VectorType,
});

// ============================================================================
// Model Blob Types
// ============================================================================

/**
 * Model blob type for serialized GP models.
 */
export const GPModelBlobType = VariantType({
    /** Gaussian Process Regressor */
    gp_regressor: StructType({
        /** Cloudpickle serialized model */
        data: BlobType,
        /** Number of input features */
        n_features: IntegerType,
        /** Kernel type name for reference */
        kernel_type: StringType,
    }),
});

// ============================================================================
// Platform Functions
// ============================================================================

/**
 * Train a Gaussian Process Regressor.
 *
 * @param X - Feature matrix
 * @param y - Target vector
 * @param config - GP configuration
 * @returns Trained GP model blob
 */
export const gp_train = East.platform(
    "gp_train",
    [MatrixType, VectorType, GPConfigType],
    GPModelBlobType
);

/**
 * Make predictions with a trained Gaussian Process.
 *
 * Returns point predictions (mean only).
 *
 * @param model - Trained GP model blob
 * @param X - Feature matrix
 * @returns Predicted values
 */
export const gp_predict = East.platform(
    "gp_predict",
    [GPModelBlobType, MatrixType],
    VectorType
);

/**
 * Make predictions with uncertainty estimates.
 *
 * Returns both mean and standard deviation.
 *
 * @param model - Trained GP model blob
 * @param X - Feature matrix
 * @returns Prediction result with mean and std
 */
export const gp_predict_std = East.platform(
    "gp_predict_std",
    [GPModelBlobType, MatrixType],
    GPPredictResultType
);

// ============================================================================
// Grouped Export
// ============================================================================

/**
 * Type definitions for GP functions.
 */
export const GPTypes = {
    /** Vector type (array of floats) */
    VectorType,
    /** Matrix type (2D array of floats) */
    MatrixType,
    /** Kernel type */
    GPKernelType,
    /** Configuration type */
    GPConfigType,
    /** Prediction result type with uncertainty */
    GPPredictResultType,
    /** Model blob type for GP models */
    ModelBlobType: GPModelBlobType,
} as const;

/**
 * Gaussian Process regression.
 *
 * Provides probabilistic regression with uncertainty quantification.
 *
 * @example
 * ```ts
 * import { East, variant } from "@elaraai/east";
 * import { GP } from "@elaraai/east-py-datascience";
 *
 * const train = East.function([], GP.Types.ModelBlobType, $ => {
 *     const X = $.let([[1.0], [2.0], [3.0], [4.0]]);
 *     const y = $.let([1.0, 4.0, 9.0, 16.0]);
 *     const config = $.let({
 *         kernel: variant('some', variant('rbf', {})),
 *         alpha: variant('some', 1e-10),
 *         n_restarts_optimizer: variant('some', 5n),
 *         normalize_y: variant('some', true),
 *         random_state: variant('some', 42n),
 *     });
 *     return $.return(GP.train(X, y, config));
 * });
 * ```
 */
export const GP = {
    /** Train GP regressor */
    train: gp_train,
    /** Make predictions (mean only) */
    predict: gp_predict,
    /** Make predictions with uncertainty */
    predictStd: gp_predict_std,
    /** Type definitions */
    Types: GPTypes,
} as const;
