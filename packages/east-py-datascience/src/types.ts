/**
 * Copyright (c) 2025 Elara AI Pty Ltd
 * Dual-licensed under AGPL-3.0 and commercial license. See LICENSE for details.
 */

/**
 * Shared type definitions for East Data Science.
 *
 * Provides common East type definitions used across data science modules
 * including vectors, matrices, and scalar function types.
 *
 * @packageDocumentation
 */

import {
    ArrayType,
    FloatType,
    IntegerType,
    FunctionType,
} from "@elaraai/east";

// Re-export commonly used types for convenience
export {
    ArrayType,
    StructType,
    VariantType,
    OptionType,
    FloatType,
    IntegerType,
    BooleanType,
    StringType,
    FunctionType,
} from "@elaraai/east";

/**
 * Vector type (1D array of floats).
 *
 * Used for optimization variables, bounds, feature vectors, and predictions.
 *
 * @example
 * ```ts
 * const x = $.let([1.0, 2.0, 3.0]); // VectorType
 * ```
 */
export const VectorType = ArrayType(FloatType);

/**
 * Matrix type (2D array of floats).
 *
 * Used for datasets, Pareto fronts, and multi-dimensional results.
 * Each element is a row (VectorType).
 *
 * @example
 * ```ts
 * const X = $.let([
 *     [1.0, 2.0],
 *     [3.0, 4.0],
 * ]); // MatrixType
 * ```
 */
export const MatrixType = ArrayType(VectorType);

/**
 * Scalar objective function type: Vector -> Float.
 *
 * Used for optimization objectives, constraints, and loss functions
 * that take a vector of decision variables and return a scalar value.
 *
 * @example
 * ```ts
 * const sumSquares = East.function([VectorType], FloatType, ($, x) => {
 *     return x.reduce((acc, xi) => acc.add(xi.multiply(xi)), East.value(0.0));
 * });
 * ```
 */
export const ScalarObjectiveType = FunctionType([VectorType], FloatType);

/**
 * Vector objective function type: Vector -> Vector.
 *
 * Used for multi-output predictions and transformations.
 *
 * @example
 * ```ts
 * const predict = East.function([VectorType], VectorType, ($, x) => {
 *     // Return multiple predictions
 *     return $.return([...]);
 * });
 * ```
 */
export const VectorObjectiveType = FunctionType([VectorType], VectorType);

/**
 * Label vector type (1D array of integers).
 *
 * Used for classification labels and cluster assignments.
 *
 * @example
 * ```ts
 * const labels = $.let([0n, 1n, 0n, 2n]); // LabelVectorType
 * ```
 */
export const LabelVectorType = ArrayType(IntegerType);
