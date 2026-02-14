/**
 * Copyright (c) 2025 Elara AI Pty Ltd
 * Dual-licensed under AGPL-3.0 and commercial license. See LICENSE for details.
 */

/**
 * Iterative coordinate descent optimization for discrete combinatorial problems.
 *
 * Provides element-wise optimization over arrays of discrete values.
 * Each element is independently optimized by trying all candidate values
 * while holding other elements fixed. Multi-start sampling improves
 * exploration of the search space.
 *
 * Ported from the Julia IterativeDecisionAlgorithm (ArrayParameterSpace branch).
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
    BooleanType,
    FloatType,
    NullType,
    FunctionType,
} from "@elaraai/east";

// ============================================================================
// Type Definitions
// ============================================================================

/**
 * Initial value strategy for parameters.
 *
 * - `first`: Use the first candidate value from each space
 * - `random`: Randomly select from each space
 */
export const InitialStrategyType = VariantType({
    first: NullType,
    random: NullType,
});

/**
 * Evaluation order for candidate values within each element's space.
 *
 * - `sequential`: Try candidates in the order they appear
 * - `random`: Shuffle candidates before trying
 */
export const EvaluationOrderType = VariantType({
    sequential: NullType,
    random: NullType,
});

/**
 * Configuration for iterative optimization.
 *
 * All fields are optional with sensible defaults.
 */
export const IterativeConfigType = StructType({
    /** Maximum coordinate descent iterations per sample (default: 100) */
    iterations: OptionType(IntegerType),
    /** Number of independent restarts (default: 1) */
    samples: OptionType(IntegerType),
    /** How to initialize parameter values (default: first) */
    initial: OptionType(InitialStrategyType),
    /** Order to evaluate candidates (default: sequential) */
    order: OptionType(EvaluationOrderType),
    /** Random seed for reproducibility */
    random_state: OptionType(IntegerType),
});

/**
 * Result of iterative optimization.
 *
 * Generic over V (the parameter value type).
 */
export const IterativeResultType = StructType({
    /** Best parameter values found */
    best_parameters: ArrayType("V"),
    /** Objective value at best parameters */
    best_objective: FloatType,
    /** Total coordinate descent iterations across all samples */
    iterations: IntegerType,
    /** Total number of objective evaluations */
    evaluations: IntegerType,
    /** Whether optimization succeeded */
    success: BooleanType,
});

// ============================================================================
// Platform Functions
// ============================================================================

/**
 * Iterative coordinate descent optimization (generic over value type V).
 *
 * Maximizes an objective function over an array of discrete parameters.
 * Each parameter position has its own set of candidate values.
 * The algorithm optimizes one element at a time (coordinate descent),
 * with multiple independent restarts (samples).
 *
 * @example
 * ```ts
 * import { East, ArrayType, IntegerType, FloatType, variant } from "@elaraai/east";
 * import { Optimization } from "@elaraai/east-py-datascience";
 *
 * // Objective: maximize sum of parameter values
 * const objective = East.function([ArrayType(IntegerType)], FloatType, ($, params) => {
 *     const total = $.let(East.value(0.0));
 *     $.for(East.value(0n), params.length(), ($, i) => {
 *         $(total.set(total.get().add(params.get(i).toFloat())));
 *     });
 *     return $.return(total.get());
 * });
 *
 * const spaces = $.let([[0n, 1n, 2n], [0n, 1n, 2n], [0n, 1n, 2n]]);
 * const config = $.let({
 *     iterations: variant('some', 10n),
 *     samples: variant('some', 3n),
 *     initial: variant('some', variant('random', null)),
 *     order: variant('some', variant('sequential', null)),
 *     random_state: variant('some', 42n),
 * });
 *
 * const result = $.let(Optimization.iterative([IntegerType], objective, spaces, config));
 * // result.best_parameters = [2n, 2n, 2n], result.best_objective = 6.0
 * ```
 */
export const optimization_iterative = East.genericPlatform(
    "optimization_iterative",
    ["V"],
    [
        FunctionType([ArrayType("V")], FloatType),  // objective: Array<V> -> Float
        ArrayType(ArrayType("V")),                   // parameter_spaces: per-element candidates
        IterativeConfigType,                         // config
    ],
    IterativeResultType
);

// ============================================================================
// Grouped Export
// ============================================================================

/**
 * Type definitions for iterative optimization.
 */
export const OptimizationTypes = {
    /** Initial value strategy variant */
    InitialStrategyType,
    /** Evaluation order variant */
    EvaluationOrderType,
    /** Configuration type */
    ConfigType: IterativeConfigType,
    /** Result type (with "V" placeholder for value type) */
    ResultType: IterativeResultType,
} as const;

/**
 * Iterative coordinate descent optimization for discrete combinatorial problems.
 *
 * Maximizes an objective by independently optimizing each element of a
 * parameter array over its candidate values. Supports multi-start sampling
 * for better exploration.
 *
 * Use cases:
 * - Task-worker assignment
 * - Scheduling and rostering
 * - Combinatorial selection problems
 * - Any discrete optimization with per-element candidate sets
 */
export const Optimization = {
    /**
     * Iterative optimization (generic over value type V).
     *
     * Call with type parameter array first, then arguments:
     * `Optimization.iterative([IntegerType], objective, spaces, config)`
     */
    iterative: optimization_iterative,

    /**
     * Type definitions for optimization functions.
     */
    Types: OptimizationTypes,
} as const;
