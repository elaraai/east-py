/**
 * Copyright (c) 2025 Elara AI Pty Ltd
 * Dual-licensed under AGPL-3.0 and commercial license. See LICENSE for details.
 */

/**
 * Iterative coordinate descent optimization for discrete combinatorial problems.
 *
 * Provides element-wise optimization over vectors of discrete integer values.
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
    VectorType,
    IntegerType,
    BooleanType,
    FloatType,
    NullType,
    FunctionType,
} from "@elaraai/east";

// ============================================================================
// Type Definitions
// ============================================================================

/** Parameter vector: Vector<Integer> */
export const ParameterVectorType = VectorType(IntegerType);

/** Objective function: Vector<Integer> -> Float */
export const IterativeObjectiveType = FunctionType([ParameterVectorType], FloatType);

/** Per-element candidate spaces: Array<Vector<Integer>> */
export const ParameterSpacesType = ArrayType(ParameterVectorType);

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
 */
export const IterativeResultType = StructType({
    /** Best parameter values found */
    best_parameters: ParameterVectorType,
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
 * Iterative coordinate descent optimization over integer parameter vectors.
 *
 * Maximizes an objective function over a vector of discrete integer parameters.
 * Each parameter position has its own set of candidate values (vector).
 * The algorithm optimizes one element at a time (coordinate descent),
 * with multiple independent restarts (samples).
 *
 * @example
 * ```ts
 * import { East, VectorType, IntegerType, FloatType, variant } from "@elaraai/east";
 * import { Optimization } from "@elaraai/east-py-datascience";
 *
 * // Objective: maximize sum of parameter values
 * const objective = East.function([VectorType(IntegerType)], FloatType, ($, params) => {
 *     const total = $.let(0.0);
 *     $.for(East.Array.range(0n, params.length()), ($, i) => {
 *         $.assign(total, total.add(params.get(i).toFloat()));
 *     });
 *     return $.return(total);
 * });
 *
 * const spaces = $.let([
 *     new BigInt64Array([0n, 1n, 2n]),
 *     new BigInt64Array([0n, 1n, 2n]),
 *     new BigInt64Array([0n, 1n, 2n]),
 * ]);
 * const config = $.let({
 *     iterations: variant('some', 10n),
 *     samples: variant('some', 3n),
 *     initial: variant('some', variant('random', null)),
 *     order: variant('some', variant('sequential', null)),
 *     random_state: variant('some', 42n),
 * });
 *
 * const result = $.let(Optimization.iterative(objective, spaces, config));
 * // result.best_objective = 6.0
 * ```
 */
export const optimization_iterative = East.platform(
    "optimization_iterative",
    [
        IterativeObjectiveType,   // objective: Vector<Integer> -> Float
        ParameterSpacesType,      // parameter_spaces: Array<Vector<Integer>>
        IterativeConfigType,      // config
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
    /** Parameter vector type */
    ParameterVectorType,
    /** Objective function type */
    ObjectiveType: IterativeObjectiveType,
    /** Parameter spaces type */
    SpacesType: ParameterSpacesType,
    /** Initial value strategy variant */
    InitialStrategyType,
    /** Evaluation order variant */
    EvaluationOrderType,
    /** Configuration type */
    ConfigType: IterativeConfigType,
    /** Result type */
    ResultType: IterativeResultType,
} as const;

/**
 * Iterative coordinate descent optimization for discrete combinatorial problems.
 *
 * Maximizes an objective by independently optimizing each element of a
 * parameter vector over its candidate values. Supports multi-start sampling
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
     * Iterative optimization over integer parameter vectors.
     *
     * `Optimization.iterative(objective, spaces, config)`
     */
    iterative: optimization_iterative,

    /**
     * Type definitions for optimization functions.
     */
    Types: OptimizationTypes,
} as const;
