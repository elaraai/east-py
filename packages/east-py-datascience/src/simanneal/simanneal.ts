/**
 * Copyright (c) 2025 Elara AI Pty Ltd
 * Dual-licensed under AGPL-3.0 and commercial license. See LICENSE for details.
 */

/**
 * Discrete optimization using Simulated Annealing.
 *
 * Provides combinatorial optimization for discrete state spaces using the
 * simanneal library. Ideal for:
 * - Permutation problems (TSP, scheduling)
 * - Subset selection (feature selection, knapsack)
 * - Assignment problems
 * - Any discrete optimization where gradient methods don't apply
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
    FunctionType,
    VectorType,
} from "@elaraai/east";

// ===========================================
// State Types
// ===========================================

/**
 * Discrete state type for simulated annealing.
 *
 * Supports different representations for combinatorial problems:
 * - int_array: Permutations, assignments, sequences of integers
 * - bool_array: Subset selections, binary decisions
 */
export const DiscreteStateType = VariantType({
    /** Integer array state (permutations, assignments) */
    int_array: VectorType(IntegerType),
    /** Boolean array state (subset selection) */
    bool_array: VectorType(BooleanType),
});

// ===========================================
// Function Types
// ===========================================

/**
 * Energy function type: state -> score.
 *
 * Computes the objective value (energy) for a given state.
 * Lower energy is better (minimization).
 */
export const EnergyFunctionType = FunctionType(
    [DiscreteStateType],
    FloatType
);

/**
 * Move function type: state -> neighbor state.
 *
 * Generates a random neighbor of the current state.
 * Should make small, local changes (e.g., swap two elements).
 */
export const MoveFunctionType = FunctionType(
    [DiscreteStateType],
    DiscreteStateType
);

/**
 * Permutation energy function type.
 *
 * Specialized for permutation-based optimization.
 */
export const PermutationEnergyType = FunctionType(
    [VectorType(IntegerType)],
    FloatType
);

/**
 * Subset energy function type.
 *
 * Specialized for subset selection optimization.
 */
export const SubsetEnergyType = FunctionType(
    [VectorType(BooleanType)],
    FloatType
);

// ===========================================
// Configuration Types
// ===========================================

/**
 * Simulated annealing configuration.
 */
export const AnnealConfigType = StructType({
    /** Starting temperature (default: 25000.0) */
    t_max: OptionType(FloatType),
    /** Ending temperature (default: 2.5) */
    t_min: OptionType(FloatType),
    /** Total iterations (default: 50000) */
    steps: OptionType(IntegerType),
    /** Progress report frequency (default: 0 = silent) */
    updates: OptionType(IntegerType),
    /** Minutes for auto-calibration (default: none) */
    auto_schedule: OptionType(FloatType),
    /** Random seed for reproducibility */
    random_state: OptionType(IntegerType),
});

// ===========================================
// Result Types
// ===========================================

/**
 * Simulated annealing result.
 */
export const AnnealResultType = StructType({
    /** Best state found */
    best_state: DiscreteStateType,
    /** Energy of best state */
    best_energy: FloatType,
    /** Actual iterations performed */
    steps_taken: IntegerType,
    /** Whether optimization completed successfully */
    success: BooleanType,
});

// ===========================================
// Platform Functions
// ===========================================

/**
 * Run simulated annealing on a discrete state space.
 *
 * Uses custom energy and move functions for flexible optimization.
 *
 * @param initial_state - Starting state
 * @param energy_fn - Function to compute state energy (lower is better)
 * @param move_fn - Function to generate neighbor states
 * @param config - Annealing schedule configuration
 * @returns Result with best state and energy
 */
export const simanneal_optimize = East.platform(
    "simanneal_optimize",
    [
        DiscreteStateType,
        EnergyFunctionType,
        MoveFunctionType,
        AnnealConfigType,
    ],
    AnnealResultType
);

/**
 * Run simulated annealing on a permutation with swap moves.
 *
 * Convenience function for permutation-based problems (TSP, scheduling).
 * Automatically uses swap moves to generate neighbors.
 *
 * @param initial_perm - Starting permutation
 * @param energy_fn - Function to compute permutation energy
 * @param config - Annealing schedule configuration
 * @returns Result with best permutation and energy
 */
export const simanneal_optimize_permutation = East.platform(
    "simanneal_optimize_permutation",
    [
        VectorType(IntegerType),
        PermutationEnergyType,
        AnnealConfigType,
    ],
    AnnealResultType
);

/**
 * Run simulated annealing on a subset selection with bit-flip moves.
 *
 * Convenience function for subset selection problems (feature selection, knapsack).
 * Automatically uses bit-flip moves to generate neighbors.
 *
 * @param initial_selection - Starting selection (boolean array)
 * @param energy_fn - Function to compute selection energy
 * @param config - Annealing schedule configuration
 * @returns Result with best selection and energy
 */
export const simanneal_optimize_subset = East.platform(
    "simanneal_optimize_subset",
    [
        VectorType(BooleanType),
        SubsetEnergyType,
        AnnealConfigType,
    ],
    AnnealResultType
);

// ===========================================
// Grouped Export
// ===========================================

/**
 * Type definitions for Simulated Annealing.
 */
export const SimAnnealTypes = {
    /** Discrete state type */
    DiscreteStateType,
    /** Energy function type */
    EnergyFunctionType,
    /** Move function type */
    MoveFunctionType,
    /** Permutation energy function type */
    PermutationEnergyType,
    /** Subset energy function type */
    SubsetEnergyType,
    /** Configuration type */
    ConfigType: AnnealConfigType,
    /** Result type */
    ResultType: AnnealResultType,
} as const;

/**
 * Discrete optimization using Simulated Annealing.
 *
 * Provides combinatorial optimization for discrete state spaces.
 * Simulated annealing is a probabilistic technique that can escape
 * local minima by occasionally accepting worse solutions.
 *
 * Ideal for:
 * - Permutation problems (TSP, job scheduling)
 * - Subset selection (feature selection, knapsack)
 * - Assignment problems
 * - Any discrete optimization with many local minima
 *
 * @example
 * ```ts
 * import { East, FloatType, variant } from "@elaraai/east";
 * import { SimAnneal } from "@elaraai/east-py-datascience";
 *
 * // TSP: minimize total route distance
 * const energy = East.function(
 *     [VectorType(IntegerType)],
 *     FloatType,
 *     ($, route) => {
 *         // Calculate total route distance
 *         return $.return(totalDistance);
 *     }
 * );
 *
 * const config = $.let({
 *     t_max: variant("some", 10000.0),
 *     t_min: variant("some", 1.0),
 *     steps: variant("some", 50000n),
 *     updates: variant("none", null),
 *     auto_schedule: variant("none", null),
 *     random_state: variant("some", 42n),
 * });
 *
 * const result = SimAnneal.optimizePermutation(initial, energy, config);
 * ```
 */
export const SimAnneal = {
    /**
     * Run simulated annealing with custom energy and move functions.
     */
    optimize: simanneal_optimize,

    /**
     * Run simulated annealing on a permutation with swap moves.
     */
    optimizePermutation: simanneal_optimize_permutation,

    /**
     * Run simulated annealing on a subset selection with bit-flip moves.
     */
    optimizeSubset: simanneal_optimize_subset,

    /**
     * Type definitions for SimAnneal functions.
     */
    Types: SimAnnealTypes,
} as const;
