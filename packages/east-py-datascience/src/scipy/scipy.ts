/**
 * Copyright (c) 2025 Elara AI Pty Ltd
 * Dual-licensed under AGPL-3.0 and commercial license. See LICENSE for details.
 */

/**
 * SciPy platform functions for East.
 *
 * Provides scientific computing utilities: statistics, optimization,
 * interpolation, and curve fitting.
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
    StringType,
    BlobType,
    NullType,
    FunctionType,
} from "@elaraai/east";
import { VectorType, MatrixType, ScalarObjectiveType } from "../types.js";

// Re-export shared types for convenience
export { VectorType, MatrixType, ScalarObjectiveType } from "../types.js";

// ============================================================================
// Enum Types
// ============================================================================

/**
 * Optimization method for scipy.optimize.minimize.
 */
export const OptimizeMethodType = VariantType({
    /** BFGS algorithm */
    bfgs: NullType,
    /** L-BFGS-B algorithm (default) */
    l_bfgs_b: NullType,
    /** Nelder-Mead simplex */
    nelder_mead: NullType,
    /** Powell's method */
    powell: NullType,
    /** Conjugate gradient */
    cg: NullType,
});

/**
 * Interpolation method for scipy.interpolate.interp1d.
 */
export const InterpolationKindType = VariantType({
    /** Linear interpolation (default) */
    linear: NullType,
    /** Cubic interpolation */
    cubic: NullType,
    /** Quadratic interpolation */
    quadratic: NullType,
});

// ============================================================================
// Config Types
// ============================================================================

/**
 * Configuration for scipy.optimize.minimize.
 */
export const OptimizeConfigType = StructType({
    /** Optimization method */
    method: OptionType(OptimizeMethodType),
    /** Maximum number of iterations */
    max_iter: OptionType(IntegerType),
    /** Tolerance for convergence */
    tol: OptionType(FloatType),
});

/**
 * Configuration for scipy.interpolate.interp1d.
 */
export const InterpolateConfigType = StructType({
    /** Interpolation method */
    kind: OptionType(InterpolationKindType),
});

/**
 * Parameter bounds for curve fitting.
 */
export const ParamBoundsType = StructType({
    /** Lower bounds for each parameter */
    lower: VectorType,
    /** Upper bounds for each parameter */
    upper: VectorType,
});

/**
 * Custom curve function type: (x: Float, params: Vector) -> Float
 */
export const CustomCurveFunctionType = FunctionType([FloatType, VectorType], FloatType);

/**
 * Curve function type for scipy_curve_fit.
 *
 * Includes built-in standard mathematical functions and a custom option
 * for user-defined functions.
 */
export const CurveFunctionType = VariantType({
    /** y = a * exp(-b * x), 2 params: [a, b] */
    exponential_decay: NullType,
    /** y = a + b * exp(-c * x), 3 params: [a, b, c] */
    exponential_with_offset: NullType,
    /** y = a * exp(b * x), 2 params: [a, b] */
    exponential_growth: NullType,
    /** y = L / (1 + exp(-k * (x - x0))), 3 params: [L, k, x0] */
    logistic: NullType,
    /** y = a * exp(-b * exp(-c * x)), 3 params: [a, b, c] */
    gompertz: NullType,
    /** y = a * x^b, 2 params: [a, b] */
    power_law: NullType,
    /** y = a + b * x, 2 params: [a, b] */
    linear: NullType,
    /** y = a + b*x + c*x^2, 3 params: [a, b, c] */
    quadratic: NullType,
    /** y = a + b*x + c*x^2 + d*x^3, 4 params: [a, b, c, d] */
    cubic: NullType,
    /** Custom function provided by user */
    custom: StructType({
        /** The curve function */
        fn: CustomCurveFunctionType,
        /** Number of parameters to optimize */
        n_params: IntegerType,
        /** Optional parameter bounds */
        param_bounds: OptionType(ParamBoundsType),
    }),
});

/**
 * Configuration for curve fitting.
 */
export const CurveFitConfigType = StructType({
    /** Maximum number of function evaluations */
    max_iter: OptionType(IntegerType),
    /** Initial guess for parameters */
    initial_guess: OptionType(VectorType),
});

/**
 * Configuration for quadratic optimization: f(x) = 0.5 * x'Ax + b'x + c
 */
export const QuadraticConfigType = StructType({
    /** Quadratic term (symmetric positive definite) */
    A: MatrixType,
    /** Linear term */
    b: VectorType,
    /** Constant term */
    c: FloatType,
});

// ============================================================================
// Result Types
// ============================================================================

/**
 * Descriptive statistics result.
 */
export const StatsDescribeResultType = StructType({
    /** Number of observations */
    count: IntegerType,
    /** Mean value */
    mean: FloatType,
    /** Variance */
    variance: FloatType,
    /** Skewness */
    skewness: FloatType,
    /** Kurtosis */
    kurtosis: FloatType,
    /** Minimum value */
    min: FloatType,
    /** Maximum value */
    max: FloatType,
});

/**
 * Correlation result (Pearson or Spearman).
 */
export const CorrelationResultType = StructType({
    /** Correlation coefficient */
    correlation: FloatType,
    /** P-value for hypothesis test */
    pvalue: FloatType,
});

/**
 * Curve fitting result.
 */
export const CurveFitResultType = StructType({
    /** Fitted parameters */
    params: VectorType,
    /** Whether fit converged */
    success: BooleanType,
    /** Coefficient of determination (R²) */
    r_squared: FloatType,
});

/**
 * Optimization result.
 */
export const OptimizeResultType = StructType({
    /** Optimal parameters */
    x: VectorType,
    /** Function value at optimum */
    fun: FloatType,
    /** Whether optimization succeeded */
    success: BooleanType,
    /** Number of iterations */
    nit: IntegerType,
});

/**
 * Model blob type for scipy interpolators.
 */
export const ScipyModelBlobType = VariantType({
    /** 1D interpolator (cloudpickle serialized) */
    scipy_interp_1d: StructType({
        /** Serialized interpolator */
        data: BlobType,
        /** Interpolation method used */
        kind: InterpolationKindType,
    }),
});

// ============================================================================
// Dual Annealing Types
// ============================================================================

/**
 * Bounds for dual annealing optimization (required).
 */
export const DualAnnealBoundsType = StructType({
    /** Lower bounds for each variable */
    lower: VectorType,
    /** Upper bounds for each variable */
    upper: VectorType,
});

/**
 * Configuration for scipy.optimize.dual_annealing.
 *
 * Combines generalized simulated annealing with local search.
 * Much faster than pure Python simanneal for continuous optimization.
 */
export const DualAnnealConfigType = StructType({
    /** Maximum function evaluations (default: 1000) */
    maxfun: OptionType(IntegerType),
    /** Maximum iterations (default: 1000) */
    maxiter: OptionType(IntegerType),
    /** Initial temperature (default: 5230) */
    initial_temp: OptionType(FloatType),
    /** Temperature restart threshold (default: 2e-5) */
    restart_temp_ratio: OptionType(FloatType),
    /** Visiting distribution parameter (default: 2.62) */
    visit: OptionType(FloatType),
    /** Acceptance distribution parameter (default: -5.0) */
    accept: OptionType(FloatType),
    /** Random seed for reproducibility */
    seed: OptionType(IntegerType),
    /** Disable local search for speed (default: false) */
    no_local_search: OptionType(BooleanType),
});

/**
 * Result from dual annealing optimization.
 */
export const DualAnnealResultType = StructType({
    /** Best solution found */
    x: VectorType,
    /** Best objective value */
    fun: FloatType,
    /** Number of function evaluations */
    nfev: IntegerType,
    /** Number of iterations */
    nit: IntegerType,
    /** Whether optimization succeeded */
    success: BooleanType,
    /** Status message */
    message: StringType,
});

// ============================================================================
// Platform Functions
// ============================================================================

/**
 * Fit a parametric curve to data using nonlinear least squares.
 */
export const scipy_curve_fit = East.platform(
    "scipy_curve_fit",
    [CurveFunctionType, VectorType, VectorType, CurveFitConfigType],
    CurveFitResultType
);

/**
 * Compute descriptive statistics for data.
 */
export const scipy_stats_describe = East.platform(
    "scipy_stats_describe",
    [VectorType],
    StatsDescribeResultType
);

/**
 * Compute Pearson correlation coefficient.
 */
export const scipy_stats_pearsonr = East.platform(
    "scipy_stats_pearsonr",
    [VectorType, VectorType],
    CorrelationResultType
);

/**
 * Compute Spearman rank correlation.
 */
export const scipy_stats_spearmanr = East.platform(
    "scipy_stats_spearmanr",
    [VectorType, VectorType],
    CorrelationResultType
);

/**
 * Fit 1D interpolator to data.
 */
export const scipy_interpolate_1d_fit = East.platform(
    "scipy_interpolate_1d_fit",
    [VectorType, VectorType, InterpolateConfigType],
    ScipyModelBlobType
);

/**
 * Evaluate 1D interpolator at given points.
 */
export const scipy_interpolate_1d_predict = East.platform(
    "scipy_interpolate_1d_predict",
    [ScipyModelBlobType, VectorType],
    VectorType
);

/**
 * Minimize a scalar function using scipy.optimize.minimize.
 */
export const scipy_optimize_minimize = East.platform(
    "scipy_optimize_minimize",
    [ScalarObjectiveType, VectorType, OptimizeConfigType],
    OptimizeResultType
);

/**
 * Minimize a quadratic function with analytical gradient.
 */
export const scipy_optimize_minimize_quadratic = East.platform(
    "scipy_optimize_minimize_quadratic",
    [VectorType, QuadraticConfigType, OptimizeConfigType],
    OptimizeResultType
);

/**
 * Global optimization using dual annealing.
 *
 * Combines generalized simulated annealing with local search.
 * Much faster than simanneal for continuous optimization problems.
 * Effective for non-convex problems with many local minima.
 *
 * @param objective_fn - Function to minimize: Vector -> Float
 * @param x0 - Optional initial guess (if none, starts from bounds center)
 * @param bounds - Required bounds for all variables
 * @param config - Algorithm configuration
 * @returns Optimization result with best solution
 */
export const scipy_optimize_dual_annealing = East.platform(
    "scipy_optimize_dual_annealing",
    [
        ScalarObjectiveType,
        OptionType(VectorType),
        DualAnnealBoundsType,
        DualAnnealConfigType,
    ],
    DualAnnealResultType
);

// ============================================================================
// Grouped Export
// ============================================================================

/**
 * Type definitions for scipy functions.
 */
export const ScipyTypes = {
    VectorType,
    MatrixType,
    ScalarObjectiveType,
    OptimizeMethodType,
    InterpolationKindType,
    OptimizeConfigType,
    InterpolateConfigType,
    ParamBoundsType,
    CustomCurveFunctionType,
    CurveFunctionType,
    CurveFitConfigType,
    QuadraticConfigType,
    StatsDescribeResultType,
    CorrelationResultType,
    CurveFitResultType,
    OptimizeResultType,
    ModelBlobType: ScipyModelBlobType,
    DualAnnealBoundsType,
    DualAnnealConfigType,
    DualAnnealResultType,
} as const;

/**
 * SciPy scientific computing utilities.
 *
 * Provides statistics, optimization, interpolation, and curve fitting.
 */
export const Scipy = {
    /** Fit parametric curve to data */
    curveFit: scipy_curve_fit,
    /** Compute descriptive statistics */
    statsDescribe: scipy_stats_describe,
    /** Compute Pearson correlation */
    statsPearsonr: scipy_stats_pearsonr,
    /** Compute Spearman correlation */
    statsSpearmanr: scipy_stats_spearmanr,
    /** Fit 1D interpolator */
    interpolate1dFit: scipy_interpolate_1d_fit,
    /** Evaluate 1D interpolator */
    interpolate1dPredict: scipy_interpolate_1d_predict,
    /** Minimize scalar function */
    optimizeMinimize: scipy_optimize_minimize,
    /** Minimize quadratic function */
    optimizeMinimizeQuadratic: scipy_optimize_minimize_quadratic,
    /** Global optimization using dual annealing */
    optimizeDualAnnealing: scipy_optimize_dual_annealing,
    /** Type definitions */
    Types: ScipyTypes,
} as const;
