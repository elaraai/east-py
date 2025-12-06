"""SciPy platform functions for East.

Provides scientific computing utilities: statistics, optimization,
interpolation, and curve fitting.
"""

from typing import Callable

import numpy as np

from east.runtime.platform import PlatformFunction
from east.types.values import EastArray, EastBlob, EastStruct, EastVariant

from east_py_datascience.types import (
    VectorType,
    ScalarObjectiveType,
    OptimizeConfigType,
    InterpolateConfigType,
    CurveFunctionType,
    CurveFitConfigType,
    QuadraticConfigType,
    StatsDescribeResultType,
    CorrelationResultType,
    CurveFitResultType,
    OptimizeResultType,
    ModelBlobType,
    FloatType,
    _get_option,
    _get_enum_tag,
    east_vector_to_numpy,
    east_matrix_to_numpy,
    numpy_to_east_vector,
)


# ============================================================================
# Native Serialization Helpers
# ============================================================================


def _serialize_native(model) -> EastBlob:
    """Serialize a model using cloudpickle."""
    import cloudpickle

    return EastBlob(cloudpickle.dumps(model))


def _deserialize_native(data: EastBlob):
    """Deserialize a model using cloudpickle."""
    import cloudpickle

    return cloudpickle.loads(bytes(data))


def _make_enum(tag: str) -> EastVariant:
    """Create an enum-like variant (tag with null value)."""
    return EastVariant(tag, None)


# ============================================================================
# Platform Function Implementations
# ============================================================================


def scipy_curve_fit_impl(
    curve_type: EastVariant,
    x: EastArray,
    y: EastArray,
    config: EastStruct,
) -> EastStruct:
    """Fit curve to data using scipy.optimize.curve_fit."""
    from scipy.optimize import curve_fit

    x_np = east_vector_to_numpy(x)
    y_np = east_vector_to_numpy(y)
    max_iter = _get_option(config.get("max_iter"), 5000)

    tag = curve_type.type

    # Built-in curves with smart defaults
    if tag == "exponential_decay":

        def model_func(x, a, b):
            return a * np.exp(-b * x)

        p0 = [float(y_np[0]), 0.1]
        bounds = ([0, 0], [np.inf, np.inf])

    elif tag == "exponential_with_offset":

        def model_func(x, a, b, c):
            return a + b * np.exp(-c * x)

        p0 = [float(y_np[-1]), float(y_np[0] - y_np[-1]), 0.1]
        bounds = ([-np.inf, -np.inf, 0], [np.inf, np.inf, np.inf])

    elif tag == "exponential_growth":

        def model_func(x, a, b):
            return a * np.exp(b * x)

        p0 = [float(y_np[0]), 0.1]
        bounds = ([0, 0], [np.inf, np.inf])

    elif tag == "logistic":

        def model_func(x, L, k, x0):
            return L / (1 + np.exp(-k * (x - x0)))

        p0 = [float(y_np.max()), 1.0, float(x_np.mean())]
        bounds = ([0, 0, -np.inf], [np.inf, np.inf, np.inf])

    elif tag == "gompertz":

        def model_func(x, a, b, c):
            return a * np.exp(-b * np.exp(-c * x))

        p0 = [float(y_np.max()), 1.0, 0.1]
        bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])

    elif tag == "power_law":

        def model_func(x, a, b):
            return a * np.power(np.maximum(x, 1e-10), b)

        p0 = [1.0, 1.0]
        bounds = ([-np.inf, -np.inf], [np.inf, np.inf])

    elif tag == "linear":

        def model_func(x, a, b):
            return a + b * x

        slope = (y_np[-1] - y_np[0]) / (x_np[-1] - x_np[0] + 1e-10)
        p0 = [float(y_np[0]), float(slope)]
        bounds = ([-np.inf, -np.inf], [np.inf, np.inf])

    elif tag == "quadratic":

        def model_func(x, a, b, c):
            return a + b * x + c * x**2

        p0 = [float(y_np[0]), 0.0, 0.0]
        bounds = ([-np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf])

    elif tag == "cubic":

        def model_func(x, a, b, c, d):
            return a + b * x + c * x**2 + d * x**3

        p0 = [float(y_np[0]), 0.0, 0.0, 0.0]
        bounds = ([-np.inf] * 4, [np.inf] * 4)

    elif tag == "custom":
        custom_config = curve_type.value
        east_fn = custom_config["fn"]  # Compiled East function
        n_params = int(custom_config["n_params"])

        # Wrap East function for scipy
        def scalar_model(x_val, *params):
            params_arr = EastArray(FloatType, [float(p) for p in params])
            return east_fn(float(x_val), params_arr)

        # Vectorize for array inputs
        model_func = np.vectorize(scalar_model, excluded=list(range(1, n_params + 1)))

        # Initial guess
        initial_guess = _get_option(config.get("initial_guess"), None)
        p0 = (
            list(east_vector_to_numpy(initial_guess))
            if initial_guess
            else [1.0] * n_params
        )

        # Bounds
        bounds_opt = _get_option(custom_config.get("param_bounds"), None)
        if bounds_opt:
            bounds = (
                list(east_vector_to_numpy(bounds_opt["lower"])),
                list(east_vector_to_numpy(bounds_opt["upper"])),
            )
        else:
            bounds = ([-np.inf] * n_params, [np.inf] * n_params)

    else:
        raise RuntimeError(f"scipy_curve_fit: Unknown curve type: {tag}")

    # Override initial guess if provided in config
    config_guess = _get_option(config.get("initial_guess"), None)
    if config_guess is not None and tag != "custom":
        p0 = list(east_vector_to_numpy(config_guess))

    try:
        params, _ = curve_fit(
            model_func, x_np, y_np, p0=p0, bounds=bounds, maxfev=int(max_iter)
        )

        # Compute R²
        y_pred = model_func(x_np, *params)
        ss_res = np.sum((y_np - y_pred) ** 2)
        ss_tot = np.sum((y_np - y_np.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0

        return EastStruct(
            {
                "params": EastArray(FloatType, [float(p) for p in params]),
                "success": True,
                "r_squared": float(np.clip(r2, -10, 1)),
            }
        )

    except Exception:
        return EastStruct(
            {
                "params": EastArray(FloatType, []),
                "success": False,
                "r_squared": 0.0,
            }
        )


def scipy_stats_describe_impl(data: EastArray) -> EastStruct:
    """Compute descriptive statistics for data."""
    from scipy import stats

    data_np = east_vector_to_numpy(data)
    result = stats.describe(data_np)

    return EastStruct(
        {
            "count": int(result.nobs),
            "mean": float(result.mean),
            "variance": float(result.variance),
            "skewness": float(result.skewness),
            "kurtosis": float(result.kurtosis),
            "min": float(result.minmax[0]),
            "max": float(result.minmax[1]),
        }
    )


def scipy_stats_pearsonr_impl(x: EastArray, y: EastArray) -> EastStruct:
    """Compute Pearson correlation coefficient."""
    from scipy import stats

    x_np = east_vector_to_numpy(x)
    y_np = east_vector_to_numpy(y)

    r, p = stats.pearsonr(x_np, y_np)

    return EastStruct(
        {
            "correlation": float(r),
            "pvalue": float(p),
        }
    )


def scipy_stats_spearmanr_impl(x: EastArray, y: EastArray) -> EastStruct:
    """Compute Spearman rank correlation."""
    from scipy import stats

    x_np = east_vector_to_numpy(x)
    y_np = east_vector_to_numpy(y)

    r, p = stats.spearmanr(x_np, y_np)

    return EastStruct(
        {
            "correlation": float(r),
            "pvalue": float(p),
        }
    )


def scipy_interpolate_1d_fit_impl(
    x: EastArray,
    y: EastArray,
    config: EastStruct,
) -> EastVariant:
    """Fit 1D interpolator."""
    from scipy import interpolate

    x_np = east_vector_to_numpy(x)
    y_np = east_vector_to_numpy(y)

    kind_variant = _get_option(config.get("kind"), None)
    kind = _get_enum_tag(kind_variant) if kind_variant else "linear"

    interp = interpolate.interp1d(x_np, y_np, kind=kind, fill_value="extrapolate")

    return EastVariant(
        "scipy_interp_1d",
        EastStruct(
            {
                "data": _serialize_native(interp),
                "kind": _make_enum(kind),
            }
        ),
    )


def scipy_interpolate_1d_predict_impl(
    model_blob: EastVariant,
    x: EastArray,
) -> EastArray:
    """Evaluate 1D interpolator at given points."""
    if model_blob.type != "scipy_interp_1d":
        raise RuntimeError(
            f"scipy_interpolate_1d_predict: Expected scipy_interp_1d, got {model_blob.type}"
        )

    interp = _deserialize_native(model_blob.value["data"])
    x_np = east_vector_to_numpy(x)

    y_np = interp(x_np)

    return numpy_to_east_vector(y_np)


def scipy_optimize_minimize_impl(
    objective_fn: Callable[[EastArray], float],
    x0: EastArray,
    config: EastStruct,
) -> EastStruct:
    """Minimize a scalar function using scipy.optimize.minimize."""
    from scipy import optimize

    x0_np = east_vector_to_numpy(x0)

    # Wrap East function for scipy
    def wrapped_objective(x):
        x_east = EastArray(FloatType, [float(v) for v in x])
        return objective_fn(x_east)

    method_variant = _get_option(config.get("method"), None)
    method = _get_enum_tag(method_variant) if method_variant else "l_bfgs_b"
    method_map = {
        "bfgs": "BFGS",
        "l_bfgs_b": "L-BFGS-B",
        "nelder_mead": "Nelder-Mead",
        "powell": "Powell",
        "cg": "CG",
    }

    result = optimize.minimize(
        wrapped_objective,
        x0_np,
        method=method_map.get(method, "L-BFGS-B"),
        options={
            "maxiter": _get_option(config.get("max_iter"), 1000),
        },
        tol=_get_option(config.get("tol"), 1e-6),
    )

    return EastStruct(
        {
            "x": numpy_to_east_vector(result.x),
            "fun": float(result.fun),
            "success": bool(result.success),
            "nit": int(result.nit),
        }
    )


def scipy_optimize_minimize_quadratic_impl(
    x0: EastArray,
    quadratic: EastStruct,
    config: EastStruct,
) -> EastStruct:
    """Minimize quadratic function: f(x) = 0.5 * x'Ax + b'x + c"""
    from scipy import optimize

    x0_np = east_vector_to_numpy(x0)
    A_np = east_matrix_to_numpy(quadratic["A"])
    b_np = east_vector_to_numpy(quadratic["b"])
    c = float(quadratic["c"])

    def objective(x):
        return 0.5 * x @ A_np @ x + b_np @ x + c

    def gradient(x):
        return A_np @ x + b_np

    method_variant = _get_option(config.get("method"), None)
    method = _get_enum_tag(method_variant) if method_variant else "l_bfgs_b"
    method_map = {
        "bfgs": "BFGS",
        "l_bfgs_b": "L-BFGS-B",
        "nelder_mead": "Nelder-Mead",
        "powell": "Powell",
        "cg": "CG",
    }

    result = optimize.minimize(
        objective,
        x0_np,
        method=method_map.get(method, "L-BFGS-B"),
        jac=gradient,
        options={
            "maxiter": _get_option(config.get("max_iter"), 1000),
        },
        tol=_get_option(config.get("tol"), 1e-6),
    )

    return EastStruct(
        {
            "x": numpy_to_east_vector(result.x),
            "fun": float(result.fun),
            "success": bool(result.success),
            "nit": int(result.nit),
        }
    )


# ============================================================================
# Platform Function Registration
# ============================================================================

scipy_impl = [
    PlatformFunction(
        name="scipy_curve_fit",
        inputs=[CurveFunctionType, VectorType, VectorType, CurveFitConfigType],
        output=CurveFitResultType,
        type="sync",
        fn=scipy_curve_fit_impl,
    ),
    PlatformFunction(
        name="scipy_stats_describe",
        inputs=[VectorType],
        output=StatsDescribeResultType,
        type="sync",
        fn=scipy_stats_describe_impl,
    ),
    PlatformFunction(
        name="scipy_stats_pearsonr",
        inputs=[VectorType, VectorType],
        output=CorrelationResultType,
        type="sync",
        fn=scipy_stats_pearsonr_impl,
    ),
    PlatformFunction(
        name="scipy_stats_spearmanr",
        inputs=[VectorType, VectorType],
        output=CorrelationResultType,
        type="sync",
        fn=scipy_stats_spearmanr_impl,
    ),
    PlatformFunction(
        name="scipy_interpolate_1d_fit",
        inputs=[VectorType, VectorType, InterpolateConfigType],
        output=ModelBlobType,
        type="sync",
        fn=scipy_interpolate_1d_fit_impl,
    ),
    PlatformFunction(
        name="scipy_interpolate_1d_predict",
        inputs=[ModelBlobType, VectorType],
        output=VectorType,
        type="sync",
        fn=scipy_interpolate_1d_predict_impl,
    ),
    PlatformFunction(
        name="scipy_optimize_minimize",
        inputs=[ScalarObjectiveType, VectorType, OptimizeConfigType],
        output=OptimizeResultType,
        type="sync",
        fn=scipy_optimize_minimize_impl,
    ),
    PlatformFunction(
        name="scipy_optimize_minimize_quadratic",
        inputs=[VectorType, QuadraticConfigType, OptimizeConfigType],
        output=OptimizeResultType,
        type="sync",
        fn=scipy_optimize_minimize_quadratic_impl,
    ),
]

__all__ = [
    "scipy_impl",
]
