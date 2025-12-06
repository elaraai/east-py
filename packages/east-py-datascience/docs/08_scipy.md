# Module 8: SciPy (`scipy_impl.py`)

## Purpose

Scientific computing utilities: statistics, optimization, interpolation, curve fitting.

## Config Types

```python
OptimizeConfigType = StructType([
    ("method", OptionType(OptimizeMethodType)),  # default l_bfgs_b
    ("max_iter", OptionType(IntegerType)),       # default 1000
    ("tol", OptionType(FloatType)),              # default 1e-6
])

InterpolateConfigType = StructType([
    ("kind", OptionType(InterpolationKindType)),  # default linear
])

# Bounds for curve fitting parameters
ParamBoundsType = StructType([
    ("lower", VectorType),
    ("upper", VectorType),
])

CurveFitConfigType = StructType([
    ("max_iter", OptionType(IntegerType)),        # default 5000
    ("initial_guess", OptionType(VectorType)),    # default: auto
])
```

## Curve Function Types

```python
# Custom curve function signature: f(x, params) -> y
# Where params is a vector of parameters to optimize
CustomCurveFunctionType = FunctionType(
    [FloatType, VectorType],  # x (scalar), params (vector)
    FloatType                  # y (scalar)
)

# Built-in curves + custom escape hatch
CurveFunctionType = VariantType([
    # Standard mathematical functions (platform knows implementation + param count)
    ("exponential_decay", NullType),        # y = a * exp(-b * x), 2 params: [a, b]
    ("exponential_with_offset", NullType),  # y = a + b * exp(-c * x), 3 params: [a, b, c]
    ("exponential_growth", NullType),       # y = a * exp(b * x), 2 params: [a, b]
    ("logistic", NullType),                 # y = L / (1 + exp(-k * (x - x0))), 3 params: [L, k, x0]
    ("gompertz", NullType),                 # y = a * exp(-b * exp(-c * x)), 3 params: [a, b, c]
    ("power_law", NullType),                # y = a * x^b, 2 params: [a, b]
    ("linear", NullType),                   # y = a + b * x, 2 params: [a, b]
    ("quadratic", NullType),                # y = a + b*x + c*x^2, 3 params: [a, b, c]
    ("cubic", NullType),                    # y = a + b*x + c*x^2 + d*x^3, 4 params: [a, b, c, d]

    # Custom: user provides their own function
    ("custom", StructType([
        ("fn", CustomCurveFunctionType),           # The curve function
        ("n_params", IntegerType),                 # Number of parameters to optimize
        ("param_bounds", OptionType(ParamBoundsType)),  # Optional bounds
    ])),
])

CurveFitResultType = StructType([
    ("params", VectorType),       # Fitted parameters
    ("success", BooleanType),     # Whether fit converged
    ("r_squared", FloatType),     # Goodness of fit
])
```

## Platform Functions

### `scipy_curve_fit`

Fit a parametric curve to data using nonlinear least squares.

Supports built-in standard mathematical functions (with smart defaults) or
custom user-defined functions passed from East code.

```python
PlatformFunction(
    name="scipy_curve_fit",
    inputs=[CurveFunctionType, VectorType, VectorType, CurveFitConfigType],
    output=CurveFitResultType,
    type="sync",
    fn=scipy_curve_fit_impl,
)

def scipy_curve_fit_impl(
    curve_type: EastVariant,
    x: Vector,
    y: Vector,
    config: EastStruct
) -> EastStruct:
    """Fit curve to data using scipy.optimize.curve_fit."""
    from scipy.optimize import curve_fit
    import numpy as np

    x_np = east_vector_to_numpy(x)
    y_np = east_vector_to_numpy(y)
    max_iter = _get_option(config.get("max_iter"), 5000)

    tag = curve_type.type

    # Built-in curves with smart defaults
    if tag == "exponential_decay":
        model_func = lambda x, a, b: a * np.exp(-b * x)
        p0 = [y_np[0], 0.1]
        bounds = ([0, 0], [np.inf, np.inf])

    elif tag == "exponential_with_offset":
        model_func = lambda x, a, b, c: a + b * np.exp(-c * x)
        p0 = [y_np[-1], y_np[0] - y_np[-1], 0.1]
        bounds = ([-np.inf, -np.inf, 0], [np.inf, np.inf, np.inf])

    elif tag == "exponential_growth":
        model_func = lambda x, a, b: a * np.exp(b * x)
        p0 = [y_np[0], 0.1]
        bounds = ([0, 0], [np.inf, np.inf])

    elif tag == "logistic":
        model_func = lambda x, L, k, x0: L / (1 + np.exp(-k * (x - x0)))
        p0 = [y_np.max(), 1.0, x_np.mean()]
        bounds = ([0, 0, -np.inf], [np.inf, np.inf, np.inf])

    elif tag == "gompertz":
        model_func = lambda x, a, b, c: a * np.exp(-b * np.exp(-c * x))
        p0 = [y_np.max(), 1.0, 0.1]
        bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])

    elif tag == "power_law":
        model_func = lambda x, a, b: a * np.power(np.maximum(x, 1e-10), b)
        p0 = [1.0, 1.0]
        bounds = ([-np.inf, -np.inf], [np.inf, np.inf])

    elif tag == "linear":
        model_func = lambda x, a, b: a + b * x
        p0 = [y_np[0], (y_np[-1] - y_np[0]) / (x_np[-1] - x_np[0] + 1e-10)]
        bounds = ([-np.inf, -np.inf], [np.inf, np.inf])

    elif tag == "quadratic":
        model_func = lambda x, a, b, c: a + b * x + c * x**2
        p0 = [y_np[0], 0.0, 0.0]
        bounds = ([-np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf])

    elif tag == "cubic":
        model_func = lambda x, a, b, c, d: a + b * x + c * x**2 + d * x**3
        p0 = [y_np[0], 0.0, 0.0, 0.0]
        bounds = ([-np.inf] * 4, [np.inf] * 4)

    elif tag == "custom":
        custom_config = curve_type.value
        east_fn = custom_config["fn"]  # Compiled East function (callable)
        n_params = custom_config["n_params"]

        # Wrap East function for scipy
        # East fn signature: (x: Float, params: Vector[Float]) -> Float
        def model_func(x_val, *params):
            params_arr = EastArray(FloatType, [float(p) for p in params])
            return east_fn(float(x_val), params_arr)

        # Vectorize for array inputs
        model_func = np.vectorize(model_func, excluded=list(range(1, n_params + 1)))

        # Initial guess
        initial_guess = _get_option(config.get("initial_guess"), None)
        p0 = list(east_vector_to_numpy(initial_guess)) if initial_guess else [1.0] * n_params

        # Bounds
        bounds_opt = _get_option(custom_config.get("param_bounds"), None)
        if bounds_opt:
            bounds = (
                list(east_vector_to_numpy(bounds_opt["lower"])),
                list(east_vector_to_numpy(bounds_opt["upper"]))
            )
        else:
            bounds = ([-np.inf] * n_params, [np.inf] * n_params)

    else:
        raise ValueError(f"Unknown curve type: {tag}")

    # Override initial guess if provided in config
    config_guess = _get_option(config.get("initial_guess"), None)
    if config_guess is not None:
        p0 = list(east_vector_to_numpy(config_guess))

    try:
        params, _ = curve_fit(
            model_func, x_np, y_np,
            p0=p0, bounds=bounds, maxfev=max_iter
        )

        # Compute R²
        y_pred = model_func(x_np, *params)
        ss_res = np.sum((y_np - y_pred) ** 2)
        ss_tot = np.sum((y_np - y_np.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0

        return EastStruct({
            "params": EastArray(FloatType, [float(p) for p in params]),
            "success": True,
            "r_squared": float(np.clip(r2, -10, 1)),
        })

    except Exception:
        return EastStruct({
            "params": EastArray(FloatType, []),
            "success": False,
            "r_squared": 0.0,
        })
```

### Usage Examples

Built-in exponential decay:
```east
let result = scipy_curve_fit(
    #exponential_decay,
    x_data,
    y_data,
    { max_iter: some(5000), initial_guess: none }
);

if result.success {
    let a = result.params[0];  // amplitude
    let b = result.params[1];  // decay rate
    print("Fitted: y = " ++ float_to_string(a) ++ " * exp(-" ++ float_to_string(b) ++ " * x)");
}
```

Custom curve function (e.g., modified Gompertz for fermentation):
```east
// Define custom decay curve: y = B_f + (B_0 - B_f) * exp(-exp(k * (x - lam)))
let gompertz_decay = fn(x: Float, params: Vector[Float]) -> Float {
    let B_f = params[0];   // final asymptote
    let B_0 = params[1];   // initial value
    let k = params[2];     // rate
    let lam = params[3];   // inflection point

    if abs(B_0 - B_f) < 0.000001 {
        B_f
    } else {
        let exponent = k * 2.718 / (B_0 - B_f) * (x - lam) - 1.0;
        B_f + (B_0 - B_f) * exp(-exp(exponent))
    }
};

let result = scipy_curve_fit(
    #custom({
        fn: gompertz_decay,
        n_params: 4,
        param_bounds: some({
            lower: [0.0, 10.0, 0.1, -2.0],   // B_f, B_0, k, lam lower bounds
            upper: [3.0, 18.0, 10.0, 6.0]    // B_f, B_0, k, lam upper bounds
        })
    }),
    time_data,
    baume_data,
    { max_iter: some(5000), initial_guess: some([1.0, 12.0, 1.0, 2.0]) }
);

if result.success {
    let B_f = result.params[0];
    let B_0 = result.params[1];
    let k = result.params[2];
    let lam = result.params[3];
    print("R² = " ++ float_to_string(result.r_squared));
}
```

---

### `scipy_stats_describe`

Compute descriptive statistics.

```python
PlatformFunction(
    name="scipy_stats_describe",
    inputs=[VectorType],
    output=StatsDescribeResultType,
    type="sync",
    fn=scipy_stats_describe_impl,
)

def scipy_stats_describe_impl(data: Vector) -> EastStruct:
    """Compute descriptive statistics for data."""
    from scipy import stats

    data_np = east_vector_to_numpy(data)
    result = stats.describe(data_np)

    return EastStruct({
        "count": int(result.nobs),
        "mean": float(result.mean),
        "variance": float(result.variance),
        "skewness": float(result.skewness),
        "kurtosis": float(result.kurtosis),
        "min": float(result.minmax[0]),
        "max": float(result.minmax[1]),
    })
```

### `scipy_stats_pearsonr`

Compute Pearson correlation coefficient.

```python
PlatformFunction(
    name="scipy_stats_pearsonr",
    inputs=[VectorType, VectorType],
    output=CorrelationResultType,
    type="sync",
    fn=scipy_stats_pearsonr_impl,
)

def scipy_stats_pearsonr_impl(x: Vector, y: Vector) -> EastStruct:
    """Compute Pearson correlation coefficient."""
    from scipy import stats

    x_np = east_vector_to_numpy(x)
    y_np = east_vector_to_numpy(y)

    r, p = stats.pearsonr(x_np, y_np)

    return EastStruct({
        "correlation": float(r),
        "pvalue": float(p),
    })
```

### `scipy_stats_spearmanr`

Compute Spearman rank correlation.

```python
PlatformFunction(
    name="scipy_stats_spearmanr",
    inputs=[VectorType, VectorType],
    output=CorrelationResultType,
    type="sync",
    fn=scipy_stats_spearmanr_impl,
)

def scipy_stats_spearmanr_impl(x: Vector, y: Vector) -> EastStruct:
    """Compute Spearman rank correlation."""
    from scipy import stats

    x_np = east_vector_to_numpy(x)
    y_np = east_vector_to_numpy(y)

    r, p = stats.spearmanr(x_np, y_np)

    return EastStruct({
        "correlation": float(r),
        "pvalue": float(p),
    })
```

### `scipy_interpolate_1d_fit`

Fit 1D interpolator.

```python
PlatformFunction(
    name="scipy_interpolate_1d_fit",
    inputs=[VectorType, VectorType, InterpolateConfigType],
    output=ModelBlobType,  # Returns "scipy_interp_1d" variant
    type="sync",
    fn=scipy_interpolate_1d_fit_impl,
)

def scipy_interpolate_1d_fit_impl(
    x: Vector,
    y: Vector,
    config: EastStruct
) -> ModelBlob:
    """Fit 1D interpolator."""
    from scipy import interpolate

    x_np = east_vector_to_numpy(x)
    y_np = east_vector_to_numpy(y)

    kind_variant = _get_option(config.get("kind"), None)
    kind = _get_enum_tag(kind_variant) if kind_variant else "linear"

    interp = interpolate.interp1d(x_np, y_np, kind=kind, fill_value="extrapolate")

    return EastVariant("scipy_interp_1d", EastStruct({
        "data": _serialize_native(interp),
        "kind": _make_enum(kind),
    }))
```

### `scipy_interpolate_1d_predict`

Evaluate 1D interpolator.

```python
PlatformFunction(
    name="scipy_interpolate_1d_predict",
    inputs=[ModelBlobType, VectorType],  # Expects "scipy_interp_1d" variant
    output=VectorType,
    type="sync",
    fn=scipy_interpolate_1d_predict_impl,
)

def scipy_interpolate_1d_predict_impl(
    model_blob: ModelBlob,
    x: Vector
) -> Vector:
    """Evaluate 1D interpolator at given points."""
    if model_blob.type != "scipy_interp_1d":
        raise ValueError(f"Expected scipy_interp_1d, got {model_blob.type}")

    interp = _deserialize_native(model_blob.value["data"])
    x_np = east_vector_to_numpy(x)

    y_np = interp(x_np)

    return numpy_to_east_vector(y_np)
```

### `scipy_optimize_minimize`

Minimize a scalar function.

```python
PlatformFunction(
    name="scipy_optimize_minimize",
    inputs=[ScalarObjectiveType, VectorType, OptimizeConfigType],
    output=OptimizeResultType,
    type="sync",
    fn=scipy_optimize_minimize_impl,
)

def scipy_optimize_minimize_impl(
    objective_fn: Callable[[EastArray], float],
    x0: Vector,
    config: EastStruct
) -> EastStruct:
    """Minimize a scalar function using scipy.optimize.minimize."""
    from scipy import optimize
    import numpy as np

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

    return EastStruct({
        "x": numpy_to_east_vector(result.x),
        "fun": float(result.fun),
        "success": bool(result.success),
        "nit": int(result.nit),
    })
```

### `scipy_optimize_minimize_quadratic`

Minimize a quadratic function (special case with analytical gradient).

```python
QuadraticConfigType = StructType([
    ("A", MatrixType),  # Quadratic term (symmetric positive definite)
    ("b", VectorType),  # Linear term
    ("c", FloatType),   # Constant term
])

PlatformFunction(
    name="scipy_optimize_minimize_quadratic",
    inputs=[VectorType, QuadraticConfigType, OptimizeConfigType],
    output=OptimizeResultType,
    type="sync",
    fn=scipy_optimize_minimize_quadratic_impl,
)

def scipy_optimize_minimize_quadratic_impl(
    x0: Vector,
    quadratic: EastStruct,
    config: EastStruct
) -> EastStruct:
    """Minimize quadratic function: f(x) = 0.5 * x'Ax + b'x + c"""
    from scipy import optimize
    import numpy as np

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

    return EastStruct({
        "x": numpy_to_east_vector(result.x),
        "fun": float(result.fun),
        "success": bool(result.success),
        "nit": int(result.nit),
    })
```
