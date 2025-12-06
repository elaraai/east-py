# Module 1: Shared Types (`types.py`)

## Imports

```python
from east.types.types import (
    ArrayType, FloatType, IntegerType, StringType, BooleanType,
    StructType, OptionType, VariantType, BlobType, NullType,
    FunctionType,
)
from east.types.values import (
    EastStruct, EastVariant, EastArray, EastBlob,
    east_null, is_east_variant,
)
```

## Matrix and Vector Types

```python
# Core data types (Type definitions)
VectorType = ArrayType(FloatType)              # 1D array of floats
MatrixType = ArrayType(ArrayType(FloatType))   # 2D array of floats
IntVectorType = ArrayType(IntegerType)         # Classification labels
StringVectorType = ArrayType(StringType)       # Feature names

# Value type aliases for function signatures
# EastArray, EastSet, EastDict, EastVariant are Generic and support type parameters
Vector = EastArray[float]
Matrix = EastArray[EastArray[float]]
IntVector = EastArray[int]
StringVector = EastArray[str]
ModelBlob = EastVariant[EastStruct]  # Model blob variants contain EastStruct with model data
```

## Enum Types (Variant with NullType values)

```python
# Scoring metric for cross-validation
ScoringMetricType = VariantType([
    # Regression metrics (negated so higher is better)
    ("neg_mean_squared_error", NullType),
    ("neg_root_mean_squared_error", NullType),
    ("neg_mean_absolute_error", NullType),
    ("neg_mean_absolute_percentage_error", NullType),
    ("neg_median_absolute_error", NullType),
    ("neg_max_error", NullType),
    ("r2", NullType),
    ("explained_variance", NullType),
    # Classification metrics
    ("accuracy", NullType),
    ("balanced_accuracy", NullType),
    ("f1", NullType),
    ("f1_micro", NullType),
    ("f1_macro", NullType),
    ("f1_weighted", NullType),
    ("precision", NullType),
    ("precision_micro", NullType),
    ("precision_macro", NullType),
    ("precision_weighted", NullType),
    ("recall", NullType),
    ("recall_micro", NullType),
    ("recall_macro", NullType),
    ("recall_weighted", NullType),
    ("roc_auc", NullType),
    ("roc_auc_ovr", NullType),
    ("roc_auc_ovo", NullType),
    ("average_precision", NullType),
    ("neg_log_loss", NullType),
    ("neg_brier_score", NullType),
    ("jaccard", NullType),
    ("jaccard_micro", NullType),
    ("jaccard_macro", NullType),
    ("jaccard_weighted", NullType),
])

# NGBoost distribution type
DistributionType = VariantType([
    ("normal", NullType),
    ("lognormal", NullType),
])

# Optuna study direction
OptimizationDirectionType = VariantType([
    ("minimize", NullType),
    ("maximize", NullType),
])

# Optuna pruner type
PrunerType = VariantType([
    ("none", NullType),
    ("median", NullType),
    ("hyperband", NullType),
])

# SciPy optimization method
OptimizeMethodType = VariantType([
    ("bfgs", NullType),
    ("l_bfgs_b", NullType),
    ("nelder_mead", NullType),
    ("powell", NullType),
    ("cg", NullType),
])

# Interpolation kind
InterpolationKindType = VariantType([
    ("linear", NullType),
    ("cubic", NullType),
    ("quadratic", NullType),
])

# Neural network activation
ActivationFunctionType = VariantType([
    ("relu", NullType),
    ("tanh", NullType),
    ("sigmoid", NullType),
    ("leaky_relu", NullType),
])

# Loss function
LossFunctionType = VariantType([
    ("mse", NullType),
    ("mae", NullType),
    ("cross_entropy", NullType),
    ("binary_cross_entropy", NullType),
])

# Optimizer type
OptimizerType = VariantType([
    ("adam", NullType),
    ("sgd", NullType),
    ("adamw", NullType),
    ("rmsprop", NullType),
])

# Gaussian Process kernel
GPKernelType = VariantType([
    ("rbf", NullType),
    ("matern32", NullType),
    ("matern52", NullType),
    ("linear", NullType),
])

# Parameter space kind (for Bayesian optimization)
ParamSpaceKindType = VariantType([
    ("int", NullType),
    ("float", NullType),
    ("categorical", NullType),
    ("log_uniform", NullType),
])
```

## Model Blob Type (Per-Model-Type Variant)

Each model type has its own variant case with specific typed fields:

```python
ModelBlobType = VariantType([
    # =========================================================================
    # Sklearn Preprocessing
    # =========================================================================
    ("standard_scaler", StructType([
        ("onnx", BlobType),
        ("n_features", IntegerType),
    ])),
    ("min_max_scaler", StructType([
        ("onnx", BlobType),
        ("n_features", IntegerType),
    ])),

    # =========================================================================
    # XGBoost
    # =========================================================================
    ("xgboost_regressor", StructType([
        ("onnx", BlobType),
        ("n_features", IntegerType),
    ])),
    ("xgboost_classifier", StructType([
        ("onnx", BlobType),
        ("n_features", IntegerType),
        ("n_classes", IntegerType),
    ])),

    # =========================================================================
    # LightGBM
    # =========================================================================
    ("lightgbm_regressor", StructType([
        ("onnx", BlobType),
        ("n_features", IntegerType),
    ])),
    ("lightgbm_classifier", StructType([
        ("onnx", BlobType),
        ("n_features", IntegerType),
        ("n_classes", IntegerType),
    ])),

    # =========================================================================
    # NGBoost (ONNX + distribution metadata)
    # =========================================================================
    ("ngboost_regressor", StructType([
        ("onnx", BlobType),
        ("distribution", DistributionType),
        ("n_features", IntegerType),
    ])),

    # =========================================================================
    # PyTorch
    # =========================================================================
    ("torch_mlp", StructType([
        ("onnx", BlobType),
        ("n_features", IntegerType),
        ("hidden_layers", ArrayType(IntegerType)),
        ("output_dim", IntegerType),
    ])),

    # =========================================================================
    # Gaussian Process (native format, no ONNX support)
    # =========================================================================
    ("gp_regressor", StructType([
        ("data", BlobType),  # cloudpickle serialized
        ("kernel", GPKernelType),
        ("n_features", IntegerType),
    ])),

    # =========================================================================
    # SciPy Interpolation (native format)
    # =========================================================================
    ("scipy_interp_1d", StructType([
        ("data", BlobType),  # cloudpickle serialized
        ("kind", InterpolationKindType),
    ])),

    # =========================================================================
    # SHAP Explainers
    # =========================================================================
    ("shap_tree_explainer", StructType([
        ("data", BlobType),  # cloudpickle serialized
        ("n_features", IntegerType),
    ])),
    ("shap_kernel_explainer", StructType([
        ("data", BlobType),  # cloudpickle serialized
        ("n_features", IntegerType),
    ])),
])
```

## Result Types

```python
# Prediction result with optional uncertainty
PredictionResultType = StructType([
    ("predictions", VectorType),
    ("std", OptionType(VectorType)),           # Standard deviation (for probabilistic)
    ("lower", OptionType(VectorType)),         # Lower confidence bound
    ("upper", OptionType(VectorType)),         # Upper confidence bound
])

# Feature importance result
FeatureImportanceType = StructType([
    ("feature_names", StringVectorType),
    ("importances", VectorType),
    ("std", OptionType(VectorType)),
])

# Cross-validation result
CrossValResultType = StructType([
    ("scores", VectorType),
    ("mean", FloatType),
    ("std", FloatType),
])

# Regression metrics result
RegressionMetricsType = StructType([
    ("mse", FloatType),
    ("rmse", FloatType),
    ("mae", FloatType),
    ("r2", FloatType),
    ("mape", FloatType),
])

# Classification metrics result
ClassificationMetricsType = StructType([
    ("accuracy", FloatType),
    ("precision", FloatType),
    ("recall", FloatType),
    ("f1", FloatType),
])

# Train/test split result
SplitResultType = StructType([
    ("X_train", MatrixType),
    ("X_test", MatrixType),
    ("y_train", VectorType),
    ("y_test", VectorType),
])

# SHAP values result
ShapResultType = StructType([
    ("shap_values", MatrixType),         # n_samples x n_features
    ("base_value", FloatType),           # Expected value
    ("feature_names", StringVectorType),
])

# PyTorch training result
TorchTrainResultType = StructType([
    ("train_losses", VectorType),
    ("val_losses", VectorType),
    ("best_epoch", IntegerType),
])

# SciPy optimization result
OptimizeResultType = StructType([
    ("x", VectorType),           # Optimal parameters
    ("fun", FloatType),          # Function value at optimum
    ("success", BooleanType),    # Whether optimization succeeded
    ("nit", IntegerType),        # Number of iterations
])

# SciPy stats describe result
StatsDescribeResultType = StructType([
    ("count", IntegerType),
    ("mean", FloatType),
    ("variance", FloatType),
    ("skewness", FloatType),
    ("kurtosis", FloatType),
    ("min", FloatType),
    ("max", FloatType),
])

# Correlation result
CorrelationResultType = StructType([
    ("correlation", FloatType),
    ("pvalue", FloatType),
])

# Curve fitting parameter bounds
ParamBoundsType = StructType([
    ("lower", VectorType),
    ("upper", VectorType),
])

# Custom curve function signature: f(x, params) -> y
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

# Curve fitting result
CurveFitResultType = StructType([
    ("params", VectorType),       # Fitted parameters
    ("success", BooleanType),     # Whether fit converged
    ("r_squared", FloatType),     # Goodness of fit
])
```

## Bayesian Optimization Types (Optuna)

```python
# Parameter value (can be int, float, string, or bool)
ParamValueType = VariantType([
    ("int", IntegerType),
    ("float", FloatType),
    ("string", StringType),
    ("bool", BooleanType),
])

# Parameter search space definition
ParamSpaceType = StructType([
    ("name", StringType),
    ("kind", ParamSpaceKindType),
    ("low", OptionType(FloatType)),
    ("high", OptionType(FloatType)),
    ("choices", OptionType(ArrayType(ParamValueType))),
])

# Named parameter (name + value pair)
NamedParamType = StructType([
    ("name", StringType),
    ("value", ParamValueType),
])

# Objective function type: East function that takes params and returns score
# Platform functions can receive FunctionType inputs as compiled Python callables
ObjectiveFunctionType = FunctionType(
    [ArrayType(NamedParamType)],  # params input
    FloatType                      # score output
)

# Optimization trial result
TrialResultType = StructType([
    ("trial_id", IntegerType),
    ("params", ArrayType(NamedParamType)),
    ("score", FloatType),
])

# Optimization study result
StudyResultType = StructType([
    ("best_params", ArrayType(NamedParamType)),
    ("best_score", FloatType),
    ("trials", ArrayType(TrialResultType)),
])

# Optuna study config
OptunaStudyConfigType = StructType([
    ("direction", OptionType(OptimizationDirectionType)),  # default minimize
    ("n_trials", IntegerType),                             # number of trials
    ("random_state", OptionType(IntegerType)),             # default None
    ("pruner", OptionType(PrunerType)),                    # default none
])
```

## MADS Types (PyNomadBBO)

```python
# Direction types for MADS
MADSDirectionType = VariantType([
    ("ortho_2n", NullType),       # 2n orthogonal directions (default)
    ("ortho_n_plus_1", NullType), # n+1 orthogonal directions
    ("lt_2n", NullType),          # Lower triangular 2n
    ("single", NullType),         # Single direction
])

# Scalar objective function: Vector -> Float
# Used for optimization (MADS, SciPy minimize) and constraints
ScalarObjectiveType = FunctionType([VectorType], FloatType)

# MADS constraint: variant where the case indicates the kind (eb/pb) and value is the function
MADSConstraintType = VariantType([
    ("eb", ScalarObjectiveType),  # Extreme barrier (infeasible points rejected)
    ("pb", ScalarObjectiveType),  # Progressive barrier (relaxed constraints)
])

# Bounds specification
MADSBoundsType = StructType([
    ("lower", VectorType),
    ("upper", VectorType),
])

# MADS optimization config
MADSConfigType = StructType([
    ("max_bb_eval", OptionType(IntegerType)),       # Max blackbox evaluations (default 100)
    ("display_degree", OptionType(IntegerType)),    # 0=silent, 1=minimal, 2=normal (default 0)
    ("direction_type", OptionType(MADSDirectionType)),  # Search direction type
    ("initial_mesh_size", OptionType(FloatType)),   # Initial mesh size
    ("min_mesh_size", OptionType(FloatType)),       # Minimum mesh size (stopping criterion)
    ("seed", OptionType(IntegerType)),              # Random seed
])

# Single-objective result type
MADSResultType = StructType([
    ("x_best", VectorType),         # Best solution found
    ("f_best", FloatType),          # Best objective value
    ("bb_eval", IntegerType),       # Number of blackbox evaluations
    ("success", BooleanType),       # Whether optimization succeeded
])

# Multi-objective result type
MADSMultiResultType = StructType([
    ("pareto_front", MatrixType),   # Pareto-optimal solutions (n_solutions x n_vars)
    ("pareto_values", MatrixType),  # Objective values (n_solutions x n_objectives)
    ("bb_eval", IntegerType),
    ("success", BooleanType),
])
```

## East Value Helpers

```python
def _get_option(value: Any, default: Any) -> Any:
    """Extract value from Option variant or return default.

    Note: Use is_east_variant (not is_east_option) because deserialized IR
    uses EastVariant with 'some'/'none' tags, not EastOption instances.
    """
    if value is None:
        return default
    if is_east_variant(value) and value.type == "some":
        return value.value
    return default


def _get_enum_tag(value: EastVariant) -> str:
    """Extract tag name from enum-like variant."""
    if isinstance(value, EastVariant):
        return value.type
    raise ValueError(f"Expected EastVariant enum, got {type(value)}")


def _make_enum(tag: str) -> EastVariant:
    """Create an enum-like variant (tag with null value)."""
    return EastVariant(tag, east_null)
```

## Numpy <-> East Conversion Helpers

```python
import numpy as np

def east_vector_to_numpy(arr: EastArray) -> np.ndarray:
    """Convert EastArray[Float] to numpy array."""
    return np.array([float(x) for x in arr], dtype=np.float32)


def east_matrix_to_numpy(arr: EastArray) -> np.ndarray:
    """Convert EastArray[EastArray[Float]] to numpy 2D array."""
    return np.array([[float(x) for x in row] for row in arr], dtype=np.float32)


def east_int_vector_to_numpy(arr: EastArray) -> np.ndarray:
    """Convert EastArray[Integer] to numpy array."""
    return np.array([int(x) for x in arr], dtype=np.int64)


def numpy_to_east_vector(arr: np.ndarray) -> EastArray:
    """Convert numpy 1D array to EastArray[Float]."""
    return EastArray(FloatType, [float(x) for x in arr.flatten()])


def numpy_to_east_matrix(arr: np.ndarray) -> EastArray:
    """Convert numpy 2D array to EastArray[EastArray[Float]]."""
    inner_type = ArrayType(FloatType)
    rows = [EastArray(FloatType, [float(x) for x in row]) for row in arr]
    return EastArray(inner_type, rows)


def numpy_to_east_int_vector(arr: np.ndarray) -> EastArray:
    """Convert numpy 1D int array to EastArray[Integer]."""
    return EastArray(IntegerType, [int(x) for x in arr.flatten()])
```

## ONNX Conversion Helpers

```python
import onnxruntime as ort
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


def _sklearn_to_onnx(model: Any, n_features: int) -> EastBlob:
    """Convert sklearn model to ONNX bytes."""
    initial_type = [("X", FloatTensorType([None, n_features]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)
    return EastBlob(onnx_model.SerializeToString())


def _xgboost_to_onnx(model: Any, n_features: int) -> EastBlob:
    """Convert XGBoost model to ONNX bytes."""
    from onnxmltools.convert import convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType

    initial_type = [("X", FloatTensorType([None, n_features]))]
    onnx_model = convert_xgboost(model, initial_types=initial_type)
    return EastBlob(onnx_model.SerializeToString())


def _lightgbm_to_onnx(model: Any, n_features: int) -> EastBlob:
    """Convert LightGBM model to ONNX bytes."""
    from onnxmltools.convert import convert_lightgbm
    from onnxmltools.convert.common.data_types import FloatTensorType

    initial_type = [("X", FloatTensorType([None, n_features]))]
    onnx_model = convert_lightgbm(model, initial_types=initial_type)
    return EastBlob(onnx_model.SerializeToString())


def _pytorch_to_onnx(model: Any, n_features: int) -> EastBlob:
    """Convert PyTorch model to ONNX bytes."""
    import torch
    import io

    model.eval()
    dummy_input = torch.randn(1, n_features)
    buffer = io.BytesIO()
    torch.onnx.export(
        model,
        dummy_input,
        buffer,
        input_names=["X"],
        output_names=["predictions"],
        dynamic_axes={"X": {0: "batch_size"}, "predictions": {0: "batch_size"}},
    )
    return EastBlob(buffer.getvalue())
```

## ONNX Inference Helpers

```python
def _onnx_predict_regression(onnx_blob: EastBlob, X: EastArray) -> EastArray:
    """Run regression inference using ONNX Runtime."""
    onnx_bytes = bytes(onnx_blob)
    X_np = east_matrix_to_numpy(X)

    session = ort.InferenceSession(onnx_bytes)
    input_name = session.get_inputs()[0].name

    outputs = session.run(None, {input_name: X_np})
    predictions = outputs[0].squeeze()

    return EastArray(FloatType, [float(p) for p in predictions])


def _onnx_predict_classification(onnx_blob: EastBlob, X: EastArray) -> EastArray:
    """Run classification inference using ONNX Runtime, return class labels."""
    onnx_bytes = bytes(onnx_blob)
    X_np = east_matrix_to_numpy(X)

    session = ort.InferenceSession(onnx_bytes)
    input_name = session.get_inputs()[0].name

    outputs = session.run(None, {input_name: X_np})
    labels = outputs[0].squeeze()

    return EastArray(IntegerType, [int(l) for l in labels])


def _onnx_predict_proba(onnx_blob: EastBlob, X: EastArray) -> EastArray:
    """Run classification inference using ONNX Runtime, return probabilities."""
    onnx_bytes = bytes(onnx_blob)
    X_np = east_matrix_to_numpy(X)

    session = ort.InferenceSession(onnx_bytes)
    input_name = session.get_inputs()[0].name

    outputs = session.run(None, {input_name: X_np})
    # For classifiers, output[1] is typically probabilities
    proba = outputs[1] if len(outputs) > 1 else outputs[0]

    return numpy_to_east_matrix(proba)


def _onnx_transform(onnx_blob: EastBlob, X: EastArray) -> EastArray:
    """Run transform (e.g., scaler) using ONNX Runtime."""
    onnx_bytes = bytes(onnx_blob)
    X_np = east_matrix_to_numpy(X)

    session = ort.InferenceSession(onnx_bytes)
    input_name = session.get_inputs()[0].name

    outputs = session.run(None, {input_name: X_np})
    X_transformed = outputs[0]

    return numpy_to_east_matrix(X_transformed)
```

## Native Model Helpers (for non-ONNX models)

```python
import cloudpickle


def _serialize_native(model: Any) -> EastBlob:
    """Serialize a model using cloudpickle."""
    return EastBlob(cloudpickle.dumps(model))


def _deserialize_native(data: EastBlob) -> Any:
    """Deserialize a model using cloudpickle."""
    return cloudpickle.loads(bytes(data))
```
