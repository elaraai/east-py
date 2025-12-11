"""Shared type definitions for East Data Science.

Provides common East type definitions used across data science modules
including vectors, matrices, and scalar function types.
"""

from typing import Any

import numpy as np

from east.types.types import (
    ArrayType,
    BlobType,
    BooleanType,
    FloatType,
    FunctionType,
    IntegerType,
    NullType,
    OptionType,
    StringType,
    StructType,
    VariantType,
)
from east.types.values import EastArray, EastVariant, is_east_variant

# ============================================================================
# Core Data Types
# ============================================================================

# Vector type (1D array of floats)
VectorType = ArrayType(FloatType)

# Matrix type (2D array of floats)
MatrixType = ArrayType(VectorType)

# Integer vector type (for classification labels)
IntVectorType = ArrayType(IntegerType)

# String vector type (for feature names)
StringVectorType = ArrayType(StringType)

# ============================================================================
# Function Types
# ============================================================================

# Scalar objective function type: Vector -> Float
ScalarObjectiveType = FunctionType([VectorType], FloatType)

# Vector objective function type: Vector -> Vector
VectorObjectiveType = FunctionType([VectorType], VectorType)

# Label vector type (1D array of integers) - alias for backwards compatibility
LabelVectorType = IntVectorType

# ============================================================================
# Enum Types (Variant with NullType values)
# ============================================================================

# SciPy optimization method
OptimizeMethodType = VariantType(
    [
        ("bfgs", NullType),
        ("l_bfgs_b", NullType),
        ("nelder_mead", NullType),
        ("powell", NullType),
        ("cg", NullType),
    ]
)

# Interpolation kind
InterpolationKindType = VariantType(
    [
        ("linear", NullType),
        ("cubic", NullType),
        ("quadratic", NullType),
    ]
)

# NGBoost distribution type
NGBoostDistributionType = VariantType(
    [
        ("normal", NullType),
        ("lognormal", NullType),
    ]
)

# Torch activation function type
TorchActivationType = VariantType(
    [
        ("relu", NullType),
        ("tanh", NullType),
        ("sigmoid", NullType),
        ("leaky_relu", NullType),
    ]
)

# Torch loss function type
TorchLossType = VariantType(
    [
        ("mse", NullType),
        ("mae", NullType),
        ("cross_entropy", NullType),
    ]
)

# Torch optimizer type
TorchOptimizerType = VariantType(
    [
        ("adam", NullType),
        ("sgd", NullType),
        ("adamw", NullType),
        ("rmsprop", NullType),
    ]
)

# GP kernel type
GPKernelType = VariantType(
    [
        ("rbf", NullType),  # Radial Basis Function (squared exponential)
        ("matern_1_2", NullType),  # Matern with nu=1/2 (exponential)
        ("matern_3_2", NullType),  # Matern with nu=3/2
        ("matern_5_2", NullType),  # Matern with nu=5/2
        ("rational_quadratic", NullType),
        ("dot_product", NullType),
    ]
)


# ============================================================================
# Config Types
# ============================================================================

# Train/test split configuration
SplitConfigType = StructType(
    [
        ("test_size", OptionType(FloatType)),  # default 0.2
        ("random_state", OptionType(IntegerType)),  # default None
        ("shuffle", OptionType(BooleanType)),  # default True
    ]
)

# 3-way train/val/test split configuration
ThreeWaySplitConfigType = StructType(
    [
        ("val_size", OptionType(FloatType)),  # default 0.15
        ("test_size", OptionType(FloatType)),  # default 0.15
        ("random_state", OptionType(IntegerType)),  # default None
        ("shuffle", OptionType(BooleanType)),  # default True
    ]
)

# SciPy optimization configuration
OptimizeConfigType = StructType(
    [
        ("method", OptionType(OptimizeMethodType)),  # default l_bfgs_b
        ("max_iter", OptionType(IntegerType)),  # default 1000
        ("tol", OptionType(FloatType)),  # default 1e-6
    ]
)

# SciPy interpolation configuration
InterpolateConfigType = StructType(
    [
        ("kind", OptionType(InterpolationKindType)),  # default linear
    ]
)

# Parameter bounds for curve fitting
ParamBoundsType = StructType(
    [
        ("lower", VectorType),
        ("upper", VectorType),
    ]
)

# Custom curve function: (x: Float, params: Vector) -> Float
CustomCurveFunctionType = FunctionType([FloatType, VectorType], FloatType)

# Curve function type (built-in + custom)
CurveFunctionType = VariantType(
    [
        # Standard mathematical functions
        ("exponential_decay", NullType),  # y = a * exp(-b * x)
        ("exponential_with_offset", NullType),  # y = a + b * exp(-c * x)
        ("exponential_growth", NullType),  # y = a * exp(b * x)
        ("logistic", NullType),  # y = L / (1 + exp(-k * (x - x0)))
        ("gompertz", NullType),  # y = a * exp(-b * exp(-c * x))
        ("power_law", NullType),  # y = a * x^b
        ("linear", NullType),  # y = a + b * x
        ("quadratic", NullType),  # y = a + b*x + c*x^2
        ("cubic", NullType),  # y = a + b*x + c*x^2 + d*x^3
        # Custom function
        (
            "custom",
            StructType(
                [
                    ("fn", CustomCurveFunctionType),
                    ("n_params", IntegerType),
                    ("param_bounds", OptionType(ParamBoundsType)),
                ]
            ),
        ),
    ]
)

# Curve fit configuration
CurveFitConfigType = StructType(
    [
        ("max_iter", OptionType(IntegerType)),  # default 5000
        ("initial_guess", OptionType(VectorType)),  # default: auto
    ]
)

# Quadratic function configuration: f(x) = 0.5 * x'Ax + b'x + c
QuadraticConfigType = StructType(
    [
        ("A", MatrixType),  # Quadratic term (symmetric positive definite)
        ("b", VectorType),  # Linear term
        ("c", FloatType),  # Constant term
    ]
)

# XGBoost configuration
XGBoostConfigType = StructType(
    [
        ("n_estimators", OptionType(IntegerType)),  # default 100
        ("max_depth", OptionType(IntegerType)),  # default 6
        ("learning_rate", OptionType(FloatType)),  # default 0.3
        ("min_child_weight", OptionType(IntegerType)),  # default 1
        ("subsample", OptionType(FloatType)),  # default 1.0
        ("colsample_bytree", OptionType(FloatType)),  # default 1.0
        ("reg_alpha", OptionType(FloatType)),  # default 0 (L1)
        ("reg_lambda", OptionType(FloatType)),  # default 1 (L2)
        ("random_state", OptionType(IntegerType)),  # default None
        ("n_jobs", OptionType(IntegerType)),  # default -1
    ]
)

# LightGBM configuration
LightGBMConfigType = StructType(
    [
        ("n_estimators", OptionType(IntegerType)),  # default 100
        ("max_depth", OptionType(IntegerType)),  # default -1 (unlimited)
        ("learning_rate", OptionType(FloatType)),  # default 0.1
        ("num_leaves", OptionType(IntegerType)),  # default 31
        ("min_child_samples", OptionType(IntegerType)),  # default 20
        ("subsample", OptionType(FloatType)),  # default 1.0
        ("colsample_bytree", OptionType(FloatType)),  # default 1.0
        ("reg_alpha", OptionType(FloatType)),  # default 0
        ("reg_lambda", OptionType(FloatType)),  # default 0
        ("random_state", OptionType(IntegerType)),  # default None
        ("n_jobs", OptionType(IntegerType)),  # default -1
    ]
)

# NGBoost configuration
NGBoostConfigType = StructType(
    [
        ("n_estimators", OptionType(IntegerType)),  # default 500
        ("learning_rate", OptionType(FloatType)),  # default 0.01
        ("minibatch_frac", OptionType(FloatType)),  # default 1.0
        ("col_sample", OptionType(FloatType)),  # default 1.0
        ("random_state", OptionType(IntegerType)),  # default None
        ("distribution", OptionType(NGBoostDistributionType)),  # default normal
    ]
)

# NGBoost prediction configuration
NGBoostPredictConfigType = StructType(
    [
        ("confidence_level", OptionType(FloatType)),  # default 0.95
    ]
)

# Torch MLP configuration
TorchMLPConfigType = StructType(
    [
        ("hidden_layers", ArrayType(IntegerType)),  # e.g., [64, 32]
        ("activation", OptionType(TorchActivationType)),  # default relu
        ("dropout", OptionType(FloatType)),  # default 0.0
        ("output_dim", OptionType(IntegerType)),  # default 1
    ]
)

# Torch training configuration
TorchTrainConfigType = StructType(
    [
        ("epochs", OptionType(IntegerType)),  # default 100
        ("batch_size", OptionType(IntegerType)),  # default 32
        ("learning_rate", OptionType(FloatType)),  # default 0.001
        ("loss", OptionType(TorchLossType)),  # default mse
        ("optimizer", OptionType(TorchOptimizerType)),  # default adam
        ("early_stopping", OptionType(IntegerType)),  # patience, 0 = disabled
        ("validation_split", OptionType(FloatType)),  # default 0.2
        ("random_state", OptionType(IntegerType)),  # for reproducibility
    ]
)

# GP configuration
GPConfigType = StructType(
    [
        ("kernel", OptionType(GPKernelType)),  # default rbf
        ("alpha", OptionType(FloatType)),  # noise level, default 1e-10
        ("n_restarts_optimizer", OptionType(IntegerType)),  # default 0
        ("normalize_y", OptionType(BooleanType)),  # default False
        ("random_state", OptionType(IntegerType)),  # for reproducibility
    ]
)

# RegressorChain base estimator config (variant carries type + config)
RegressorChainBaseConfigType = VariantType(
    [
        ("xgboost", XGBoostConfigType),
        ("lightgbm", LightGBMConfigType),
        ("ngboost", NGBoostConfigType),
        ("gp", GPConfigType),
    ]
)

# RegressorChain configuration
RegressorChainConfigType = StructType(
    [
        ("base_estimator", RegressorChainBaseConfigType),  # Base estimator with config
        (
            "order",
            OptionType(ArrayType(IntegerType)),
        ),  # Chain order (default: None = 0,1,2,...)
        ("random_state", OptionType(IntegerType)),  # Random seed
    ]
)

# ============================================================================
# Result Types
# ============================================================================

# Train/test split result
SplitResultType = StructType(
    [
        ("X_train", MatrixType),
        ("X_test", MatrixType),
        ("y_train", VectorType),
        ("y_test", VectorType),
    ]
)

# 3-way train/val/test split result
ThreeWaySplitResultType = StructType(
    [
        ("X_train", MatrixType),
        ("X_val", MatrixType),
        ("X_test", MatrixType),
        ("Y_train", MatrixType),
        ("Y_val", MatrixType),
        ("Y_test", MatrixType),
    ]
)

# ============================================================================
# Flexible Metrics Types
# ============================================================================

# Regression metric variant (flexible)
RegressionMetricType = VariantType(
    [
        ("mse", NullType),
        ("rmse", NullType),
        ("mae", NullType),
        ("r2", NullType),
        ("mape", NullType),
        ("explained_variance", NullType),
        ("max_error", NullType),
        ("median_ae", NullType),
    ]
)

# Single metric result
MetricResultType = StructType(
    [
        ("metric", RegressionMetricType),
        ("value", FloatType),
    ]
)

# Multiple metrics result
MetricsResultType = ArrayType(MetricResultType)

# Metric aggregation type
MetricAggregationType = VariantType(
    [
        ("per_target", NullType),
        ("uniform_average", NullType),
    ]
)

# Multi-target metrics config
MultiMetricsConfigType = StructType(
    [
        ("aggregation", OptionType(MetricAggregationType)),
    ]
)

# Multi-target metric value (scalar or per-target)
MultiMetricValueType = VariantType(
    [
        ("scalar", FloatType),
        ("per_target", VectorType),
    ]
)

# Multi-target metric result
MultiMetricResultType = StructType(
    [
        ("metric", RegressionMetricType),
        ("value", MultiMetricValueType),
    ]
)

# Multi-target metrics result
MultiMetricsResultType = ArrayType(MultiMetricResultType)

# Classification metric variant
ClassificationMetricType = VariantType(
    [
        ("accuracy", NullType),
        ("balanced_accuracy", NullType),
        ("precision", NullType),
        ("recall", NullType),
        ("f1", NullType),
        ("matthews_corrcoef", NullType),
        ("cohen_kappa", NullType),
        ("jaccard", NullType),
    ]
)

# Classification averaging type
ClassificationAverageType = VariantType(
    [
        ("macro", NullType),
        ("micro", NullType),
        ("weighted", NullType),
        ("binary", NullType),
    ]
)

# Classification metrics config
ClassificationMetricsConfigType = StructType(
    [
        ("average", OptionType(ClassificationAverageType)),
    ]
)

# Single classification metric result
ClassificationMetricResultType = StructType(
    [
        ("metric", ClassificationMetricType),
        ("value", FloatType),
    ]
)

# Multiple classification metrics result
ClassificationMetricResultsType = ArrayType(ClassificationMetricResultType)

# Multi-target classification config
MultiClassificationConfigType = StructType(
    [
        ("average", OptionType(ClassificationAverageType)),
        ("aggregation", OptionType(MetricAggregationType)),
    ]
)

# Multi-target classification metric result
MultiClassificationMetricResultType = StructType(
    [
        ("metric", ClassificationMetricType),
        ("value", MultiMetricValueType),
    ]
)

# Multi-target classification metrics result
MultiClassificationMetricResultsType = ArrayType(MultiClassificationMetricResultType)

# SciPy stats describe result
StatsDescribeResultType = StructType(
    [
        ("count", IntegerType),
        ("mean", FloatType),
        ("variance", FloatType),
        ("skewness", FloatType),
        ("kurtosis", FloatType),
        ("min", FloatType),
        ("max", FloatType),
    ]
)

# Correlation result
CorrelationResultType = StructType(
    [
        ("correlation", FloatType),
        ("pvalue", FloatType),
    ]
)

# Curve fitting result
CurveFitResultType = StructType(
    [
        ("params", VectorType),
        ("success", BooleanType),
        ("r_squared", FloatType),
    ]
)

# SciPy optimization result
OptimizeResultType = StructType(
    [
        ("x", VectorType),  # Optimal parameters
        ("fun", FloatType),  # Function value at optimum
        ("success", BooleanType),  # Whether optimization succeeded
        ("nit", IntegerType),  # Number of iterations
    ]
)

# SciPy dual annealing bounds (required)
DualAnnealBoundsType = StructType(
    [
        ("lower", VectorType),  # Lower bounds for each variable
        ("upper", VectorType),  # Upper bounds for each variable
    ]
)

# SciPy dual annealing configuration
DualAnnealConfigType = StructType(
    [
        ("maxfun", OptionType(IntegerType)),  # Max function evals (default 1000)
        ("maxiter", OptionType(IntegerType)),  # Max iterations (default 1000)
        ("initial_temp", OptionType(FloatType)),  # Initial temperature (default 5230)
        (
            "restart_temp_ratio",
            OptionType(FloatType),
        ),  # Restart threshold (default 2e-5)
        ("visit", OptionType(FloatType)),  # Visiting distribution param (default 2.62)
        ("accept", OptionType(FloatType)),  # Acceptance param (default -5.0)
        ("seed", OptionType(IntegerType)),  # Random seed
        ("no_local_search", OptionType(BooleanType)),  # Disable local search
    ]
)

# SciPy dual annealing result
DualAnnealResultType = StructType(
    [
        ("x", VectorType),  # Best solution found
        ("fun", FloatType),  # Best objective value
        ("nfev", IntegerType),  # Number of function evaluations
        ("nit", IntegerType),  # Number of iterations
        ("success", BooleanType),  # Whether optimization succeeded
        ("message", StringType),  # Status message
    ]
)

# NGBoost prediction result (with uncertainty)
NGBoostPredictResultType = StructType(
    [
        ("predictions", VectorType),  # Point predictions (mean)
        ("std", OptionType(VectorType)),  # Standard deviation
        ("lower", OptionType(VectorType)),  # Lower confidence interval
        ("upper", OptionType(VectorType)),  # Upper confidence interval
    ]
)

# SHAP values result
ShapResultType = StructType(
    [
        ("shap_values", MatrixType),  # SHAP values (n_samples x n_features)
        ("base_value", FloatType),  # Expected value (base prediction)
        ("feature_names", StringVectorType),  # Feature names
    ]
)

# Feature importance result
FeatureImportanceType = StructType(
    [
        ("feature_names", StringVectorType),  # Feature names
        ("importances", VectorType),  # Mean |SHAP| for each feature
        ("std", OptionType(VectorType)),  # Std of |SHAP| for each feature
    ]
)

# Torch training result
TorchTrainResultType = StructType(
    [
        ("train_losses", VectorType),  # Training loss per epoch
        ("val_losses", VectorType),  # Validation loss per epoch
        ("best_epoch", IntegerType),  # Best epoch (for early stopping)
    ]
)

# GP prediction result (with uncertainty)
GPPredictResultType = StructType(
    [
        ("mean", VectorType),  # Predicted mean
        ("std", VectorType),  # Predicted standard deviation
    ]
)

# ============================================================================
# Model Blob Type
# ============================================================================

# Model blob type - each model type has its own variant case
ModelBlobType = VariantType(
    [
        # Sklearn Preprocessing
        (
            "standard_scaler",
            StructType(
                [
                    ("onnx", BlobType),
                    ("n_features", IntegerType),
                ]
            ),
        ),
        (
            "min_max_scaler",
            StructType(
                [
                    ("onnx", BlobType),
                    ("n_features", IntegerType),
                ]
            ),
        ),
        # SciPy Interpolation (native format)
        (
            "scipy_interp_1d",
            StructType(
                [
                    ("data", BlobType),  # cloudpickle serialized
                    ("kind", InterpolationKindType),
                ]
            ),
        ),
        # XGBoost models (cloudpickle serialized)
        (
            "xgboost_regressor",
            StructType(
                [
                    ("data", BlobType),
                    ("n_features", IntegerType),
                ]
            ),
        ),
        (
            "xgboost_classifier",
            StructType(
                [
                    ("data", BlobType),
                    ("n_features", IntegerType),
                    ("n_classes", IntegerType),
                ]
            ),
        ),
        # LightGBM models (cloudpickle serialized)
        (
            "lightgbm_regressor",
            StructType(
                [
                    ("data", BlobType),
                    ("n_features", IntegerType),
                ]
            ),
        ),
        (
            "lightgbm_classifier",
            StructType(
                [
                    ("data", BlobType),
                    ("n_features", IntegerType),
                    ("n_classes", IntegerType),
                ]
            ),
        ),
        # NGBoost models (cloudpickle serialized)
        (
            "ngboost_regressor",
            StructType(
                [
                    ("data", BlobType),
                    ("distribution", NGBoostDistributionType),
                    ("n_features", IntegerType),
                ]
            ),
        ),
        # SHAP explainers (cloudpickle serialized)
        (
            "shap_tree_explainer",
            StructType(
                [
                    ("data", BlobType),
                    ("n_features", IntegerType),
                ]
            ),
        ),
        (
            "shap_kernel_explainer",
            StructType(
                [
                    ("data", BlobType),
                    ("n_features", IntegerType),
                ]
            ),
        ),
        # PyTorch models (cloudpickle serialized)
        (
            "torch_mlp",
            StructType(
                [
                    ("data", BlobType),
                    ("n_features", IntegerType),
                    ("hidden_layers", ArrayType(IntegerType)),
                    ("output_dim", IntegerType),
                ]
            ),
        ),
        # RegressorChain (cloudpickle serialized)
        (
            "regressor_chain",
            StructType(
                [
                    ("data", BlobType),
                    ("n_features", IntegerType),
                    ("n_targets", IntegerType),
                    (
                        "base_estimator_type",
                        StringType,
                    ),  # "xgboost", "lightgbm", "ngboost", or "gp"
                ]
            ),
        ),
        # Gaussian Process (cloudpickle serialized)
        (
            "gp_regressor",
            StructType(
                [
                    ("data", BlobType),
                    ("n_features", IntegerType),
                    ("kernel_type", StringType),  # kernel name for reference
                ]
            ),
        ),
    ]
)

# ============================================================================
# Helper Functions
# ============================================================================


def _get_option(opt: EastVariant | None, default: Any) -> Any:
    """Extract value from Option variant, returning default if None."""
    if opt is None:
        return default
    if is_east_variant(opt) and opt.type == "some":
        return opt.value
    return default


def _get_enum_tag(variant: EastVariant) -> str:
    """Get tag name from enum-like variant."""
    if isinstance(variant, EastVariant):
        return variant.type
    raise ValueError(f"Expected EastVariant, got {type(variant)}")


# ============================================================================
# Numpy <-> East Conversion Helpers
# ============================================================================


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
    rows: list[EastArray] = [
        EastArray(FloatType, [float(x) for x in row]) for row in arr
    ]
    return EastArray(inner_type, rows)


def numpy_to_east_int_vector(arr: np.ndarray) -> EastArray:
    """Convert numpy 1D int array to EastArray[Integer]."""
    return EastArray(IntegerType, [int(x) for x in arr.flatten()])


__all__ = [
    # Core Types
    "VectorType",
    "MatrixType",
    "IntVectorType",
    "StringVectorType",
    "ScalarObjectiveType",
    "VectorObjectiveType",
    "LabelVectorType",
    # Sklearn Types
    "SplitConfigType",
    "SplitResultType",
    "ThreeWaySplitConfigType",
    "ThreeWaySplitResultType",
    # Flexible Metrics Types
    "RegressionMetricType",
    "MetricResultType",
    "MetricsResultType",
    "MetricAggregationType",
    "MultiMetricsConfigType",
    "MultiMetricValueType",
    "MultiMetricResultType",
    "MultiMetricsResultType",
    "ClassificationMetricType",
    "ClassificationAverageType",
    "ClassificationMetricsConfigType",
    "ClassificationMetricResultType",
    "ClassificationMetricResultsType",
    "MultiClassificationConfigType",
    "MultiClassificationMetricResultType",
    "MultiClassificationMetricResultsType",
    # Scipy Types
    "OptimizeMethodType",
    "InterpolationKindType",
    "OptimizeConfigType",
    "InterpolateConfigType",
    "ParamBoundsType",
    "CustomCurveFunctionType",
    "CurveFunctionType",
    "CurveFitConfigType",
    "QuadraticConfigType",
    # XGBoost Types
    "XGBoostConfigType",
    # LightGBM Types
    "LightGBMConfigType",
    # NGBoost Types
    "NGBoostDistributionType",
    "NGBoostConfigType",
    "NGBoostPredictConfigType",
    "NGBoostPredictResultType",
    # SHAP Types
    "ShapResultType",
    "FeatureImportanceType",
    # Torch Types
    "TorchActivationType",
    "TorchLossType",
    "TorchOptimizerType",
    "TorchMLPConfigType",
    "TorchTrainConfigType",
    "TorchTrainResultType",
    # RegressorChain Types
    "RegressorChainBaseConfigType",
    "RegressorChainConfigType",
    # GP Types
    "GPKernelType",
    "GPConfigType",
    "GPPredictResultType",
    # Scipy Result Types
    "StatsDescribeResultType",
    "CorrelationResultType",
    "CurveFitResultType",
    "OptimizeResultType",
    # Scipy Dual Annealing Types
    "DualAnnealBoundsType",
    "DualAnnealConfigType",
    "DualAnnealResultType",
    # Model Blob
    "ModelBlobType",
    # Helpers
    "_get_option",
    "_get_enum_tag",
    "east_vector_to_numpy",
    "east_matrix_to_numpy",
    "east_int_vector_to_numpy",
    "numpy_to_east_vector",
    "numpy_to_east_matrix",
    "numpy_to_east_int_vector",
]
