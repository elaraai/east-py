#
# Copyright (c) 2025 Elara AI Pty Ltd
# Licensed under the Business Source License 1.1. See LICENSE.md for details.
#
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
        ("kl_div", NullType),
        ("bce", NullType),
        ("bce_with_logits", NullType),
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

# Torch output activation type
TorchOutputActivationType = VariantType(
    [
        ("none", NullType),
        ("softmax", NullType),
        ("sigmoid", NullType),
    ]
)

# Per-row output constraint type
RowConstraintType = VariantType(
    [
        # Independent binary outputs (sigmoid), optionally masked
        (
            "binary",
            StructType(
                [
                    ("mask", OptionType(ArrayType(BooleanType))),  # User-specified mask
                    (
                        "data_mask",
                        OptionType(ArrayType(BooleanType)),
                    ),  # Data-derived static mask
                ]
            ),
        ),
        # Mutually exclusive - at most one position active (softmax)
        (
            "mutex",
            StructType(
                [
                    ("mask", OptionType(ArrayType(BooleanType))),  # User-specified mask
                    ("allow_none", OptionType(BooleanType)),
                    (
                        "data_mask",
                        OptionType(ArrayType(BooleanType)),
                    ),  # Data-derived static mask
                ]
            ),
        ),
        # At most N positions active
        (
            "at_most",
            StructType(
                [
                    ("max_count", IntegerType),
                    ("mask", OptionType(ArrayType(BooleanType))),  # User-specified mask
                    (
                        "data_mask",
                        OptionType(ArrayType(BooleanType)),
                    ),  # Data-derived static mask
                ]
            ),
        ),
    ]
)

# Positive weight type for class imbalance handling
PosWeightType = VariantType(
    [
        ("scalar", FloatType),  # Single weight applied to all outputs
        (
            "per_output",
            ArrayType(FloatType),
        ),  # Per-output weights (length = output_dim)
    ]
)

# Prior regularization configuration
PriorConfigType = StructType(
    [
        ("values", ArrayType(FloatType)),  # Prior probabilities per output
        ("weight", FloatType),  # Lambda weight for MSE regularization
    ]
)

# Per-sample constraints configuration
SampleConstraintsConfigType = StructType(
    [
        # Per-sample boolean masks: (n_samples, n_rows, n_cols)
        # True = allowed, False = masked (output forced to 0/-inf)
        ("masks", OptionType(ArrayType(ArrayType(ArrayType(BooleanType))))),
        # Per-sample positive weights: (n_samples, output_dim)
        ("pos_weights", OptionType(ArrayType(ArrayType(FloatType)))),
        # Per-sample prior values: (n_samples, output_dim)
        ("priors", OptionType(ArrayType(ArrayType(FloatType)))),
    ]
)

# Constrained output configuration
ConstrainedOutputConfigType = StructType(
    [
        ("row_constraints", ArrayType(RowConstraintType)),
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

# Class weight mode type
ClassWeightModeType = VariantType(
    [
        ("balanced", NullType),
    ]
)

# Confusion matrix result type
ConfusionMatrixResultType = StructType(
    [
        ("matrix", MatrixType),  # n_classes x n_classes
        ("classes", IntVectorType),  # class labels
    ]
)

# N-way split configuration (unified for 2-way, 3-way, 4-way, etc.)
SplitConfigType = StructType(
    [
        # Split proportions (must sum to 1.0), e.g., [0.7, 0.15, 0.15] for train/val/test
        ("split_sizes", ArrayType(FloatType)),
        ("random_state", OptionType(IntegerType)),  # default None
        ("shuffle", OptionType(BooleanType)),  # default True
        # Multi-column stratification: each inner array is one column of labels
        # E.g., [[origin1, origin2, ...], [mpf1, mpf2, ...]] stratifies on origin × mpf
        ("stratify", OptionType(ArrayType(ArrayType(IntegerType)))),
        # Minimum samples per compound stratum (default = n_splits)
        ("min_stratify_samples", OptionType(IntegerType)),
        # Columns that must have overlapping values in all splits (but not used for stratification)
        # Samples with values that don't appear in all splits are rejected
        ("overlap", OptionType(ArrayType(ArrayType(IntegerType)))),
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

# Custom curve function: (x: Float, params: Vector, fixed_params: Vector) -> Float
# The params are optimized, fixed_params are passed through unchanged.
CustomCurveFunctionType = FunctionType([FloatType, VectorType, VectorType], FloatType)

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
                    ("fixed_params", OptionType(VectorType)),  # Passed to fn but not optimized
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
        ("gamma", OptionType(FloatType)),  # default 0 (min split loss)
        ("random_state", OptionType(IntegerType)),  # default None
        ("n_jobs", OptionType(IntegerType)),  # default -1
        ("sample_weight", OptionType(VectorType)),  # sample weights (default uniform)
        ("categorical_features", OptionType(IntVectorType)),  # categorical column indices
        ("max_cat_to_onehot", OptionType(IntegerType)),  # default 4
        ("max_cat_threshold", OptionType(IntegerType)),  # default 64
    ]
)

# XGBoost quantile configuration
XGBoostQuantileConfigType = StructType(
    [
        ("quantiles", VectorType),  # quantiles to predict, e.g., [0.1, 0.5, 0.9]
        ("n_estimators", OptionType(IntegerType)),  # default 100
        ("max_depth", OptionType(IntegerType)),  # default 6
        ("learning_rate", OptionType(FloatType)),  # default 0.3
        ("min_child_weight", OptionType(IntegerType)),  # default 1
        ("subsample", OptionType(FloatType)),  # default 1.0
        ("colsample_bytree", OptionType(FloatType)),  # default 1.0
        ("reg_alpha", OptionType(FloatType)),  # default 0 (L1)
        ("reg_lambda", OptionType(FloatType)),  # default 1 (L2)
        ("gamma", OptionType(FloatType)),  # default 0 (min split loss)
        ("random_state", OptionType(IntegerType)),  # default None
        ("n_jobs", OptionType(IntegerType)),  # default -1
        ("sample_weight", OptionType(VectorType)),  # sample weights (default uniform)
        ("categorical_features", OptionType(IntVectorType)),  # categorical column indices
        ("max_cat_to_onehot", OptionType(IntegerType)),  # default 4
        ("max_cat_threshold", OptionType(IntegerType)),  # default 64
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
        ("output_activation", OptionType(TorchOutputActivationType)),  # default none
        ("dropout", OptionType(FloatType)),  # default 0.0
        ("output_dim", OptionType(IntegerType)),  # default 1
        (
            "output_constraints",
            OptionType(ConstrainedOutputConfigType),
        ),  # per-row constraints
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
        (
            "pos_weight",
            OptionType(PosWeightType),
        ),  # for BCE class imbalance (scalar or per-output)
        ("prior", OptionType(PriorConfigType)),  # prior regularization config
        (
            "sample_constraints",
            OptionType(SampleConstraintsConfigType),
        ),  # per-sample constraints
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

# N-way split result (unified)
SplitResultType = StructType(
    [
        # Array of feature matrices, one per split (in order of split_sizes)
        ("X_splits", ArrayType(MatrixType)),
        # Array of target matrices, one per split (in order of split_sizes)
        ("Y_splits", ArrayType(MatrixType)),
        # Indices of rows rejected due to rare stratify classes or missing overlap values
        ("rejected_indices", ArrayType(IntegerType)),
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
        ("mean_error", NullType),  # Bias: mean(pred - true)
        ("pinball_loss", FloatType),  # Quantile loss (param = alpha)
        ("huber", FloatType),  # Robust loss (param = delta)
        ("mean_tweedie_deviance", FloatType),  # Skewed distributions (param = power)
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

# Cohen's Kappa weights type
CohenKappaWeightsType = VariantType(
    [
        ("none", NullType),
        ("linear", NullType),
        ("quadratic", NullType),
    ]
)

# Classification metric variant
ClassificationMetricType = VariantType(
    [
        ("accuracy", NullType),
        ("balanced_accuracy", NullType),
        ("precision", NullType),
        ("recall", NullType),
        ("f1", NullType),
        ("matthews_corrcoef", NullType),
        ("cohen_kappa", CohenKappaWeightsType),
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

# ROC AUC multi-class strategy type
RocAucMultiClassType = VariantType(
    [
        ("ovr", NullType),  # One-vs-rest
        ("ovo", NullType),  # One-vs-one
    ]
)

# ROC AUC configuration type
RocAucConfigType = StructType(
    [
        ("multi_class", OptionType(RocAucMultiClassType)),  # default ovr
        ("average", OptionType(ClassificationAverageType)),  # default macro
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

# Robust statistics result (median-based, outlier-resistant)
RobustStatsResultType = StructType(
    [
        ("median", FloatType),
        ("iqr", FloatType),
        ("mad", FloatType),
        ("q1", FloatType),
        ("q3", FloatType),
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

# XGBoost quantile prediction result
XGBoostQuantilePredictResultType = StructType(
    [
        ("quantiles", VectorType),  # Quantile values that were predicted
        ("predictions", MatrixType),  # Predictions matrix (n_samples x n_quantiles)
    ]
)

# SHAP values variant type - 2D (regression/binary) or 3D (multi-class)
ShapValuesType = VariantType(
    [
        ("matrix_2d", MatrixType),  # 2D for regression/binary (n_samples x n_features)
        (
            "tensor_3d",
            ArrayType(MatrixType),
        ),  # 3D for multi-class (n_samples x n_features x n_classes)
    ]
)

# SHAP base value variant type - single (regression/binary) or per-class (multi-class)
ShapBaseValueType = VariantType(
    [
        ("single", FloatType),  # Single base value for regression/binary
        ("per_class", VectorType),  # Per-class base values for multi-class
    ]
)

# SHAP values result
ShapResultType = StructType(
    [
        ("shap_values", ShapValuesType),  # SHAP values (2D or 3D)
        ("base_value", ShapBaseValueType),  # Base value(s)
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

# ============================================================================
# Lightning Types
# ============================================================================

# Lightning output mode - determines loss function
LightningOutputType = VariantType(
    [
        # Regression: MSE loss, no activation
        ("regression", NullType),
        # Binary: BCE loss, sigmoid activation
        (
            "binary",
            StructType(
                [
                    ("pos_weight", OptionType(FloatType)),
                ]
            ),
        ),
        # Multiclass: CrossEntropy loss, softmax activation
        (
            "multiclass",
            StructType(
                [
                    ("n_classes", IntegerType),
                    ("class_weights", OptionType(VectorType)),
                ]
            ),
        ),
        # Multi-head categorical: N independent CrossEntropy heads
        (
            "multi_head",
            StructType(
                [
                    ("n_heads", IntegerType),
                    ("n_classes_per_head", IntegerType),
                    ("class_weights", OptionType(MatrixType)),  # (n_heads, n_classes)
                ]
            ),
        ),
    ]
)

# Lightning architecture
LightningArchitectureType = VariantType(
    [
        (
            "mlp",
            StructType(
                [
                    ("hidden_layers", ArrayType(IntegerType)),
                ]
            ),
        ),
        (
            "autoencoder",
            StructType(
                [
                    ("encoder_layers", ArrayType(IntegerType)),
                    ("latent_dim", IntegerType),
                    ("decoder_layers", ArrayType(IntegerType)),
                ]
            ),
        ),
    ]
)

# Lightning epoch callback: (epoch, train_loss, val_loss) -> void
LightningEpochCallbackType = FunctionType([IntegerType, FloatType, FloatType], NullType)

# Lightning training configuration
LightningConfigType = StructType(
    [
        ("architecture", LightningArchitectureType),
        ("output", LightningOutputType),
        ("learning_rate", OptionType(FloatType)),  # default 1e-3
        ("max_epochs", OptionType(IntegerType)),  # default 100
        ("patience", OptionType(IntegerType)),  # early stopping, default 10
        ("batch_size", OptionType(IntegerType)),  # default 32
        ("dropout", OptionType(FloatType)),  # default 0.1
        ("gradient_clip", OptionType(FloatType)),  # default 1.0
        ("weight_decay", OptionType(FloatType)),  # L2 regularization, default 0
        ("random_state", OptionType(IntegerType)),  # for reproducibility
        ("epoch_callback", OptionType(LightningEpochCallbackType)),  # called each epoch
    ]
)

# Lightning training result
LightningResultType = StructType(
    [
        ("model", BlobType),  # Serialized model (state_dict + hparams)
        ("train_loss", FloatType),
        ("val_loss", FloatType),
        ("best_epoch", IntegerType),
    ]
)

# Group-based weights for Lightning (binary or multi_head)
GroupWeightsType = StructType(
    [
        # Weights variant - shape depends on output type
        (
            "weights",
            VariantType(
                [
                    # For binary: pos_weight vector per group [n_groups][output_dim]
                    ("binary", ArrayType(ArrayType(FloatType))),
                    # For multi_head: class_weight matrix per group [n_groups][n_heads][n_classes]
                    ("multi_head", ArrayType(ArrayType(ArrayType(FloatType)))),
                ]
            ),
        ),
        # Group index per sample: [n_samples]
        ("sample_groups", ArrayType(IntegerType)),
    ]
)

# Lightning sequence generation configuration
LightningGenerateConfigType = StructType(
    [
        ("n_steps", IntegerType),  # Number of steps to generate
        ("temperature", FloatType),  # 0.0 = argmax, > 0 = scaled sampling
        ("return_probs", BooleanType),  # If true, return probabilities instead of samples
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
        (
            "robust_scaler",
            StructType(
                [
                    ("onnx", BlobType),
                    ("n_features", IntegerType),
                ]
            ),
        ),
        # Sklearn Encoders (cloudpickle serialized)
        (
            "label_encoder",
            StructType(
                [
                    ("data", BlobType),
                    ("n_classes", IntegerType),
                ]
            ),
        ),
        (
            "ordinal_encoder",
            StructType(
                [
                    ("data", BlobType),
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
                    ("categorical_features", OptionType(ArrayType(IntegerType))),
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
                    ("categorical_features", OptionType(ArrayType(IntegerType))),
                ]
            ),
        ),
        # XGBoost quantile regressor (cloudpickle serialized, one model per quantile)
        (
            "xgboost_quantile",
            StructType(
                [
                    ("data", BlobType),
                    ("quantiles", VectorType),
                    ("n_features", IntegerType),
                    ("categorical_features", OptionType(IntVectorType)),
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
        # Lightning models (state_dict + hparams serialized)
        (
            "lightning",
            StructType(
                [
                    ("data", BlobType),  # pickle serialized state_dict + hparams
                    ("n_features", IntegerType),
                    ("output_dim", IntegerType),
                    ("architecture_type", StringType),  # "mlp" or "autoencoder"
                    ("output_type", StringType),  # "regression", "binary", "multiclass", "multi_head"
                    ("latent_dim", OptionType(IntegerType)),  # for autoencoder only
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
    "ClassWeightModeType",
    "ConfusionMatrixResultType",
    "SplitConfigType",
    "SplitResultType",
    # Flexible Metrics Types
    "RegressionMetricType",
    "MetricResultType",
    "MetricsResultType",
    "MetricAggregationType",
    "MultiMetricsConfigType",
    "MultiMetricValueType",
    "MultiMetricResultType",
    "MultiMetricsResultType",
    "CohenKappaWeightsType",
    "ClassificationMetricType",
    "ClassificationAverageType",
    "RocAucMultiClassType",
    "RocAucConfigType",
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
    "XGBoostQuantileConfigType",
    "XGBoostQuantilePredictResultType",
    # LightGBM Types
    "LightGBMConfigType",
    # NGBoost Types
    "NGBoostDistributionType",
    "NGBoostConfigType",
    "NGBoostPredictConfigType",
    "NGBoostPredictResultType",
    # SHAP Types
    "ShapValuesType",
    "ShapBaseValueType",
    "ShapResultType",
    "FeatureImportanceType",
    # Torch Types
    "TorchActivationType",
    "TorchOutputActivationType",
    "TorchLossType",
    "TorchOptimizerType",
    "RowConstraintType",
    "ConstrainedOutputConfigType",
    "PosWeightType",
    "PriorConfigType",
    "SampleConstraintsConfigType",
    "TorchMLPConfigType",
    "TorchTrainConfigType",
    "TorchTrainResultType",
    # Lightning Types
    "LightningOutputType",
    "LightningArchitectureType",
    "LightningEpochCallbackType",
    "LightningConfigType",
    "LightningResultType",
    "GroupWeightsType",
    "LightningGenerateConfigType",
    # RegressorChain Types
    "RegressorChainBaseConfigType",
    "RegressorChainConfigType",
    # GP Types
    "GPKernelType",
    "GPConfigType",
    "GPPredictResultType",
    # Scipy Result Types
    "StatsDescribeResultType",
    "RobustStatsResultType",
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
