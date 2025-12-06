# East-Py Data Science Package Design

## Overview

`east-py-data-science` is a Python package providing platform functions for data science and machine learning operations in the East programming language. It enables East programs to train models, make predictions, perform optimization, and compute feature importance using industry-standard libraries.

## Design Principles

1. **East Type System Compliance**: All inputs/outputs use East types defined in `east-py`
2. **No Generics Workaround**: Use `ArrayType(ArrayType(FloatType))` for matrices and `ArrayType(FloatType)` for vectors
3. **Configuration via Structs**: Use `StructType` with `OptionType` for optional parameters
4. **Sync Operations**: ML operations are computationally intensive but CPU-bound, so use `type="sync"`
5. **Handle-Based Resources**: Models return opaque handles (integers) for stateful operations
6. **Serialization**: Models can be serialized to `BlobType` for persistence

## Package Structure

```
packages/
└── east-py-data-science/
    ├── pyproject.toml
    ├── README.md
    └── east_py_data_science/
        ├── __init__.py           # Main exports, python_data_science_platform
        ├── types.py              # Shared type definitions
        ├── scikit.py             # scikit-learn operations
        ├── xgboost_impl.py       # XGBoost gradient boosting
        ├── lightgbm_impl.py      # LightGBM gradient boosting
        ├── ngboost_impl.py       # NGBoost probabilistic predictions
        ├── optuna_impl.py        # Hyperparameter optimization
        ├── shap_impl.py          # Feature importance/explainability
        ├── scipy_impl.py         # Scientific computing utilities
        ├── torch_impl.py         # PyTorch neural networks
        └── gp_impl.py            # Gaussian Process regression
```

## Dependencies

```toml
[project]
dependencies = [
    "east-py",
    # Core ML
    "scikit-learn>=1.3.0",
    "scipy>=1.11.0",
    # Gradient boosting
    "xgboost>=2.0.0",
    "lightgbm>=4.0.0",
    "ngboost>=0.5.0",
    # Hyperparameter optimization
    "optuna>=3.0.0",
    # Explainability
    "shap>=0.42.0",
    # Deep learning (optional)
    "torch>=2.0.0",
    # Gaussian processes (optional)
    "gpflow>=2.9.0",
]
```

---

## Module 1: Shared Types (`types.py`)

### Matrix and Vector Types

```python
from east.types.types import (
    ArrayType, FloatType, IntegerType, StringType, BooleanType,
    StructType, OptionType, VariantType, BlobType, NullType
)

# Core data types
VectorType = ArrayType(FloatType)           # 1D array of floats
MatrixType = ArrayType(ArrayType(FloatType))  # 2D array of floats
LabelVectorType = ArrayType(IntegerType)    # Classification labels
StringVectorType = ArrayType(StringType)    # Feature names

# Model handle (opaque reference to trained model)
ModelHandleType = IntegerType

# Prediction result with optional uncertainty
PredictionResultType = StructType([
    ("predictions", VectorType),
    ("std", OptionType(VectorType)),         # Standard deviation (for probabilistic)
    ("lower", OptionType(VectorType)),       # Lower confidence bound
    ("upper", OptionType(VectorType)),       # Upper confidence bound
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

# Hyperparameter value (can be int, float, string, or bool)
HyperparamValueType = VariantType([
    ("int", IntegerType),
    ("float", FloatType),
    ("string", StringType),
    ("bool", BooleanType),
])

# =============================================================================
# Enum Types (Variant with NullType values - like enums)
# =============================================================================

# Hyperparameter space type
HyperparamSpaceKindType = VariantType([
    ("int", NullType),
    ("float", NullType),
    ("categorical", NullType),
    ("log_uniform", NullType),
])

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

# =============================================================================
# Compound Types
# =============================================================================

# Hyperparameter search space definition
HyperparamSpaceType = StructType([
    ("name", StringType),
    ("kind", HyperparamSpaceKindType),
    ("low", OptionType(FloatType)),
    ("high", OptionType(FloatType)),
    ("choices", OptionType(ArrayType(HyperparamValueType))),
])

# Optimization trial result
TrialResultType = StructType([
    ("trial_id", IntegerType),
    ("params", ArrayType(StructType([("name", StringType), ("value", HyperparamValueType)]))),
    ("score", FloatType),
])

# Optimization study result
StudyResultType = StructType([
    ("best_params", ArrayType(StructType([("name", StringType), ("value", HyperparamValueType)]))),
    ("best_score", FloatType),
    ("trials", ArrayType(TrialResultType)),
])
```

---

## Module 2: Scikit-Learn (`scikit.py`)

### Purpose
Core machine learning utilities: preprocessing, model selection, metrics.

### Platform Functions

#### `sklearn_train_test_split`
Split data into train/test sets.

```python
# Type Definition
SklearnSplitConfigType = StructType([
    ("test_size", OptionType(FloatType)),      # default 0.2
    ("random_state", OptionType(IntegerType)), # default None
    ("shuffle", OptionType(BooleanType)),      # default True
])

# PlatformFunction
PlatformFunction(
    name="sklearn_train_test_split",
    inputs=[MatrixType, VectorType, SklearnSplitConfigType],
    output=StructType([
        ("X_train", MatrixType),
        ("X_test", MatrixType),
        ("y_train", VectorType),
        ("y_test", VectorType),
    ]),
    type="sync",
    fn=sklearn_train_test_split_impl,
)

# Implementation
def sklearn_train_test_split_impl(
    X: list[list[float]],
    y: list[float],
    config: dict
) -> dict:
    """Split arrays into train and test subsets."""
    from sklearn.model_selection import train_test_split

    test_size = _get_option(config.get("test_size"), 0.2)
    random_state = _get_option(config.get("random_state"), None)
    shuffle = _get_option(config.get("shuffle"), True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=shuffle
    )

    return {
        "X_train": [list(row) for row in X_train],
        "X_test": [list(row) for row in X_test],
        "y_train": list(y_train),
        "y_test": list(y_test),
    }
```

#### `sklearn_standard_scaler_fit`
Fit a standard scaler to training data.

```python
PlatformFunction(
    name="sklearn_standard_scaler_fit",
    inputs=[MatrixType],
    output=ModelHandleType,
    type="sync",
    fn=sklearn_standard_scaler_fit_impl,
)

# Implementation stores scaler in global registry, returns handle
def sklearn_standard_scaler_fit_impl(X: list[list[float]]) -> int:
    """Fit StandardScaler and return handle."""
    from sklearn.preprocessing import StandardScaler
    import numpy as np

    scaler = StandardScaler()
    scaler.fit(np.array(X))

    handle = _register_model(scaler)
    return handle
```

#### `sklearn_standard_scaler_transform`
Transform data using fitted scaler.

```python
PlatformFunction(
    name="sklearn_standard_scaler_transform",
    inputs=[ModelHandleType, MatrixType],
    output=MatrixType,
    type="sync",
    fn=sklearn_standard_scaler_transform_impl,
)

def sklearn_standard_scaler_transform_impl(handle: int, X: list[list[float]]) -> list[list[float]]:
    """Transform data using fitted scaler."""
    import numpy as np

    scaler = _get_model(handle)
    X_scaled = scaler.transform(np.array(X))
    return [list(row) for row in X_scaled]
```

#### `sklearn_cross_val_score`
Perform k-fold cross-validation.

```python
SklearnCrossValConfigType = StructType([
    ("cv", OptionType(IntegerType)),           # default 5
    ("scoring", OptionType(ScoringMetricType)), # default neg_mean_squared_error
    ("shuffle", OptionType(BooleanType)),      # default True
    ("random_state", OptionType(IntegerType)), # default None
])

PlatformFunction(
    name="sklearn_cross_val_score",
    inputs=[ModelHandleType, MatrixType, VectorType, SklearnCrossValConfigType],
    output=CrossValResultType,
    type="sync",
    fn=sklearn_cross_val_score_impl,
)

def sklearn_cross_val_score_impl(
    handle: int,
    X: list[list[float]],
    y: list[float],
    config: dict
) -> dict:
    """Perform cross-validation and return scores."""
    from sklearn.model_selection import cross_val_score, KFold
    import numpy as np

    model = _get_model(handle)
    cv = _get_option(config.get("cv"), 5)
    scoring = _get_option(config.get("scoring"), "neg_mean_squared_error")
    shuffle = _get_option(config.get("shuffle"), True)
    random_state = _get_option(config.get("random_state"), None)

    kfold = KFold(n_splits=cv, shuffle=shuffle, random_state=random_state)
    scores = cross_val_score(model, np.array(X), np.array(y), cv=kfold, scoring=scoring)

    return {
        "scores": list(scores),
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
    }
```

#### `sklearn_metrics_regression`
Compute regression metrics.

```python
PlatformFunction(
    name="sklearn_metrics_regression",
    inputs=[VectorType, VectorType],
    output=StructType([
        ("mse", FloatType),
        ("rmse", FloatType),
        ("mae", FloatType),
        ("r2", FloatType),
        ("mape", FloatType),
    ]),
    type="sync",
    fn=sklearn_metrics_regression_impl,
)

def sklearn_metrics_regression_impl(y_true: list[float], y_pred: list[float]) -> dict:
    """Compute regression metrics."""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import numpy as np

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # MAPE (avoid division by zero)
    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) if mask.any() else 0.0

    return {
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mae),
        "r2": float(r2),
        "mape": mape,
    }
```

#### `sklearn_metrics_classification`
Compute classification metrics.

```python
PlatformFunction(
    name="sklearn_metrics_classification",
    inputs=[LabelVectorType, LabelVectorType],
    output=StructType([
        ("accuracy", FloatType),
        ("precision", FloatType),
        ("recall", FloatType),
        ("f1", FloatType),
    ]),
    type="sync",
    fn=sklearn_metrics_classification_impl,
)

def sklearn_metrics_classification_impl(y_true: list[int], y_pred: list[int]) -> dict:
    """Compute classification metrics."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
```

---

## Module 3: XGBoost (`xgboost_impl.py`)

### Purpose
Gradient boosting for regression and classification with XGBoost.

### Platform Functions

#### `xgboost_train_regressor`
Train XGBoost regression model.

```python
XGBoostRegressorConfigType = StructType([
    ("n_estimators", OptionType(IntegerType)),      # default 100
    ("max_depth", OptionType(IntegerType)),         # default 6
    ("learning_rate", OptionType(FloatType)),       # default 0.3
    ("min_child_weight", OptionType(IntegerType)),  # default 1
    ("subsample", OptionType(FloatType)),           # default 1.0
    ("colsample_bytree", OptionType(FloatType)),    # default 1.0
    ("reg_alpha", OptionType(FloatType)),           # default 0 (L1)
    ("reg_lambda", OptionType(FloatType)),          # default 1 (L2)
    ("random_state", OptionType(IntegerType)),      # default None
    ("n_jobs", OptionType(IntegerType)),            # default -1
])

PlatformFunction(
    name="xgboost_train_regressor",
    inputs=[MatrixType, VectorType, XGBoostRegressorConfigType],
    output=ModelHandleType,
    type="sync",
    fn=xgboost_train_regressor_impl,
)

def xgboost_train_regressor_impl(
    X: list[list[float]],
    y: list[float],
    config: dict
) -> int:
    """Train XGBoost regressor and return model handle."""
    import xgboost as xgb
    import numpy as np

    model = xgb.XGBRegressor(
        n_estimators=_get_option(config.get("n_estimators"), 100),
        max_depth=_get_option(config.get("max_depth"), 6),
        learning_rate=_get_option(config.get("learning_rate"), 0.3),
        min_child_weight=_get_option(config.get("min_child_weight"), 1),
        subsample=_get_option(config.get("subsample"), 1.0),
        colsample_bytree=_get_option(config.get("colsample_bytree"), 1.0),
        reg_alpha=_get_option(config.get("reg_alpha"), 0.0),
        reg_lambda=_get_option(config.get("reg_lambda"), 1.0),
        random_state=_get_option(config.get("random_state"), None),
        n_jobs=_get_option(config.get("n_jobs"), -1),
    )

    model.fit(np.array(X), np.array(y))
    return _register_model(model)
```

#### `xgboost_train_classifier`
Train XGBoost classification model.

```python
XGBoostClassifierConfigType = StructType([
    ("n_estimators", OptionType(IntegerType)),
    ("max_depth", OptionType(IntegerType)),
    ("learning_rate", OptionType(FloatType)),
    ("min_child_weight", OptionType(IntegerType)),
    ("subsample", OptionType(FloatType)),
    ("colsample_bytree", OptionType(FloatType)),
    ("reg_alpha", OptionType(FloatType)),
    ("reg_lambda", OptionType(FloatType)),
    ("random_state", OptionType(IntegerType)),
    ("n_jobs", OptionType(IntegerType)),
])

PlatformFunction(
    name="xgboost_train_classifier",
    inputs=[MatrixType, LabelVectorType, XGBoostClassifierConfigType],
    output=ModelHandleType,
    type="sync",
    fn=xgboost_train_classifier_impl,
)

def xgboost_train_classifier_impl(
    X: list[list[float]],
    y: list[int],
    config: dict
) -> int:
    """Train XGBoost classifier and return model handle."""
    import xgboost as xgb
    import numpy as np

    model = xgb.XGBClassifier(
        n_estimators=_get_option(config.get("n_estimators"), 100),
        max_depth=_get_option(config.get("max_depth"), 6),
        learning_rate=_get_option(config.get("learning_rate"), 0.3),
        # ... same params as regressor
        use_label_encoder=False,
        eval_metric="logloss",
    )

    model.fit(np.array(X), np.array(y))
    return _register_model(model)
```

#### `xgboost_predict`
Make predictions with trained XGBoost model.

```python
PlatformFunction(
    name="xgboost_predict",
    inputs=[ModelHandleType, MatrixType],
    output=VectorType,
    type="sync",
    fn=xgboost_predict_impl,
)

def xgboost_predict_impl(handle: int, X: list[list[float]]) -> list[float]:
    """Make predictions with XGBoost model."""
    import numpy as np

    model = _get_model(handle)
    predictions = model.predict(np.array(X))
    return [float(p) for p in predictions]
```

#### `xgboost_predict_proba`
Get class probabilities for classification.

```python
PlatformFunction(
    name="xgboost_predict_proba",
    inputs=[ModelHandleType, MatrixType],
    output=MatrixType,  # n_samples x n_classes
    type="sync",
    fn=xgboost_predict_proba_impl,
)

def xgboost_predict_proba_impl(handle: int, X: list[list[float]]) -> list[list[float]]:
    """Get class probabilities from XGBoost classifier."""
    import numpy as np

    model = _get_model(handle)
    proba = model.predict_proba(np.array(X))
    return [list(row) for row in proba]
```

#### `xgboost_feature_importance`
Get feature importance from trained model.

```python
PlatformFunction(
    name="xgboost_feature_importance",
    inputs=[ModelHandleType, StringVectorType],  # model, feature_names
    output=FeatureImportanceType,
    type="sync",
    fn=xgboost_feature_importance_impl,
)

def xgboost_feature_importance_impl(handle: int, feature_names: list[str]) -> dict:
    """Get feature importance from XGBoost model."""
    model = _get_model(handle)
    importances = model.feature_importances_

    return {
        "feature_names": feature_names,
        "importances": [float(i) for i in importances],
        "std": None,  # XGBoost doesn't provide std
    }
```

#### `xgboost_save` / `xgboost_load`
Serialize/deserialize model to blob.

```python
PlatformFunction(
    name="xgboost_save",
    inputs=[ModelHandleType],
    output=BlobType,
    type="sync",
    fn=xgboost_save_impl,
)

def xgboost_save_impl(handle: int) -> bytes:
    """Serialize XGBoost model to bytes."""
    import io
    import pickle

    model = _get_model(handle)
    buffer = io.BytesIO()
    pickle.dump(model, buffer)
    return buffer.getvalue()

PlatformFunction(
    name="xgboost_load",
    inputs=[BlobType],
    output=ModelHandleType,
    type="sync",
    fn=xgboost_load_impl,
)

def xgboost_load_impl(data: bytes) -> int:
    """Load XGBoost model from bytes."""
    import io
    import pickle

    buffer = io.BytesIO(data)
    model = pickle.load(buffer)
    return _register_model(model)
```

---

## Module 4: LightGBM (`lightgbm_impl.py`)

### Purpose
Fast gradient boosting with LightGBM (better for large datasets).

### Platform Functions

#### `lightgbm_train_regressor`

```python
LightGBMRegressorConfigType = StructType([
    ("n_estimators", OptionType(IntegerType)),      # default 100
    ("max_depth", OptionType(IntegerType)),         # default -1 (unlimited)
    ("learning_rate", OptionType(FloatType)),       # default 0.1
    ("num_leaves", OptionType(IntegerType)),        # default 31
    ("min_child_samples", OptionType(IntegerType)), # default 20
    ("subsample", OptionType(FloatType)),           # default 1.0
    ("colsample_bytree", OptionType(FloatType)),    # default 1.0
    ("reg_alpha", OptionType(FloatType)),           # default 0
    ("reg_lambda", OptionType(FloatType)),          # default 0
    ("random_state", OptionType(IntegerType)),      # default None
    ("n_jobs", OptionType(IntegerType)),            # default -1
])

PlatformFunction(
    name="lightgbm_train_regressor",
    inputs=[MatrixType, VectorType, LightGBMRegressorConfigType],
    output=ModelHandleType,
    type="sync",
    fn=lightgbm_train_regressor_impl,
)

def lightgbm_train_regressor_impl(
    X: list[list[float]],
    y: list[float],
    config: dict
) -> int:
    """Train LightGBM regressor and return model handle."""
    import lightgbm as lgb
    import numpy as np

    model = lgb.LGBMRegressor(
        n_estimators=_get_option(config.get("n_estimators"), 100),
        max_depth=_get_option(config.get("max_depth"), -1),
        learning_rate=_get_option(config.get("learning_rate"), 0.1),
        num_leaves=_get_option(config.get("num_leaves"), 31),
        min_child_samples=_get_option(config.get("min_child_samples"), 20),
        subsample=_get_option(config.get("subsample"), 1.0),
        colsample_bytree=_get_option(config.get("colsample_bytree"), 1.0),
        reg_alpha=_get_option(config.get("reg_alpha"), 0.0),
        reg_lambda=_get_option(config.get("reg_lambda"), 0.0),
        random_state=_get_option(config.get("random_state"), None),
        n_jobs=_get_option(config.get("n_jobs"), -1),
        verbose=-1,
    )

    model.fit(np.array(X), np.array(y))
    return _register_model(model)
```

#### `lightgbm_train_classifier`

```python
PlatformFunction(
    name="lightgbm_train_classifier",
    inputs=[MatrixType, LabelVectorType, LightGBMClassifierConfigType],
    output=ModelHandleType,
    type="sync",
    fn=lightgbm_train_classifier_impl,
)
```

#### `lightgbm_predict`

```python
PlatformFunction(
    name="lightgbm_predict",
    inputs=[ModelHandleType, MatrixType],
    output=VectorType,
    type="sync",
    fn=lightgbm_predict_impl,
)
```

#### `lightgbm_predict_proba`

```python
PlatformFunction(
    name="lightgbm_predict_proba",
    inputs=[ModelHandleType, MatrixType],
    output=MatrixType,
    type="sync",
    fn=lightgbm_predict_proba_impl,
)
```

#### `lightgbm_feature_importance`

```python
PlatformFunction(
    name="lightgbm_feature_importance",
    inputs=[ModelHandleType, StringVectorType],
    output=FeatureImportanceType,
    type="sync",
    fn=lightgbm_feature_importance_impl,
)
```

#### `lightgbm_save` / `lightgbm_load`

```python
PlatformFunction(
    name="lightgbm_save",
    inputs=[ModelHandleType],
    output=BlobType,
    type="sync",
    fn=lightgbm_save_impl,
)

PlatformFunction(
    name="lightgbm_load",
    inputs=[BlobType],
    output=ModelHandleType,
    type="sync",
    fn=lightgbm_load_impl,
)
```

---

## Module 5: NGBoost (`ngboost_impl.py`)

### Purpose
Probabilistic predictions with uncertainty quantification using NGBoost.

### Platform Functions

#### `ngboost_train_regressor`
Train NGBoost model with natural gradient boosting.

```python
NGBoostRegressorConfigType = StructType([
    ("n_estimators", OptionType(IntegerType)),      # default 500
    ("learning_rate", OptionType(FloatType)),       # default 0.01
    ("minibatch_frac", OptionType(FloatType)),      # default 1.0
    ("col_sample", OptionType(FloatType)),          # default 1.0
    ("random_state", OptionType(IntegerType)),      # default None
    ("distribution", OptionType(DistributionType)), # default normal
])

PlatformFunction(
    name="ngboost_train_regressor",
    inputs=[MatrixType, VectorType, NGBoostRegressorConfigType],
    output=ModelHandleType,
    type="sync",
    fn=ngboost_train_regressor_impl,
)

def ngboost_train_regressor_impl(
    X: list[list[float]],
    y: list[float],
    config: dict
) -> int:
    """Train NGBoost regressor for probabilistic predictions."""
    from ngboost import NGBRegressor
    from ngboost.distns import Normal, LogNormal
    import numpy as np

    dist_name = _get_option(config.get("distribution"), "normal")
    dist = Normal if dist_name == "normal" else LogNormal

    model = NGBRegressor(
        Dist=dist,
        n_estimators=_get_option(config.get("n_estimators"), 500),
        learning_rate=_get_option(config.get("learning_rate"), 0.01),
        minibatch_frac=_get_option(config.get("minibatch_frac"), 1.0),
        col_sample=_get_option(config.get("col_sample"), 1.0),
        random_state=_get_option(config.get("random_state"), None),
        verbose=False,
    )

    model.fit(np.array(X), np.array(y))
    return _register_model(model)
```

#### `ngboost_predict`
Get point predictions.

```python
PlatformFunction(
    name="ngboost_predict",
    inputs=[ModelHandleType, MatrixType],
    output=VectorType,
    type="sync",
    fn=ngboost_predict_impl,
)
```

#### `ngboost_predict_dist`
Get full predictive distribution (mean, std, confidence intervals).

```python
NGBoostPredictConfigType = StructType([
    ("confidence_level", OptionType(FloatType)),  # default 0.95
])

PlatformFunction(
    name="ngboost_predict_dist",
    inputs=[ModelHandleType, MatrixType, NGBoostPredictConfigType],
    output=PredictionResultType,
    type="sync",
    fn=ngboost_predict_dist_impl,
)

def ngboost_predict_dist_impl(
    handle: int,
    X: list[list[float]],
    config: dict
) -> dict:
    """Get predictions with uncertainty from NGBoost."""
    import numpy as np
    from scipy import stats

    model = _get_model(handle)
    X_arr = np.array(X)

    # Get distribution predictions
    dist_pred = model.pred_dist(X_arr)

    # Extract mean and std
    loc = dist_pred.loc  # mean
    scale = dist_pred.scale  # std

    # Compute confidence intervals
    confidence = _get_option(config.get("confidence_level"), 0.95)
    alpha = 1 - confidence
    z = stats.norm.ppf(1 - alpha / 2)

    lower = loc - z * scale
    upper = loc + z * scale

    return EastStruct({
        "predictions": EastArray(FloatType, [float(p) for p in loc]),
        "std": EastSome(EastArray(FloatType, [float(s) for s in scale])),
        "lower": EastSome(EastArray(FloatType, [float(l) for l in lower])),
        "upper": EastSome(EastArray(FloatType, [float(u) for u in upper])),
    })
```

#### `ngboost_save` / `ngboost_load`

```python
PlatformFunction(
    name="ngboost_save",
    inputs=[ModelHandleType],
    output=BlobType,
    type="sync",
    fn=ngboost_save_impl,
)

PlatformFunction(
    name="ngboost_load",
    inputs=[BlobType],
    output=ModelHandleType,
    type="sync",
    fn=ngboost_load_impl,
)
```

---

## Module 6: Optuna (`optuna_impl.py`)

### Purpose
Hyperparameter optimization using Optuna's TPE sampler.

### Platform Functions

#### `optuna_create_study`
Create a new optimization study.

```python
OptunaStudyConfigType = StructType([
    ("direction", OptionType(OptimizationDirectionType)), # default minimize
    ("random_state", OptionType(IntegerType)),     # default None
    ("pruner", OptionType(PrunerType)),             # default none
])

PlatformFunction(
    name="optuna_create_study",
    inputs=[OptunaStudyConfigType],
    output=ModelHandleType,  # Study handle
    type="sync",
    fn=optuna_create_study_impl,
)

def optuna_create_study_impl(config: dict) -> int:
    """Create Optuna study for hyperparameter optimization."""
    import optuna

    direction = _get_option(config.get("direction"), "minimize")
    random_state = _get_option(config.get("random_state"), None)
    pruner_name = _get_option(config.get("pruner"), "none")

    if pruner_name == "median":
        pruner = optuna.pruners.MedianPruner()
    elif pruner_name == "hyperband":
        pruner = optuna.pruners.HyperbandPruner()
    else:
        pruner = optuna.pruners.NopPruner()

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction=direction, sampler=sampler, pruner=pruner)

    return _register_model(study)
```

#### `optuna_suggest_params`
Suggest hyperparameters for a trial based on search space.

```python
PlatformFunction(
    name="optuna_suggest_params",
    inputs=[
        ModelHandleType,  # study handle
        IntegerType,      # trial number
        ArrayType(HyperparamSpaceType),  # search space
    ],
    output=ArrayType(StructType([("name", StringType), ("value", HyperparamValueType)])),
    type="sync",
    fn=optuna_suggest_params_impl,
)

def optuna_suggest_params_impl(
    study_handle: int,
    trial_num: int,
    space: list[dict]
) -> list[dict]:
    """Suggest hyperparameters for a trial."""
    import optuna

    study = _get_model(study_handle)
    trial = study.ask()  # Get next trial

    params = []
    for param_def in space:
        name = param_def["name"]
        param_type = param_def["type"]

        if param_type == "int":
            low = int(_get_option(param_def.get("low"), 1))
            high = int(_get_option(param_def.get("high"), 100))
            value = trial.suggest_int(name, low, high)
            params.append({"name": name, "value": {"type": "int", "value": value}})

        elif param_type == "float":
            low = _get_option(param_def.get("low"), 0.0)
            high = _get_option(param_def.get("high"), 1.0)
            value = trial.suggest_float(name, low, high)
            params.append({"name": name, "value": {"type": "float", "value": value}})

        elif param_type == "log_uniform":
            low = _get_option(param_def.get("low"), 1e-6)
            high = _get_option(param_def.get("high"), 1.0)
            value = trial.suggest_float(name, low, high, log=True)
            params.append({"name": name, "value": {"type": "float", "value": value}})

        elif param_type == "categorical":
            choices = _get_option(param_def.get("choices"), [])
            # Convert HyperparamValueType to Python values
            py_choices = [_hyperparamvalue_to_py(c) for c in choices]
            value = trial.suggest_categorical(name, py_choices)
            params.append({"name": name, "value": _py_to_hyperparamvalue(value)})

    # Store trial for later completion
    _register_model(trial, key=f"trial_{study_handle}_{trial_num}")

    return params
```

#### `optuna_complete_trial`
Report trial result.

```python
PlatformFunction(
    name="optuna_complete_trial",
    inputs=[ModelHandleType, IntegerType, FloatType],  # study, trial_num, score
    output=NullType,
    type="sync",
    fn=optuna_complete_trial_impl,
)

def optuna_complete_trial_impl(study_handle: int, trial_num: int, score: float) -> None:
    """Complete a trial with its score."""
    study = _get_model(study_handle)
    trial = _get_model(f"trial_{study_handle}_{trial_num}")
    study.tell(trial, score)
```

#### `optuna_get_best`
Get best hyperparameters from study.

```python
PlatformFunction(
    name="optuna_get_best",
    inputs=[ModelHandleType],
    output=StudyResultType,
    type="sync",
    fn=optuna_get_best_impl,
)

def optuna_get_best_impl(study_handle: int) -> dict:
    """Get best trial results from study."""
    study = _get_model(study_handle)

    # Best params
    best_params = [
        {"name": k, "value": _py_to_hyperparamvalue(v)}
        for k, v in study.best_params.items()
    ]

    # All trials
    trials = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            trial_params = [
                {"name": k, "value": _py_to_hyperparamvalue(v)}
                for k, v in trial.params.items()
            ]
            trials.append({
                "trial_id": trial.number,
                "params": trial_params,
                "score": trial.value,
            })

    return {
        "best_params": best_params,
        "best_score": study.best_value,
        "trials": trials,
    }
```

---

## Module 7: SHAP (`shap_impl.py`)

### Purpose
Model-agnostic feature importance and explainability using SHAP values.

### Platform Functions

#### `shap_tree_explainer_create`
Create TreeExplainer for tree-based models.

```python
PlatformFunction(
    name="shap_tree_explainer_create",
    inputs=[ModelHandleType],
    output=ModelHandleType,  # Explainer handle
    type="sync",
    fn=shap_tree_explainer_create_impl,
)

def shap_tree_explainer_create_impl(model_handle: int) -> int:
    """Create SHAP TreeExplainer for tree-based models."""
    import shap

    model = _get_model(model_handle)
    explainer = shap.TreeExplainer(model)
    return _register_model(explainer)
```

#### `shap_compute_values`
Compute SHAP values for samples.

```python
ShapResultType = StructType([
    ("shap_values", MatrixType),         # n_samples x n_features
    ("base_value", FloatType),           # Expected value
    ("feature_names", StringVectorType),
])

PlatformFunction(
    name="shap_compute_values",
    inputs=[ModelHandleType, MatrixType, StringVectorType],  # explainer, X, feature_names
    output=ShapResultType,
    type="sync",
    fn=shap_compute_values_impl,
)

def shap_compute_values_impl(
    explainer_handle: int,
    X: list[list[float]],
    feature_names: list[str]
) -> dict:
    """Compute SHAP values for samples."""
    import numpy as np

    explainer = _get_model(explainer_handle)
    X_arr = np.array(X)

    shap_values = explainer.shap_values(X_arr)

    # Handle multi-output (classification)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Take positive class for binary

    return {
        "shap_values": [list(row) for row in shap_values],
        "base_value": float(explainer.expected_value) if np.isscalar(explainer.expected_value)
                      else float(explainer.expected_value[1]),
        "feature_names": feature_names,
    }
```

#### `shap_feature_importance`
Get global feature importance from SHAP values.

```python
PlatformFunction(
    name="shap_feature_importance",
    inputs=[MatrixType, StringVectorType],  # shap_values, feature_names
    output=FeatureImportanceType,
    type="sync",
    fn=shap_feature_importance_impl,
)

def shap_feature_importance_impl(
    shap_values: EastArray,      # EastArray[EastArray[float]] - Matrix
    feature_names: EastArray     # EastArray[str]
) -> EastStruct:
    """Compute global feature importance from SHAP values."""
    import numpy as np

    # Convert EastArray to numpy
    shap_list = [[x for x in row] for row in shap_values]
    shap_arr = np.array(shap_list)
    mean_abs_shap = np.abs(shap_arr).mean(axis=0)
    std_shap = np.abs(shap_arr).std(axis=0)

    return EastStruct({
        "feature_names": EastArray(StringType, list(feature_names)),
        "importances": EastArray(FloatType, [float(i) for i in mean_abs_shap]),
        "std": EastSome(EastArray(FloatType, [float(s) for s in std_shap])),
    })
```

#### `shap_kernel_explainer_create`
Create KernelExplainer for any model (model-agnostic).

```python
PlatformFunction(
    name="shap_kernel_explainer_create",
    inputs=[ModelHandleType, MatrixType],  # model, background_data
    output=ModelHandleType,
    type="sync",
    fn=shap_kernel_explainer_create_impl,
)

def shap_kernel_explainer_create_impl(model_handle: int, background: list[list[float]]) -> int:
    """Create SHAP KernelExplainer for any model."""
    import shap
    import numpy as np

    model = _get_model(model_handle)
    explainer = shap.KernelExplainer(model.predict, np.array(background))
    return _register_model(explainer)
```

---

## Module 8: SciPy (`scipy_impl.py`)

### Purpose
Scientific computing utilities: statistics, optimization, interpolation.

### Platform Functions

#### `scipy_stats_describe`
Compute descriptive statistics.

```python
StatsDescribeResultType = StructType([
    ("count", IntegerType),
    ("mean", FloatType),
    ("variance", FloatType),
    ("skewness", FloatType),
    ("kurtosis", FloatType),
    ("min", FloatType),
    ("max", FloatType),
])

PlatformFunction(
    name="scipy_stats_describe",
    inputs=[VectorType],
    output=StatsDescribeResultType,
    type="sync",
    fn=scipy_stats_describe_impl,
)

def scipy_stats_describe_impl(data: list[float]) -> dict:
    """Compute descriptive statistics for data."""
    from scipy import stats
    import numpy as np

    result = stats.describe(data)

    return {
        "count": int(result.nobs),
        "mean": float(result.mean),
        "variance": float(result.variance),
        "skewness": float(result.skewness),
        "kurtosis": float(result.kurtosis),
        "min": float(result.minmax[0]),
        "max": float(result.minmax[1]),
    }
```

#### `scipy_stats_pearsonr`
Compute Pearson correlation coefficient.

```python
PlatformFunction(
    name="scipy_stats_pearsonr",
    inputs=[VectorType, VectorType],
    output=StructType([
        ("correlation", FloatType),
        ("pvalue", FloatType),
    ]),
    type="sync",
    fn=scipy_stats_pearsonr_impl,
)

def scipy_stats_pearsonr_impl(x: list[float], y: list[float]) -> dict:
    """Compute Pearson correlation coefficient."""
    from scipy import stats

    r, p = stats.pearsonr(x, y)
    return {"correlation": float(r), "pvalue": float(p)}
```

#### `scipy_stats_spearmanr`
Compute Spearman rank correlation.

```python
PlatformFunction(
    name="scipy_stats_spearmanr",
    inputs=[VectorType, VectorType],
    output=StructType([
        ("correlation", FloatType),
        ("pvalue", FloatType),
    ]),
    type="sync",
    fn=scipy_stats_spearmanr_impl,
)
```

#### `scipy_optimize_minimize`
Minimize a scalar function.

```python
ScipyMinimizeConfigType = StructType([
    ("method", OptionType(OptimizeMethodType)),     # default l_bfgs_b
    ("max_iter", OptionType(IntegerType)),         # default 1000
    ("tol", OptionType(FloatType)),                # default 1e-6
])

ScipyMinimizeResultType = StructType([
    ("x", VectorType),           # Optimal parameters
    ("fun", FloatType),          # Function value at optimum
    ("success", BooleanType),    # Whether optimization succeeded
    ("nit", IntegerType),        # Number of iterations
])

PlatformFunction(
    name="scipy_optimize_minimize",
    inputs=[
        ModelHandleType,         # Objective function handle
        VectorType,              # Initial guess x0
        ScipyMinimizeConfigType,
    ],
    output=ScipyMinimizeResultType,
    type="sync",
    fn=scipy_optimize_minimize_impl,
)

def scipy_optimize_minimize_impl(
    objective_handle: int,
    x0: list[float],
    config: dict
) -> dict:
    """Minimize a scalar function."""
    from scipy import optimize
    import numpy as np

    objective_fn = _get_model(objective_handle)

    result = optimize.minimize(
        objective_fn,
        np.array(x0),
        method=_get_option(config.get("method"), "L-BFGS-B"),
        options={
            "maxiter": _get_option(config.get("max_iter"), 1000),
        },
        tol=_get_option(config.get("tol"), 1e-6),
    )

    return {
        "x": list(result.x),
        "fun": float(result.fun),
        "success": bool(result.success),
        "nit": int(result.nit),
    }
```

#### `scipy_interpolate_1d`
1D interpolation.

```python
Scipy1DInterpolateConfigType = StructType([
    ("kind", OptionType(InterpolationKindType)),  # default linear
])

PlatformFunction(
    name="scipy_interpolate_1d_fit",
    inputs=[VectorType, VectorType, Scipy1DInterpolateConfigType],  # x, y, config
    output=ModelHandleType,
    type="sync",
    fn=scipy_interpolate_1d_fit_impl,
)

PlatformFunction(
    name="scipy_interpolate_1d_predict",
    inputs=[ModelHandleType, VectorType],
    output=VectorType,
    type="sync",
    fn=scipy_interpolate_1d_predict_impl,
)
```

---

## Module 9: PyTorch (`torch_impl.py`)

### Purpose
Neural network models using PyTorch.

### Platform Functions

#### `torch_mlp_create`
Create Multi-Layer Perceptron model.

```python
TorchMLPConfigType = StructType([
    ("hidden_layers", ArrayType(IntegerType)),     # e.g., [64, 32]
    ("activation", OptionType(ActivationFunctionType)), # default relu
    ("dropout", OptionType(FloatType)),             # default 0.0
    ("output_dim", OptionType(IntegerType)),        # default 1 (regression)
])

PlatformFunction(
    name="torch_mlp_create",
    inputs=[IntegerType, TorchMLPConfigType],  # input_dim, config
    output=ModelHandleType,
    type="sync",
    fn=torch_mlp_create_impl,
)

def torch_mlp_create_impl(input_dim: int, config: dict) -> int:
    """Create PyTorch MLP model."""
    import torch
    import torch.nn as nn

    hidden_layers = config.get("hidden_layers", [64, 32])
    activation_name = _get_option(config.get("activation"), "relu")
    dropout = _get_option(config.get("dropout"), 0.0)
    output_dim = _get_option(config.get("output_dim"), 1)

    activation = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
    }[activation_name]

    layers = []
    prev_dim = input_dim
    for hidden_dim in hidden_layers:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(activation())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev_dim = hidden_dim

    layers.append(nn.Linear(prev_dim, output_dim))

    model = nn.Sequential(*layers)
    return _register_model(model)
```

#### `torch_train`
Train PyTorch model.

```python
TorchTrainConfigType = StructType([
    ("epochs", OptionType(IntegerType)),           # default 100
    ("batch_size", OptionType(IntegerType)),       # default 32
    ("learning_rate", OptionType(FloatType)),      # default 0.001
    ("loss", OptionType(LossFunctionType)),         # default mse
    ("optimizer", OptionType(OptimizerType)),      # default adam
    ("early_stopping", OptionType(IntegerType)),   # patience, 0 = disabled
    ("validation_split", OptionType(FloatType)),   # default 0.2
])

TorchTrainResultType = StructType([
    ("train_losses", VectorType),
    ("val_losses", VectorType),
    ("best_epoch", IntegerType),
])

PlatformFunction(
    name="torch_train",
    inputs=[ModelHandleType, MatrixType, VectorType, TorchTrainConfigType],
    output=TorchTrainResultType,
    type="sync",
    fn=torch_train_impl,
)

def torch_train_impl(
    model_handle: int,
    X: list[list[float]],
    y: list[float],
    config: dict
) -> dict:
    """Train PyTorch model."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    import numpy as np

    model = _get_model(model_handle)

    # Config
    epochs = _get_option(config.get("epochs"), 100)
    batch_size = _get_option(config.get("batch_size"), 32)
    lr = _get_option(config.get("learning_rate"), 0.001)
    loss_name = _get_option(config.get("loss"), "mse")
    optimizer_name = _get_option(config.get("optimizer"), "adam")
    patience = _get_option(config.get("early_stopping"), 0)
    val_split = _get_option(config.get("validation_split"), 0.2)

    # Convert to tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y).unsqueeze(1)

    # Train/val split
    n = len(X_tensor)
    n_val = int(n * val_split)
    indices = torch.randperm(n)

    X_train = X_tensor[indices[n_val:]]
    y_train = y_tensor[indices[n_val:]]
    X_val = X_tensor[indices[:n_val]]
    y_val = y_tensor[indices[:n_val]]

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)

    # Loss and optimizer
    criterion = nn.MSELoss() if loss_name == "mse" else nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) if optimizer_name == "adam" else torch.optim.SGD(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0

    for epoch in range(epochs):
        # Train
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        train_losses.append(epoch_loss / len(train_loader))

        # Validate
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val).item()
        val_losses.append(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
        elif patience > 0:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_epoch": best_epoch,
    }
```

#### `torch_predict`

```python
PlatformFunction(
    name="torch_predict",
    inputs=[ModelHandleType, MatrixType],
    output=VectorType,
    type="sync",
    fn=torch_predict_impl,
)

def torch_predict_impl(model_handle: int, X: list[list[float]]) -> list[float]:
    """Make predictions with PyTorch model."""
    import torch

    model = _get_model(model_handle)
    model.eval()

    with torch.no_grad():
        X_tensor = torch.FloatTensor(X)
        predictions = model(X_tensor).squeeze().tolist()

    if isinstance(predictions, float):
        predictions = [predictions]

    return predictions
```

#### `torch_save` / `torch_load`

```python
PlatformFunction(
    name="torch_save",
    inputs=[ModelHandleType],
    output=BlobType,
    type="sync",
    fn=torch_save_impl,
)

PlatformFunction(
    name="torch_load",
    inputs=[BlobType],
    output=ModelHandleType,
    type="sync",
    fn=torch_load_impl,
)
```

---

## Module 10: Gaussian Process (`gp_impl.py`)

### Purpose
Gaussian Process regression for probabilistic predictions with uncertainty.

### Platform Functions

#### `gp_train`
Train Gaussian Process model.

```python
GPConfigType = StructType([
    ("kernel", OptionType(GPKernelType)),           # default rbf
    ("noise_variance", OptionType(FloatType)),     # default 0.1
    ("length_scale", OptionType(FloatType)),       # default 1.0
    ("max_iter", OptionType(IntegerType)),         # default 1000
])

PlatformFunction(
    name="gp_train",
    inputs=[MatrixType, VectorType, GPConfigType],
    output=ModelHandleType,
    type="sync",
    fn=gp_train_impl,
)

def gp_train_impl(X: list[list[float]], y: list[float], config: dict) -> int:
    """Train Gaussian Process model using GPflow."""
    import numpy as np
    import gpflow

    X_arr = np.array(X, dtype=np.float64)
    y_arr = np.array(y, dtype=np.float64).reshape(-1, 1)

    kernel_name = _get_option(config.get("kernel"), "rbf")

    if kernel_name == "rbf":
        kernel = gpflow.kernels.SquaredExponential()
    elif kernel_name == "matern32":
        kernel = gpflow.kernels.Matern32()
    elif kernel_name == "matern52":
        kernel = gpflow.kernels.Matern52()
    else:
        kernel = gpflow.kernels.SquaredExponential()

    model = gpflow.models.GPR(
        data=(X_arr, y_arr),
        kernel=kernel,
        noise_variance=_get_option(config.get("noise_variance"), 0.1),
    )

    # Optimize hyperparameters
    opt = gpflow.optimizers.Scipy()
    opt.minimize(
        model.training_loss,
        model.trainable_variables,
        options={"maxiter": _get_option(config.get("max_iter"), 1000)},
    )

    return _register_model(model)
```

#### `gp_predict`
Get predictions with uncertainty.

```python
GPPredictConfigType = StructType([
    ("confidence_level", OptionType(FloatType)),  # default 0.95
])

PlatformFunction(
    name="gp_predict",
    inputs=[ModelHandleType, MatrixType, GPPredictConfigType],
    output=PredictionResultType,
    type="sync",
    fn=gp_predict_impl,
)

def gp_predict_impl(handle: int, X: list[list[float]], config: dict) -> dict:
    """Predict with Gaussian Process model."""
    import numpy as np
    from scipy import stats

    model = _get_model(handle)
    X_arr = np.array(X, dtype=np.float64)

    mean, var = model.predict_f(X_arr)
    std = np.sqrt(var.numpy())
    mean = mean.numpy().flatten()
    std = std.flatten()

    confidence = _get_option(config.get("confidence_level"), 0.95)
    z = stats.norm.ppf(1 - (1 - confidence) / 2)

    lower = mean - z * std
    upper = mean + z * std

    return EastStruct({
        "predictions": EastArray(FloatType, [float(m) for m in mean]),
        "std": EastSome(EastArray(FloatType, [float(s) for s in std])),
        "lower": EastSome(EastArray(FloatType, [float(l) for l in lower])),
        "upper": EastSome(EastArray(FloatType, [float(u) for u in upper])),
    })
```

---

## Main Export (`__init__.py`)

```python
"""East Python Data Science Platform Functions.

Python implementation of data science platform functions for the East programming language.
Provides machine learning, optimization, and explainability operations.
"""

from east_py_data_science.scikit import sklearn_impl
from east_py_data_science.xgboost_impl import xgboost_impl
from east_py_data_science.lightgbm_impl import lightgbm_impl
from east_py_data_science.ngboost_impl import ngboost_impl
from east_py_data_science.optuna_impl import optuna_impl
from east_py_data_science.shap_impl import shap_impl
from east_py_data_science.scipy_impl import scipy_impl
from east_py_data_science.torch_impl import torch_impl
from east_py_data_science.gp_impl import gp_impl

__version__ = "0.1.0"

# Complete Python Data Science platform implementation
python_data_science_platform = [
    *sklearn_impl,
    *xgboost_impl,
    *lightgbm_impl,
    *ngboost_impl,
    *optuna_impl,
    *shap_impl,
    *scipy_impl,
    *torch_impl,
    *gp_impl,
]

__all__ = [
    "__version__",
    "python_data_science_platform",
    # Module exports
    "sklearn_impl",
    "xgboost_impl",
    "lightgbm_impl",
    "ngboost_impl",
    "optuna_impl",
    "shap_impl",
    "scipy_impl",
    "torch_impl",
    "gp_impl",
]
```

---

## Model Registry and Helper Functions

All modules use a shared model registry and helper functions for working with East values:

```python
# In types.py or a dedicated registry.py

from typing import Any
import threading

from east.types.values import (
    EastStruct,      # Immutable struct (extends dict, hashable)
    EastVariant,     # Tagged union (has .type and .value properties)
    EastArray,       # Mutable array with element_type tracking
    EastBlob,        # Immutable binary data (extends bytes)
    east_null,       # Singleton null value
    EastSome,        # Create Option "some" variant: EastSome(value)
    EastNone,        # Create Option "none" variant: EastNone()
)
from east.types.types import FloatType, ArrayType

# =============================================================================
# Model Registry
# =============================================================================

_model_registry: dict[int | str, Any] = {}
_next_handle = 1
_lock = threading.Lock()

def _register_model(model: Any, key: int | str | None = None) -> int:
    """Register a model and return its handle."""
    global _next_handle
    with _lock:
        if key is None:
            handle = _next_handle
            _next_handle += 1
        else:
            handle = key
        _model_registry[handle] = model
    return handle if isinstance(handle, int) else 0

def _get_model(handle: int | str) -> Any:
    """Retrieve a model by handle."""
    if handle not in _model_registry:
        raise ValueError(f"Model handle {handle} not found")
    return _model_registry[handle]

def _delete_model(handle: int | str) -> None:
    """Delete a model from registry."""
    with _lock:
        if handle in _model_registry:
            del _model_registry[handle]

# =============================================================================
# East Value Helpers
# =============================================================================

def _get_option(value: Any, default: Any) -> Any:
    """Extract value from Option variant (EastVariant) or return default.

    East Option types are EastVariant with type="some" or type="none".
    """
    if value is None:
        return default
    # Handle EastVariant directly
    if isinstance(value, EastVariant):
        if value.type == "some":
            return value.value
        elif value.type == "none":
            return default
    # Handle dict-style variant (for compatibility)
    if isinstance(value, dict) and "type" in value:
        if value["type"] == "some":
            return value.get("value", default)
        elif value["type"] == "none":
            return default
    return value

def _get_enum_tag(value: Any) -> str:
    """Extract tag name from enum-like variant (Variant with NullType values).

    Enum variants in East are EastVariant where type is the enum case name
    and value is east_null.
    """
    if isinstance(value, EastVariant):
        return value.type
    if isinstance(value, dict) and "type" in value:
        return value["type"]
    raise ValueError(f"Expected EastVariant enum, got {type(value)}")

def _make_enum(tag: str) -> EastVariant:
    """Create an enum-like variant (tag with null value)."""
    return EastVariant(tag, east_null)
```

### East Type Conversion Patterns

All platform function implementations MUST use proper East value types:

```python
# Creating vectors (1D arrays)
EastArray(FloatType, [1.0, 2.0, 3.0])
EastArray(IntegerType, [1, 2, 3])
EastArray(StringType, ["a", "b", "c"])

# Creating matrices (2D arrays)
inner_type = ArrayType(FloatType)
rows = [EastArray(FloatType, row) for row in [[1.0, 2.0], [3.0, 4.0]]]
matrix = EastArray(inner_type, rows)

# Creating structs
EastStruct({
    "field1": 42,
    "field2": "hello",
    "nested": EastStruct({"x": 1.0, "y": 2.0}),
})

# Creating Option types
EastSome(42)           # some(42)
EastNone()             # none

# Creating enum variants (variant with null value)
EastVariant("relu", east_null)           # ActivationFunctionType.relu
EastVariant("minimize", east_null)       # OptimizationDirectionType.minimize

# Converting numpy arrays to East
def numpy_to_east_vector(arr: np.ndarray) -> EastArray:
    return EastArray(FloatType, arr.tolist())

def numpy_to_east_matrix(arr: np.ndarray) -> EastArray:
    inner_type = ArrayType(FloatType)
    rows = [EastArray(FloatType, row.tolist()) for row in arr]
    return EastArray(inner_type, rows)

# Converting East arrays to numpy
def east_vector_to_numpy(arr: EastArray) -> np.ndarray:
    return np.array(list(arr))

def east_matrix_to_numpy(arr: EastArray) -> np.ndarray:
    return np.array([list(row) for row in arr])
```

### Example: Complete Implementation with East Types

Here's a complete example showing proper East type usage:

```python
def sklearn_train_test_split_impl(
    X: EastArray,          # EastArray[EastArray[float]] - Matrix
    y: EastArray,          # EastArray[float] - Vector
    config: EastStruct     # EastStruct with Option fields
) -> EastStruct:
    """Split arrays into train and test subsets."""
    from sklearn.model_selection import train_test_split
    import numpy as np

    # Extract config options (EastVariant some/none)
    test_size = _get_option(config.get("test_size"), 0.2)
    random_state = _get_option(config.get("random_state"), None)
    shuffle = _get_option(config.get("shuffle"), True)

    # Convert East arrays to numpy
    X_np = east_matrix_to_numpy(X)
    y_np = east_vector_to_numpy(y)

    # Perform split
    X_train, X_test, y_train, y_test = train_test_split(
        X_np, y_np, test_size=test_size, random_state=random_state, shuffle=shuffle
    )

    # Return EastStruct with EastArray values
    return EastStruct({
        "X_train": numpy_to_east_matrix(X_train),
        "X_test": numpy_to_east_matrix(X_test),
        "y_train": numpy_to_east_vector(y_train),
        "y_test": numpy_to_east_vector(y_test),
    })
```

### Example: Using Enum Variants

```python
def sklearn_cross_val_score_impl(
    handle: int,
    X: EastArray,
    y: EastArray,
    config: EastStruct
) -> EastStruct:
    """Perform cross-validation and return scores."""
    from sklearn.model_selection import cross_val_score, KFold
    import numpy as np

    model = _get_model(handle)
    cv = _get_option(config.get("cv"), 5)

    # Extract enum tag from ScoringMetricType variant
    scoring_variant = _get_option(config.get("scoring"), None)
    if scoring_variant is not None:
        scoring = _get_enum_tag(scoring_variant)  # e.g., "neg_mean_squared_error"
    else:
        scoring = "neg_mean_squared_error"

    shuffle = _get_option(config.get("shuffle"), True)
    random_state = _get_option(config.get("random_state"), None)

    X_np = east_matrix_to_numpy(X)
    y_np = east_vector_to_numpy(y)

    kfold = KFold(n_splits=cv, shuffle=shuffle, random_state=random_state)
    scores = cross_val_score(model, X_np, y_np, cv=kfold, scoring=scoring)

    return EastStruct({
        "scores": EastArray(FloatType, scores.tolist()),
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
    })
```

### Example: Returning Option Values

```python
def gp_predict_impl(
    handle: int,
    X: EastArray,
    config: EastStruct
) -> EastStruct:
    """Predict with Gaussian Process model."""
    import numpy as np
    from scipy import stats

    model = _get_model(handle)
    X_np = east_matrix_to_numpy(X)

    mean, var = model.predict_f(X_np)
    std = np.sqrt(var.numpy())
    mean_arr = mean.numpy().flatten()
    std_arr = std.flatten()

    confidence = _get_option(config.get("confidence_level"), 0.95)
    z = stats.norm.ppf(1 - (1 - confidence) / 2)

    lower = mean_arr - z * std_arr
    upper = mean_arr + z * std_arr

    # Return EastStruct with Option fields using EastSome/EastNone
    return EastStruct({
        "predictions": EastArray(FloatType, mean_arr.tolist()),
        "std": EastSome(EastArray(FloatType, std_arr.tolist())),
        "lower": EastSome(EastArray(FloatType, lower.tolist())),
        "upper": EastSome(EastArray(FloatType, upper.tolist())),
    })
```
---

## Implementation Checklist

### Phase 1: Core Infrastructure
- [ ] Create package structure
- [ ] Implement types.py with shared types
- [ ] Implement model registry
- [ ] Set up pyproject.toml with dependencies

### Phase 2: Scikit-Learn Module
- [ ] `sklearn_train_test_split`
- [ ] `sklearn_standard_scaler_fit`
- [ ] `sklearn_standard_scaler_transform`
- [ ] `sklearn_cross_val_score`
- [ ] `sklearn_metrics_regression`
- [ ] `sklearn_metrics_classification`

### Phase 3: Gradient Boosting
- [ ] XGBoost: train, predict, save/load
- [ ] LightGBM: train, predict, save/load
- [ ] NGBoost: train, predict_dist, save/load

### Phase 4: Optimization
- [ ] Optuna: create_study, suggest_params, complete_trial, get_best

### Phase 5: Explainability
- [ ] SHAP: tree_explainer, kernel_explainer, compute_values, feature_importance

### Phase 6: Scientific Computing
- [ ] SciPy: stats, optimize, interpolate

### Phase 7: Deep Learning (Optional)
- [ ] PyTorch: mlp_create, train, predict, save/load

### Phase 8: Gaussian Processes (Optional)
- [ ] GPflow: train, predict with uncertainty

### Phase 9: Testing & Documentation
- [ ] Unit tests for each module
- [ ] Integration tests with East IR
- [ ] README with usage examples

---

## Usage Example (East Code)

```east
// Train XGBoost model
let config = {
    n_estimators: some(100),
    max_depth: some(6),
    learning_rate: some(0.1),
    random_state: some(42),
    // ... other fields are none
};

let model = xgboost_train_regressor(X_train, y_train, config);

// Make predictions
let predictions = xgboost_predict(model, X_test);

// Get feature importance
let importance = xgboost_feature_importance(model, feature_names);

// Save model
let model_blob = xgboost_save(model);

// Later: load model
let loaded_model = xgboost_load(model_blob);
```
