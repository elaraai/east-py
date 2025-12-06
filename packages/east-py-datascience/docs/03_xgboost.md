# Module 3: XGBoost (`xgboost_impl.py`)

## Purpose

Gradient boosting for regression and classification with XGBoost.

## Config Types

```python
XGBoostConfigType = StructType([
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
```

## Platform Functions

### `xgboost_train_regressor`

Train XGBoost regression model.

```python
PlatformFunction(
    name="xgboost_train_regressor",
    inputs=[MatrixType, VectorType, XGBoostConfigType],
    output=ModelBlobType,  # Returns "xgboost_regressor" variant
    type="sync",
    fn=xgboost_train_regressor_impl,
)

def xgboost_train_regressor_impl(
    X: Matrix,
    y: Vector,
    config: EastStruct
) -> ModelBlob:
    """Train XGBoost regressor and return model blob."""
    import xgboost as xgb

    X_np = east_matrix_to_numpy(X)
    y_np = east_vector_to_numpy(y)
    n_features = X_np.shape[1]

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
    model.fit(X_np, y_np)

    onnx_data = _xgboost_to_onnx(model, n_features)

    return EastVariant("xgboost_regressor", EastStruct({
        "onnx": onnx_data,
        "n_features": n_features,
    }))
```

### `xgboost_train_classifier`

Train XGBoost classification model.

```python
PlatformFunction(
    name="xgboost_train_classifier",
    inputs=[MatrixType, IntVectorType, XGBoostConfigType],
    output=ModelBlobType,  # Returns "xgboost_classifier" variant
    type="sync",
    fn=xgboost_train_classifier_impl,
)

def xgboost_train_classifier_impl(
    X: Matrix,
    y: IntVector,
    config: EastStruct
) -> ModelBlob:
    """Train XGBoost classifier and return model blob."""
    import xgboost as xgb
    import numpy as np

    X_np = east_matrix_to_numpy(X)
    y_np = east_int_vector_to_numpy(y)
    n_features = X_np.shape[1]
    n_classes = len(np.unique(y_np))

    model = xgb.XGBClassifier(
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
        use_label_encoder=False,
        eval_metric="logloss",
    )
    model.fit(X_np, y_np)

    onnx_data = _xgboost_to_onnx(model, n_features)

    return EastVariant("xgboost_classifier", EastStruct({
        "onnx": onnx_data,
        "n_features": n_features,
        "n_classes": n_classes,
    }))
```

### `xgboost_predict`

Make predictions with trained XGBoost regressor.

```python
PlatformFunction(
    name="xgboost_predict",
    inputs=[ModelBlobType, MatrixType],  # Expects "xgboost_regressor" variant
    output=VectorType,
    type="sync",
    fn=xgboost_predict_impl,
)

def xgboost_predict_impl(
    model_blob: ModelBlob,
    X: Matrix
) -> Vector:
    """Make predictions with XGBoost regressor."""
    if model_blob.type != "xgboost_regressor":
        raise ValueError(f"Expected xgboost_regressor, got {model_blob.type}")

    onnx_blob = model_blob.value["onnx"]
    return _onnx_predict_regression(onnx_blob, X)
```

### `xgboost_predict_class`

Predict class labels with XGBoost classifier.

```python
PlatformFunction(
    name="xgboost_predict_class",
    inputs=[ModelBlobType, MatrixType],  # Expects "xgboost_classifier" variant
    output=IntVectorType,
    type="sync",
    fn=xgboost_predict_class_impl,
)

def xgboost_predict_class_impl(
    model_blob: ModelBlob,
    X: Matrix
) -> IntVector:
    """Predict class labels with XGBoost classifier."""
    if model_blob.type != "xgboost_classifier":
        raise ValueError(f"Expected xgboost_classifier, got {model_blob.type}")

    onnx_blob = model_blob.value["onnx"]
    return _onnx_predict_classification(onnx_blob, X)
```

### `xgboost_predict_proba`

Get class probabilities from XGBoost classifier.

```python
PlatformFunction(
    name="xgboost_predict_proba",
    inputs=[ModelBlobType, MatrixType],  # Expects "xgboost_classifier" variant
    output=MatrixType,  # n_samples x n_classes
    type="sync",
    fn=xgboost_predict_proba_impl,
)

def xgboost_predict_proba_impl(
    model_blob: ModelBlob,
    X: Matrix
) -> Matrix:
    """Get class probabilities from XGBoost classifier."""
    if model_blob.type != "xgboost_classifier":
        raise ValueError(f"Expected xgboost_classifier, got {model_blob.type}")

    onnx_blob = model_blob.value["onnx"]
    return _onnx_predict_proba(onnx_blob, X)
```

Note: Since we're using ONNX format, native XGBoost feature importance isn't available. Use SHAP (`shap_kernel_explainer_create` + `shap_feature_importance`) for feature importance with ONNX models.
