# Module 4: LightGBM (`lightgbm_impl.py`)

## Purpose

Fast gradient boosting with LightGBM (better for large datasets).

## Config Types

```python
LightGBMConfigType = StructType([
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
```

## Platform Functions

### `lightgbm_train_regressor`

Train LightGBM regression model.

```python
PlatformFunction(
    name="lightgbm_train_regressor",
    inputs=[MatrixType, VectorType, LightGBMConfigType],
    output=ModelBlobType,  # Returns "lightgbm_regressor" variant
    type="sync",
    fn=lightgbm_train_regressor_impl,
)

def lightgbm_train_regressor_impl(
    X: Matrix,
    y: Vector,
    config: EastStruct
) -> ModelBlob:
    """Train LightGBM regressor and return model blob."""
    import lightgbm as lgb

    X_np = east_matrix_to_numpy(X)
    y_np = east_vector_to_numpy(y)
    n_features = X_np.shape[1]

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
    model.fit(X_np, y_np)

    onnx_data = _lightgbm_to_onnx(model, n_features)

    return EastVariant("lightgbm_regressor", EastStruct({
        "onnx": onnx_data,
        "n_features": n_features,
    }))
```

### `lightgbm_train_classifier`

Train LightGBM classification model.

```python
PlatformFunction(
    name="lightgbm_train_classifier",
    inputs=[MatrixType, IntVectorType, LightGBMConfigType],
    output=ModelBlobType,  # Returns "lightgbm_classifier" variant
    type="sync",
    fn=lightgbm_train_classifier_impl,
)

def lightgbm_train_classifier_impl(
    X: Matrix,
    y: Vector,
    config: EastStruct
) -> ModelBlob:
    """Train LightGBM classifier and return model blob."""
    import lightgbm as lgb
    import numpy as np

    X_np = east_matrix_to_numpy(X)
    y_np = east_int_vector_to_numpy(y)
    n_features = X_np.shape[1]
    n_classes = len(np.unique(y_np))

    model = lgb.LGBMClassifier(
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
    model.fit(X_np, y_np)

    onnx_data = _lightgbm_to_onnx(model, n_features)

    return EastVariant("lightgbm_classifier", EastStruct({
        "onnx": onnx_data,
        "n_features": n_features,
        "n_classes": n_classes,
    }))
```

### `lightgbm_predict`

Make predictions with trained LightGBM regressor.

```python
PlatformFunction(
    name="lightgbm_predict",
    inputs=[ModelBlobType, MatrixType],  # Expects "lightgbm_regressor" variant
    output=VectorType,
    type="sync",
    fn=lightgbm_predict_impl,
)

def lightgbm_predict_impl(
    model_blob: ModelBlob,
    X: Matrix
) -> Vector:
    """Make predictions with LightGBM regressor."""
    if model_blob.type != "lightgbm_regressor":
        raise ValueError(f"Expected lightgbm_regressor, got {model_blob.type}")

    onnx_blob = model_blob.value["onnx"]
    return _onnx_predict_regression(onnx_blob, X)
```

### `lightgbm_predict_class`

Predict class labels with LightGBM classifier.

```python
PlatformFunction(
    name="lightgbm_predict_class",
    inputs=[ModelBlobType, MatrixType],  # Expects "lightgbm_classifier" variant
    output=IntVectorType,
    type="sync",
    fn=lightgbm_predict_class_impl,
)

def lightgbm_predict_class_impl(
    model_blob: ModelBlob,
    X: Matrix
) -> Vector:
    """Predict class labels with LightGBM classifier."""
    if model_blob.type != "lightgbm_classifier":
        raise ValueError(f"Expected lightgbm_classifier, got {model_blob.type}")

    onnx_blob = model_blob.value["onnx"]
    return _onnx_predict_classification(onnx_blob, X)
```

### `lightgbm_predict_proba`

Get class probabilities from LightGBM classifier.

```python
PlatformFunction(
    name="lightgbm_predict_proba",
    inputs=[ModelBlobType, MatrixType],  # Expects "lightgbm_classifier" variant
    output=MatrixType,  # n_samples x n_classes
    type="sync",
    fn=lightgbm_predict_proba_impl,
)

def lightgbm_predict_proba_impl(
    model_blob: ModelBlob,
    X: Matrix
) -> Vector:
    """Get class probabilities from LightGBM classifier."""
    if model_blob.type != "lightgbm_classifier":
        raise ValueError(f"Expected lightgbm_classifier, got {model_blob.type}")

    onnx_blob = model_blob.value["onnx"]
    return _onnx_predict_proba(onnx_blob, X)
```
