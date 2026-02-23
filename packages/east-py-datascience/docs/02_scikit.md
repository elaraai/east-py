# Module 2: Scikit-Learn (`scikit.py`)

## Purpose

Core machine learning utilities: preprocessing, model selection, metrics, clustering (GMM).

## Config Types

```python
SplitConfigType = StructType([
    ("test_size", OptionType(FloatType)),      # default 0.2
    ("random_state", OptionType(IntegerType)), # default None
    ("shuffle", OptionType(BooleanType)),      # default True
])
```

## Platform Functions

### `sklearn_train_test_split`

Split data into train/test sets.

```python
PlatformFunction(
    name="sklearn_train_test_split",
    inputs=[MatrixType, VectorType, SplitConfigType],
    output=SplitResultType,
    type="sync",
    fn=sklearn_train_test_split_impl,
)

def sklearn_train_test_split_impl(
    X: Matrix,
    y: Vector,
    config: EastStruct
) -> EastStruct:
    """Split arrays into train and test subsets."""
    from sklearn.model_selection import train_test_split

    test_size = _get_option(config.get("test_size"), 0.2)
    random_state = _get_option(config.get("random_state"), None)
    shuffle = _get_option(config.get("shuffle"), True)

    X_np = east_matrix_to_numpy(X)
    y_np = east_vector_to_numpy(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_np, y_np, test_size=test_size, random_state=random_state, shuffle=shuffle
    )

    return EastStruct({
        "X_train": numpy_to_east_matrix(X_train),
        "X_test": numpy_to_east_matrix(X_test),
        "y_train": numpy_to_east_vector(y_train),
        "y_test": numpy_to_east_vector(y_test),
    })
```

### `sklearn_standard_scaler_fit`

Fit a standard scaler to training data.

```python
PlatformFunction(
    name="sklearn_standard_scaler_fit",
    inputs=[MatrixType],
    output=ModelBlobType,  # Returns "standard_scaler" variant
    type="sync",
    fn=sklearn_standard_scaler_fit_impl,
)

def sklearn_standard_scaler_fit_impl(X: Matrix) -> ModelBlob:
    """Fit StandardScaler and return model blob."""
    from sklearn.preprocessing import StandardScaler

    X_np = east_matrix_to_numpy(X)
    n_features = X_np.shape[1]

    scaler = StandardScaler()
    scaler.fit(X_np)

    onnx_data = _sklearn_to_onnx(scaler, n_features)

    return EastVariant("standard_scaler", EastStruct({
        "onnx": onnx_data,
        "n_features": n_features,
    }))
```

### `sklearn_standard_scaler_transform`

Transform data using fitted scaler.

```python
PlatformFunction(
    name="sklearn_standard_scaler_transform",
    inputs=[ModelBlobType, MatrixType],  # Expects "standard_scaler" variant
    output=MatrixType,
    type="sync",
    fn=sklearn_standard_scaler_transform_impl,
)

def sklearn_standard_scaler_transform_impl(
    model_blob: ModelBlob,
    X: Matrix
) -> Matrix:
    """Transform data using fitted scaler."""
    if model_blob.type != "standard_scaler":
        raise ValueError(f"Expected standard_scaler, got {model_blob.type}")

    onnx_blob = model_blob.value["onnx"]
    return _onnx_transform(onnx_blob, X)
```

### `sklearn_min_max_scaler_fit`

Fit a min-max scaler to training data.

```python
PlatformFunction(
    name="sklearn_min_max_scaler_fit",
    inputs=[MatrixType],
    output=ModelBlobType,  # Returns "min_max_scaler" variant
    type="sync",
    fn=sklearn_min_max_scaler_fit_impl,
)

def sklearn_min_max_scaler_fit_impl(X: Matrix) -> ModelBlob:
    """Fit MinMaxScaler and return model blob."""
    from sklearn.preprocessing import MinMaxScaler

    X_np = east_matrix_to_numpy(X)
    n_features = X_np.shape[1]

    scaler = MinMaxScaler()
    scaler.fit(X_np)

    onnx_data = _sklearn_to_onnx(scaler, n_features)

    return EastVariant("min_max_scaler", EastStruct({
        "onnx": onnx_data,
        "n_features": n_features,
    }))
```

### `sklearn_min_max_scaler_transform`

Transform data using fitted min-max scaler.

```python
PlatformFunction(
    name="sklearn_min_max_scaler_transform",
    inputs=[ModelBlobType, MatrixType],  # Expects "min_max_scaler" variant
    output=MatrixType,
    type="sync",
    fn=sklearn_min_max_scaler_transform_impl,
)

def sklearn_min_max_scaler_transform_impl(
    model_blob: ModelBlob,
    X: Matrix
) -> Matrix:
    """Transform data using fitted min-max scaler."""
    if model_blob.type != "min_max_scaler":
        raise ValueError(f"Expected min_max_scaler, got {model_blob.type}")

    onnx_blob = model_blob.value["onnx"]
    return _onnx_transform(onnx_blob, X)
```

### `sklearn_metrics_regression`

Compute regression metrics.

```python
PlatformFunction(
    name="sklearn_metrics_regression",
    inputs=[VectorType, VectorType],
    output=RegressionMetricsType,
    type="sync",
    fn=sklearn_metrics_regression_impl,
)

def sklearn_metrics_regression_impl(
    y_true: Vector,
    y_pred: Vector
) -> EastStruct:
    """Compute regression metrics."""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import numpy as np

    y_true_np = east_vector_to_numpy(y_true)
    y_pred_np = east_vector_to_numpy(y_pred)

    mse = mean_squared_error(y_true_np, y_pred_np)
    mae = mean_absolute_error(y_true_np, y_pred_np)
    r2 = r2_score(y_true_np, y_pred_np)

    # MAPE (avoid division by zero)
    mask = y_true_np != 0
    if mask.any():
        mape = float(np.mean(np.abs((y_true_np[mask] - y_pred_np[mask]) / y_true_np[mask])) * 100)
    else:
        mape = 0.0

    return EastStruct({
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mae),
        "r2": float(r2),
        "mape": mape,
    })
```

### `sklearn_metrics_classification`

Compute classification metrics.

```python
PlatformFunction(
    name="sklearn_metrics_classification",
    inputs=[IntVectorType, IntVectorType],
    output=ClassificationMetricsType,
    type="sync",
    fn=sklearn_metrics_classification_impl,
)

def sklearn_metrics_classification_impl(
    y_true: IntVector,
    y_pred: IntVector
) -> EastStruct:
    """Compute classification metrics."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    y_true_np = east_int_vector_to_numpy(y_true)
    y_pred_np = east_int_vector_to_numpy(y_pred)

    return EastStruct({
        "accuracy": float(accuracy_score(y_true_np, y_pred_np)),
        "precision": float(precision_score(y_true_np, y_pred_np, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true_np, y_pred_np, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_true_np, y_pred_np, average="weighted", zero_division=0)),
    })
```
