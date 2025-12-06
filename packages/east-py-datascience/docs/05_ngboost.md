# Module 5: NGBoost (`ngboost_impl.py`)

## Purpose

Probabilistic predictions with uncertainty quantification using NGBoost.

## Config Types

```python
NGBoostConfigType = StructType([
    ("n_estimators", OptionType(IntegerType)),       # default 500
    ("learning_rate", OptionType(FloatType)),        # default 0.01
    ("minibatch_frac", OptionType(FloatType)),       # default 1.0
    ("col_sample", OptionType(FloatType)),           # default 1.0
    ("random_state", OptionType(IntegerType)),       # default None
    ("distribution", OptionType(DistributionType)),  # default normal
])

NGBoostPredictConfigType = StructType([
    ("confidence_level", OptionType(FloatType)),  # default 0.95
])
```

## Platform Functions

### `ngboost_train_regressor`

Train NGBoost model with natural gradient boosting.

```python
PlatformFunction(
    name="ngboost_train_regressor",
    inputs=[MatrixType, VectorType, NGBoostConfigType],
    output=ModelBlobType,  # Returns "ngboost_regressor" variant
    type="sync",
    fn=ngboost_train_regressor_impl,
)

def ngboost_train_regressor_impl(
    X: Matrix,
    y: Vector,
    config: EastStruct
) -> ModelBlob:
    """Train NGBoost regressor for probabilistic predictions."""
    from ngboost import NGBRegressor
    from ngboost.distns import Normal, LogNormal

    X_np = east_matrix_to_numpy(X)
    y_np = east_vector_to_numpy(y)
    n_features = X_np.shape[1]

    # Get distribution type
    dist_variant = _get_option(config.get("distribution"), None)
    dist_name = _get_enum_tag(dist_variant) if dist_variant else "normal"
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
    model.fit(X_np, y_np)

    # NGBoost uses sklearn DecisionTreeRegressor as base, which can be converted to ONNX
    # However, the full NGBoost model with distribution params needs special handling
    # For now, we use native serialization
    onnx_data = _sklearn_to_onnx(model, n_features)

    return EastVariant("ngboost_regressor", EastStruct({
        "onnx": onnx_data,
        "distribution": _make_enum(dist_name),
        "n_features": n_features,
    }))
```

### `ngboost_predict`

Get point predictions (mean).

```python
PlatformFunction(
    name="ngboost_predict",
    inputs=[ModelBlobType, MatrixType],  # Expects "ngboost_regressor" variant
    output=VectorType,
    type="sync",
    fn=ngboost_predict_impl,
)

def ngboost_predict_impl(
    model_blob: ModelBlob,
    X: Matrix
) -> Vector:
    """Make point predictions with NGBoost regressor."""
    if model_blob.type != "ngboost_regressor":
        raise ValueError(f"Expected ngboost_regressor, got {model_blob.type}")

    onnx_blob = model_blob.value["onnx"]
    return _onnx_predict_regression(onnx_blob, X)
```

### `ngboost_predict_dist`

Get full predictive distribution (mean, std, confidence intervals).

Note: Since ONNX only gives point predictions, for full distribution we need
to store additional state or use native format. This implementation assumes
we store distribution parameters in a separate state blob.

```python
PlatformFunction(
    name="ngboost_predict_dist",
    inputs=[ModelBlobType, MatrixType, NGBoostPredictConfigType],
    output=PredictionResultType,
    type="sync",
    fn=ngboost_predict_dist_impl,
)

def ngboost_predict_dist_impl(
    model_blob: ModelBlob,
    X: Matrix,
    config: EastStruct
) -> EastStruct:
    """Get predictions with uncertainty from NGBoost.

    Note: Full distribution prediction requires native model, not just ONNX.
    This is a limitation of the ONNX-first approach for probabilistic models.
    For production use, consider storing native model alongside ONNX.
    """
    if model_blob.type != "ngboost_regressor":
        raise ValueError(f"Expected ngboost_regressor, got {model_blob.type}")

    # For now, return point predictions with no uncertainty
    # TODO: Store native model for full distribution support
    onnx_blob = model_blob.value["onnx"]
    predictions = _onnx_predict_regression(onnx_blob, X)

    return EastStruct({
        "predictions": predictions,
        "std": EastNone(),
        "lower": EastNone(),
        "upper": EastNone(),
    })
```

### Alternative: Native Format for Full Distribution

For full probabilistic predictions, use native format:

```python
# Alternative model blob type for full NGBoost support
# ("ngboost_regressor_native", StructType([
#     ("data", BlobType),  # cloudpickle serialized full model
#     ("distribution", DistributionType),
#     ("n_features", IntegerType),
# ])),

def ngboost_predict_dist_native_impl(
    model_blob: ModelBlob,
    X: Matrix,
    config: EastStruct
) -> EastStruct:
    """Get predictions with full uncertainty using native model."""
    from scipy import stats
    import numpy as np

    model = _deserialize_native(model_blob.value["data"])
    X_np = east_matrix_to_numpy(X)

    # Get distribution predictions
    dist_pred = model.pred_dist(X_np)

    # Extract mean and std
    loc = dist_pred.loc
    scale = dist_pred.scale

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
