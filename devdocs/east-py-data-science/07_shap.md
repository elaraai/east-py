# Module 7: SHAP (`shap_impl.py`)

## Purpose

Model-agnostic feature importance and explainability using SHAP values.

Note: SHAP explainers are stateful and use native cloudpickle serialization.

## Platform Functions

### `shap_tree_explainer_create`

Create TreeExplainer for tree-based models (XGBoost, LightGBM, etc.).

Note: TreeExplainer works with the native model, not ONNX. For ONNX models,
we need to use KernelExplainer or deserialize the original model.

```python
PlatformFunction(
    name="shap_tree_explainer_create",
    inputs=[ModelBlobType],  # Tree-based model blob
    output=ModelBlobType,    # Returns "shap_tree_explainer" variant
    type="sync",
    fn=shap_tree_explainer_create_impl,
)

def shap_tree_explainer_create_impl(model_blob: ModelBlob) -> ModelBlob:
    """Create SHAP TreeExplainer for tree-based models.

    Note: This requires the original model, not ONNX. For ONNX-only models,
    use shap_kernel_explainer_create instead.
    """
    import shap

    # TreeExplainer needs native model - ONNX not supported
    # This would require storing native model alongside ONNX
    raise NotImplementedError(
        "TreeExplainer requires native model format. "
        "Use shap_kernel_explainer_create for ONNX models."
    )
```

### `shap_kernel_explainer_create`

Create KernelExplainer for any model (model-agnostic, works with ONNX).

```python
PlatformFunction(
    name="shap_kernel_explainer_create",
    inputs=[ModelBlobType, MatrixType],  # model, background_data
    output=ModelBlobType,                 # Returns "shap_kernel_explainer" variant
    type="sync",
    fn=shap_kernel_explainer_create_impl,
)

def shap_kernel_explainer_create_impl(
    model_blob: ModelBlob,
    background: Matrix
) -> ModelBlob:
    """Create SHAP KernelExplainer for any model (including ONNX)."""
    import shap

    background_np = east_matrix_to_numpy(background)
    n_features = background_np.shape[1]

    # Create a prediction function that uses ONNX
    def predict_fn(X):
        X_east = numpy_to_east_matrix(X)

        # Dispatch based on model type
        if model_blob.type in ("xgboost_regressor", "lightgbm_regressor", "ngboost_regressor"):
            result = _onnx_predict_regression(model_blob.value["onnx"], X_east)
        elif model_blob.type in ("xgboost_classifier", "lightgbm_classifier"):
            result = _onnx_predict_proba(model_blob.value["onnx"], X_east)
            # Return probabilities as numpy
            return east_matrix_to_numpy(result)
        else:
            raise ValueError(f"Unsupported model type for SHAP: {model_blob.type}")

        return east_vector_to_numpy(result)

    explainer = shap.KernelExplainer(predict_fn, background_np)

    return EastVariant("shap_kernel_explainer", EastStruct({
        "data": _serialize_native(explainer),
        "n_features": n_features,
    }))
```

### `shap_compute_values`

Compute SHAP values for samples.

```python
PlatformFunction(
    name="shap_compute_values",
    inputs=[ModelBlobType, MatrixType, StringVectorType],  # explainer, X, feature_names
    output=ShapResultType,
    type="sync",
    fn=shap_compute_values_impl,
)

def shap_compute_values_impl(
    explainer_blob: ModelBlob,
    X: Matrix,
    feature_names: StringVector
) -> EastStruct:
    """Compute SHAP values for samples."""
    import numpy as np

    if explainer_blob.type not in ("shap_tree_explainer", "shap_kernel_explainer"):
        raise ValueError(f"Expected SHAP explainer, got {explainer_blob.type}")

    explainer = _deserialize_native(explainer_blob.value["data"])
    X_np = east_matrix_to_numpy(X)

    shap_values = explainer.shap_values(X_np)

    # Handle multi-output (classification)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Take positive class for binary

    # Get base value
    base_value = explainer.expected_value
    if isinstance(base_value, np.ndarray):
        base_value = float(base_value[1]) if len(base_value) > 1 else float(base_value[0])
    else:
        base_value = float(base_value)

    return EastStruct({
        "shap_values": numpy_to_east_matrix(shap_values),
        "base_value": base_value,
        "feature_names": EastArray(StringType, list(feature_names)),
    })
```

### `shap_feature_importance`

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
    shap_values: Matrix,
    feature_names: StringVector
) -> EastStruct:
    """Compute global feature importance from SHAP values."""
    import numpy as np

    shap_np = east_matrix_to_numpy(shap_values)
    mean_abs_shap = np.abs(shap_np).mean(axis=0)
    std_shap = np.abs(shap_np).std(axis=0)

    return EastStruct({
        "feature_names": EastArray(StringType, list(feature_names)),
        "importances": EastArray(FloatType, [float(i) for i in mean_abs_shap]),
        "std": EastSome(EastArray(FloatType, [float(s) for s in std_shap])),
    })
```
