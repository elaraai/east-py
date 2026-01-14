#
# Copyright (c) 2025 Elara AI Pty Ltd
# Licensed under the Business Source License 1.1. See LICENSE.md for details.
#
"""SHAP platform functions for East.

Provides model-agnostic feature importance and explainability using SHAP values.
Uses cloudpickle for explainer serialization.
"""

import warnings

import numpy as np

from east.runtime.platform import PlatformFunction
from east.types.values import EastArray, EastBlob, EastStruct, EastVariant

from east_py_datascience.types import (
    MatrixType,
    StringVectorType,
    ShapValuesType,
    ShapResultType,
    FeatureImportanceType,
    ModelBlobType,
    east_matrix_to_numpy,
    numpy_to_east_matrix,
    numpy_to_east_vector,
)


# ============================================================================
# Serialization Helpers
# ============================================================================


def _serialize_explainer(explainer) -> EastBlob:
    """Serialize SHAP explainer using cloudpickle."""
    try:
        import cloudpickle
    except ImportError as e:
        raise RuntimeError(
            "_serialize_explainer: cloudpickle not installed. "
            "Install with: pip install cloudpickle"
        ) from e

    try:
        return EastBlob(cloudpickle.dumps(explainer))
    except Exception as e:
        raise RuntimeError(
            f"_serialize_explainer: Failed to serialize explainer - {e}"
        ) from e


def _deserialize_explainer(blob: EastBlob):
    """Deserialize SHAP explainer using cloudpickle."""
    try:
        import cloudpickle
    except ImportError as e:
        raise RuntimeError(
            "_deserialize_explainer: cloudpickle not installed. "
            "Install with: pip install cloudpickle"
        ) from e

    try:
        return cloudpickle.loads(bytes(blob))
    except Exception as e:
        raise RuntimeError(
            f"_deserialize_explainer: Failed to deserialize explainer - {e}"
        ) from e


def _deserialize_model(blob: EastBlob):
    """Deserialize model using cloudpickle."""
    try:
        import cloudpickle
    except ImportError as e:
        raise RuntimeError(
            "_deserialize_model: cloudpickle not installed. "
            "Install with: pip install cloudpickle"
        ) from e

    try:
        return cloudpickle.loads(bytes(blob))
    except Exception as e:
        raise RuntimeError(
            f"_deserialize_model: Failed to deserialize model - {e}"
        ) from e


# ============================================================================
# Platform Function Implementations
# ============================================================================


def shap_tree_explainer_create_impl(
    model_blob: EastVariant,
) -> EastVariant:
    """Create SHAP TreeExplainer for tree-based models."""
    try:
        import shap
    except ImportError as e:
        raise RuntimeError(
            "shap_tree_explainer_create: shap not installed. "
            "Install with: pip install shap"
        ) from e

    function_name = "shap_tree_explainer_create"

    # Get model type and validate
    # Note: LightGBM is not supported due to SHAP compatibility issues
    model_type = model_blob.type
    if model_type not in (
        "xgboost_regressor",
        "xgboost_classifier",
        "xgboost_quantile",
    ):
        raise RuntimeError(
            f"{function_name}: TreeExplainer requires XGBoost model, got {model_type}. "
            "Use KernelExplainer for other model types."
        )

    try:
        model_data = _deserialize_model(model_blob.value["data"])
        n_features = int(model_blob.value["n_features"])

        # For quantile models, select the median quantile (or closest to 0.5)
        if model_type == "xgboost_quantile":
            # model_data is a dict of {quantile: model}
            quantiles = sorted(model_data.keys())
            # Find quantile closest to 0.5 (median)
            median_q = min(quantiles, key=lambda q: abs(q - 0.5))
            model = model_data[median_q]
        else:
            model = model_data
    except Exception as e:
        raise RuntimeError(f"{function_name}: Invalid input data - {e}") from e

    # Create TreeExplainer
    try:
        # Suppress SHAP warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=Warning)
            explainer = shap.TreeExplainer(model)
    except Exception as e:
        raise RuntimeError(
            f"{function_name}: Failed to create TreeExplainer - {e}"
        ) from e

    return EastVariant(
        "shap_tree_explainer",
        EastStruct(
            {
                "data": _serialize_explainer(explainer),
                "n_features": n_features,
            }
        ),
    )


def _get_predict_fn(model, model_type: str, categorical_features=None):
    """Get appropriate predict function for model type."""
    try:
        if model_type == "torch_mlp":
            # Torch model needs special handling
            try:
                import torch
            except ImportError as e:
                raise RuntimeError(
                    "_get_predict_fn: Failed to import torch library. "
                    "Install with: pip install torch"
                ) from e

            def predict_fn(X):
                try:
                    model.eval()
                    with torch.no_grad():
                        X_tensor = torch.tensor(X, dtype=torch.float32)
                        output = model(X_tensor).numpy()
                    return output.flatten() if output.shape[1] == 1 else output
                except Exception as e:
                    raise RuntimeError(
                        f"_get_predict_fn: Torch prediction failed - {e}"
                    ) from e

            return predict_fn
        elif model_type == "ngboost_regressor":
            # NGBoost predict returns mean
            return lambda X: model.predict(X)
        elif model_type == "gp_regressor":
            # GP predict returns mean
            return lambda X: model.predict(X)
        elif model_type == "regressor_chain":
            # RegressorChain predict returns multi-target predictions
            # For SHAP, we return the first target (or could average)
            return (
                lambda X: model.predict(X)[:, 0]
                if model.predict(X).ndim > 1
                else model.predict(X)
            )
        elif model_type in ("mapie_split", "mapie_cross", "mapie_cqr"):
            # MAPIE regressor - returns point prediction from predict_interval
            import warnings

            def predict_fn(X):
                # Apply categorical features if needed
                X_pred = _apply_categorical_for_predict(X, categorical_features)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=Warning)
                    y_pred, _ = model.predict_interval(X_pred)
                return y_pred

            return predict_fn
        elif model_type == "mapie_classifier":
            # MAPIE classifier - returns class probabilities
            import warnings

            def predict_fn(X):
                # Apply categorical features if needed
                X_pred = _apply_categorical_for_predict(X, categorical_features)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=Warning)
                    # Get base classifier for probabilities
                    if hasattr(model, "estimator") and hasattr(model.estimator, "predict_proba"):
                        return model.estimator.predict_proba(X_pred)
                    else:
                        # Fallback to predict_set probabilities via stored base_clf
                        y_pred, _ = model.predict_set(X_pred)
                        return y_pred

            return predict_fn
        elif model_type == "mapie_interval_width":
            # Uncertainty predictor for MAPIE regressor - returns interval width
            import warnings

            def predict_fn(X):
                X_pred = _apply_categorical_for_predict(X, categorical_features)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=Warning)
                    _, y_intervals = model.predict_interval(X_pred)
                    # Interval width = upper - lower
                    lower = y_intervals[:, 0, 0]
                    upper = y_intervals[:, 1, 0]
                    return upper - lower

            return predict_fn
        elif model_type == "mapie_set_size":
            # Uncertainty predictor for MAPIE classifier - returns set size
            import warnings

            def predict_fn(X):
                X_pred = _apply_categorical_for_predict(X, categorical_features)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=Warning)
                    _, y_sets = model.predict_set(X_pred)
                    # Set size = number of classes in prediction set
                    set_sizes = y_sets[:, :, 0].sum(axis=1).astype(float)
                    return set_sizes

            return predict_fn
        else:
            # Tree-based models
            return lambda X: model.predict(X)
    except Exception as e:
        raise RuntimeError(
            f"_get_predict_fn: Failed to create predict function - {e}"
        ) from e


def _apply_categorical_for_predict(X, categorical_features):
    """Apply categorical dtype conversion for prediction if needed.

    Matches the categorical handling in mapie_impl.py and xgboost_impl.py.
    """
    if categorical_features is None:
        return X

    import pandas as pd

    if isinstance(X, pd.DataFrame):
        X_np = X.values
    else:
        X_np = X

    df = pd.DataFrame(X_np)
    for idx in categorical_features:
        if idx < 0 or idx >= X_np.shape[1]:
            continue  # Skip invalid indices silently during SHAP computation
        col = df[idx]
        # Convert to int then category (matching mapie_impl pattern)
        df[idx] = col.astype(int).astype("category")

    return df


def _extract_model_from_blob(model_blob: EastVariant, function_name: str):
    """Extract model, n_features, and categorical_features from model blob.

    Handles different model structures:
    - Standard models: { data: blob, n_features, ... }
    - MAPIE regressors (split/cross): { data: EastVariant(xgboost/lightgbm, blob), n_features, ... }
    - MAPIE CQR: { data: blob, n_features, ... }
    - MAPIE classifier: { data: EastVariant(xgboost/lightgbm, blob), n_features, n_classes, ... }
    - Uncertainty predictors: { data: blob, n_features }
    """
    model_type = model_blob.type
    model_data = model_blob.value
    categorical_features = None

    # Handle MAPIE models with nested variant structure
    if model_type in ("mapie_split", "mapie_cross", "mapie_cqr"):
        # data is a variant (xgboost/lightgbm/histogram) containing the blob
        data_variant = model_data["data"]
        model_bytes = data_variant.value  # Extract blob from variant
        combined = _deserialize_model(model_bytes)
        model = combined["mapie"]
        categorical_features = combined.get("categorical_features")
        n_features = int(model_data["n_features"])
    elif model_type == "mapie_classifier":
        # Classifier is a struct (not variant), data is a variant
        data_variant = model_data["data"]
        model_bytes = data_variant.value
        combined = _deserialize_model(model_bytes)
        model = combined["mapie"]
        categorical_features = combined.get("categorical_features")
        n_features = int(model_data["n_features"])
    elif model_type in ("mapie_interval_width", "mapie_set_size"):
        # Uncertainty predictors - data contains { mapie, categorical_features }
        combined = _deserialize_model(model_data["data"])
        model = combined["mapie"]
        categorical_features = combined.get("categorical_features")
        n_features = int(model_data["n_features"])
    else:
        # Standard model blobs
        deserialized = _deserialize_model(model_data["data"])
        # Handle Torch model package format (dict with "model" key)
        if (
            model_type == "torch_mlp"
            and isinstance(deserialized, dict)
            and "model" in deserialized
        ):
            model = deserialized["model"]
        else:
            model = deserialized
        n_features = int(model_data["n_features"])

    return model, n_features, categorical_features, model_type


def shap_kernel_explainer_create_impl(
    model_blob: EastVariant,
    X_background: EastArray,
) -> EastVariant:
    """Create SHAP KernelExplainer for any model."""
    try:
        import shap
    except ImportError as e:
        raise RuntimeError(
            "shap_kernel_explainer_create: shap not installed. "
            "Install with: pip install shap"
        ) from e

    function_name = "shap_kernel_explainer_create"

    try:
        model, n_features, categorical_features, model_type = _extract_model_from_blob(
            model_blob, function_name
        )
    except Exception as e:
        raise RuntimeError(f"{function_name}: Invalid input data - {e}") from e

    try:
        X_bg = east_matrix_to_numpy(X_background)
    except Exception as e:
        raise RuntimeError(f"{function_name}: Invalid input data - {e}") from e

    # Get predict function for the model type
    predict_fn = _get_predict_fn(model, model_type, categorical_features)

    # Create KernelExplainer with background data
    try:
        # Suppress SHAP warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=Warning)
            explainer = shap.KernelExplainer(predict_fn, X_bg)
    except Exception as e:
        raise RuntimeError(
            f"{function_name}: Failed to create KernelExplainer - {e}"
        ) from e

    return EastVariant(
        "shap_kernel_explainer",
        EastStruct(
            {
                "data": _serialize_explainer(explainer),
                "n_features": n_features,
            }
        ),
    )


def shap_compute_values_impl(
    explainer_blob: EastVariant,
    X: EastArray,
    feature_names: EastArray,
) -> EastStruct:
    """Compute SHAP values for samples."""
    function_name = "shap_compute_values"

    if explainer_blob.type not in ("shap_tree_explainer", "shap_kernel_explainer"):
        raise RuntimeError(
            f"{function_name}: Expected SHAP explainer, got {explainer_blob.type}"
        )

    try:
        explainer = _deserialize_explainer(explainer_blob.value["data"])
    except Exception as e:
        raise RuntimeError(f"{function_name}: Invalid input data - {e}") from e

    try:
        X_np = east_matrix_to_numpy(X)
    except Exception as e:
        raise RuntimeError(f"{function_name}: Invalid input data - {e}") from e

    # Compute SHAP values
    try:
        # Suppress SHAP warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=Warning)
            shap_values = explainer.shap_values(X_np)
    except Exception as e:
        raise RuntimeError(
            f"{function_name}: Failed to compute SHAP values - {e}"
        ) from e

    # Determine if multi-class (more than 2 classes)
    try:
        base_value_raw = explainer.expected_value
        is_multiclass = False
        n_classes = 1

        # Check for multi-class: list/array with > 2 elements or 3D shap_values
        if isinstance(shap_values, list):
            n_classes = len(shap_values)
            is_multiclass = n_classes > 2
        elif shap_values.ndim == 3:
            n_classes = shap_values.shape[2]
            is_multiclass = n_classes > 2

        # Convert feature names
        names_list = [str(name) for name in feature_names]
        from east.types.types import StringType

        if is_multiclass:
            # Multi-class: return 3D tensor and per-class base values
            if isinstance(shap_values, list):
                # Convert list of 2D arrays to 3D array (n_samples, n_features, n_classes)
                shap_3d = np.stack(shap_values, axis=2)
            else:
                shap_3d = shap_values  # Already 3D

            # Convert 3D array to list of matrices (one per sample)
            # Shape: (n_samples, n_features, n_classes) -> list of (n_features, n_classes) matrices
            tensor_3d_list = [
                numpy_to_east_matrix(shap_3d[i])  # (n_features, n_classes)
                for i in range(shap_3d.shape[0])
            ]

            # Base values per class
            if isinstance(base_value_raw, (np.ndarray, list)):
                base_values = [float(v) for v in base_value_raw]
            else:
                base_values = [float(base_value_raw)] * n_classes

            return EastStruct(
                {
                    "shap_values": EastVariant(
                        "tensor_3d",
                        EastArray(MatrixType, tensor_3d_list),
                    ),
                    "base_value": EastVariant(
                        "per_class",
                        numpy_to_east_vector(np.array(base_values)),
                    ),
                    "feature_names": EastArray(StringType, names_list),
                }
            )
        else:
            # Regression or binary classification: return 2D matrix
            if isinstance(shap_values, list):
                # Binary: take positive class (index 1)
                shap_2d = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            elif shap_values.ndim == 3:
                # Binary with 3D output: take positive class
                shap_2d = (
                    shap_values[:, :, 1]
                    if shap_values.shape[2] > 1
                    else shap_values[:, :, 0]
                )
            else:
                shap_2d = shap_values

            # Ensure 2D
            if shap_2d.ndim == 1:
                shap_2d = shap_2d.reshape(1, -1)

            # Single base value (for binary, use positive class)
            if isinstance(base_value_raw, (np.ndarray, list)):
                base_value = (
                    float(base_value_raw[1])
                    if len(base_value_raw) > 1
                    else float(base_value_raw[0])
                )
            else:
                base_value = float(base_value_raw)

            return EastStruct(
                {
                    "shap_values": EastVariant(
                        "matrix_2d",
                        numpy_to_east_matrix(shap_2d),
                    ),
                    "base_value": EastVariant("single", base_value),
                    "feature_names": EastArray(StringType, names_list),
                }
            )
    except Exception as e:
        raise RuntimeError(
            f"{function_name}: Failed to process SHAP results - {e}"
        ) from e


def shap_feature_importance_impl(
    shap_values: EastVariant,
    feature_names: EastArray,
) -> EastStruct:
    """Compute global feature importance from SHAP values.

    Accepts either matrix_2d (regression/binary) or tensor_3d (multi-class) variant.
    For multi-class, computes mean(|SHAP|) across both samples and classes.
    """
    function_name = "shap_feature_importance"

    try:
        variant_type = shap_values.type
        if variant_type == "matrix_2d":
            # 2D: (n_samples, n_features)
            shap_np = east_matrix_to_numpy(shap_values.value)
            mean_abs_shap = np.abs(shap_np).mean(axis=0)
            std_shap = np.abs(shap_np).std(axis=0)
        elif variant_type == "tensor_3d":
            # 3D: list of (n_features, n_classes) matrices, one per sample
            # Convert to numpy 3D array
            tensor_list = list(shap_values.value)
            shap_3d = np.stack([east_matrix_to_numpy(m) for m in tensor_list], axis=0)
            # Shape: (n_samples, n_features, n_classes)
            # Mean across samples and classes to get per-feature importance
            mean_abs_shap = np.abs(shap_3d).mean(axis=(0, 2))
            std_shap = np.abs(shap_3d).std(axis=(0, 2))
        else:
            raise RuntimeError(
                f"{function_name}: Expected matrix_2d or tensor_3d variant, got {variant_type}"
            )
    except Exception as e:
        raise RuntimeError(f"{function_name}: Invalid input data - {e}") from e

    # Mean absolute SHAP value per feature
    try:
        # Convert feature names
        names_list = [str(name) for name in feature_names]

        from east.types.types import StringType

        return EastStruct(
            {
                "feature_names": EastArray(StringType, names_list),
                "importances": numpy_to_east_vector(mean_abs_shap),
                "std": EastVariant("some", numpy_to_east_vector(std_shap)),
            }
        )
    except Exception as e:
        raise RuntimeError(
            f"{function_name}: Failed to compute feature importance - {e}"
        ) from e


# ============================================================================
# Platform Function Registration
# ============================================================================

shap_impl = [
    PlatformFunction(
        name="shap_tree_explainer_create",
        inputs=[ModelBlobType],
        output=ModelBlobType,
        type="sync",
        fn=shap_tree_explainer_create_impl,
    ),
    PlatformFunction(
        name="shap_kernel_explainer_create",
        inputs=[ModelBlobType, MatrixType],
        output=ModelBlobType,
        type="sync",
        fn=shap_kernel_explainer_create_impl,
    ),
    PlatformFunction(
        name="shap_compute_values",
        inputs=[ModelBlobType, MatrixType, StringVectorType],
        output=ShapResultType,
        type="sync",
        fn=shap_compute_values_impl,
    ),
    PlatformFunction(
        name="shap_feature_importance",
        inputs=[ShapValuesType, StringVectorType],
        output=FeatureImportanceType,
        type="sync",
        fn=shap_feature_importance_impl,
    ),
]

__all__ = [
    "shap_impl",
]
