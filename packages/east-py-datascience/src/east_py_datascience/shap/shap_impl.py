"""SHAP platform functions for East.

Provides model-agnostic feature importance and explainability using SHAP values.
Uses cloudpickle for explainer serialization.
"""

import numpy as np

from east.runtime.platform import PlatformFunction
from east.types.values import EastArray, EastBlob, EastStruct, EastVariant

from east_py_datascience.types import (
    MatrixType,
    StringVectorType,
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
    model_type = model_blob.type
    if model_type not in (
        "xgboost_regressor",
        "xgboost_classifier",
        "lightgbm_regressor",
        "lightgbm_classifier",
    ):
        raise RuntimeError(
            f"{function_name}: TreeExplainer requires tree-based model, got {model_type}"
        )

    try:
        model = _deserialize_model(model_blob.value["data"])
        n_features = int(model_blob.value["n_features"])
    except Exception as e:
        raise RuntimeError(f"{function_name}: Invalid input data - {e}") from e

    # Create TreeExplainer
    try:
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


def _get_predict_fn(model, model_type: str):
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
        else:
            # Tree-based models
            return lambda X: model.predict(X)
    except Exception as e:
        raise RuntimeError(
            f"_get_predict_fn: Failed to create predict function - {e}"
        ) from e


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
        model_type = model_blob.type
        model = _deserialize_model(model_blob.value["data"])
        n_features = int(model_blob.value["n_features"])
    except Exception as e:
        raise RuntimeError(f"{function_name}: Invalid input data - {e}") from e

    try:
        X_bg = east_matrix_to_numpy(X_background)
    except Exception as e:
        raise RuntimeError(f"{function_name}: Invalid input data - {e}") from e

    # Get predict function for the model type
    predict_fn = _get_predict_fn(model, model_type)

    # Create KernelExplainer with background data
    try:
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
        shap_values = explainer.shap_values(X_np)
    except Exception as e:
        raise RuntimeError(
            f"{function_name}: Failed to compute SHAP values - {e}"
        ) from e

    # Handle multi-output (classification) - take positive class for binary
    try:
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

        # Ensure 2D
        if shap_values.ndim == 1:
            shap_values = shap_values.reshape(1, -1)

        # Get base value
        base_value = explainer.expected_value
        if isinstance(base_value, np.ndarray):
            base_value = (
                float(base_value[1]) if len(base_value) > 1 else float(base_value[0])
            )
        else:
            base_value = float(base_value)

        # Convert feature names
        names_list = [str(name) for name in feature_names]

        from east.types.types import StringType

        return EastStruct(
            {
                "shap_values": numpy_to_east_matrix(shap_values),
                "base_value": base_value,
                "feature_names": EastArray(StringType, names_list),
            }
        )
    except Exception as e:
        raise RuntimeError(
            f"{function_name}: Failed to process SHAP results - {e}"
        ) from e


def shap_feature_importance_impl(
    shap_values: EastArray,
    feature_names: EastArray,
) -> EastStruct:
    """Compute global feature importance from SHAP values."""
    function_name = "shap_feature_importance"

    try:
        shap_np = east_matrix_to_numpy(shap_values)
    except Exception as e:
        raise RuntimeError(f"{function_name}: Invalid input data - {e}") from e

    # Mean absolute SHAP value per feature
    try:
        mean_abs_shap = np.abs(shap_np).mean(axis=0)
        std_shap = np.abs(shap_np).std(axis=0)

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
        inputs=[MatrixType, StringVectorType],
        output=FeatureImportanceType,
        type="sync",
        fn=shap_feature_importance_impl,
    ),
]

__all__ = [
    "shap_impl",
]
