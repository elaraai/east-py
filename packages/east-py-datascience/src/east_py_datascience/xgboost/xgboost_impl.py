#
# Copyright (c) 2025 Elara AI Pty Ltd
# Licensed under the Business Source License 1.1. See LICENSE.md for details.
#
"""XGBoost platform functions for East.

Provides gradient boosting for regression and classification.
Uses cloudpickle for model serialization to enable portable inference.
"""

import warnings

import numpy as np

from east.runtime.platform import PlatformFunction
from east.types.values import EastArray, EastBlob, EastStruct, EastVariant

from east_py_datascience.types import (
    MatrixType,
    VectorType,
    IntVectorType,
    XGBoostConfigType,
    XGBoostQuantileConfigType,
    XGBoostQuantilePredictResultType,
    ModelBlobType,
    _get_option,
    east_matrix_to_numpy,
    east_vector_to_numpy,
    east_int_vector_to_numpy,
    numpy_to_east_vector,
    numpy_to_east_matrix,
    numpy_to_east_int_vector,
)


# ============================================================================
# Serialization Helpers
# ============================================================================


def _serialize_model(model) -> EastBlob:
    """Serialize model using cloudpickle."""
    import cloudpickle

    return EastBlob(cloudpickle.dumps(model))


def _deserialize_model(blob: EastBlob):
    """Deserialize model using cloudpickle."""
    import cloudpickle

    return cloudpickle.loads(bytes(blob))


# ============================================================================
# Platform Function Implementations
# ============================================================================


def xgboost_train_regressor_impl(
    X: EastArray,
    y: EastArray,
    config: EastStruct,
) -> EastVariant:
    """Train XGBoost regressor and return model blob."""
    try:
        import xgboost as xgb
    except ImportError as e:
        raise RuntimeError(
            "xgboost_train_regressor: xgboost not installed. "
            "Install with: pip install xgboost"
        ) from e

    try:
        X_np = east_matrix_to_numpy(X)
        y_np = east_vector_to_numpy(y)
    except Exception as e:
        raise RuntimeError(f"xgboost_train_regressor: Invalid input data - {e}") from e

    if X_np.shape[0] != y_np.shape[0]:
        raise RuntimeError(
            f"xgboost_train_regressor: X has {X_np.shape[0]} samples "
            f"but y has {y_np.shape[0]} samples"
        )

    n_features = X_np.shape[1]

    # Extract sample weights if provided
    sample_weight_opt = _get_option(config.get("sample_weight"), None)
    sample_weight_np = None
    if sample_weight_opt is not None:
        sample_weight_np = east_vector_to_numpy(sample_weight_opt)
        if sample_weight_np.shape[0] != X_np.shape[0]:
            raise RuntimeError(
                f"xgboost_train_regressor: sample_weight has {sample_weight_np.shape[0]} "
                f"elements but X has {X_np.shape[0]} samples"
            )

    try:
        random_state = _get_option(config.get("random_state"), None)
        if random_state is not None:
            random_state = int(random_state)

        n_jobs = _get_option(config.get("n_jobs"), -1)
        if n_jobs is not None:
            n_jobs = int(n_jobs)

        # Suppress XGBoost warnings during training
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=Warning)
            model = xgb.XGBRegressor(
                n_estimators=int(_get_option(config.get("n_estimators"), 100)),
                max_depth=int(_get_option(config.get("max_depth"), 6)),
                learning_rate=float(_get_option(config.get("learning_rate"), 0.3)),
                min_child_weight=int(_get_option(config.get("min_child_weight"), 1)),
                subsample=float(_get_option(config.get("subsample"), 1.0)),
                colsample_bytree=float(
                    _get_option(config.get("colsample_bytree"), 1.0)
                ),
                reg_alpha=float(_get_option(config.get("reg_alpha"), 0.0)),
                reg_lambda=float(_get_option(config.get("reg_lambda"), 1.0)),
                random_state=random_state,
                n_jobs=n_jobs,
            )
            model.fit(X_np, y_np, sample_weight=sample_weight_np)
    except Exception as e:
        raise RuntimeError(
            f"xgboost_train_regressor: Training failed with X shape {X_np.shape} - {e}"
        ) from e

    model_data = _serialize_model(model)

    return EastVariant(
        "xgboost_regressor",
        EastStruct(
            {
                "data": model_data,
                "n_features": n_features,
            }
        ),
    )


def xgboost_train_classifier_impl(
    X: EastArray,
    y: EastArray,
    config: EastStruct,
) -> EastVariant:
    """Train XGBoost classifier and return model blob."""
    try:
        import xgboost as xgb
    except ImportError as e:
        raise RuntimeError(
            "xgboost_train_classifier: xgboost not installed. "
            "Install with: pip install xgboost"
        ) from e

    try:
        X_np = east_matrix_to_numpy(X)
        y_np = east_int_vector_to_numpy(y)
    except Exception as e:
        raise RuntimeError(f"xgboost_train_classifier: Invalid input data - {e}") from e

    if X_np.shape[0] != y_np.shape[0]:
        raise RuntimeError(
            f"xgboost_train_classifier: X has {X_np.shape[0]} samples "
            f"but y has {y_np.shape[0]} samples"
        )

    n_features = X_np.shape[1]
    n_classes = len(np.unique(y_np))

    # Extract sample weights if provided
    sample_weight_opt = _get_option(config.get("sample_weight"), None)
    sample_weight_np = None
    if sample_weight_opt is not None:
        sample_weight_np = east_vector_to_numpy(sample_weight_opt)
        if sample_weight_np.shape[0] != X_np.shape[0]:
            raise RuntimeError(
                f"xgboost_train_classifier: sample_weight has {sample_weight_np.shape[0]} "
                f"elements but X has {X_np.shape[0]} samples"
            )

    try:
        random_state = _get_option(config.get("random_state"), None)
        if random_state is not None:
            random_state = int(random_state)

        n_jobs = _get_option(config.get("n_jobs"), -1)
        if n_jobs is not None:
            n_jobs = int(n_jobs)

        # Suppress XGBoost warnings during training
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=Warning)
            model = xgb.XGBClassifier(
                n_estimators=int(_get_option(config.get("n_estimators"), 100)),
                max_depth=int(_get_option(config.get("max_depth"), 6)),
                learning_rate=float(_get_option(config.get("learning_rate"), 0.3)),
                min_child_weight=int(_get_option(config.get("min_child_weight"), 1)),
                subsample=float(_get_option(config.get("subsample"), 1.0)),
                colsample_bytree=float(
                    _get_option(config.get("colsample_bytree"), 1.0)
                ),
                reg_alpha=float(_get_option(config.get("reg_alpha"), 0.0)),
                reg_lambda=float(_get_option(config.get("reg_lambda"), 1.0)),
                random_state=random_state,
                n_jobs=n_jobs,
            )
            model.fit(X_np, y_np, sample_weight=sample_weight_np)
    except Exception as e:
        raise RuntimeError(
            f"xgboost_train_classifier: Training failed with X shape {X_np.shape} - {e}"
        ) from e

    model_data = _serialize_model(model)

    return EastVariant(
        "xgboost_classifier",
        EastStruct(
            {
                "data": model_data,
                "n_features": n_features,
                "n_classes": n_classes,
            }
        ),
    )


def xgboost_predict_impl(
    model_blob: EastVariant,
    X: EastArray,
) -> EastArray:
    """Make predictions with XGBoost regressor."""
    if model_blob.type != "xgboost_regressor":
        raise RuntimeError(
            f"xgboost_predict: Expected xgboost_regressor, got {model_blob.type}"
        )

    try:
        X_np = east_matrix_to_numpy(X)
    except Exception as e:
        raise RuntimeError(f"xgboost_predict: Invalid input data - {e}") from e

    try:
        model = _deserialize_model(model_blob.value["data"])
        # Suppress warnings during prediction
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=Warning)
            y_pred = model.predict(X_np)
        return numpy_to_east_vector(y_pred)
    except Exception as e:
        raise RuntimeError(
            f"xgboost_predict: Prediction failed with X shape {X_np.shape} - {e}"
        ) from e


def xgboost_predict_class_impl(
    model_blob: EastVariant,
    X: EastArray,
) -> EastArray:
    """Predict class labels with XGBoost classifier."""
    if model_blob.type != "xgboost_classifier":
        raise RuntimeError(
            f"xgboost_predict_class: Expected xgboost_classifier, got {model_blob.type}"
        )

    try:
        X_np = east_matrix_to_numpy(X)
    except Exception as e:
        raise RuntimeError(f"xgboost_predict_class: Invalid input data - {e}") from e

    try:
        model = _deserialize_model(model_blob.value["data"])
        # Suppress warnings during prediction
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=Warning)
            y_pred = model.predict(X_np)
        return numpy_to_east_int_vector(y_pred)
    except Exception as e:
        raise RuntimeError(
            f"xgboost_predict_class: Prediction failed with X shape {X_np.shape} - {e}"
        ) from e


def xgboost_predict_proba_impl(
    model_blob: EastVariant,
    X: EastArray,
) -> EastArray:
    """Get class probabilities from XGBoost classifier."""
    if model_blob.type != "xgboost_classifier":
        raise RuntimeError(
            f"xgboost_predict_proba: Expected xgboost_classifier, got {model_blob.type}"
        )

    try:
        X_np = east_matrix_to_numpy(X)
    except Exception as e:
        raise RuntimeError(f"xgboost_predict_proba: Invalid input data - {e}") from e

    try:
        model = _deserialize_model(model_blob.value["data"])
        # Suppress warnings during prediction
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=Warning)
            proba = model.predict_proba(X_np)
        return numpy_to_east_matrix(proba)
    except Exception as e:
        raise RuntimeError(
            f"xgboost_predict_proba: Prediction failed with X shape {X_np.shape} - {e}"
        ) from e


def xgboost_train_quantile_impl(
    X: EastArray,
    y: EastArray,
    config: EastStruct,
) -> EastVariant:
    """Train XGBoost quantile regression models (one per quantile)."""
    try:
        import xgboost as xgb
    except ImportError as e:
        raise RuntimeError(
            "xgboost_train_quantile: xgboost not installed. "
            "Install with: pip install xgboost"
        ) from e

    try:
        X_np = east_matrix_to_numpy(X)
        y_np = east_vector_to_numpy(y)
    except Exception as e:
        raise RuntimeError(f"xgboost_train_quantile: Invalid input data - {e}") from e

    if X_np.shape[0] != y_np.shape[0]:
        raise RuntimeError(
            f"xgboost_train_quantile: X has {X_np.shape[0]} samples "
            f"but y has {y_np.shape[0]} samples"
        )

    n_features = X_np.shape[1]

    # Get quantiles from config
    quantiles_arr = config.get("quantiles")
    quantiles = [float(q) for q in quantiles_arr]

    # Validate quantiles
    for q in quantiles:
        if not 0 < q < 1:
            raise RuntimeError(
                f"xgboost_train_quantile: Quantiles must be in (0, 1), got {q}"
            )

    # Extract sample weights if provided
    sample_weight_opt = _get_option(config.get("sample_weight"), None)
    sample_weight_np = None
    if sample_weight_opt is not None:
        sample_weight_np = east_vector_to_numpy(sample_weight_opt)
        if sample_weight_np.shape[0] != X_np.shape[0]:
            raise RuntimeError(
                f"xgboost_train_quantile: sample_weight has {sample_weight_np.shape[0]} "
                f"elements but X has {X_np.shape[0]} samples"
            )

    try:
        random_state = _get_option(config.get("random_state"), None)
        if random_state is not None:
            random_state = int(random_state)

        n_jobs = _get_option(config.get("n_jobs"), -1)
        if n_jobs is not None:
            n_jobs = int(n_jobs)

        # Base parameters for all quantile models
        base_params = {
            "n_estimators": int(_get_option(config.get("n_estimators"), 100)),
            "max_depth": int(_get_option(config.get("max_depth"), 6)),
            "learning_rate": float(_get_option(config.get("learning_rate"), 0.3)),
            "min_child_weight": int(_get_option(config.get("min_child_weight"), 1)),
            "subsample": float(_get_option(config.get("subsample"), 1.0)),
            "colsample_bytree": float(_get_option(config.get("colsample_bytree"), 1.0)),
            "reg_alpha": float(_get_option(config.get("reg_alpha"), 0.0)),
            "reg_lambda": float(_get_option(config.get("reg_lambda"), 1.0)),
            "random_state": random_state,
            "n_jobs": n_jobs,
            "verbosity": 0,
        }

        # Train one model per quantile
        models = {}
        # Suppress XGBoost warnings during training
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=Warning)
            for q in quantiles:
                model = xgb.XGBRegressor(
                    objective="reg:quantileerror",
                    quantile_alpha=q,
                    **base_params,
                )
                model.fit(X_np, y_np, sample_weight=sample_weight_np)
                models[q] = model

    except Exception as e:
        raise RuntimeError(
            f"xgboost_train_quantile: Training failed with X shape {X_np.shape} - {e}"
        ) from e

    model_data = _serialize_model(models)

    return EastVariant(
        "xgboost_quantile",
        EastStruct(
            {
                "data": model_data,
                "quantiles": numpy_to_east_vector(np.array(quantiles)),
                "n_features": n_features,
            }
        ),
    )


def xgboost_predict_quantile_impl(
    model_blob: EastVariant,
    X: EastArray,
) -> EastStruct:
    """Predict quantiles with XGBoost quantile regressor."""
    if model_blob.type != "xgboost_quantile":
        raise RuntimeError(
            f"xgboost_predict_quantile: Expected xgboost_quantile, got {model_blob.type}"
        )

    try:
        X_np = east_matrix_to_numpy(X)
    except Exception as e:
        raise RuntimeError(f"xgboost_predict_quantile: Invalid input data - {e}") from e

    n_samples = X_np.shape[0]

    try:
        # Deserialize models dict
        models = _deserialize_model(model_blob.value["data"])

        # Use the model dict keys directly (they are the original quantile values)
        # This avoids float precision issues from serialization/deserialization
        quantiles_list = sorted(models.keys())
        n_quantiles = len(quantiles_list)

        # Predict each quantile
        predictions = np.zeros((n_samples, n_quantiles))
        # Suppress warnings during prediction
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=Warning)
            for i, q in enumerate(quantiles_list):
                predictions[:, i] = models[q].predict(X_np)

        return EastStruct(
            {
                "quantiles": numpy_to_east_vector(np.array(quantiles_list)),
                "predictions": numpy_to_east_matrix(predictions),
            }
        )
    except Exception as e:
        raise RuntimeError(
            f"xgboost_predict_quantile: Prediction failed with X shape {X_np.shape} - {e}"
        ) from e


# ============================================================================
# Platform Function Registration
# ============================================================================

xgboost_impl = [
    PlatformFunction(
        name="xgboost_train_regressor",
        inputs=[MatrixType, VectorType, XGBoostConfigType],
        output=ModelBlobType,
        type="sync",
        fn=xgboost_train_regressor_impl,
    ),
    PlatformFunction(
        name="xgboost_train_classifier",
        inputs=[MatrixType, IntVectorType, XGBoostConfigType],
        output=ModelBlobType,
        type="sync",
        fn=xgboost_train_classifier_impl,
    ),
    PlatformFunction(
        name="xgboost_predict",
        inputs=[ModelBlobType, MatrixType],
        output=VectorType,
        type="sync",
        fn=xgboost_predict_impl,
    ),
    PlatformFunction(
        name="xgboost_predict_class",
        inputs=[ModelBlobType, MatrixType],
        output=IntVectorType,
        type="sync",
        fn=xgboost_predict_class_impl,
    ),
    PlatformFunction(
        name="xgboost_predict_proba",
        inputs=[ModelBlobType, MatrixType],
        output=MatrixType,
        type="sync",
        fn=xgboost_predict_proba_impl,
    ),
    PlatformFunction(
        name="xgboost_train_quantile",
        inputs=[MatrixType, VectorType, XGBoostQuantileConfigType],
        output=ModelBlobType,
        type="sync",
        fn=xgboost_train_quantile_impl,
    ),
    PlatformFunction(
        name="xgboost_predict_quantile",
        inputs=[ModelBlobType, MatrixType],
        output=XGBoostQuantilePredictResultType,
        type="sync",
        fn=xgboost_predict_quantile_impl,
    ),
]

__all__ = [
    "xgboost_impl",
]
