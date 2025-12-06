"""XGBoost platform functions for East.

Provides gradient boosting for regression and classification.
Uses cloudpickle for model serialization to enable portable inference.
"""

import numpy as np

from east.runtime.platform import PlatformFunction
from east.types.values import EastArray, EastBlob, EastStruct, EastVariant

from east_py_datascience.types import (
    MatrixType,
    VectorType,
    IntVectorType,
    XGBoostConfigType,
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

    try:
        random_state = _get_option(config.get("random_state"), None)
        if random_state is not None:
            random_state = int(random_state)

        n_jobs = _get_option(config.get("n_jobs"), -1)
        if n_jobs is not None:
            n_jobs = int(n_jobs)

        model = xgb.XGBRegressor(
            n_estimators=int(_get_option(config.get("n_estimators"), 100)),
            max_depth=int(_get_option(config.get("max_depth"), 6)),
            learning_rate=float(_get_option(config.get("learning_rate"), 0.3)),
            min_child_weight=int(_get_option(config.get("min_child_weight"), 1)),
            subsample=float(_get_option(config.get("subsample"), 1.0)),
            colsample_bytree=float(_get_option(config.get("colsample_bytree"), 1.0)),
            reg_alpha=float(_get_option(config.get("reg_alpha"), 0.0)),
            reg_lambda=float(_get_option(config.get("reg_lambda"), 1.0)),
            random_state=random_state,
            n_jobs=n_jobs,
        )
        model.fit(X_np, y_np)
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

    try:
        random_state = _get_option(config.get("random_state"), None)
        if random_state is not None:
            random_state = int(random_state)

        n_jobs = _get_option(config.get("n_jobs"), -1)
        if n_jobs is not None:
            n_jobs = int(n_jobs)

        model = xgb.XGBClassifier(
            n_estimators=int(_get_option(config.get("n_estimators"), 100)),
            max_depth=int(_get_option(config.get("max_depth"), 6)),
            learning_rate=float(_get_option(config.get("learning_rate"), 0.3)),
            min_child_weight=int(_get_option(config.get("min_child_weight"), 1)),
            subsample=float(_get_option(config.get("subsample"), 1.0)),
            colsample_bytree=float(_get_option(config.get("colsample_bytree"), 1.0)),
            reg_alpha=float(_get_option(config.get("reg_alpha"), 0.0)),
            reg_lambda=float(_get_option(config.get("reg_lambda"), 1.0)),
            random_state=random_state,
            n_jobs=n_jobs,
        )
        model.fit(X_np, y_np)
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
        proba = model.predict_proba(X_np)
        return numpy_to_east_matrix(proba)
    except Exception as e:
        raise RuntimeError(
            f"xgboost_predict_proba: Prediction failed with X shape {X_np.shape} - {e}"
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
]

__all__ = [
    "xgboost_impl",
]
