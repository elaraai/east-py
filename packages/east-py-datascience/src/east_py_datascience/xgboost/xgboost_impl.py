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
# Categorical Feature Helpers
# ============================================================================


def _prepare_categorical_features(X_np, categorical_features, func_name: str):
    """Prepare feature matrix with categorical columns.

    Args:
        X_np: numpy array of features
        categorical_features: list of column indices that are categorical, or None
        func_name: name of the calling function for error messages

    Returns:
        Tuple of (X_prepared, cat_indices, enable_categorical) where:
        - X_prepared is either the original numpy array or a pandas DataFrame
        - cat_indices is the list of categorical indices or None
        - enable_categorical is True if categorical features are used
    """
    if categorical_features is None:
        return X_np, None, False

    cat_indices = [int(i) for i in categorical_features]

    # Validate indices
    for idx in cat_indices:
        if idx < 0 or idx >= X_np.shape[1]:
            raise RuntimeError(
                f"{func_name}: categorical_features index {idx} "
                f"out of bounds for {X_np.shape[1]} features"
            )

    # Convert to DataFrame with categorical columns
    # XGBoost requires integer category indices, so convert floats to ints first
    import pandas as pd

    df = pd.DataFrame(X_np)
    for idx in cat_indices:
        col = df[idx]
        # Check that all values are whole numbers (can be safely converted to int)
        non_integer_mask = col != col.astype(int)
        if non_integer_mask.any():
            bad_row = non_integer_mask.idxmax()
            bad_value = col[bad_row]
            raise RuntimeError(
                f"{func_name}: categorical column {idx} contains non-integer value "
                f"{bad_value} at row {bad_row}. Categorical features must contain "
                f"whole numbers (0.0, 1.0, 2.0, ...) representing category indices."
            )
        df[idx] = col.astype(int).astype("category")

    return df, cat_indices, True


def _apply_categorical_features(X_np, categorical_features, func_name: str):
    """Apply categorical dtypes to feature matrix for prediction.

    Args:
        X_np: numpy array of features
        categorical_features: EastArray of column indices or None
        func_name: name of the calling function for error messages

    Returns:
        X_prepared - either the original numpy array or a pandas DataFrame
    """
    cat_features_opt = _get_option(categorical_features, None)
    if cat_features_opt is None:
        return X_np

    cat_indices = east_int_vector_to_numpy(cat_features_opt)

    import pandas as pd

    df = pd.DataFrame(X_np)
    for idx in cat_indices:
        if idx < 0 or idx >= X_np.shape[1]:
            raise RuntimeError(
                f"{func_name}: categorical_features index {idx} "
                f"out of bounds for {X_np.shape[1]} features"
            )
        col = df[idx]
        # Check that all values are whole numbers (can be safely converted to int)
        non_integer_mask = col != col.astype(int)
        if non_integer_mask.any():
            bad_row = non_integer_mask.idxmax()
            bad_value = col[bad_row]
            raise RuntimeError(
                f"{func_name}: categorical column {idx} contains non-integer value "
                f"{bad_value} at row {bad_row}. Categorical features must contain "
                f"whole numbers (0.0, 1.0, 2.0, ...) representing category indices."
            )
        df[idx] = col.astype(int).astype("category")

    return df


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

    # Extract categorical features config
    categorical_features = _get_option(config.get("categorical_features"), None)
    X_train, cat_indices, enable_categorical = _prepare_categorical_features(
        X_np, categorical_features, "xgboost_train_regressor"
    )

    # Extract categorical config options
    max_cat_to_onehot = _get_option(config.get("max_cat_to_onehot"), None)
    max_cat_threshold = _get_option(config.get("max_cat_threshold"), None)

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
                enable_categorical=enable_categorical,
                max_cat_to_onehot=int(max_cat_to_onehot) if max_cat_to_onehot else 4,
                max_cat_threshold=int(max_cat_threshold) if max_cat_threshold else 64,
            )
            model.fit(X_train, y_np, sample_weight=sample_weight_np)
    except Exception as e:
        raise RuntimeError(
            f"xgboost_train_regressor: Training failed with X shape {X_np.shape} - {e}"
        ) from e

    model_data = _serialize_model(model)

    # Store categorical features for prediction
    cat_features_blob = None
    if cat_indices is not None:
        cat_features_blob = EastVariant(
            "some", numpy_to_east_int_vector(np.array(cat_indices))
        )
    else:
        cat_features_blob = EastVariant("none", None)

    return EastVariant(
        "xgboost_regressor",
        EastStruct(
            {
                "data": model_data,
                "n_features": n_features,
                "categorical_features": cat_features_blob,
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

    # Extract categorical features config
    categorical_features = _get_option(config.get("categorical_features"), None)
    X_train, cat_indices, enable_categorical = _prepare_categorical_features(
        X_np, categorical_features, "xgboost_train_classifier"
    )

    # Extract categorical config options
    max_cat_to_onehot = _get_option(config.get("max_cat_to_onehot"), None)
    max_cat_threshold = _get_option(config.get("max_cat_threshold"), None)

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
                enable_categorical=enable_categorical,
                max_cat_to_onehot=int(max_cat_to_onehot) if max_cat_to_onehot else 4,
                max_cat_threshold=int(max_cat_threshold) if max_cat_threshold else 64,
            )
            model.fit(X_train, y_np, sample_weight=sample_weight_np)
    except Exception as e:
        raise RuntimeError(
            f"xgboost_train_classifier: Training failed with X shape {X_np.shape} - {e}"
        ) from e

    model_data = _serialize_model(model)

    # Store categorical features for prediction
    cat_features_blob = None
    if cat_indices is not None:
        cat_features_blob = EastVariant(
            "some", numpy_to_east_int_vector(np.array(cat_indices))
        )
    else:
        cat_features_blob = EastVariant("none", None)

    return EastVariant(
        "xgboost_classifier",
        EastStruct(
            {
                "data": model_data,
                "n_features": n_features,
                "n_classes": n_classes,
                "categorical_features": cat_features_blob,
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

    # Apply categorical features if present
    X_pred = _apply_categorical_features(
        X_np, model_blob.value.get("categorical_features"), "xgboost_predict"
    )

    try:
        model = _deserialize_model(model_blob.value["data"])
        # Suppress warnings during prediction
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=Warning)
            y_pred = model.predict(X_pred)
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

    # Apply categorical features if present
    X_pred = _apply_categorical_features(
        X_np, model_blob.value.get("categorical_features"), "xgboost_predict_class"
    )

    try:
        model = _deserialize_model(model_blob.value["data"])
        # Suppress warnings during prediction
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=Warning)
            y_pred = model.predict(X_pred)
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

    # Apply categorical features if present
    X_pred = _apply_categorical_features(
        X_np, model_blob.value.get("categorical_features"), "xgboost_predict_proba"
    )

    try:
        model = _deserialize_model(model_blob.value["data"])
        # Suppress warnings during prediction
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=Warning)
            proba = model.predict_proba(X_pred)
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

    # Extract categorical features config
    categorical_features = _get_option(config.get("categorical_features"), None)
    X_train, cat_indices, enable_categorical = _prepare_categorical_features(
        X_np, categorical_features, "xgboost_train_quantile"
    )

    # Extract categorical config options
    max_cat_to_onehot = _get_option(config.get("max_cat_to_onehot"), None)
    max_cat_threshold = _get_option(config.get("max_cat_threshold"), None)

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
            "enable_categorical": enable_categorical,
            "max_cat_to_onehot": int(max_cat_to_onehot) if max_cat_to_onehot else 4,
            "max_cat_threshold": int(max_cat_threshold) if max_cat_threshold else 64,
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
                model.fit(X_train, y_np, sample_weight=sample_weight_np)
                models[q] = model

    except Exception as e:
        raise RuntimeError(
            f"xgboost_train_quantile: Training failed with X shape {X_np.shape} - {e}"
        ) from e

    model_data = _serialize_model(models)

    # Store categorical features for prediction
    cat_features_blob = None
    if cat_indices is not None:
        cat_features_blob = EastVariant(
            "some", numpy_to_east_int_vector(np.array(cat_indices))
        )
    else:
        cat_features_blob = EastVariant("none", None)

    return EastVariant(
        "xgboost_quantile",
        EastStruct(
            {
                "data": model_data,
                "quantiles": numpy_to_east_vector(np.array(quantiles)),
                "n_features": n_features,
                "categorical_features": cat_features_blob,
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

    # Apply categorical features if present
    X_pred = _apply_categorical_features(
        X_np, model_blob.value.get("categorical_features"), "xgboost_predict_quantile"
    )

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
                predictions[:, i] = models[q].predict(X_pred)

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
