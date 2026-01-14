#
# Copyright (c) 2025 Elara AI Pty Ltd
# Licensed under the Business Source License 1.1. See LICENSE.md for details.
#
"""MAPIE conformal prediction implementation (MAPIE 1.2.0 API).

Provides prediction intervals with coverage guarantees using
conformal prediction methods.
"""

import warnings

import numpy as np

from east.runtime.platform import PlatformFunction
from east.types.values import EastArray, EastBlob, EastStruct, EastVariant

from east_py_datascience.types import (
    MatrixType,
    VectorType,
    IntVectorType,
    _get_option,
    east_matrix_to_numpy,
    east_vector_to_numpy,
    east_int_vector_to_numpy,
    numpy_to_east_vector,
    numpy_to_east_matrix,
    numpy_to_east_int_vector,
)


# ============================================================================
# Type Definitions for MAPIE
# ============================================================================

from east.types.types import (
    ArrayType,
    BlobType,
    FloatType,
    IntegerType,
    NullType,
    OptionType,
    StructType,
    VariantType,
)

# Tagged model data - variant tag indicates base model type, value is the blob
MAPIEBaseModelDataType = VariantType(
    [
        ("xgboost", BlobType),
        ("lightgbm", BlobType),
        ("histogram", BlobType),
    ]
)

# MAPIE regressor model blob type (MAPIE 1.2.0)
MAPIERegressorBlobType = VariantType(
    [
        (
            "mapie_split",
            StructType(
                [
                    ("data", MAPIEBaseModelDataType),
                    ("n_features", IntegerType),
                    ("confidence_level", FloatType),
                ]
            ),
        ),
        (
            "mapie_cross",
            StructType(
                [
                    ("data", MAPIEBaseModelDataType),
                    ("n_features", IntegerType),
                    ("confidence_level", FloatType),
                ]
            ),
        ),
        (
            "mapie_cqr",
            StructType(
                [
                    ("data", MAPIEBaseModelDataType),
                    ("n_features", IntegerType),
                    ("confidence_level", FloatType),
                ]
            ),
        ),
    ]
)

# MAPIE classifier model blob type (single-case variant for consistency)
MAPIEClassifierBlobType = VariantType(
    [
        (
            "mapie_classifier",
            StructType(
                [
                    ("data", MAPIEBaseModelDataType),
                    ("n_features", IntegerType),
                    ("n_classes", IntegerType),
                    ("classes", ArrayType(IntegerType)),
                    ("confidence_level", FloatType),
                ]
            ),
        ),
    ]
)

# Interval result type
IntervalResultType = StructType(
    [
        ("lower", VectorType),
        ("pred", VectorType),
        ("upper", VectorType),
    ]
)

# Prediction set result type
PredictionSetResultType = StructType(
    [
        ("pred", ArrayType(IntegerType)),
        ("sets", ArrayType(ArrayType(IntegerType))),
        ("probabilities", MatrixType),
        ("set_sizes", ArrayType(IntegerType)),
    ]
)

# Uncertainty predictor type (for SHAP integration)
UncertaintyPredictorType = VariantType(
    [
        (
            "mapie_interval_width",
            StructType([("data", BlobType), ("n_features", IntegerType)]),
        ),
        (
            "mapie_set_size",
            StructType([("data", BlobType), ("n_features", IntegerType)]),
        ),
    ]
)

# Config types - use full XGBoost/LightGBM config types for complete parameter support
XGBoostConfigType = StructType(
    [
        ("n_estimators", OptionType(IntegerType)),
        ("max_depth", OptionType(IntegerType)),
        ("learning_rate", OptionType(FloatType)),
        ("min_child_weight", OptionType(IntegerType)),
        ("subsample", OptionType(FloatType)),
        ("colsample_bytree", OptionType(FloatType)),
        ("reg_alpha", OptionType(FloatType)),
        ("reg_lambda", OptionType(FloatType)),
        ("gamma", OptionType(FloatType)),
        ("random_state", OptionType(IntegerType)),
        ("n_jobs", OptionType(IntegerType)),
        ("sample_weight", OptionType(VectorType)),
        ("categorical_features", OptionType(ArrayType(IntegerType))),
        ("max_cat_to_onehot", OptionType(IntegerType)),
        ("max_cat_threshold", OptionType(IntegerType)),
    ]
)

LightGBMConfigType = StructType(
    [
        ("n_estimators", OptionType(IntegerType)),
        ("max_depth", OptionType(IntegerType)),
        ("learning_rate", OptionType(FloatType)),
        ("num_leaves", OptionType(IntegerType)),
        ("min_child_samples", OptionType(IntegerType)),
        ("subsample", OptionType(FloatType)),
        ("colsample_bytree", OptionType(FloatType)),
        ("reg_alpha", OptionType(FloatType)),
        ("reg_lambda", OptionType(FloatType)),
        ("random_state", OptionType(IntegerType)),
        ("n_jobs", OptionType(IntegerType)),
    ]
)

BaseModelType = VariantType(
    [
        ("xgboost", XGBoostConfigType),
        ("lightgbm", LightGBMConfigType),
    ]
)

ConformalMethodType = VariantType(
    [
        ("split", NullType),
        ("cross", NullType),
    ]
)

MAPIEConfigType = StructType(
    [
        ("base_model", BaseModelType),
        ("method", OptionType(ConformalMethodType)),
        ("confidence_level", OptionType(FloatType)),
        ("cv_folds", OptionType(IntegerType)),
        ("random_state", OptionType(IntegerType)),
    ]
)

MAPIECQRConfigType = StructType(
    [
        ("xgboost_config", XGBoostConfigType),
        ("confidence_level", OptionType(FloatType)),
        ("random_state", OptionType(IntegerType)),
    ]
)

ClassificationMethodType = VariantType(
    [
        ("lac", NullType),
        ("aps", NullType),
    ]
)

BaseClassifierType = VariantType(
    [
        ("xgboost", XGBoostConfigType),
        ("lightgbm", LightGBMConfigType),
    ]
)

MAPIEClassifierConfigType = StructType(
    [
        ("base_model", BaseClassifierType),
        ("method", OptionType(ClassificationMethodType)),
        ("confidence_level", OptionType(FloatType)),
        ("random_state", OptionType(IntegerType)),
    ]
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
# Categorical Feature Helpers
# ============================================================================


def _prepare_categorical_features(X_np, categorical_features, func_name: str):
    """Prepare feature matrix with categorical columns for training.

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
        categorical_features: list of column indices that are categorical, or None
        func_name: name of the calling function for error messages

    Returns:
        X_prepared - either the original numpy array or a pandas DataFrame
    """
    if categorical_features is None:
        return X_np

    import pandas as pd

    df = pd.DataFrame(X_np)
    for idx in categorical_features:
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
# Base Model Creation
# ============================================================================


def _create_base_regressor(base_model_variant: EastVariant, random_state):
    """Create base sklearn-compatible regressor from config."""
    model_type = base_model_variant.type
    config = base_model_variant.value

    if model_type == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as e:
            raise RuntimeError(
                "xgboost not installed. Install with: pip install xgboost"
            ) from e

        # Get n_jobs from config, default to -1
        n_jobs = _get_option(config.get("n_jobs"), -1)
        if n_jobs is not None:
            n_jobs = int(n_jobs)
        else:
            n_jobs = -1

        # Get categorical features config
        categorical_features = _get_option(config.get("categorical_features"), None)
        if categorical_features is not None:
            categorical_features = [int(x) for x in categorical_features]

        max_cat_to_onehot = _get_option(config.get("max_cat_to_onehot"), None)
        if max_cat_to_onehot is not None:
            max_cat_to_onehot = int(max_cat_to_onehot)

        max_cat_threshold = _get_option(config.get("max_cat_threshold"), None)
        if max_cat_threshold is not None:
            max_cat_threshold = int(max_cat_threshold)

        # Build kwargs
        kwargs = {
            "n_estimators": int(_get_option(config.get("n_estimators"), 100)),
            "max_depth": int(_get_option(config.get("max_depth"), 6)),
            "learning_rate": float(_get_option(config.get("learning_rate"), 0.3)),
            "min_child_weight": int(_get_option(config.get("min_child_weight"), 1)),
            "subsample": float(_get_option(config.get("subsample"), 1.0)),
            "colsample_bytree": float(_get_option(config.get("colsample_bytree"), 1.0)),
            "reg_alpha": float(_get_option(config.get("reg_alpha"), 0)),
            "reg_lambda": float(_get_option(config.get("reg_lambda"), 1)),
            "gamma": float(_get_option(config.get("gamma"), 0)),
            "random_state": random_state,
            "n_jobs": n_jobs,
            "verbosity": 0,
        }

        # Add categorical features params if specified
        if categorical_features is not None:
            kwargs["enable_categorical"] = True
        if max_cat_to_onehot is not None:
            kwargs["max_cat_to_onehot"] = max_cat_to_onehot
        if max_cat_threshold is not None:
            kwargs["max_cat_threshold"] = max_cat_threshold

        return XGBRegressor(**kwargs), categorical_features

    elif model_type == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as e:
            raise RuntimeError(
                "lightgbm not installed. Install with: pip install lightgbm"
            ) from e

        # Get n_jobs from config, default to -1
        n_jobs = _get_option(config.get("n_jobs"), -1)
        if n_jobs is not None:
            n_jobs = int(n_jobs)
        else:
            n_jobs = -1

        return LGBMRegressor(
            n_estimators=int(_get_option(config.get("n_estimators"), 100)),
            max_depth=int(_get_option(config.get("max_depth"), -1)),
            learning_rate=float(_get_option(config.get("learning_rate"), 0.1)),
            num_leaves=int(_get_option(config.get("num_leaves"), 31)),
            min_child_samples=int(_get_option(config.get("min_child_samples"), 20)),
            subsample=float(_get_option(config.get("subsample"), 1.0)),
            colsample_bytree=float(_get_option(config.get("colsample_bytree"), 1.0)),
            reg_alpha=float(_get_option(config.get("reg_alpha"), 0)),
            reg_lambda=float(_get_option(config.get("reg_lambda"), 0)),
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=-1,
        ), None  # No categorical features for LightGBM in this context
    else:
        raise ValueError(f"Unknown base model type: {model_type}")


def _create_base_classifier(base_model_variant: EastVariant, random_state):
    """Create base sklearn-compatible classifier from config."""
    model_type = base_model_variant.type
    config = base_model_variant.value

    if model_type == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as e:
            raise RuntimeError(
                "xgboost not installed. Install with: pip install xgboost"
            ) from e

        # Get n_jobs from config, default to -1
        n_jobs = _get_option(config.get("n_jobs"), -1)
        if n_jobs is not None:
            n_jobs = int(n_jobs)
        else:
            n_jobs = -1

        # Get categorical features config
        categorical_features = _get_option(config.get("categorical_features"), None)
        if categorical_features is not None:
            categorical_features = [int(x) for x in categorical_features]

        max_cat_to_onehot = _get_option(config.get("max_cat_to_onehot"), None)
        if max_cat_to_onehot is not None:
            max_cat_to_onehot = int(max_cat_to_onehot)

        max_cat_threshold = _get_option(config.get("max_cat_threshold"), None)
        if max_cat_threshold is not None:
            max_cat_threshold = int(max_cat_threshold)

        # Build kwargs
        kwargs = {
            "n_estimators": int(_get_option(config.get("n_estimators"), 100)),
            "max_depth": int(_get_option(config.get("max_depth"), 6)),
            "learning_rate": float(_get_option(config.get("learning_rate"), 0.3)),
            "min_child_weight": int(_get_option(config.get("min_child_weight"), 1)),
            "subsample": float(_get_option(config.get("subsample"), 1.0)),
            "colsample_bytree": float(_get_option(config.get("colsample_bytree"), 1.0)),
            "reg_alpha": float(_get_option(config.get("reg_alpha"), 0)),
            "reg_lambda": float(_get_option(config.get("reg_lambda"), 1)),
            "gamma": float(_get_option(config.get("gamma"), 0)),
            "random_state": random_state,
            "n_jobs": n_jobs,
            "verbosity": 0,
        }

        # Add categorical features params if specified
        if categorical_features is not None:
            kwargs["enable_categorical"] = True
        if max_cat_to_onehot is not None:
            kwargs["max_cat_to_onehot"] = max_cat_to_onehot
        if max_cat_threshold is not None:
            kwargs["max_cat_threshold"] = max_cat_threshold

        return XGBClassifier(**kwargs), categorical_features

    elif model_type == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as e:
            raise RuntimeError(
                "lightgbm not installed. Install with: pip install lightgbm"
            ) from e

        # Get n_jobs from config, default to -1
        n_jobs = _get_option(config.get("n_jobs"), -1)
        if n_jobs is not None:
            n_jobs = int(n_jobs)
        else:
            n_jobs = -1

        return LGBMClassifier(
            n_estimators=int(_get_option(config.get("n_estimators"), 100)),
            max_depth=int(_get_option(config.get("max_depth"), -1)),
            learning_rate=float(_get_option(config.get("learning_rate"), 0.1)),
            num_leaves=int(_get_option(config.get("num_leaves"), 31)),
            min_child_samples=int(_get_option(config.get("min_child_samples"), 20)),
            subsample=float(_get_option(config.get("subsample"), 1.0)),
            colsample_bytree=float(_get_option(config.get("colsample_bytree"), 1.0)),
            reg_alpha=float(_get_option(config.get("reg_alpha"), 0)),
            reg_lambda=float(_get_option(config.get("reg_lambda"), 0)),
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=-1,
        ), None  # No categorical features for LightGBM in this context
    else:
        raise ValueError(f"Unknown base classifier type: {model_type}")


# ============================================================================
# Regression Implementations (MAPIE 1.2.0 API)
# ============================================================================


def mapie_train_conformal_regressor_impl(
    X_train: EastArray,
    y_train: EastArray,
    X_calib: EastArray,
    y_calib: EastArray,
    config: EastStruct,
) -> EastVariant:
    """Train a MAPIE conformal regressor (MAPIE 1.2.0 API)."""
    try:
        from mapie.regression import SplitConformalRegressor, CrossConformalRegressor
        from mapie.conformity_scores import AbsoluteConformityScore
    except ImportError as e:
        raise RuntimeError(
            f"mapie_train_conformal_regressor: MAPIE not installed or wrong version. "
            f"Install with: pip install mapie>=1.2.0. Error: {e}"
        )

    # Convert inputs
    try:
        X_train_np = east_matrix_to_numpy(X_train)
        y_train_np = east_vector_to_numpy(y_train)
        X_calib_np = east_matrix_to_numpy(X_calib)
        y_calib_np = east_vector_to_numpy(y_calib)
    except Exception as e:
        raise RuntimeError(
            f"mapie_train_conformal_regressor: Invalid input data - {e}"
        ) from e

    # Validate shapes
    if X_train_np.shape[0] != y_train_np.shape[0]:
        raise RuntimeError(
            f"mapie_train_conformal_regressor: X_train has {X_train_np.shape[0]} samples "
            f"but y_train has {y_train_np.shape[0]} samples"
        )
    if X_calib_np.shape[0] != y_calib_np.shape[0]:
        raise RuntimeError(
            f"mapie_train_conformal_regressor: X_calib has {X_calib_np.shape[0]} samples "
            f"but y_calib has {y_calib_np.shape[0]} samples"
        )
    if X_train_np.shape[1] != X_calib_np.shape[1]:
        raise RuntimeError(
            f"mapie_train_conformal_regressor: X_train has {X_train_np.shape[1]} features "
            f"but X_calib has {X_calib_np.shape[1]} features"
        )

    # Extract config
    base_model_config = config.get("base_model")
    method_variant = _get_option(config.get("method"), None)
    confidence_level = float(_get_option(config.get("confidence_level"), 0.9))
    cv_folds = int(_get_option(config.get("cv_folds"), 5))
    random_state = _get_option(config.get("random_state"), None)
    if random_state is not None:
        random_state = int(random_state)

    # Determine method
    if method_variant is None:
        method = "split"
    else:
        method = method_variant.type

    # Create base model (returns tuple: model, categorical_features)
    base_model, categorical_features = _create_base_regressor(base_model_config, random_state)
    base_model_type = base_model_config.type

    # Extract sample_weight from base model config if provided
    base_config_value = base_model_config.value
    sample_weight_raw = _get_option(base_config_value.get("sample_weight"), None)
    fit_params = {}
    if sample_weight_raw is not None:
        fit_params["sample_weight"] = east_vector_to_numpy(sample_weight_raw)

    # Prepare categorical features for XGBoost (validates and converts to category dtype)
    X_train_np, categorical_features, _ = _prepare_categorical_features(
        X_train_np, categorical_features, "mapie_train_conformal_regressor"
    )
    X_calib_np = _apply_categorical_features(
        X_calib_np, categorical_features, "mapie_train_conformal_regressor"
    )

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=Warning)

            if method == "split":
                # Split conformal: train base model, then conformalize
                base_model.fit(X_train_np, y_train_np, **fit_params)
                # Create conformity score with relaxed eps for numerical precision
                conformity_score = AbsoluteConformityScore(sym=True)
                conformity_score.eps = 1e-06  # Relax consistency check tolerance
                mapie = SplitConformalRegressor(
                    estimator=base_model,
                    confidence_level=confidence_level,
                    conformity_score=conformity_score,
                    prefit=True,
                )
                mapie.conformalize(X_calib_np, y_calib_np)
                variant_type = "mapie_split"
            elif method == "cross":
                # Cross conformal: combine train and calib, use cross-validation
                if categorical_features is not None:
                    import pandas as pd
                    X_all = pd.concat([X_train_np, X_calib_np], ignore_index=True)
                else:
                    X_all = np.vstack([X_train_np, X_calib_np])
                y_all = np.hstack([y_train_np, y_calib_np])
                mapie = CrossConformalRegressor(
                    estimator=base_model,
                    confidence_level=confidence_level,
                    cv=cv_folds,
                )
                # For cross conformal, pass fit_params through fit_conformalize
                mapie.fit_conformalize(X_all, y_all, fit_params=fit_params if fit_params else None)
                variant_type = "mapie_cross"
            else:
                raise RuntimeError(
                    f"mapie_train_conformal_regressor: Unknown method '{method}'"
                )

    except Exception as e:
        raise RuntimeError(
            f"mapie_train_conformal_regressor: Training failed - {e}"
        ) from e

    # Serialize model with categorical features for prediction-time conversion
    import cloudpickle
    combined_model = {
        "mapie": mapie,
        "categorical_features": categorical_features,  # Store for prediction
    }
    model_bytes = EastBlob(cloudpickle.dumps(combined_model))
    n_features = X_train_np.shape[1]

    return EastVariant(
        variant_type,
        EastStruct(
            {
                "data": EastVariant(base_model_type, model_bytes),
                "n_features": n_features,
                "confidence_level": confidence_level,
            }
        ),
    )


def mapie_train_cqr_impl(
    X_train: EastArray,
    y_train: EastArray,
    X_calib: EastArray,
    y_calib: EastArray,
    config: EastStruct,
) -> EastVariant:
    """Train a MAPIE CQR (Conformalized Quantile Regression) model.

    Uses sklearn's HistGradientBoostingRegressor which has native quantile
    support and is natively supported by MAPIE's CQR.
    """
    try:
        from mapie.regression import ConformalizedQuantileRegressor
    except ImportError as e:
        raise RuntimeError(
            f"mapie_train_cqr: MAPIE not installed or wrong version. "
            f"Install with: pip install mapie>=1.2.0. Error: {e}"
        )
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
    except ImportError as e:
        raise RuntimeError(f"mapie_train_cqr: scikit-learn not installed. Error: {e}")

    # Convert inputs
    try:
        X_train_np = east_matrix_to_numpy(X_train)
        y_train_np = east_vector_to_numpy(y_train)
        X_calib_np = east_matrix_to_numpy(X_calib)
        y_calib_np = east_vector_to_numpy(y_calib)
    except Exception as e:
        raise RuntimeError(f"mapie_train_cqr: Invalid input data - {e}") from e

    # Validate shapes
    if X_train_np.shape[0] != y_train_np.shape[0]:
        raise RuntimeError(
            f"mapie_train_cqr: X_train has {X_train_np.shape[0]} samples "
            f"but y_train has {y_train_np.shape[0]} samples"
        )
    if X_calib_np.shape[0] != y_calib_np.shape[0]:
        raise RuntimeError(
            f"mapie_train_cqr: X_calib has {X_calib_np.shape[0]} samples "
            f"but y_calib has {y_calib_np.shape[0]} samples"
        )

    # Extract config - use xgboost_config params to configure HistGradientBoosting
    xgb_config = config.get("xgboost_config")
    confidence_level = float(_get_option(config.get("confidence_level"), 0.9))
    random_state = _get_option(config.get("random_state"), None)
    if random_state is not None:
        random_state = int(random_state)

    # Create HistGradientBoostingRegressor with quantile loss (native CQR support)
    # Map XGBoost-like params to HistGradientBoosting params
    hgb_params = {
        "loss": "quantile",  # Required for CQR
        "max_iter": int(_get_option(xgb_config.get("n_estimators"), 100)),
        "max_depth": int(_get_option(xgb_config.get("max_depth"), 6)),
        "learning_rate": float(_get_option(xgb_config.get("learning_rate"), 0.1)),
        "l2_regularization": float(_get_option(xgb_config.get("reg_lambda"), 0)),
        "random_state": random_state,
    }

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=Warning)

            # Combine train and calib for CQR (it handles internal splitting)
            X_all = np.vstack([X_train_np, X_calib_np])
            y_all = np.hstack([y_train_np, y_calib_np])

            # CQR with HistGradientBoostingRegressor - natively supported by MAPIE
            base_model = HistGradientBoostingRegressor(**hgb_params)
            mapie_cqr = ConformalizedQuantileRegressor(
                estimator=base_model,
                confidence_level=confidence_level,
                prefit=False,
            )
            mapie_cqr.fit(X_all, y_all)
            mapie_cqr.conformalize(X_all, y_all)

    except Exception as e:
        raise RuntimeError(f"mapie_train_cqr: Training failed - {e}") from e

    # Serialize model with categorical_features (None for CQR, but needed for
    # compatibility with mapie_predict_interval which expects a dict)
    import cloudpickle
    combined_model = {
        "mapie": mapie_cqr,
        "categorical_features": None,  # CQR doesn't support categorical features yet
    }
    model_bytes = EastBlob(cloudpickle.dumps(combined_model))
    n_features = X_train_np.shape[1]

    return EastVariant(
        "mapie_cqr",
        EastStruct(
            {
                "data": EastVariant("histogram", model_bytes),
                "n_features": n_features,
                "confidence_level": confidence_level,
            }
        ),
    )


def mapie_predict_interval_impl(
    model_blob: EastVariant,
    X: EastArray,
) -> EastStruct:
    """Predict with intervals using the model's calibrated confidence level."""
    model_type = model_blob.type
    model_data = model_blob.value

    # Load model - data is always a variant (tag=base_model_type, value=blob)
    data_field = model_data.get("data")
    model_bytes = data_field.value  # Extract blob from variant
    combined_model = _deserialize_model(model_bytes)
    model = combined_model["mapie"]
    categorical_features = combined_model.get("categorical_features")

    n_features = model_data.get("n_features")

    # Convert input
    try:
        X_np = east_matrix_to_numpy(X)
    except Exception as e:
        raise RuntimeError(f"mapie_predict_interval: Invalid input data - {e}") from e

    # Validate
    if X_np.shape[1] != n_features:
        raise RuntimeError(
            f"mapie_predict_interval: Model trained with {n_features} features "
            f"but X has {X_np.shape[1]} features"
        )

    # Apply categorical feature conversion if model was trained with them
    X_np = _apply_categorical_features(
        X_np, categorical_features, "mapie_predict_interval"
    )

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=Warning)

            # MAPIE 1.2.0: predict_interval() returns (predictions, intervals)
            # intervals shape: (n_samples, 2, n_confidence_levels)
            y_pred, y_intervals = model.predict_interval(X_np)

            # Extract lower and upper bounds (first confidence level)
            lower = y_intervals[:, 0, 0]
            upper = y_intervals[:, 1, 0]

    except Exception as e:
        raise RuntimeError(f"mapie_predict_interval: Prediction failed - {e}") from e

    return EastStruct(
        {
            "lower": numpy_to_east_vector(lower),
            "pred": numpy_to_east_vector(y_pred),
            "upper": numpy_to_east_vector(upper),
        }
    )


# ============================================================================
# Classification Implementations (MAPIE 1.2.0 API)
# ============================================================================


def mapie_train_conformal_classifier_impl(
    X_train: EastArray,
    y_train: EastArray,
    X_calib: EastArray,
    y_calib: EastArray,
    config: EastStruct,
) -> EastStruct:
    """Train a MAPIE conformal classifier (MAPIE 1.2.0 API)."""
    try:
        from mapie.classification import SplitConformalClassifier
    except ImportError as e:
        raise RuntimeError(
            f"mapie_train_conformal_classifier: MAPIE not installed or wrong version. "
            f"Install with: pip install mapie>=1.2.0. Error: {e}"
        )

    # Convert inputs
    try:
        X_train_np = east_matrix_to_numpy(X_train)
        y_train_np = east_int_vector_to_numpy(y_train)
        X_calib_np = east_matrix_to_numpy(X_calib)
        y_calib_np = east_int_vector_to_numpy(y_calib)
    except Exception as e:
        raise RuntimeError(
            f"mapie_train_conformal_classifier: Invalid input data - {e}"
        ) from e

    # Validate shapes
    if X_train_np.shape[0] != y_train_np.shape[0]:
        raise RuntimeError(
            f"mapie_train_conformal_classifier: X_train has {X_train_np.shape[0]} samples "
            f"but y_train has {y_train_np.shape[0]} samples"
        )
    if X_calib_np.shape[0] != y_calib_np.shape[0]:
        raise RuntimeError(
            f"mapie_train_conformal_classifier: X_calib has {X_calib_np.shape[0]} samples "
            f"but y_calib has {y_calib_np.shape[0]} samples"
        )
    if X_train_np.shape[1] != X_calib_np.shape[1]:
        raise RuntimeError(
            f"mapie_train_conformal_classifier: X_train has {X_train_np.shape[1]} features "
            f"but X_calib has {X_calib_np.shape[1]} features"
        )

    # Extract config
    base_model_config = config.get("base_model")
    method_variant = _get_option(config.get("method"), None)
    confidence_level = float(_get_option(config.get("confidence_level"), 0.9))
    random_state = _get_option(config.get("random_state"), None)
    if random_state is not None:
        random_state = int(random_state)

    # Determine conformity score method
    if method_variant is None:
        conformity_score = "lac"
    else:
        conformity_score = method_variant.type

    # Create base classifier (returns tuple: model, categorical_features)
    base_clf, categorical_features = _create_base_classifier(base_model_config, random_state)
    base_model_type = base_model_config.type

    # Extract sample_weight from base model config if provided
    base_config_value = base_model_config.value
    sample_weight_raw = _get_option(base_config_value.get("sample_weight"), None)
    fit_params = {}
    if sample_weight_raw is not None:
        fit_params["sample_weight"] = east_vector_to_numpy(sample_weight_raw)

    # Prepare categorical features for XGBoost (validates and converts to category dtype)
    X_train_np, categorical_features, _ = _prepare_categorical_features(
        X_train_np, categorical_features, "mapie_train_conformal_classifier"
    )
    X_calib_np = _apply_categorical_features(
        X_calib_np, categorical_features, "mapie_train_conformal_classifier"
    )

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=Warning)

            # Train classifier on training set
            base_clf.fit(X_train_np, y_train_np, **fit_params)

            # Create SplitConformalClassifier with prefit estimator
            mapie_clf = SplitConformalClassifier(
                estimator=base_clf,
                confidence_level=confidence_level,
                conformity_score=conformity_score,
                prefit=True,
            )
            mapie_clf.conformalize(X_calib_np, y_calib_np)

    except Exception as e:
        raise RuntimeError(
            f"mapie_train_conformal_classifier: Training failed - {e}"
        ) from e

    # Get class information
    classes = base_clf.classes_.tolist()
    n_classes = len(classes)

    # Serialize both models - MAPIE wrapper and base classifier
    # We need to store the base classifier separately because MAPIE 1.2.0
    # doesn't expose it via a public attribute after conformalize()
    # Also store categorical_features for prediction-time conversion
    import cloudpickle
    combined_model = {
        "mapie": mapie_clf,
        "base_clf": base_clf,
        "categorical_features": categorical_features,  # Store for prediction
    }
    model_bytes = EastBlob(cloudpickle.dumps(combined_model))
    n_features = X_train_np.shape[1]

    return EastVariant(
        "mapie_classifier",
        EastStruct(
            {
                "data": EastVariant(base_model_type, model_bytes),
                "n_features": n_features,
                "n_classes": n_classes,
                "classes": numpy_to_east_int_vector(np.array(classes)),
                "confidence_level": confidence_level,
            }
        ),
    )


def mapie_predict_set_impl(
    model_blob: EastVariant,
    X: EastArray,
) -> EastStruct:
    """Predict with prediction sets using the model's calibrated confidence level."""
    # Extract struct from variant (mapie_classifier)
    model_data = model_blob.value
    # Load model - data is a variant where tag=base_model_type, value=blob
    data_variant = model_data.get("data")
    model_bytes = data_variant.value  # Extract blob from variant
    combined_model = _deserialize_model(model_bytes)
    mapie_model = combined_model["mapie"]
    base_clf = combined_model["base_clf"]
    categorical_features = combined_model.get("categorical_features")  # May be None
    n_features = model_data.get("n_features")

    # Convert input
    try:
        X_np = east_matrix_to_numpy(X)
    except Exception as e:
        raise RuntimeError(f"mapie_predict_set: Invalid input data - {e}") from e

    # Validate
    if X_np.shape[1] != n_features:
        raise RuntimeError(
            f"mapie_predict_set: Model trained with {n_features} features "
            f"but X has {X_np.shape[1]} features"
        )

    # Apply categorical feature conversion if model was trained with them
    X_np = _apply_categorical_features(
        X_np, categorical_features, "mapie_predict_set"
    )

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=Warning)

            # MAPIE 1.2.0: predict_set() returns (predictions, sets)
            # sets shape: (n_samples, n_classes, n_confidence_levels)
            y_pred, y_sets = mapie_model.predict_set(X_np)

            # y_sets shape: (n_samples, n_classes, 1) for single confidence level
            sets_matrix = y_sets[:, :, 0].astype(int)  # (n_samples, n_classes)
            set_sizes = sets_matrix.sum(axis=1)  # Number of classes in each set

            # Get class probabilities via the stored base classifier
            proba = base_clf.predict_proba(X_np)  # (n_samples, n_classes)

    except Exception as e:
        raise RuntimeError(f"mapie_predict_set: Prediction failed - {e}") from e

    return EastStruct(
        {
            "pred": numpy_to_east_int_vector(y_pred),
            "sets": EastArray(
                ArrayType(ArrayType(IntegerType)),
                [
                    EastArray(ArrayType(IntegerType), [int(x) for x in row])
                    for row in sets_matrix
                ],
            ),
            "probabilities": numpy_to_east_matrix(proba),
            "set_sizes": numpy_to_east_int_vector(set_sizes),
        }
    )


# ============================================================================
# Uncertainty Predictor Implementations (for SHAP integration)
# ============================================================================


def mapie_uncertainty_predictor_regressor_impl(
    model_blob: EastVariant,
) -> EastVariant:
    """Create an uncertainty predictor from a MAPIE regressor.

    The resulting model blob predicts interval width (upper - lower) instead of
    point predictions. Use with SHAP KernelExplainer to explain uncertainty.
    """
    model_type = model_blob.type
    model_data = model_blob.value

    if model_type not in ("mapie_split", "mapie_cross", "mapie_cqr"):
        raise RuntimeError(
            f"mapie_uncertainty_predictor_regressor: Expected MAPIE regressor, "
            f"got {model_type}"
        )

    # Extract the serialized model data - data is always a variant now
    data_variant = model_data.get("data")
    model_bytes = data_variant.value
    n_features = model_data.get("n_features")

    return EastVariant(
        "mapie_interval_width",
        EastStruct(
            {
                "data": model_bytes,
                "n_features": n_features,
            }
        ),
    )


def mapie_uncertainty_predictor_classifier_impl(
    model_blob: EastVariant,
) -> EastVariant:
    """Create an uncertainty predictor from a MAPIE classifier.

    The resulting model blob predicts prediction set size instead of
    class probabilities. Use with SHAP KernelExplainer to explain uncertainty.
    """
    # Classifier is a variant with single case "mapie_classifier"
    model_data = model_blob.value
    # data is a variant (xgboost/lightgbm)
    data_variant = model_data.get("data")
    model_bytes = data_variant.value
    n_features = model_data.get("n_features")

    return EastVariant(
        "mapie_set_size",
        EastStruct(
            {
                "data": model_bytes,
                "n_features": n_features,
            }
        ),
    )


# ============================================================================
# Platform Function Registration
# ============================================================================

mapie_impl = [
    # Regression
    PlatformFunction(
        name="mapie_train_conformal_regressor",
        inputs=[MatrixType, VectorType, MatrixType, VectorType, MAPIEConfigType],
        output=MAPIERegressorBlobType,
        type="sync",
        fn=mapie_train_conformal_regressor_impl,
    ),
    PlatformFunction(
        name="mapie_train_cqr",
        inputs=[MatrixType, VectorType, MatrixType, VectorType, MAPIECQRConfigType],
        output=MAPIERegressorBlobType,
        type="sync",
        fn=mapie_train_cqr_impl,
    ),
    PlatformFunction(
        name="mapie_predict_interval",
        inputs=[MAPIERegressorBlobType, MatrixType],
        output=IntervalResultType,
        type="sync",
        fn=mapie_predict_interval_impl,
    ),
    # Classification
    PlatformFunction(
        name="mapie_train_conformal_classifier",
        inputs=[
            MatrixType,
            IntVectorType,
            MatrixType,
            IntVectorType,
            MAPIEClassifierConfigType,
        ],
        output=MAPIEClassifierBlobType,
        type="sync",
        fn=mapie_train_conformal_classifier_impl,
    ),
    PlatformFunction(
        name="mapie_predict_set",
        inputs=[MAPIEClassifierBlobType, MatrixType],
        output=PredictionSetResultType,
        type="sync",
        fn=mapie_predict_set_impl,
    ),
    # SHAP integration (uncertainty predictors)
    PlatformFunction(
        name="mapie_uncertainty_predictor_regressor",
        inputs=[MAPIERegressorBlobType],
        output=UncertaintyPredictorType,
        type="sync",
        fn=mapie_uncertainty_predictor_regressor_impl,
    ),
    PlatformFunction(
        name="mapie_uncertainty_predictor_classifier",
        inputs=[MAPIEClassifierBlobType],
        output=UncertaintyPredictorType,
        type="sync",
        fn=mapie_uncertainty_predictor_classifier_impl,
    ),
]

__all__ = [
    "mapie_impl",
]
