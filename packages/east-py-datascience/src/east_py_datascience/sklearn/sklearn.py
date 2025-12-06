"""Sklearn platform functions for East.

Provides core machine learning utilities: preprocessing, model selection, and metrics.
Uses ONNX for model serialization to enable portable inference.
"""

import numpy as np

from east.runtime.platform import PlatformFunction
from east.types.types import ArrayType, FloatType
from east.types.values import EastArray, EastBlob, EastStruct, EastVariant

from east_py_datascience.types import (
    MatrixType,
    VectorType,
    IntVectorType,
    SplitConfigType,
    SplitResultType,
    RegressionMetricsType,
    ClassificationMetricsType,
    ModelBlobType,
    RegressorChainConfigType,
    _get_option,
    _get_enum_tag,
    east_vector_to_numpy,
    east_matrix_to_numpy,
    east_int_vector_to_numpy,
    numpy_to_east_vector,
    numpy_to_east_matrix,
)

# ============================================================================
# ONNX Helpers
# ============================================================================


def _sklearn_to_onnx(model, n_features: int) -> EastBlob:
    """Convert sklearn model to ONNX bytes."""
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    initial_type = [("X", FloatTensorType([None, n_features]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)
    return EastBlob(onnx_model.SerializeToString())


def _onnx_transform(onnx_blob: EastBlob, X: EastArray) -> EastArray:
    """Run transform (e.g., scaler) using ONNX Runtime."""
    import onnxruntime as ort

    onnx_bytes = bytes(onnx_blob)
    X_np = east_matrix_to_numpy(X)

    session = ort.InferenceSession(onnx_bytes)
    input_name = session.get_inputs()[0].name

    outputs = session.run(None, {input_name: X_np})
    X_transformed = outputs[0]

    return numpy_to_east_matrix(X_transformed)


# ============================================================================
# Platform Function Implementations
# ============================================================================


def sklearn_train_test_split_impl(
    X: EastArray,
    y: EastArray,
    config: EastStruct,
) -> EastStruct:
    """Split arrays into train and test subsets."""
    try:
        from sklearn.model_selection import train_test_split
    except ImportError as e:
        raise RuntimeError(
            "sklearn_train_test_split: scikit-learn not installed. "
            "Install with: pip install scikit-learn"
        ) from e

    try:
        X_np = east_matrix_to_numpy(X)
        y_np = east_vector_to_numpy(y)
    except Exception as e:
        raise RuntimeError(f"sklearn_train_test_split: Invalid input data - {e}") from e

    if X_np.shape[0] != y_np.shape[0]:
        raise RuntimeError(
            f"sklearn_train_test_split: X has {X_np.shape[0]} samples "
            f"but y has {y_np.shape[0]} samples"
        )

    test_size = _get_option(config.get("test_size"), 0.2)
    random_state = _get_option(config.get("random_state"), None)
    shuffle = _get_option(config.get("shuffle"), True)

    if random_state is not None:
        random_state = int(random_state)

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_np, y_np, test_size=test_size, random_state=random_state, shuffle=shuffle
        )
    except Exception as e:
        raise RuntimeError(
            f"sklearn_train_test_split: Split failed with X shape {X_np.shape} - {e}"
        ) from e

    return EastStruct(
        {
            "X_train": numpy_to_east_matrix(X_train),
            "X_test": numpy_to_east_matrix(X_test),
            "y_train": numpy_to_east_vector(y_train),
            "y_test": numpy_to_east_vector(y_test),
        }
    )


def sklearn_standard_scaler_fit_impl(X: EastArray) -> EastVariant:
    """Fit StandardScaler and return model blob."""
    try:
        from sklearn.preprocessing import StandardScaler
    except ImportError as e:
        raise RuntimeError(
            "sklearn_standard_scaler_fit: scikit-learn not installed. "
            "Install with: pip install scikit-learn"
        ) from e

    try:
        X_np = east_matrix_to_numpy(X)
    except Exception as e:
        raise RuntimeError(
            f"sklearn_standard_scaler_fit: Invalid input data - {e}"
        ) from e

    n_features = X_np.shape[1]

    try:
        scaler = StandardScaler()
        scaler.fit(X_np)
        onnx_data = _sklearn_to_onnx(scaler, n_features)
    except Exception as e:
        raise RuntimeError(
            f"sklearn_standard_scaler_fit: Fitting failed with X shape {X_np.shape} - {e}"
        ) from e

    return EastVariant(
        "standard_scaler",
        EastStruct(
            {
                "onnx": onnx_data,
                "n_features": n_features,
            }
        ),
    )


def sklearn_standard_scaler_transform_impl(
    model_blob: EastVariant,
    X: EastArray,
) -> EastArray:
    """Transform data using fitted scaler."""
    if model_blob.type != "standard_scaler":
        raise RuntimeError(
            f"sklearn_standard_scaler_transform: Expected standard_scaler, got {model_blob.type}"
        )

    try:
        onnx_blob = model_blob.value["onnx"]
        return _onnx_transform(onnx_blob, X)
    except Exception as e:
        raise RuntimeError(
            f"sklearn_standard_scaler_transform: Transform failed - {e}"
        ) from e


def sklearn_min_max_scaler_fit_impl(X: EastArray) -> EastVariant:
    """Fit MinMaxScaler and return model blob."""
    try:
        from sklearn.preprocessing import MinMaxScaler
    except ImportError as e:
        raise RuntimeError(
            "sklearn_min_max_scaler_fit: scikit-learn not installed. "
            "Install with: pip install scikit-learn"
        ) from e

    try:
        X_np = east_matrix_to_numpy(X)
    except Exception as e:
        raise RuntimeError(
            f"sklearn_min_max_scaler_fit: Invalid input data - {e}"
        ) from e

    n_features = X_np.shape[1]

    try:
        scaler = MinMaxScaler()
        scaler.fit(X_np)
        onnx_data = _sklearn_to_onnx(scaler, n_features)
    except Exception as e:
        raise RuntimeError(
            f"sklearn_min_max_scaler_fit: Fitting failed with X shape {X_np.shape} - {e}"
        ) from e

    return EastVariant(
        "min_max_scaler",
        EastStruct(
            {
                "onnx": onnx_data,
                "n_features": n_features,
            }
        ),
    )


def sklearn_min_max_scaler_transform_impl(
    model_blob: EastVariant,
    X: EastArray,
) -> EastArray:
    """Transform data using fitted min-max scaler."""
    if model_blob.type != "min_max_scaler":
        raise RuntimeError(
            f"sklearn_min_max_scaler_transform: Expected min_max_scaler, got {model_blob.type}"
        )

    try:
        onnx_blob = model_blob.value["onnx"]
        return _onnx_transform(onnx_blob, X)
    except Exception as e:
        raise RuntimeError(
            f"sklearn_min_max_scaler_transform: Transform failed - {e}"
        ) from e


def sklearn_metrics_regression_impl(
    y_true: EastArray,
    y_pred: EastArray,
) -> EastStruct:
    """Compute regression metrics."""
    try:
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    except ImportError as e:
        raise RuntimeError(
            "sklearn_metrics_regression: scikit-learn not installed. "
            "Install with: pip install scikit-learn"
        ) from e

    try:
        y_true_np = east_vector_to_numpy(y_true)
        y_pred_np = east_vector_to_numpy(y_pred)
    except Exception as e:
        raise RuntimeError(
            f"sklearn_metrics_regression: Invalid input data - {e}"
        ) from e

    if y_true_np.shape[0] != y_pred_np.shape[0]:
        raise RuntimeError(
            f"sklearn_metrics_regression: y_true has {y_true_np.shape[0]} samples "
            f"but y_pred has {y_pred_np.shape[0]} samples"
        )

    try:
        mse = mean_squared_error(y_true_np, y_pred_np)
        mae = mean_absolute_error(y_true_np, y_pred_np)
        r2 = r2_score(y_true_np, y_pred_np)

        # MAPE (avoid division by zero)
        mask = y_true_np != 0
        if mask.any():
            mape = float(
                np.mean(np.abs((y_true_np[mask] - y_pred_np[mask]) / y_true_np[mask]))
                * 100
            )
        else:
            mape = 0.0
    except Exception as e:
        raise RuntimeError(
            f"sklearn_metrics_regression: Metric computation failed - {e}"
        ) from e

    return EastStruct(
        {
            "mse": float(mse),
            "rmse": float(np.sqrt(mse)),
            "mae": float(mae),
            "r2": float(r2),
            "mape": mape,
        }
    )


def sklearn_metrics_classification_impl(
    y_true: EastArray,
    y_pred: EastArray,
) -> EastStruct:
    """Compute classification metrics."""
    try:
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
        )
    except ImportError as e:
        raise RuntimeError(
            "sklearn_metrics_classification: scikit-learn not installed. "
            "Install with: pip install scikit-learn"
        ) from e

    try:
        y_true_np = east_int_vector_to_numpy(y_true)
        y_pred_np = east_int_vector_to_numpy(y_pred)
    except Exception as e:
        raise RuntimeError(
            f"sklearn_metrics_classification: Invalid input data - {e}"
        ) from e

    if y_true_np.shape[0] != y_pred_np.shape[0]:
        raise RuntimeError(
            f"sklearn_metrics_classification: y_true has {y_true_np.shape[0]} samples "
            f"but y_pred has {y_pred_np.shape[0]} samples"
        )

    try:
        return EastStruct(
            {
                "accuracy": float(accuracy_score(y_true_np, y_pred_np)),
                "precision": float(
                    precision_score(
                        y_true_np, y_pred_np, average="weighted", zero_division=0
                    )
                ),
                "recall": float(
                    recall_score(
                        y_true_np, y_pred_np, average="weighted", zero_division=0
                    )
                ),
                "f1": float(
                    f1_score(y_true_np, y_pred_np, average="weighted", zero_division=0)
                ),
            }
        )
    except Exception as e:
        raise RuntimeError(
            f"sklearn_metrics_classification: Metric computation failed - {e}"
        ) from e


# ============================================================================
# RegressorChain Helpers
# ============================================================================


def _serialize_model(model) -> EastBlob:
    """Serialize model using cloudpickle."""
    import cloudpickle

    return EastBlob(cloudpickle.dumps(model))


def _deserialize_model(blob: EastBlob):
    """Deserialize model using cloudpickle."""
    import cloudpickle

    return cloudpickle.loads(bytes(blob))


def _create_base_estimator(estimator_variant: EastVariant):
    """Create a sklearn-compatible base estimator from config variant."""
    estimator_type = estimator_variant.type
    config = estimator_variant.value

    if estimator_type == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=int(_get_option(config.get("n_estimators"), 100)),
            max_depth=int(_get_option(config.get("max_depth"), 6)),
            learning_rate=float(_get_option(config.get("learning_rate"), 0.3)),
            min_child_weight=int(_get_option(config.get("min_child_weight"), 1)),
            subsample=float(_get_option(config.get("subsample"), 1.0)),
            colsample_bytree=float(_get_option(config.get("colsample_bytree"), 1.0)),
            reg_alpha=float(_get_option(config.get("reg_alpha"), 0)),
            reg_lambda=float(_get_option(config.get("reg_lambda"), 1)),
            random_state=_get_option(config.get("random_state"), None),
            n_jobs=int(_get_option(config.get("n_jobs"), -1)),
        )

    elif estimator_type == "lightgbm":
        from lightgbm import LGBMRegressor

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
            random_state=_get_option(config.get("random_state"), None),
            n_jobs=int(_get_option(config.get("n_jobs"), -1)),
            verbosity=-1,
        )

    elif estimator_type == "ngboost":
        from ngboost import NGBRegressor
        from ngboost.distns import Normal, LogNormal

        dist_variant = _get_option(config.get("distribution"), None)
        dist_type = _get_enum_tag(dist_variant) if dist_variant else "normal"
        dist = LogNormal if dist_type == "lognormal" else Normal

        return NGBRegressor(
            n_estimators=int(_get_option(config.get("n_estimators"), 500)),
            learning_rate=float(_get_option(config.get("learning_rate"), 0.01)),
            minibatch_frac=float(_get_option(config.get("minibatch_frac"), 1.0)),
            col_sample=float(_get_option(config.get("col_sample"), 1.0)),
            random_state=_get_option(config.get("random_state"), None),
            Dist=dist,
            verbose=False,
        )

    elif estimator_type == "gp":
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import (
            RBF,
            Matern,
            RationalQuadratic,
            DotProduct,
            ConstantKernel,
        )

        kernel_variant = _get_option(config.get("kernel"), None)
        kernel_type = _get_enum_tag(kernel_variant) if kernel_variant else "rbf"

        kernel_map = {
            "rbf": ConstantKernel() * RBF(),
            "matern_1_2": ConstantKernel() * Matern(nu=0.5),
            "matern_3_2": ConstantKernel() * Matern(nu=1.5),
            "matern_5_2": ConstantKernel() * Matern(nu=2.5),
            "rational_quadratic": ConstantKernel() * RationalQuadratic(),
            "dot_product": ConstantKernel() * DotProduct(),
        }
        kernel = kernel_map.get(kernel_type, ConstantKernel() * RBF())

        alpha = _get_option(config.get("alpha"), 1e-10)
        if alpha is not None:
            alpha = float(alpha)
        else:
            alpha = 1e-10

        n_restarts = _get_option(config.get("n_restarts_optimizer"), 0)
        if n_restarts is not None:
            n_restarts = int(n_restarts)
        else:
            n_restarts = 0

        normalize_y = _get_option(config.get("normalize_y"), False)
        if normalize_y is not None:
            normalize_y = bool(normalize_y)
        else:
            normalize_y = False

        random_state = _get_option(config.get("random_state"), None)
        if random_state is not None:
            random_state = int(random_state)

        return GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha,
            n_restarts_optimizer=n_restarts,
            normalize_y=normalize_y,
            random_state=random_state,
        )

    else:
        raise RuntimeError(
            f"_create_base_estimator: Unknown estimator type: {estimator_type}"
        )


def sklearn_regressor_chain_train_impl(
    X: EastArray,
    Y: EastArray,
    config: EastStruct,
) -> EastVariant:
    """Train a RegressorChain for multi-target regression."""
    try:
        from sklearn.multioutput import RegressorChain
    except ImportError as e:
        raise RuntimeError(
            "sklearn_regressor_chain_train: scikit-learn not installed. "
            "Install with: pip install scikit-learn"
        ) from e

    try:
        X_np = east_matrix_to_numpy(X)
        # Y is a matrix (n_samples x n_targets)
        Y_np = np.array([[float(x) for x in row] for row in Y], dtype=np.float32)
    except Exception as e:
        raise RuntimeError(
            f"sklearn_regressor_chain_train: Invalid input data - {e}"
        ) from e

    if X_np.shape[0] != Y_np.shape[0]:
        raise RuntimeError(
            f"sklearn_regressor_chain_train: X has {X_np.shape[0]} samples "
            f"but Y has {Y_np.shape[0]} samples"
        )

    n_features = X_np.shape[1]
    n_targets = Y_np.shape[1]

    # Get base estimator config
    base_estimator_variant = config.get("base_estimator")
    try:
        base_estimator = _create_base_estimator(base_estimator_variant)
    except Exception as e:
        raise RuntimeError(
            f"sklearn_regressor_chain_train: Failed to create base estimator - {e}"
        ) from e

    base_estimator_type = base_estimator_variant.type

    # Get order
    order = _get_option(config.get("order"), None)
    if order is not None:
        order = [int(x) for x in order]

    # Get random_state
    random_state = _get_option(config.get("random_state"), None)
    if random_state is not None:
        random_state = int(random_state)

    try:
        # Create and train chain
        chain = RegressorChain(
            base_estimator=base_estimator,
            order=order,
            random_state=random_state,
        )
        chain.fit(X_np, Y_np)
    except Exception as e:
        raise RuntimeError(
            f"sklearn_regressor_chain_train: Training failed with X shape {X_np.shape} - {e}"
        ) from e

    return EastVariant(
        "regressor_chain",
        EastStruct(
            {
                "data": _serialize_model(chain),
                "n_features": n_features,
                "n_targets": n_targets,
                "base_estimator_type": base_estimator_type,
            }
        ),
    )


def sklearn_regressor_chain_predict_impl(
    model_blob: EastVariant,
    X: EastArray,
) -> EastArray:
    """Predict using a fitted RegressorChain."""
    if model_blob.type != "regressor_chain":
        raise RuntimeError(
            f"sklearn_regressor_chain_predict: Expected regressor_chain, got {model_blob.type}"
        )

    try:
        X_np = east_matrix_to_numpy(X)
    except Exception as e:
        raise RuntimeError(
            f"sklearn_regressor_chain_predict: Invalid input data - {e}"
        ) from e

    try:
        chain = _deserialize_model(model_blob.value["data"])
        predictions = chain.predict(X_np)
    except Exception as e:
        raise RuntimeError(
            f"sklearn_regressor_chain_predict: Prediction failed with X shape {X_np.shape} - {e}"
        ) from e

    # Return as matrix (n_samples x n_targets)
    return numpy_to_east_matrix(predictions)


# ============================================================================
# Platform Function Registration
# ============================================================================

# Multi-target type for RegressorChain
MultiTargetType = ArrayType(ArrayType(FloatType))

sklearn_impl = [
    PlatformFunction(
        name="sklearn_train_test_split",
        inputs=[MatrixType, VectorType, SplitConfigType],
        output=SplitResultType,
        type="sync",
        fn=sklearn_train_test_split_impl,
    ),
    PlatformFunction(
        name="sklearn_standard_scaler_fit",
        inputs=[MatrixType],
        output=ModelBlobType,
        type="sync",
        fn=sklearn_standard_scaler_fit_impl,
    ),
    PlatformFunction(
        name="sklearn_standard_scaler_transform",
        inputs=[ModelBlobType, MatrixType],
        output=MatrixType,
        type="sync",
        fn=sklearn_standard_scaler_transform_impl,
    ),
    PlatformFunction(
        name="sklearn_min_max_scaler_fit",
        inputs=[MatrixType],
        output=ModelBlobType,
        type="sync",
        fn=sklearn_min_max_scaler_fit_impl,
    ),
    PlatformFunction(
        name="sklearn_min_max_scaler_transform",
        inputs=[ModelBlobType, MatrixType],
        output=MatrixType,
        type="sync",
        fn=sklearn_min_max_scaler_transform_impl,
    ),
    PlatformFunction(
        name="sklearn_metrics_regression",
        inputs=[VectorType, VectorType],
        output=RegressionMetricsType,
        type="sync",
        fn=sklearn_metrics_regression_impl,
    ),
    PlatformFunction(
        name="sklearn_metrics_classification",
        inputs=[IntVectorType, IntVectorType],
        output=ClassificationMetricsType,
        type="sync",
        fn=sklearn_metrics_classification_impl,
    ),
    PlatformFunction(
        name="sklearn_regressor_chain_train",
        inputs=[MatrixType, MultiTargetType, RegressorChainConfigType],
        output=ModelBlobType,
        type="sync",
        fn=sklearn_regressor_chain_train_impl,
    ),
    PlatformFunction(
        name="sklearn_regressor_chain_predict",
        inputs=[ModelBlobType, MatrixType],
        output=MultiTargetType,
        type="sync",
        fn=sklearn_regressor_chain_predict_impl,
    ),
]

__all__ = [
    "sklearn_impl",
    # Re-export types from types.py
    "SplitConfigType",
    "SplitResultType",
    "RegressionMetricsType",
    "ClassificationMetricsType",
    "ModelBlobType",
]
