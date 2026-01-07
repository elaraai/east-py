#
# Copyright (c) 2025 Elara AI Pty Ltd
# Licensed under the Business Source License 1.1. See LICENSE.md for details.
#
"""Sklearn platform functions for East.

Provides core machine learning utilities: preprocessing, model selection, and metrics.
Uses ONNX for model serialization to enable portable inference.
"""

import warnings

# Suppress sklearn warnings
warnings.filterwarnings("ignore", module="sklearn")

import numpy as np  # noqa: E402

from east.runtime.platform import PlatformFunction  # noqa: E402
from east.types.types import ArrayType  # noqa: E402
from east.types.values import EastArray, EastBlob, EastStruct, EastVariant  # noqa: E402

from east_py_datascience.types import (  # noqa: E402
    MatrixType,
    VectorType,
    IntVectorType,
    SplitConfigType,
    SplitResultType,
    ThreeWaySplitConfigType,
    ThreeWaySplitResultType,
    ModelBlobType,
    RegressorChainConfigType,
    # Flexible metrics types
    RegressionMetricType,
    MetricResultType,
    MetricsResultType,
    MultiMetricsConfigType,
    MultiMetricResultType,
    MultiMetricsResultType,
    ClassificationMetricType,
    ClassificationMetricsConfigType,
    ClassificationMetricResultType,
    ClassificationMetricResultsType,
    MultiClassificationConfigType,
    MultiClassificationMetricResultType,
    MultiClassificationMetricResultsType,
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
    """Split arrays into train and test subsets.

    When stratify is provided, automatically filters out classes with fewer than
    2 samples (minimum required for train/test split). Returns rejected_indices
    containing the original indices of filtered samples.
    """
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
    stratify_labels = _get_option(config.get("stratify"), None)

    if random_state is not None:
        random_state = int(random_state)

    # Track rejected indices (originally empty)
    rejected_indices = []

    # Get minimum samples per stratify class (default 2 for 2-way split)
    min_stratify_samples = _get_option(config.get("min_stratify_samples"), 2)
    if min_stratify_samples is not None:
        min_stratify_samples = int(min_stratify_samples)
    else:
        min_stratify_samples = 2

    # Convert stratify labels to numpy array if provided
    stratify_arr = None
    if stratify_labels is not None:
        stratify_arr = np.array([int(x) for x in stratify_labels])
        if len(stratify_arr) != X_np.shape[0]:
            raise RuntimeError(
                f"sklearn_train_test_split: stratify has {len(stratify_arr)} labels "
                f"but X has {X_np.shape[0]} samples"
            )

        # Filter rare classes: need at least min_stratify_samples for split
        unique_classes, counts = np.unique(stratify_arr, return_counts=True)
        rare_classes = set(unique_classes[counts < min_stratify_samples])

        if rare_classes:
            # Find indices to keep and reject
            keep_mask = np.array([c not in rare_classes for c in stratify_arr])
            rejected_indices = np.where(~keep_mask)[0].tolist()

            # Filter data
            X_np = X_np[keep_mask]
            y_np = y_np[keep_mask]
            stratify_arr = stratify_arr[keep_mask]

    # Track indices through splits to map back to original positions
    if stratify_labels is not None:
        original_indices = np.arange(len(stratify_labels))
        keep_mask_initial = np.array([int(x) not in set(np.unique(np.array([int(x) for x in stratify_labels]))[np.unique(np.array([int(x) for x in stratify_labels]), return_counts=True)[1] < min_stratify_samples]) for x in stratify_labels])
        kept_indices = original_indices[keep_mask_initial] if len(rejected_indices) > 0 else original_indices
    else:
        kept_indices = np.arange(X_np.shape[0])

    try:
        if stratify_arr is not None:
            indices = np.arange(X_np.shape[0])
            X_train, X_test, y_train, y_test, strat_train, strat_test, idx_train, idx_test = train_test_split(
                X_np, y_np, stratify_arr, indices,
                test_size=test_size, random_state=random_state, shuffle=shuffle,
                stratify=stratify_arr
            )

            # Post-split validation: ensure all stratify classes appear in BOTH splits
            classes_in_train = set(strat_train)
            classes_in_test = set(strat_test)

            # Find classes missing from any split
            all_classes = set(stratify_arr)
            missing_classes = all_classes - (classes_in_train & classes_in_test)

            if missing_classes:
                # Remove samples of missing classes from all splits
                train_keep = np.array([c not in missing_classes for c in strat_train])
                test_keep = np.array([c not in missing_classes for c in strat_test])

                # Add rejected indices (map back to original positions)
                train_rejected = kept_indices[idx_train[~train_keep]].tolist()
                test_rejected = kept_indices[idx_test[~test_keep]].tolist()
                rejected_indices.extend(train_rejected + test_rejected)

                # Filter the splits
                X_train, y_train = X_train[train_keep], y_train[train_keep]
                X_test, y_test = X_test[test_keep], y_test[test_keep]
        else:
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
            "rejected_indices": EastArray(ArrayType("integer"), [int(i) for i in rejected_indices]),
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


def sklearn_train_val_test_split_impl(
    X: EastArray,
    Y: EastArray,
    config: EastStruct,
) -> EastStruct:
    """Split arrays into train, validation, and test subsets.

    When stratify is provided, automatically filters out classes with fewer than
    min_stratify_samples (default 3 for 3-way split). Returns rejected_indices
    containing the original indices of filtered samples.
    """
    try:
        from sklearn.model_selection import train_test_split
    except ImportError as e:
        raise RuntimeError(
            "sklearn_train_val_test_split: scikit-learn not installed. "
            "Install with: pip install scikit-learn"
        ) from e

    try:
        X_np = east_matrix_to_numpy(X)
        Y_np = east_matrix_to_numpy(Y)
    except Exception as e:
        raise RuntimeError(
            f"sklearn_train_val_test_split: Invalid input data - {e}"
        ) from e

    if X_np.shape[0] != Y_np.shape[0]:
        raise RuntimeError(
            f"sklearn_train_val_test_split: X has {X_np.shape[0]} samples "
            f"but Y has {Y_np.shape[0]} samples"
        )

    val_size = _get_option(config.get("val_size"), 0.15)
    test_size = _get_option(config.get("test_size"), 0.15)
    random_state = _get_option(config.get("random_state"), None)
    shuffle = _get_option(config.get("shuffle"), True)
    stratify_labels = _get_option(config.get("stratify"), None)

    if random_state is not None:
        random_state = int(random_state)

    # Track rejected indices (originally empty)
    rejected_indices = []

    # Get minimum samples per stratify class (default 3 for 3-way split)
    min_stratify_samples = _get_option(config.get("min_stratify_samples"), 3)
    if min_stratify_samples is not None:
        min_stratify_samples = int(min_stratify_samples)
    else:
        min_stratify_samples = 3

    # Convert stratify labels to numpy array if provided
    stratify_arr = None
    if stratify_labels is not None:
        stratify_arr = np.array([int(x) for x in stratify_labels])
        if len(stratify_arr) != X_np.shape[0]:
            raise RuntimeError(
                f"sklearn_train_val_test_split: stratify has {len(stratify_arr)} labels "
                f"but X has {X_np.shape[0]} samples"
            )

        # Filter rare classes: need at least min_stratify_samples for 3-way split
        unique_classes, counts = np.unique(stratify_arr, return_counts=True)
        rare_classes = set(unique_classes[counts < min_stratify_samples])

        if rare_classes:
            # Find indices to keep and reject
            keep_mask = np.array([c not in rare_classes for c in stratify_arr])
            rejected_indices = np.where(~keep_mask)[0].tolist()

            # Filter data
            X_np = X_np[keep_mask]
            Y_np = Y_np[keep_mask]
            stratify_arr = stratify_arr[keep_mask]

    # Track indices through splits to map back to original positions
    # After pre-filtering rare classes, create index mapping
    if stratify_labels is not None:
        # kept_indices maps filtered position -> original position
        original_indices = np.arange(len(stratify_labels))
        keep_mask_initial = np.array([int(x) not in set(np.unique(np.array([int(x) for x in stratify_labels]))[np.unique(np.array([int(x) for x in stratify_labels]), return_counts=True)[1] < min_stratify_samples]) for x in stratify_labels])
        kept_indices = original_indices[keep_mask_initial] if len(rejected_indices) > 0 else original_indices
    else:
        kept_indices = np.arange(X_np.shape[0])

    try:
        # First split: separate test set
        if stratify_arr is not None:
            indices_temp = np.arange(X_np.shape[0])
            X_temp, X_test, Y_temp, Y_test, strat_temp, strat_test, idx_temp, idx_test = train_test_split(
                X_np, Y_np, stratify_arr, indices_temp,
                test_size=test_size, random_state=random_state, shuffle=shuffle,
                stratify=stratify_arr
            )
        else:
            indices_temp = np.arange(X_np.shape[0])
            X_temp, X_test, Y_temp, Y_test, idx_temp, idx_test = train_test_split(
                X_np, Y_np, indices_temp,
                test_size=test_size, random_state=random_state, shuffle=shuffle
            )
            strat_temp = None
            strat_test = None

        # Second split: separate validation from training
        # Adjust val_ratio for remaining data
        val_ratio = val_size / (1.0 - test_size)
        if strat_temp is not None:
            X_train, X_val, Y_train, Y_val, strat_train, strat_val, idx_train, idx_val = train_test_split(
                X_temp, Y_temp, strat_temp, idx_temp,
                test_size=val_ratio, random_state=random_state, shuffle=shuffle,
                stratify=strat_temp
            )
        else:
            X_train, X_val, Y_train, Y_val, idx_train, idx_val = train_test_split(
                X_temp,
                Y_temp,
                idx_temp,
                test_size=val_ratio,
                random_state=random_state,
                shuffle=shuffle,
            )
            strat_train = None
            strat_val = None

        # Post-split validation: ensure all stratify classes appear in ALL splits
        if stratify_arr is not None:
            classes_in_train = set(strat_train)
            classes_in_val = set(strat_val)
            classes_in_test = set(strat_test)

            # Find classes missing from any split
            all_classes = set(stratify_arr)
            missing_classes = all_classes - (classes_in_train & classes_in_val & classes_in_test)

            if missing_classes:
                # Remove samples of missing classes from all splits
                train_keep = np.array([c not in missing_classes for c in strat_train])
                val_keep = np.array([c not in missing_classes for c in strat_val])
                test_keep = np.array([c not in missing_classes for c in strat_test])

                # Add rejected indices (map back to original positions)
                train_rejected = kept_indices[idx_train[~train_keep]].tolist()
                val_rejected = kept_indices[idx_val[~val_keep]].tolist()
                test_rejected = kept_indices[idx_test[~test_keep]].tolist()
                rejected_indices.extend(train_rejected + val_rejected + test_rejected)

                # Filter the splits
                X_train, Y_train = X_train[train_keep], Y_train[train_keep]
                X_val, Y_val = X_val[val_keep], Y_val[val_keep]
                X_test, Y_test = X_test[test_keep], Y_test[test_keep]

    except Exception as e:
        raise RuntimeError(
            f"sklearn_train_val_test_split: Split failed with X shape {X_np.shape} - {e}"
        ) from e

    return EastStruct(
        {
            "X_train": numpy_to_east_matrix(X_train),
            "X_val": numpy_to_east_matrix(X_val),
            "X_test": numpy_to_east_matrix(X_test),
            "Y_train": numpy_to_east_matrix(Y_train),
            "Y_val": numpy_to_east_matrix(Y_val),
            "Y_test": numpy_to_east_matrix(Y_test),
            "rejected_indices": EastArray(ArrayType("integer"), [int(i) for i in rejected_indices]),
        }
    )


# ============================================================================
# Flexible Metrics Implementation
# ============================================================================

# Regression metric function mapping
REGRESSION_METRIC_FUNCTIONS = {
    "mse": lambda y_true, y_pred: float(np.mean((y_true - y_pred) ** 2)),
    "rmse": lambda y_true, y_pred: float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
    "mae": lambda y_true, y_pred: float(np.mean(np.abs(y_true - y_pred))),
    "r2": None,  # Uses sklearn
    "mape": None,  # Custom implementation
    "explained_variance": None,  # Uses sklearn
    "max_error": lambda y_true, y_pred: float(np.max(np.abs(y_true - y_pred))),
    "median_ae": lambda y_true, y_pred: float(np.median(np.abs(y_true - y_pred))),
}


def _compute_regression_metric(
    metric_name: str, y_true: np.ndarray, y_pred: np.ndarray, param: float = None
) -> float:
    """Compute a single regression metric.

    Args:
        metric_name: Name of the metric to compute
        y_true: Ground truth values
        y_pred: Predicted values
        param: Optional parameter for metrics that need it (alpha for pinball, delta for huber, power for tweedie)
    """
    from sklearn import metrics as sklearn_metrics

    if metric_name == "mse":
        return float(sklearn_metrics.mean_squared_error(y_true, y_pred))
    elif metric_name == "rmse":
        return float(np.sqrt(sklearn_metrics.mean_squared_error(y_true, y_pred)))
    elif metric_name == "mae":
        return float(sklearn_metrics.mean_absolute_error(y_true, y_pred))
    elif metric_name == "r2":
        return float(sklearn_metrics.r2_score(y_true, y_pred))
    elif metric_name == "mape":
        # Avoid division by zero
        mask = y_true != 0
        if mask.any():
            return float(
                np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            )
        return 0.0
    elif metric_name == "explained_variance":
        return float(sklearn_metrics.explained_variance_score(y_true, y_pred))
    elif metric_name == "max_error":
        return float(sklearn_metrics.max_error(y_true, y_pred))
    elif metric_name == "median_ae":
        return float(sklearn_metrics.median_absolute_error(y_true, y_pred))
    elif metric_name == "mean_error":
        # Bias: mean(pred - true), should be ~0 for unbiased predictions
        return float(np.mean(y_pred - y_true))
    elif metric_name == "pinball_loss":
        # Proper scoring rule for quantile regression
        alpha = param if param is not None else 0.5  # default to median
        return float(sklearn_metrics.mean_pinball_loss(y_true, y_pred, alpha=alpha))
    elif metric_name == "huber":
        # Huber loss: robust to outliers
        delta = param if param is not None else 1.0
        residuals = y_pred - y_true
        abs_residuals = np.abs(residuals)
        quadratic = np.minimum(abs_residuals, delta)
        linear = abs_residuals - quadratic
        return float(np.mean(0.5 * quadratic**2 + delta * linear))
    elif metric_name == "mean_tweedie_deviance":
        # For skewed distributions (power=0: normal, power=1: Poisson, power=2: Gamma)
        power = param if param is not None else 0.0
        return float(sklearn_metrics.mean_tweedie_deviance(y_true, y_pred, power=power))
    else:
        raise ValueError(f"Unknown regression metric: {metric_name}")


def sklearn_compute_metrics_impl(
    y_true: EastArray,
    y_pred: EastArray,
    metrics: EastArray,
) -> EastArray:
    """Compute regression metrics for single-target predictions."""
    try:
        y_true_np = east_vector_to_numpy(y_true)
        y_pred_np = east_vector_to_numpy(y_pred)
    except Exception as e:
        raise RuntimeError(f"sklearn_compute_metrics: Invalid input data - {e}") from e

    if y_true_np.shape[0] != y_pred_np.shape[0]:
        raise RuntimeError(
            f"sklearn_compute_metrics: y_true has {y_true_np.shape[0]} samples "
            f"but y_pred has {y_pred_np.shape[0]} samples"
        )

    results = []
    for metric_variant in metrics:
        metric_name = metric_variant.type
        # Extract param for metrics that need it (pinball_loss, huber, mean_tweedie_deviance)
        param = metric_variant.value if metric_variant.value is not None else None
        try:
            value = _compute_regression_metric(metric_name, y_true_np, y_pred_np, param)
            results.append(
                EastStruct(
                    {
                        "metric": EastVariant(metric_name, param),
                        "value": value,
                    }
                )
            )
        except Exception:
            pass  # Skip metrics that fail

    return EastArray(MetricResultType, results)


def sklearn_compute_metrics_multi_impl(
    Y_true: EastArray,
    Y_pred: EastArray,
    metrics: EastArray,
    config: EastStruct,
) -> EastArray:
    """Compute regression metrics for multi-target predictions."""
    try:
        Y_true_np = east_matrix_to_numpy(Y_true)
        Y_pred_np = east_matrix_to_numpy(Y_pred)
    except Exception as e:
        raise RuntimeError(
            f"sklearn_compute_metrics_multi: Invalid input data - {e}"
        ) from e

    if Y_true_np.shape != Y_pred_np.shape:
        raise RuntimeError(
            f"sklearn_compute_metrics_multi: Y_true has shape {Y_true_np.shape} "
            f"but Y_pred has shape {Y_pred_np.shape}"
        )

    n_targets = Y_true_np.shape[1]

    # Get aggregation mode
    agg_opt = _get_option(config.get("aggregation"), None)
    aggregation = agg_opt.type if agg_opt else "per_target"

    results = []
    for metric_variant in metrics:
        metric_name = metric_variant.type
        # Extract param for metrics that need it
        param = metric_variant.value if metric_variant.value is not None else None
        try:
            # Compute per target
            per_target_values = []
            for i in range(n_targets):
                val = _compute_regression_metric(
                    metric_name, Y_true_np[:, i], Y_pred_np[:, i], param
                )
                per_target_values.append(val)

            # Format based on aggregation
            if aggregation == "per_target":
                result_value = EastVariant(
                    "per_target", numpy_to_east_vector(np.array(per_target_values))
                )
            else:  # uniform_average
                result_value = EastVariant("scalar", float(np.mean(per_target_values)))

            results.append(
                EastStruct(
                    {
                        "metric": EastVariant(metric_name, param),
                        "value": result_value,
                    }
                )
            )
        except Exception:
            pass  # Skip metrics that fail

    return EastArray(MultiMetricResultType, results)


# Classification metric function mapping
CLASSIFICATION_METRICS_WITH_AVERAGE = {"precision", "recall", "f1", "jaccard"}


def _compute_classification_metric(
    metric_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "macro",
) -> float:
    """Compute a single classification metric."""
    from sklearn import metrics as sklearn_metrics

    kwargs = {}
    if metric_name in CLASSIFICATION_METRICS_WITH_AVERAGE:
        kwargs["average"] = average
        kwargs["zero_division"] = 0

    if metric_name == "accuracy":
        return float(sklearn_metrics.accuracy_score(y_true, y_pred))
    elif metric_name == "balanced_accuracy":
        return float(sklearn_metrics.balanced_accuracy_score(y_true, y_pred))
    elif metric_name == "precision":
        return float(sklearn_metrics.precision_score(y_true, y_pred, **kwargs))
    elif metric_name == "recall":
        return float(sklearn_metrics.recall_score(y_true, y_pred, **kwargs))
    elif metric_name == "f1":
        return float(sklearn_metrics.f1_score(y_true, y_pred, **kwargs))
    elif metric_name == "matthews_corrcoef":
        return float(sklearn_metrics.matthews_corrcoef(y_true, y_pred))
    elif metric_name == "cohen_kappa":
        return float(sklearn_metrics.cohen_kappa_score(y_true, y_pred))
    elif metric_name == "jaccard":
        return float(sklearn_metrics.jaccard_score(y_true, y_pred, **kwargs))
    else:
        raise ValueError(f"Unknown classification metric: {metric_name}")


def sklearn_compute_classification_metrics_impl(
    y_true: EastArray,
    y_pred: EastArray,
    metrics: EastArray,
    config: EastStruct,
) -> EastArray:
    """Compute classification metrics for single-target predictions."""
    try:
        y_true_np = east_int_vector_to_numpy(y_true)
        y_pred_np = east_int_vector_to_numpy(y_pred)
    except Exception as e:
        raise RuntimeError(
            f"sklearn_compute_classification_metrics: Invalid input data - {e}"
        ) from e

    if y_true_np.shape[0] != y_pred_np.shape[0]:
        raise RuntimeError(
            f"sklearn_compute_classification_metrics: y_true has {y_true_np.shape[0]} samples "
            f"but y_pred has {y_pred_np.shape[0]} samples"
        )

    # Get average mode
    avg_opt = _get_option(config.get("average"), None)
    average = avg_opt.type if avg_opt else "macro"

    results = []
    for metric_variant in metrics:
        metric_name = metric_variant.type
        try:
            value = _compute_classification_metric(
                metric_name, y_true_np, y_pred_np, average
            )
            results.append(
                EastStruct(
                    {
                        "metric": EastVariant(metric_name, None),
                        "value": value,
                    }
                )
            )
        except Exception:
            pass  # Skip metrics that fail

    return EastArray(ClassificationMetricResultType, results)


def sklearn_compute_classification_metrics_multi_impl(
    Y_true: EastArray,
    Y_pred: EastArray,
    metrics: EastArray,
    config: EastStruct,
) -> EastArray:
    """Compute classification metrics for multi-target predictions."""
    try:
        Y_true_np = east_matrix_to_numpy(Y_true).astype(int)
        Y_pred_np = east_matrix_to_numpy(Y_pred).astype(int)
    except Exception as e:
        raise RuntimeError(
            f"sklearn_compute_classification_metrics_multi: Invalid input data - {e}"
        ) from e

    if Y_true_np.shape != Y_pred_np.shape:
        raise RuntimeError(
            f"sklearn_compute_classification_metrics_multi: Y_true has shape {Y_true_np.shape} "
            f"but Y_pred has shape {Y_pred_np.shape}"
        )

    n_targets = Y_true_np.shape[1]

    # Get config options
    avg_opt = _get_option(config.get("average"), None)
    average = avg_opt.type if avg_opt else "macro"
    agg_opt = _get_option(config.get("aggregation"), None)
    aggregation = agg_opt.type if agg_opt else "per_target"

    results = []
    for metric_variant in metrics:
        metric_name = metric_variant.type
        try:
            # Compute per target
            per_target_values = []
            for i in range(n_targets):
                val = _compute_classification_metric(
                    metric_name, Y_true_np[:, i], Y_pred_np[:, i], average
                )
                per_target_values.append(val)

            # Format based on aggregation
            if aggregation == "per_target":
                result_value = EastVariant(
                    "per_target", numpy_to_east_vector(np.array(per_target_values))
                )
            else:  # uniform_average
                result_value = EastVariant("scalar", float(np.mean(per_target_values)))

            results.append(
                EastStruct(
                    {
                        "metric": EastVariant(metric_name, None),
                        "value": result_value,
                    }
                )
            )
        except Exception:
            pass  # Skip metrics that fail

    return EastArray(MultiClassificationMetricResultType, results)


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
            estimator=base_estimator,
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

sklearn_impl = [
    PlatformFunction(
        name="sklearn_train_test_split",
        inputs=[MatrixType, VectorType, SplitConfigType],
        output=SplitResultType,
        type="sync",
        fn=sklearn_train_test_split_impl,
    ),
    PlatformFunction(
        name="sklearn_train_val_test_split",
        inputs=[MatrixType, MatrixType, ThreeWaySplitConfigType],
        output=ThreeWaySplitResultType,
        type="sync",
        fn=sklearn_train_val_test_split_impl,
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
    # Flexible regression metrics
    PlatformFunction(
        name="sklearn_compute_metrics",
        inputs=[VectorType, VectorType, ArrayType(RegressionMetricType)],
        output=MetricsResultType,
        type="sync",
        fn=sklearn_compute_metrics_impl,
    ),
    PlatformFunction(
        name="sklearn_compute_metrics_multi",
        inputs=[
            MatrixType,
            MatrixType,
            ArrayType(RegressionMetricType),
            MultiMetricsConfigType,
        ],
        output=MultiMetricsResultType,
        type="sync",
        fn=sklearn_compute_metrics_multi_impl,
    ),
    # Flexible classification metrics
    PlatformFunction(
        name="sklearn_compute_classification_metrics",
        inputs=[
            IntVectorType,
            IntVectorType,
            ArrayType(ClassificationMetricType),
            ClassificationMetricsConfigType,
        ],
        output=ClassificationMetricResultsType,
        type="sync",
        fn=sklearn_compute_classification_metrics_impl,
    ),
    PlatformFunction(
        name="sklearn_compute_classification_metrics_multi",
        inputs=[
            MatrixType,
            MatrixType,
            ArrayType(ClassificationMetricType),
            MultiClassificationConfigType,
        ],
        output=MultiClassificationMetricResultsType,
        type="sync",
        fn=sklearn_compute_classification_metrics_multi_impl,
    ),
    # RegressorChain
    PlatformFunction(
        name="sklearn_regressor_chain_train",
        inputs=[MatrixType, MatrixType, RegressorChainConfigType],
        output=ModelBlobType,
        type="sync",
        fn=sklearn_regressor_chain_train_impl,
    ),
    PlatformFunction(
        name="sklearn_regressor_chain_predict",
        inputs=[ModelBlobType, MatrixType],
        output=MatrixType,
        type="sync",
        fn=sklearn_regressor_chain_predict_impl,
    ),
]

__all__ = [
    "sklearn_impl",
    # Re-export types from types.py
    "SplitConfigType",
    "SplitResultType",
    "ThreeWaySplitConfigType",
    "ThreeWaySplitResultType",
    "ModelBlobType",
]
