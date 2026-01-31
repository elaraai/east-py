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
from east.types.types import ArrayType, FloatType  # noqa: E402
from east.types.values import EastArray, EastBlob, EastStruct, EastVariant  # noqa: E402

from east_py_datascience.types import (  # noqa: E402
    MatrixType,
    VectorType,
    IntVectorType,
    SplitConfigType,
    SplitResultType,
    ModelBlobType,
    RegressorChainConfigType,
    ClassWeightModeType,
    ConfusionMatrixResultType,
    RocAucConfigType,
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
    numpy_to_east_int_vector,
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


def _combine_stratify_columns(columns: list[np.ndarray]) -> np.ndarray:
    """Combine multiple stratification columns into compound strata.

    Uses dynamic multipliers based on actual ranges to avoid collisions.
    For columns [A, B, C], computes: A * (max_B+1) * (max_C+1) + B * (max_C+1) + C
    """
    if len(columns) == 1:
        return columns[0]

    # Normalize each column to start from 0 and compute bases
    normalized = []
    bases = []
    for col in columns:
        col_min = col.min()
        normalized.append(col - col_min)
        bases.append(int(col.max() - col_min + 1))

    # Compute cumulative multipliers (from right to left)
    # For columns [A, B, C] with bases [bA, bB, bC]:
    # multipliers = [bB * bC, bC, 1]
    multipliers = [1] * len(columns)
    for i in range(len(columns) - 2, -1, -1):
        multipliers[i] = multipliers[i + 1] * bases[i + 1]

    # Combine: sum(normalized[i] * multipliers[i])
    combined = np.zeros(len(columns[0]), dtype=np.int64)
    for i, (col, mult) in enumerate(zip(normalized, multipliers)):
        combined += col.astype(np.int64) * mult

    return combined


def sklearn_split_impl(
    X: EastArray,
    Y: EastArray,
    config: EastStruct,
) -> EastStruct:
    """Split arrays into N subsets (train/test, train/val/test, etc.).

    Supports multi-column stratification (combined into compound strata) and
    overlap columns (values must appear in all splits).
    """
    try:
        from sklearn.model_selection import train_test_split
    except ImportError as e:
        raise RuntimeError(
            "sklearn_split: scikit-learn not installed. "
            "Install with: pip install scikit-learn"
        ) from e

    try:
        X_np = east_matrix_to_numpy(X)
        Y_np = east_matrix_to_numpy(Y)
    except Exception as e:
        raise RuntimeError(f"sklearn_split: Invalid input data - {e}") from e

    if X_np.shape[0] != Y_np.shape[0]:
        raise RuntimeError(
            f"sklearn_split: X has {X_np.shape[0]} samples "
            f"but Y has {Y_np.shape[0]} samples"
        )

    # Parse config
    split_sizes = [float(s) for s in config.get("split_sizes")]
    n_splits = len(split_sizes)

    if n_splits < 2:
        raise RuntimeError("sklearn_split: split_sizes must have at least 2 elements")

    if abs(sum(split_sizes) - 1.0) > 0.01:
        raise RuntimeError(f"sklearn_split: split_sizes must sum to 1.0, got {sum(split_sizes)}")

    random_state = _get_option(config.get("random_state"), None)
    shuffle = _get_option(config.get("shuffle"), True)
    stratify_columns = _get_option(config.get("stratify"), None)
    overlap_columns = _get_option(config.get("overlap"), None)

    if random_state is not None:
        random_state = int(random_state)

    # Default min_stratify_samples = n_splits (need at least 1 sample per split)
    min_stratify_samples = _get_option(config.get("min_stratify_samples"), n_splits)
    if min_stratify_samples is not None:
        min_stratify_samples = int(min_stratify_samples)
    else:
        min_stratify_samples = n_splits

    n_samples = X_np.shape[0]
    rejected_indices = []
    original_indices = np.arange(n_samples)

    # Build compound stratification from multiple columns
    stratify_arr = None
    if stratify_columns is not None:
        columns = [np.array([int(x) for x in col]) for col in stratify_columns]
        for i, col in enumerate(columns):
            if len(col) != n_samples:
                raise RuntimeError(
                    f"sklearn_split: stratify column {i} has {len(col)} labels "
                    f"but X has {n_samples} samples"
                )
        stratify_arr = _combine_stratify_columns(columns)

        # Filter rare compound strata
        unique_strata, counts = np.unique(stratify_arr, return_counts=True)
        rare_strata = set(unique_strata[counts < min_stratify_samples])

        if rare_strata:
            keep_mask = np.array([s not in rare_strata for s in stratify_arr])
            rejected_indices = original_indices[~keep_mask].tolist()

            X_np = X_np[keep_mask]
            Y_np = Y_np[keep_mask]
            stratify_arr = stratify_arr[keep_mask]
            original_indices = original_indices[keep_mask]

    # Build overlap columns array if provided
    overlap_arr = None
    if overlap_columns is not None:
        overlap_cols = [np.array([int(x) for x in col]) for col in overlap_columns]
        for i, col in enumerate(overlap_cols):
            if len(col) != n_samples:
                raise RuntimeError(
                    f"sklearn_split: overlap column {i} has {len(col)} labels "
                    f"but X has {n_samples} samples"
                )
        # Combine overlap columns too (we'll check after splitting)
        overlap_arr = _combine_stratify_columns(overlap_cols)
        # Filter to match current data after stratify filtering
        if len(rejected_indices) > 0:
            keep_mask = np.isin(np.arange(len(overlap_columns[0])), original_indices)
            overlap_arr = overlap_arr[keep_mask]

    try:
        # Perform N-way split iteratively
        # Start with all data, split off each subset one at a time
        remaining_X = X_np
        remaining_Y = Y_np
        remaining_strat = stratify_arr
        remaining_overlap = overlap_arr
        remaining_idx = np.arange(len(X_np))  # Indices into current filtered data

        X_splits = []
        Y_splits = []
        strat_splits = []
        overlap_splits = []
        idx_splits = []  # Track indices for each split

        remaining_fraction = 1.0

        for i in range(n_splits - 1):
            # Fraction of remaining data for this split
            this_split_size = split_sizes[i] / remaining_fraction
            remaining_fraction -= split_sizes[i]

            if remaining_strat is not None:
                (this_X, remaining_X,
                 this_Y, remaining_Y,
                 this_strat, remaining_strat,
                 this_overlap, remaining_overlap,
                 this_idx, remaining_idx) = train_test_split(
                    remaining_X, remaining_Y, remaining_strat,
                    remaining_overlap if remaining_overlap is not None else remaining_strat,
                    remaining_idx,
                    test_size=1.0 - this_split_size,
                    random_state=random_state,
                    shuffle=shuffle,
                    stratify=remaining_strat
                )
                if overlap_arr is None:
                    this_overlap = None
                    remaining_overlap = None
            else:
                if remaining_overlap is not None:
                    (this_X, remaining_X,
                     this_Y, remaining_Y,
                     this_overlap, remaining_overlap,
                     this_idx, remaining_idx) = train_test_split(
                        remaining_X, remaining_Y, remaining_overlap, remaining_idx,
                        test_size=1.0 - this_split_size,
                        random_state=random_state,
                        shuffle=shuffle
                    )
                else:
                    (this_X, remaining_X,
                     this_Y, remaining_Y,
                     this_idx, remaining_idx) = train_test_split(
                        remaining_X, remaining_Y, remaining_idx,
                        test_size=1.0 - this_split_size,
                        random_state=random_state,
                        shuffle=shuffle
                    )
                    this_overlap = None

            X_splits.append(this_X)
            Y_splits.append(this_Y)
            strat_splits.append(this_strat if remaining_strat is not None else None)
            overlap_splits.append(this_overlap)
            idx_splits.append(this_idx)

        # Last split is whatever remains
        X_splits.append(remaining_X)
        Y_splits.append(remaining_Y)
        strat_splits.append(remaining_strat)
        overlap_splits.append(remaining_overlap)
        idx_splits.append(remaining_idx)

        # Post-split validation for stratify: ensure all strata appear in ALL splits
        if stratify_arr is not None:
            strata_per_split = [set(s) for s in strat_splits if s is not None]
            if strata_per_split:
                common_strata = strata_per_split[0]
                for s in strata_per_split[1:]:
                    common_strata = common_strata & s

                all_strata = set(stratify_arr)
                missing_strata = all_strata - common_strata

                if missing_strata:
                    # Remove samples with missing strata from all splits
                    for i in range(n_splits):
                        keep_mask = np.array([s in common_strata for s in strat_splits[i]])
                        # Track rejected (map back to original indices)
                        split_rejected = original_indices[idx_splits[i][~keep_mask]].tolist()
                        rejected_indices.extend(split_rejected)

                        X_splits[i] = X_splits[i][keep_mask]
                        Y_splits[i] = Y_splits[i][keep_mask]
                        if overlap_splits[i] is not None:
                            overlap_splits[i] = overlap_splits[i][keep_mask]
                        idx_splits[i] = idx_splits[i][keep_mask]
                        strat_splits[i] = strat_splits[i][keep_mask]

        # Post-split validation for overlap: ensure overlap values appear in ALL splits
        if overlap_arr is not None:
            overlap_per_split = [set(o) for o in overlap_splits if o is not None]
            if overlap_per_split:
                common_overlap = overlap_per_split[0]
                for o in overlap_per_split[1:]:
                    common_overlap = common_overlap & o

                # Remove samples with non-common overlap values
                for i in range(n_splits):
                    if overlap_splits[i] is not None:
                        keep_mask = np.array([o in common_overlap for o in overlap_splits[i]])
                        # Track rejected
                        split_rejected = original_indices[idx_splits[i][~keep_mask]].tolist()
                        rejected_indices.extend(split_rejected)

                        X_splits[i] = X_splits[i][keep_mask]
                        Y_splits[i] = Y_splits[i][keep_mask]

    except Exception as e:
        raise RuntimeError(
            f"sklearn_split: Split failed with X shape {X_np.shape} - {e}"
        ) from e

    return EastStruct(
        {
            "X_splits": EastArray(
                ArrayType(MatrixType),
                [numpy_to_east_matrix(x) for x in X_splits]
            ),
            "Y_splits": EastArray(
                ArrayType(MatrixType),
                [numpy_to_east_matrix(y) for y in Y_splits]
            ),
            "rejected_indices": EastArray(
                ArrayType("integer"),
                [int(i) for i in sorted(set(rejected_indices))]
            ),
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


def sklearn_robust_scaler_fit_impl(X: EastArray) -> EastVariant:
    """Fit RobustScaler and return model blob.

    RobustScaler scales features using statistics that are robust to outliers.
    It centers data using the median and scales using the interquartile range (IQR).
    """
    try:
        from sklearn.preprocessing import RobustScaler
    except ImportError as e:
        raise RuntimeError(
            "sklearn_robust_scaler_fit: scikit-learn not installed. "
            "Install with: pip install scikit-learn"
        ) from e

    try:
        X_np = east_matrix_to_numpy(X)
    except Exception as e:
        raise RuntimeError(
            f"sklearn_robust_scaler_fit: Invalid input data - {e}"
        ) from e

    n_features = X_np.shape[1]

    try:
        scaler = RobustScaler()
        scaler.fit(X_np)
        onnx_data = _sklearn_to_onnx(scaler, n_features)
    except Exception as e:
        raise RuntimeError(
            f"sklearn_robust_scaler_fit: Fitting failed with X shape {X_np.shape} - {e}"
        ) from e

    return EastVariant(
        "robust_scaler",
        EastStruct(
            {
                "onnx": onnx_data,
                "n_features": n_features,
            }
        ),
    )


def sklearn_robust_scaler_transform_impl(
    model_blob: EastVariant,
    X: EastArray,
) -> EastArray:
    """Transform data using fitted robust scaler."""
    if model_blob.type != "robust_scaler":
        raise RuntimeError(
            f"sklearn_robust_scaler_transform: Expected robust_scaler, got {model_blob.type}"
        )

    try:
        onnx_blob = model_blob.value["onnx"]
        return _onnx_transform(onnx_blob, X)
    except Exception as e:
        raise RuntimeError(
            f"sklearn_robust_scaler_transform: Transform failed - {e}"
        ) from e


def sklearn_label_encoder_fit_impl(y: EastArray) -> EastVariant:
    """Fit LabelEncoder to labels and return model blob."""
    import cloudpickle

    try:
        from sklearn.preprocessing import LabelEncoder
    except ImportError as e:
        raise RuntimeError(
            "sklearn_label_encoder_fit: scikit-learn not installed. "
            "Install with: pip install scikit-learn"
        ) from e

    try:
        y_np = east_int_vector_to_numpy(y)
    except Exception as e:
        raise RuntimeError(
            f"sklearn_label_encoder_fit: Invalid input data - {e}"
        ) from e

    try:
        encoder = LabelEncoder()
        encoder.fit(y_np)
        n_classes = len(encoder.classes_)
    except Exception as e:
        raise RuntimeError(
            f"sklearn_label_encoder_fit: Fitting failed - {e}"
        ) from e

    return EastVariant(
        "label_encoder",
        EastStruct(
            {
                "data": EastBlob(cloudpickle.dumps(encoder)),
                "n_classes": n_classes,
            }
        ),
    )


def sklearn_label_encoder_transform_impl(
    model_blob: EastVariant,
    y: EastArray,
) -> EastArray:
    """Transform labels using fitted LabelEncoder."""
    import cloudpickle

    if model_blob.type != "label_encoder":
        raise RuntimeError(
            f"sklearn_label_encoder_transform: Expected label_encoder, got {model_blob.type}"
        )

    try:
        y_np = east_int_vector_to_numpy(y)
        encoder = cloudpickle.loads(bytes(model_blob.value["data"]))
        y_encoded = encoder.transform(y_np)
    except Exception as e:
        raise RuntimeError(
            f"sklearn_label_encoder_transform: Transform failed - {e}"
        ) from e

    return numpy_to_east_int_vector(y_encoded)


def sklearn_label_encoder_inverse_transform_impl(
    model_blob: EastVariant,
    y: EastArray,
) -> EastArray:
    """Inverse transform encoded labels back to original values."""
    import cloudpickle

    if model_blob.type != "label_encoder":
        raise RuntimeError(
            f"sklearn_label_encoder_inverse_transform: Expected label_encoder, got {model_blob.type}"
        )

    try:
        y_np = east_int_vector_to_numpy(y)
        encoder = cloudpickle.loads(bytes(model_blob.value["data"]))
        y_original = encoder.inverse_transform(y_np)
    except Exception as e:
        raise RuntimeError(
            f"sklearn_label_encoder_inverse_transform: Inverse transform failed - {e}"
        ) from e

    return numpy_to_east_int_vector(y_original)


def sklearn_ordinal_encoder_fit_impl(X: EastArray) -> EastVariant:
    """Fit OrdinalEncoder to features and return model blob."""
    import cloudpickle

    try:
        from sklearn.preprocessing import OrdinalEncoder
    except ImportError as e:
        raise RuntimeError(
            "sklearn_ordinal_encoder_fit: scikit-learn not installed. "
            "Install with: pip install scikit-learn"
        ) from e

    try:
        X_np = east_matrix_to_numpy(X)
    except Exception as e:
        raise RuntimeError(
            f"sklearn_ordinal_encoder_fit: Invalid input data - {e}"
        ) from e

    n_features = X_np.shape[1]

    try:
        encoder = OrdinalEncoder()
        encoder.fit(X_np)
    except Exception as e:
        raise RuntimeError(
            f"sklearn_ordinal_encoder_fit: Fitting failed - {e}"
        ) from e

    return EastVariant(
        "ordinal_encoder",
        EastStruct(
            {
                "data": EastBlob(cloudpickle.dumps(encoder)),
                "n_features": n_features,
            }
        ),
    )


def sklearn_ordinal_encoder_transform_impl(
    model_blob: EastVariant,
    X: EastArray,
) -> EastArray:
    """Transform features using fitted OrdinalEncoder."""
    import cloudpickle

    if model_blob.type != "ordinal_encoder":
        raise RuntimeError(
            f"sklearn_ordinal_encoder_transform: Expected ordinal_encoder, got {model_blob.type}"
        )

    try:
        X_np = east_matrix_to_numpy(X)
        encoder = cloudpickle.loads(bytes(model_blob.value["data"]))
        X_encoded = encoder.transform(X_np)
    except Exception as e:
        raise RuntimeError(
            f"sklearn_ordinal_encoder_transform: Transform failed - {e}"
        ) from e

    return numpy_to_east_matrix(X_encoded)


def sklearn_compute_class_weight_impl(
    mode: EastVariant,
    y: EastArray,
) -> EastArray:
    """Compute class weights for balanced training.

    Calculates weights inversely proportional to class frequencies.
    """
    try:
        from sklearn.utils.class_weight import compute_class_weight
    except ImportError as e:
        raise RuntimeError(
            "sklearn_compute_class_weight: scikit-learn not installed. "
            "Install with: pip install scikit-learn"
        ) from e

    try:
        y_np = east_int_vector_to_numpy(y)
    except Exception as e:
        raise RuntimeError(
            f"sklearn_compute_class_weight: Invalid input data - {e}"
        ) from e

    mode_type = mode.type
    if mode_type != "balanced":
        raise RuntimeError(
            f"sklearn_compute_class_weight: Unknown mode type: {mode_type}"
        )

    try:
        classes = np.unique(y_np)
        weights = compute_class_weight("balanced", classes=classes, y=y_np)
    except Exception as e:
        raise RuntimeError(
            f"sklearn_compute_class_weight: Computing weights failed - {e}"
        ) from e

    return numpy_to_east_vector(weights)


def sklearn_confusion_matrix_impl(
    y_true: EastArray,
    y_pred: EastArray,
) -> EastStruct:
    """Compute confusion matrix for classification results.

    Returns matrix where entry [i,j] is the count of samples with true label i
    that were predicted as label j.
    """
    try:
        from sklearn.metrics import confusion_matrix
    except ImportError as e:
        raise RuntimeError(
            "sklearn_confusion_matrix: scikit-learn not installed. "
            "Install with: pip install scikit-learn"
        ) from e

    try:
        y_true_np = east_int_vector_to_numpy(y_true)
        y_pred_np = east_int_vector_to_numpy(y_pred)
    except Exception as e:
        raise RuntimeError(
            f"sklearn_confusion_matrix: Invalid input data - {e}"
        ) from e

    if y_true_np.shape[0] != y_pred_np.shape[0]:
        raise RuntimeError(
            f"sklearn_confusion_matrix: y_true has {y_true_np.shape[0]} samples "
            f"but y_pred has {y_pred_np.shape[0]} samples"
        )

    try:
        classes = np.unique(np.concatenate([y_true_np, y_pred_np]))
        cm = confusion_matrix(y_true_np, y_pred_np, labels=classes)
    except Exception as e:
        raise RuntimeError(
            f"sklearn_confusion_matrix: Computing matrix failed - {e}"
        ) from e

    return EastStruct(
        {
            "matrix": numpy_to_east_matrix(cm.astype(np.float32)),
            "classes": numpy_to_east_int_vector(classes),
        }
    )


def sklearn_roc_auc_score_impl(
    y_true: EastArray,
    y_proba: EastArray,
    config: EastStruct,
) -> float:
    """Compute ROC AUC score for classification results."""
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError as e:
        raise RuntimeError(
            "sklearn_roc_auc_score: scikit-learn not installed. "
            "Install with: pip install scikit-learn"
        ) from e

    try:
        y_true_np = east_int_vector_to_numpy(y_true)
        y_proba_np = east_matrix_to_numpy(y_proba)
    except Exception as e:
        raise RuntimeError(
            f"sklearn_roc_auc_score: Invalid input data - {e}"
        ) from e

    if y_true_np.shape[0] != y_proba_np.shape[0]:
        raise RuntimeError(
            f"sklearn_roc_auc_score: y_true has {y_true_np.shape[0]} samples "
            f"but y_proba has {y_proba_np.shape[0]} samples"
        )

    # Get config options
    multi_class_opt = config.get("multi_class")
    multi_class = "ovr"  # default
    if multi_class_opt is not None and hasattr(multi_class_opt, "type"):
        if multi_class_opt.type == "some":
            multi_class = _get_enum_tag(multi_class_opt.value)

    average_opt = config.get("average")
    average = "macro"  # default
    if average_opt is not None and hasattr(average_opt, "type"):
        if average_opt.type == "some":
            average = _get_enum_tag(average_opt.value)

    try:
        n_classes = len(np.unique(y_true_np))
        if n_classes == 2:
            # Binary classification - use positive class probabilities
            score = roc_auc_score(y_true_np, y_proba_np[:, 1])
        else:
            # Multi-class classification
            score = roc_auc_score(
                y_true_np,
                y_proba_np,
                multi_class=multi_class,
                average=average,
            )
    except Exception as e:
        raise RuntimeError(
            f"sklearn_roc_auc_score: Computing score failed - {e}"
        ) from e

    return float(score)


def sklearn_log_loss_impl(
    y_true: EastArray,
    y_proba: EastArray,
) -> float:
    """Compute log loss (cross-entropy) for classification results."""
    try:
        from sklearn.metrics import log_loss
    except ImportError as e:
        raise RuntimeError(
            "sklearn_log_loss: scikit-learn not installed. "
            "Install with: pip install scikit-learn"
        ) from e

    try:
        y_true_np = east_int_vector_to_numpy(y_true)
        y_proba_np = east_matrix_to_numpy(y_proba)
    except Exception as e:
        raise RuntimeError(
            f"sklearn_log_loss: Invalid input data - {e}"
        ) from e

    if y_true_np.shape[0] != y_proba_np.shape[0]:
        raise RuntimeError(
            f"sklearn_log_loss: y_true has {y_true_np.shape[0]} samples "
            f"but y_proba has {y_proba_np.shape[0]} samples"
        )

    try:
        loss = log_loss(y_true_np, y_proba_np)
    except Exception as e:
        raise RuntimeError(
            f"sklearn_log_loss: Computing loss failed - {e}"
        ) from e

    return float(loss)




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
    cohen_kappa_weights: str | None = None,
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
        # Handle weights parameter
        weights = None
        if cohen_kappa_weights and cohen_kappa_weights != "none":
            weights = cohen_kappa_weights  # "linear" or "quadratic"
        return float(sklearn_metrics.cohen_kappa_score(y_true, y_pred, weights=weights))
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
        # Extract cohen_kappa weights if present
        cohen_kappa_weights = None
        if metric_name == "cohen_kappa" and metric_variant.value is not None:
            if hasattr(metric_variant.value, "type"):
                cohen_kappa_weights = metric_variant.value.type
        try:
            value = _compute_classification_metric(
                metric_name, y_true_np, y_pred_np, average, cohen_kappa_weights
            )
            results.append(
                EastStruct(
                    {
                        "metric": metric_variant,
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
        # Extract cohen_kappa weights if present
        cohen_kappa_weights = None
        if metric_name == "cohen_kappa" and metric_variant.value is not None:
            if hasattr(metric_variant.value, "type"):
                cohen_kappa_weights = metric_variant.value.type
        try:
            # Compute per target
            per_target_values = []
            for i in range(n_targets):
                val = _compute_classification_metric(
                    metric_name, Y_true_np[:, i], Y_pred_np[:, i], average, cohen_kappa_weights
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
                        "metric": metric_variant,
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
        name="sklearn_split",
        inputs=[MatrixType, MatrixType, SplitConfigType],
        output=SplitResultType,
        type="sync",
        fn=sklearn_split_impl,
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
        name="sklearn_robust_scaler_fit",
        inputs=[MatrixType],
        output=ModelBlobType,
        type="sync",
        fn=sklearn_robust_scaler_fit_impl,
    ),
    PlatformFunction(
        name="sklearn_robust_scaler_transform",
        inputs=[ModelBlobType, MatrixType],
        output=MatrixType,
        type="sync",
        fn=sklearn_robust_scaler_transform_impl,
    ),
    PlatformFunction(
        name="sklearn_compute_class_weight",
        inputs=[ClassWeightModeType, IntVectorType],
        output=VectorType,
        type="sync",
        fn=sklearn_compute_class_weight_impl,
    ),
    PlatformFunction(
        name="sklearn_confusion_matrix",
        inputs=[IntVectorType, IntVectorType],
        output=ConfusionMatrixResultType,
        type="sync",
        fn=sklearn_confusion_matrix_impl,
    ),
    PlatformFunction(
        name="sklearn_roc_auc_score",
        inputs=[IntVectorType, MatrixType, RocAucConfigType],
        output=FloatType,
        type="sync",
        fn=sklearn_roc_auc_score_impl,
    ),
    PlatformFunction(
        name="sklearn_log_loss",
        inputs=[IntVectorType, MatrixType],
        output=FloatType,
        type="sync",
        fn=sklearn_log_loss_impl,
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
    # LabelEncoder
    PlatformFunction(
        name="sklearn_label_encoder_fit",
        inputs=[IntVectorType],
        output=ModelBlobType,
        type="sync",
        fn=sklearn_label_encoder_fit_impl,
    ),
    PlatformFunction(
        name="sklearn_label_encoder_transform",
        inputs=[ModelBlobType, IntVectorType],
        output=IntVectorType,
        type="sync",
        fn=sklearn_label_encoder_transform_impl,
    ),
    PlatformFunction(
        name="sklearn_label_encoder_inverse_transform",
        inputs=[ModelBlobType, IntVectorType],
        output=IntVectorType,
        type="sync",
        fn=sklearn_label_encoder_inverse_transform_impl,
    ),
    # OrdinalEncoder
    PlatformFunction(
        name="sklearn_ordinal_encoder_fit",
        inputs=[MatrixType],
        output=ModelBlobType,
        type="sync",
        fn=sklearn_ordinal_encoder_fit_impl,
    ),
    PlatformFunction(
        name="sklearn_ordinal_encoder_transform",
        inputs=[ModelBlobType, MatrixType],
        output=MatrixType,
        type="sync",
        fn=sklearn_ordinal_encoder_transform_impl,
    ),
]

__all__ = [
    "sklearn_impl",
    # Re-export types from types.py
    "SplitConfigType",
    "SplitResultType",
    "ModelBlobType",
]
