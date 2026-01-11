# Design: MAPIE Conformal Prediction Intervals

## Purpose

Add MAPIE (Model Agnostic Prediction Interval Estimator) support for tree-based models to provide:

- **Regression**: Prediction intervals with coverage guarantees
- **Classification**: Prediction sets with coverage guarantees

Unlike raw quantile regression or standard classification, conformal prediction ensures that if you request 90% coverage, approximately 90% of test points will fall within those intervals/sets.

## Current State

- XGBoost has `trainQuantile`/`predictQuantile` for quantile regression, but no coverage guarantees
- LightGBM has no quantile/interval support
- NGBoost provides distributional predictions (already probabilistic)

### Why MAPIE?

| Method | Coverage Guarantee | Calibration Required |
|--------|-------------------|---------------------|
| Raw Quantile Regression | No - often under/over-covers | No |
| MAPIE Split Conformal | Yes - provable guarantees | Yes - needs held-out set |
| MAPIE CQR | Yes - combines quantile + conformal | Yes |

## Design

### MAPIE 1.2.0 API

MAPIE 1.2.0 uses a new class structure with separate classes for different conformal methods:

#### Regression Classes

| Class | Description | Methods |
|-------|-------------|---------|
| `SplitConformalRegressor` | Split conformal with prefit model | `conformalize()`, `predict_interval()` |
| `CrossConformalRegressor` | Cross-validation based conformal | `fit_conformalize()`, `predict_interval()` |
| `ConformalizedQuantileRegressor` | CQR with quantile models | `fit()`, `conformalize()`, `predict_interval()` |

#### Classification Classes

| Class | Description | Methods |
|-------|-------------|---------|
| `SplitConformalClassifier` | Split conformal with prefit classifier | `conformalize()`, `predict_set()` |
| `CrossConformalClassifier` | Cross-validation based conformal | `fit_conformalize()`, `predict_set()` |

#### Key API Differences from Legacy

1. **`confidence_level` instead of `alpha`**: MAPIE 1.2.0 uses `confidence_level` (e.g., 0.9 for 90% coverage) instead of `alpha` (e.g., 0.1 for 90% coverage). The relationship is: `confidence_level = 1 - alpha`.

2. **Confidence level set at training time**: Multiple confidence levels can be specified during training via `confidence_level=[0.9, 0.95]`, and intervals at all levels are returned at prediction time.

3. **Method names**: `conformalize()` for split conformal, `fit_conformalize()` for cross conformal.

4. **Return shapes**: `predict_interval()` returns `(predictions, intervals)` where intervals shape is `(n_samples, 2, n_confidence_levels)`.

### Approach

Create a new `MAPIE` module that supports both regression and classification:

**Regression:**
1. Trains a base model (XGBoost or LightGBM regressor)
2. Calibrates it using MAPIE's conformal prediction on a held-out calibration set
3. Returns prediction intervals with coverage guarantees

**Classification:**
1. Trains a base classifier (XGBoost or LightGBM classifier)
2. Calibrates it using MAPIE's conformal classification on a held-out calibration set
3. Returns prediction sets with coverage guarantees

### Supported Base Models

#### Regression

| Base Model | MAPIE Class | Notes |
|------------|-------------|-------|
| XGBoost Regressor | `SplitConformalRegressor` | Split conformal with prefit |
| LightGBM Regressor | `SplitConformalRegressor` | Split conformal with prefit |
| XGBoost Regressor | `CrossConformalRegressor` | Cross-validation based |
| XGBoost Quantile | `ConformalizedQuantileRegressor` | Uses quantile models for CQR |

#### Classification

| Base Model | MAPIE Class | Notes |
|------------|-------------|-------|
| XGBoost Classifier | `SplitConformalClassifier` | Prediction sets with coverage guarantees |
| LightGBM Classifier | `SplitConformalClassifier` | Prediction sets with coverage guarantees |

### Conformal Methods

#### Regression Methods

| Method | MAPIE 1.2.0 Class | Description | Use Case |
|--------|-------------------|-------------|----------|
| `split` | `SplitConformalRegressor` | Split conformal - simple holdout | Fast, requires calibration set |
| `cross` | `CrossConformalRegressor` | Cross-validation based | Better intervals, slower training |
| `cqr` | `ConformalizedQuantileRegressor` | Conformalized Quantile Regression | Best for heteroscedastic data |

#### Classification Methods

| Method | Conformity Score | Description | Use Case |
|--------|------------------|-------------|----------|
| `lac` | Least Ambiguous Classifier | Smallest prediction sets | Default, good for most cases |
| `aps` | Adaptive Prediction Sets | Adapts to probabilities | Better calibrated coverage |

### File Changes

#### 1. `src/mapie/mapie.ts` (new file)

```typescript
/**
 * MAPIE conformal prediction intervals for East.
 *
 * Provides prediction intervals with coverage guarantees using
 * conformal prediction methods (MAPIE 1.2.0 API).
 *
 * @packageDocumentation
 */

import {
    East,
    StructType,
    VariantType,
    OptionType,
    ArrayType,
    IntegerType,
    FloatType,
    BlobType,
    NullType,
} from "@elaraai/east";
import { VectorType, MatrixType, LabelVectorType } from "../types.js";

// Re-export shared types for convenience
export { VectorType, MatrixType, LabelVectorType } from "../types.js";

// ============================================================================
// Config Types
// ============================================================================

/**
 * Conformal prediction method for regression.
 */
export const ConformalMethodType = VariantType({
    /** Split conformal - requires separate calibration set */
    split: NullType,
    /** Cross conformal - uses CV for calibration (combines train + calib) */
    cross: NullType,
});

/**
 * Configuration for XGBoost base model (subset for MAPIE).
 */
export const MAPIEXGBoostConfigType = StructType({
    /** Number of boosting rounds (default 100) */
    n_estimators: OptionType(IntegerType),
    /** Maximum tree depth (default 6) */
    max_depth: OptionType(IntegerType),
    /** Learning rate / step size shrinkage (default 0.3) */
    learning_rate: OptionType(FloatType),
    /** Minimum sum of instance weight needed in a child (default 1) */
    min_child_weight: OptionType(IntegerType),
    /** Subsample ratio of training instances (default 1.0) */
    subsample: OptionType(FloatType),
    /** Subsample ratio of columns when constructing trees (default 1.0) */
    colsample_bytree: OptionType(FloatType),
    /** L1 regularization term (default 0) */
    reg_alpha: OptionType(FloatType),
    /** L2 regularization term (default 1) */
    reg_lambda: OptionType(FloatType),
    /** Minimum loss reduction required to make a further partition (default 0) */
    gamma: OptionType(FloatType),
    /** Random seed for reproducibility */
    random_state: OptionType(IntegerType),
});

/**
 * Configuration for LightGBM base model (subset for MAPIE).
 */
export const MAPIELightGBMConfigType = StructType({
    /** Number of boosting rounds (default 100) */
    n_estimators: OptionType(IntegerType),
    /** Maximum tree depth, -1 for unlimited (default -1) */
    max_depth: OptionType(IntegerType),
    /** Learning rate / step size shrinkage (default 0.1) */
    learning_rate: OptionType(FloatType),
    /** Maximum number of leaves in one tree (default 31) */
    num_leaves: OptionType(IntegerType),
    /** Minimum number of samples required in a leaf (default 20) */
    min_child_samples: OptionType(IntegerType),
    /** Subsample ratio of training instances (default 1.0) */
    subsample: OptionType(FloatType),
    /** Subsample ratio of columns when constructing trees (default 1.0) */
    colsample_bytree: OptionType(FloatType),
    /** L1 regularization term (default 0) */
    reg_alpha: OptionType(FloatType),
    /** L2 regularization term (default 0) */
    reg_lambda: OptionType(FloatType),
    /** Random seed for reproducibility */
    random_state: OptionType(IntegerType),
});

/**
 * Base model type for MAPIE regression.
 */
export const BaseModelType = VariantType({
    /** XGBoost regressor as base model */
    xgboost: MAPIEXGBoostConfigType,
    /** LightGBM regressor as base model */
    lightgbm: MAPIELightGBMConfigType,
});

/**
 * Configuration for MAPIE conformal prediction.
 */
export const MAPIEConfigType = StructType({
    /** Base model configuration */
    base_model: BaseModelType,
    /** Conformal method (default: split) */
    method: OptionType(ConformalMethodType),
    /** Confidence level: coverage probability (default 0.9 = 90% intervals) */
    confidence_level: OptionType(FloatType),
    /** Number of CV folds for cross method (default 5) */
    cv_folds: OptionType(IntegerType),
    /** Random seed for reproducibility */
    random_state: OptionType(IntegerType),
});

/**
 * Configuration for CQR (Conformalized Quantile Regression).
 * Requires a base model that supports quantile regression (XGBoost).
 */
export const MAPIECQRConfigType = StructType({
    /** XGBoost config for the base quantile model */
    xgboost_config: MAPIEXGBoostConfigType,
    /** Confidence level: coverage probability (default 0.9 = 90% intervals) */
    confidence_level: OptionType(FloatType),
    /** Random seed for reproducibility */
    random_state: OptionType(IntegerType),
});

// ============================================================================
// Classification Config Types
// ============================================================================

/**
 * Classification conformal method (conformity score).
 */
export const ClassificationMethodType = VariantType({
    /** Least Ambiguous set-valued Classifier - smallest sets */
    lac: NullType,
    /** Adaptive Prediction Sets - adapts to probabilities */
    aps: NullType,
});

/**
 * Base classifier type for MAPIE classification.
 */
export const BaseClassifierType = VariantType({
    /** XGBoost classifier as base model */
    xgboost: MAPIEXGBoostConfigType,
    /** LightGBM classifier as base model */
    lightgbm: MAPIELightGBMConfigType,
});

/**
 * Configuration for MAPIE conformal classification.
 */
export const MAPIEClassifierConfigType = StructType({
    /** Base classifier configuration */
    base_model: BaseClassifierType,
    /** Classification conformity score method (default: lac) */
    method: OptionType(ClassificationMethodType),
    /** Confidence level: coverage probability (default 0.9 = 90% coverage) */
    confidence_level: OptionType(FloatType),
    /** Random seed for reproducibility */
    random_state: OptionType(IntegerType),
});

// ============================================================================
// Model Blob Types
// ============================================================================

/** Base model type indicator */
const BaseModelTypeIndicator = VariantType({
    xgboost: NullType,
    lightgbm: NullType,
});

/**
 * Model blob for MAPIE conformal regressor.
 */
export const MAPIERegressorBlobType = VariantType({
    /** MAPIE regressor with split conformal */
    mapie_split: StructType({
        /** Cloudpickle serialized SplitConformalRegressor */
        data: BlobType,
        /** Number of input features */
        n_features: IntegerType,
        /** Confidence level used during calibration */
        confidence_level: FloatType,
        /** Base model type ('xgboost' or 'lightgbm') */
        base_model_type: BaseModelTypeIndicator,
    }),
    /** MAPIE regressor with cross conformal */
    mapie_cross: StructType({
        /** Cloudpickle serialized CrossConformalRegressor */
        data: BlobType,
        /** Number of input features */
        n_features: IntegerType,
        /** Confidence level used during calibration */
        confidence_level: FloatType,
        /** Base model type ('xgboost' or 'lightgbm') */
        base_model_type: BaseModelTypeIndicator,
    }),
    /** MAPIE CQR regressor */
    mapie_cqr: StructType({
        /** Cloudpickle serialized ConformalizedQuantileRegressor */
        data: BlobType,
        /** Number of input features */
        n_features: IntegerType,
        /** Confidence level used during calibration */
        confidence_level: FloatType,
    }),
});

/**
 * Model blob for MAPIE conformal classifier.
 */
export const MAPIEClassifierBlobType = StructType({
    /** Cloudpickle serialized SplitConformalClassifier */
    data: BlobType,
    /** Number of input features */
    n_features: IntegerType,
    /** Number of classes */
    n_classes: IntegerType,
    /** Class labels */
    classes: ArrayType(IntegerType),
    /** Confidence level used during calibration */
    confidence_level: FloatType,
    /** Base model type ('xgboost' or 'lightgbm') */
    base_model_type: BaseModelTypeIndicator,
});

// ============================================================================
// Result Types
// ============================================================================

/**
 * Prediction interval result (regression).
 */
export const IntervalResultType = StructType({
    /** Lower bound of prediction interval */
    lower: VectorType,
    /** Point prediction (median/mean) */
    pred: VectorType,
    /** Upper bound of prediction interval */
    upper: VectorType,
});

/**
 * Prediction set result (classification).
 * For each sample, contains the set of classes included in the prediction set.
 */
export const PredictionSetResultType = StructType({
    /** Predicted class (argmax of probabilities) */
    pred: ArrayType(IntegerType),
    /** Prediction set membership matrix (n_samples x n_classes, 1 if class in set) */
    sets: ArrayType(ArrayType(IntegerType)),
    /** Class probabilities (n_samples x n_classes) */
    probabilities: MatrixType,
    /** Size of each prediction set */
    set_sizes: ArrayType(IntegerType),
});

// ============================================================================
// Platform Functions
// ============================================================================

// --------------------------------
// Regression Functions
// --------------------------------

/**
 * Train a MAPIE conformal regressor.
 *
 * For split conformal, uses X_calib/y_calib for calibration.
 * For cross conformal, combines train and calib data, uses CV for calibration.
 *
 * @param X_train - Training feature matrix
 * @param y_train - Training target vector
 * @param X_calib - Calibration feature matrix
 * @param y_calib - Calibration target vector
 * @param config - MAPIE configuration
 * @returns Model blob containing calibrated MAPIE regressor
 */
export const mapie_train_conformal_regressor = East.platform(
    "mapie_train_conformal_regressor",
    [MatrixType, VectorType, MatrixType, VectorType, MAPIEConfigType],
    MAPIERegressorBlobType
);

/**
 * Train a MAPIE CQR (Conformalized Quantile Regression) model.
 *
 * CQR combines quantile regression with conformal prediction for
 * adaptive intervals that are wider where uncertainty is higher.
 *
 * @param X_train - Training feature matrix
 * @param y_train - Training target vector
 * @param X_calib - Calibration feature matrix
 * @param y_calib - Calibration target vector
 * @param config - CQR configuration
 * @returns Model blob containing calibrated CQR model
 */
export const mapie_train_cqr = East.platform(
    "mapie_train_cqr",
    [MatrixType, VectorType, MatrixType, VectorType, MAPIECQRConfigType],
    MAPIERegressorBlobType
);

/**
 * Predict with intervals using a MAPIE regressor.
 *
 * Returns intervals at the confidence level specified during training.
 *
 * @param model - Trained MAPIE regressor blob
 * @param X - Feature matrix to predict
 * @returns Prediction intervals (lower, pred, upper)
 */
export const mapie_predict_interval = East.platform(
    "mapie_predict_interval",
    [MAPIERegressorBlobType, MatrixType],
    IntervalResultType
);

// --------------------------------
// Classification Functions
// --------------------------------

/**
 * Train a MAPIE conformal classifier.
 *
 * Uses split conformal prediction with calibration set for classification.
 *
 * @param X_train - Training feature matrix
 * @param y_train - Training labels (integers)
 * @param X_calib - Calibration feature matrix
 * @param y_calib - Calibration labels
 * @param config - Classifier configuration
 * @returns Model blob containing calibrated MAPIE classifier
 */
export const mapie_train_conformal_classifier = East.platform(
    "mapie_train_conformal_classifier",
    [MatrixType, LabelVectorType, MatrixType, LabelVectorType, MAPIEClassifierConfigType],
    MAPIEClassifierBlobType
);

/**
 * Predict with prediction sets using a MAPIE classifier.
 *
 * Returns prediction sets at the confidence level specified during training.
 *
 * @param model - Trained MAPIE classifier blob
 * @param X - Feature matrix to predict
 * @returns Prediction sets (pred, sets, probabilities, set_sizes)
 */
export const mapie_predict_set = East.platform(
    "mapie_predict_set",
    [MAPIEClassifierBlobType, MatrixType],
    PredictionSetResultType
);

// ============================================================================
// Module Export
// ============================================================================

export const Types = {
    // Regression
    ConformalMethodType,
    BaseModelType,
    MAPIEConfigType,
    MAPIECQRConfigType,
    MAPIERegressorBlobType,
    IntervalResultType,
    MAPIEXGBoostConfigType,
    MAPIELightGBMConfigType,
    // Classification
    ClassificationMethodType,
    BaseClassifierType,
    MAPIEClassifierConfigType,
    MAPIEClassifierBlobType,
    PredictionSetResultType,
};
```

#### 2. `src/east_py_datascience/mapie/__init__.py` (new file)

```python
"""MAPIE conformal prediction for East - regression and classification."""

from .mapie_impl import (
    mapie_impl,
)

__all__ = [
    "mapie_impl",
]
```

#### 3. `src/east_py_datascience/mapie/mapie_impl.py` (new file)

```python
"""MAPIE conformal prediction implementation (MAPIE 1.2.0 API)."""

import numpy as np
import cloudpickle

from east_py.types import (
    EastArray,
    EastStruct,
    EastVariant,
    ArrayType,
)


def _get_option(opt, default):
    """Extract value from East OptionType."""
    if opt is None:
        return default
    if hasattr(opt, "type"):
        if opt.type == "none":
            return default
        return opt.value
    return opt


def _create_base_regressor(base_model_variant, random_state):
    """Create base sklearn-compatible regressor from config."""
    model_type = base_model_variant.type
    config = base_model_variant.value

    if model_type == "xgboost":
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
            gamma=float(_get_option(config.get("gamma"), 0)),
            random_state=random_state,
            n_jobs=-1,
        )
    elif model_type == "lightgbm":
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
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
    else:
        raise ValueError(f"Unknown base model type: {model_type}")


def _create_base_classifier(base_model_variant, random_state):
    """Create base sklearn-compatible classifier from config."""
    model_type = base_model_variant.type
    config = base_model_variant.value

    if model_type == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=int(_get_option(config.get("n_estimators"), 100)),
            max_depth=int(_get_option(config.get("max_depth"), 6)),
            learning_rate=float(_get_option(config.get("learning_rate"), 0.3)),
            min_child_weight=int(_get_option(config.get("min_child_weight"), 1)),
            subsample=float(_get_option(config.get("subsample"), 1.0)),
            colsample_bytree=float(_get_option(config.get("colsample_bytree"), 1.0)),
            reg_alpha=float(_get_option(config.get("reg_alpha"), 0)),
            reg_lambda=float(_get_option(config.get("reg_lambda"), 1)),
            gamma=float(_get_option(config.get("gamma"), 0)),
            random_state=random_state,
            n_jobs=-1,
        )
    elif model_type == "lightgbm":
        from lightgbm import LGBMClassifier

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
            n_jobs=-1,
            verbose=-1,
        )
    else:
        raise ValueError(f"Unknown base classifier type: {model_type}")


# ============================================================================
# Regression Implementation
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
    except ImportError as e:
        raise RuntimeError(
            f"mapie_train_conformal_regressor: MAPIE not installed. "
            f"Install with: pip install mapie>=1.2.0. Error: {e}"
        )

    # Convert inputs
    X_train_np = np.array([[float(x) for x in row] for row in X_train])
    y_train_np = np.array([float(y) for y in y_train])
    X_calib_np = np.array([[float(x) for x in row] for row in X_calib])
    y_calib_np = np.array([float(y) for y in y_calib])

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

    # Create base model
    base_model = _create_base_regressor(base_model_config, random_state)
    base_model_type = base_model_config.type

    try:
        if method == "split":
            # Split conformal: train on X_train, conformalize on X_calib
            base_model.fit(X_train_np, y_train_np)
            mapie = SplitConformalRegressor(
                estimator=base_model,
                confidence_level=confidence_level,
                prefit=True,
            )
            mapie.conformalize(X_calib_np, y_calib_np)
            variant_type = "mapie_split"
        elif method == "cross":
            # Cross conformal: combine train and calib, use cross-validation
            X_all = np.vstack([X_train_np, X_calib_np])
            y_all = np.hstack([y_train_np, y_calib_np])
            mapie = CrossConformalRegressor(
                estimator=base_model,
                confidence_level=confidence_level,
                cv=cv_folds,
            )
            mapie.fit_conformalize(X_all, y_all)
            variant_type = "mapie_cross"
        else:
            raise RuntimeError(f"mapie_train_conformal_regressor: Unknown method '{method}'")

    except Exception as e:
        raise RuntimeError(f"mapie_train_conformal_regressor: Training failed - {e}")

    # Serialize model
    model_bytes = cloudpickle.dumps(mapie)
    n_features = X_train_np.shape[1]

    return EastVariant(
        variant_type,
        EastStruct({
            "data": model_bytes,
            "n_features": n_features,
            "confidence_level": confidence_level,
            "base_model_type": EastVariant(base_model_type, None),
        }),
    )


def mapie_train_cqr_impl(
    X_train: EastArray,
    y_train: EastArray,
    X_calib: EastArray,
    y_calib: EastArray,
    config: EastStruct,
) -> EastVariant:
    """Train a MAPIE CQR (Conformalized Quantile Regression) model."""
    try:
        from mapie.regression import ConformalizedQuantileRegressor
    except ImportError as e:
        raise RuntimeError(
            f"mapie_train_cqr: MAPIE not installed. "
            f"Install with: pip install mapie>=1.2.0. Error: {e}"
        )
    try:
        from xgboost import XGBRegressor
    except ImportError as e:
        raise RuntimeError(f"mapie_train_cqr: XGBoost not installed. Error: {e}")

    # Convert inputs
    X_train_np = np.array([[float(x) for x in row] for row in X_train])
    y_train_np = np.array([float(y) for y in y_train])
    X_calib_np = np.array([[float(x) for x in row] for row in X_calib])
    y_calib_np = np.array([float(y) for y in y_calib])

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

    # Extract config
    xgb_config = config.get("xgboost_config")
    confidence_level = float(_get_option(config.get("confidence_level"), 0.9))
    random_state = _get_option(config.get("random_state"), None)
    if random_state is not None:
        random_state = int(random_state)

    # Create XGBoost quantile model
    xgb_params = {
        "n_estimators": int(_get_option(xgb_config.get("n_estimators"), 100)),
        "max_depth": int(_get_option(xgb_config.get("max_depth"), 6)),
        "learning_rate": float(_get_option(xgb_config.get("learning_rate"), 0.3)),
        "min_child_weight": int(_get_option(xgb_config.get("min_child_weight"), 1)),
        "subsample": float(_get_option(xgb_config.get("subsample"), 1.0)),
        "colsample_bytree": float(_get_option(xgb_config.get("colsample_bytree"), 1.0)),
        "reg_alpha": float(_get_option(xgb_config.get("reg_alpha"), 0)),
        "reg_lambda": float(_get_option(xgb_config.get("reg_lambda"), 1)),
        "random_state": random_state,
        "n_jobs": -1,
        "objective": "reg:quantileerror",
    }

    try:
        # CQR needs to train on train data, conformalize on calib data
        base_model = XGBRegressor(**xgb_params)
        base_model.fit(X_train_np, y_train_np)

        mapie_cqr = ConformalizedQuantileRegressor(
            estimator=base_model,
            confidence_level=confidence_level,
            prefit=True,
        )
        mapie_cqr.conformalize(X_calib_np, y_calib_np)

    except Exception as e:
        raise RuntimeError(f"mapie_train_cqr: Training failed - {e}")

    # Serialize model
    model_bytes = cloudpickle.dumps(mapie_cqr)
    n_features = X_train_np.shape[1]

    return EastVariant(
        "mapie_cqr",
        EastStruct({
            "data": model_bytes,
            "n_features": n_features,
            "confidence_level": confidence_level,
        }),
    )


def mapie_predict_interval_impl(
    model_blob: EastVariant,
    X: EastArray,
) -> EastStruct:
    """Predict with intervals using the model's calibrated confidence level."""
    model_type = model_blob.type
    model_data = model_blob.value

    # Load model
    model_bytes = model_data.get("data")
    model = cloudpickle.loads(model_bytes)
    n_features = model_data.get("n_features")

    # Convert input
    X_np = np.array([[float(x) for x in row] for row in X])

    # Validate
    if X_np.shape[1] != n_features:
        raise RuntimeError(
            f"mapie_predict_interval: Model trained with {n_features} features "
            f"but X has {X_np.shape[1]} features"
        )

    try:
        # MAPIE 1.2.0: predict_interval() returns (predictions, intervals)
        # intervals shape: (n_samples, 2, n_confidence_levels)
        y_pred, y_intervals = model.predict_interval(X_np)

        # Extract lower and upper bounds (first confidence level)
        lower = y_intervals[:, 0, 0]
        upper = y_intervals[:, 1, 0]

    except Exception as e:
        raise RuntimeError(f"mapie_predict_interval: Prediction failed - {e}")

    return EastStruct({
        "lower": EastArray(ArrayType("float"), [float(x) for x in lower]),
        "pred": EastArray(ArrayType("float"), [float(x) for x in y_pred]),
        "upper": EastArray(ArrayType("float"), [float(x) for x in upper]),
    })


# ============================================================================
# Classification Implementation
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
            f"mapie_train_conformal_classifier: MAPIE not installed. "
            f"Install with: pip install mapie>=1.2.0. Error: {e}"
        )

    # Convert inputs
    X_train_np = np.array([[float(x) for x in row] for row in X_train])
    y_train_np = np.array([int(y) for y in y_train])
    X_calib_np = np.array([[float(x) for x in row] for row in X_calib])
    y_calib_np = np.array([int(y) for y in y_calib])

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

    # Create base classifier
    base_clf = _create_base_classifier(base_model_config, random_state)
    base_model_type = base_model_config.type

    try:
        # Train classifier on training set
        base_clf.fit(X_train_np, y_train_np)

        # Create SplitConformalClassifier with prefit estimator
        mapie_clf = SplitConformalClassifier(
            estimator=base_clf,
            confidence_level=confidence_level,
            conformity_score=conformity_score,
            prefit=True,
        )
        mapie_clf.conformalize(X_calib_np, y_calib_np)

    except Exception as e:
        raise RuntimeError(f"mapie_train_conformal_classifier: Training failed - {e}")

    # Get class information
    classes = base_clf.classes_.tolist()
    n_classes = len(classes)

    # Serialize model
    model_bytes = cloudpickle.dumps(mapie_clf)
    n_features = X_train_np.shape[1]

    return EastStruct({
        "data": model_bytes,
        "n_features": n_features,
        "n_classes": n_classes,
        "classes": EastArray(ArrayType("integer"), [int(c) for c in classes]),
        "confidence_level": confidence_level,
        "base_model_type": EastVariant(base_model_type, None),
    })


def mapie_predict_set_impl(
    model_blob: EastStruct,
    X: EastArray,
) -> EastStruct:
    """Predict with prediction sets using the model's calibrated confidence level."""
    # Load model
    model_bytes = model_blob.get("data")
    model = cloudpickle.loads(model_bytes)
    n_features = model_blob.get("n_features")

    # Convert input
    X_np = np.array([[float(x) for x in row] for row in X])

    # Validate
    if X_np.shape[1] != n_features:
        raise RuntimeError(
            f"mapie_predict_set: Model trained with {n_features} features "
            f"but X has {X_np.shape[1]} features"
        )

    try:
        # MAPIE 1.2.0: predict_set() returns (predictions, sets)
        # sets shape: (n_samples, n_classes, n_confidence_levels)
        y_pred, y_sets = model.predict_set(X_np)

        # y_sets shape: (n_samples, n_classes, 1) for single confidence level
        sets_matrix = y_sets[:, :, 0].astype(int)  # (n_samples, n_classes)
        set_sizes = sets_matrix.sum(axis=1)  # Number of classes in each set

        # Get class probabilities via the underlying estimator
        proba = model.estimator_.predict_proba(X_np)  # (n_samples, n_classes)

    except Exception as e:
        raise RuntimeError(f"mapie_predict_set: Prediction failed - {e}")

    return EastStruct({
        "pred": EastArray(ArrayType("integer"), [int(x) for x in y_pred]),
        "sets": EastArray(ArrayType(ArrayType("integer")),
                         [[int(x) for x in row] for row in sets_matrix]),
        "probabilities": EastArray(ArrayType(ArrayType("float")),
                                   [[float(x) for x in row] for row in proba]),
        "set_sizes": EastArray(ArrayType("integer"), [int(x) for x in set_sizes]),
    })


# ============================================================================
# Function Registry
# ============================================================================

mapie_impl = {
    "mapie_train_conformal_regressor": mapie_train_conformal_regressor_impl,
    "mapie_train_cqr": mapie_train_cqr_impl,
    "mapie_predict_interval": mapie_predict_interval_impl,
    "mapie_train_conformal_classifier": mapie_train_conformal_classifier_impl,
    "mapie_predict_set": mapie_predict_set_impl,
}
```

#### 4. `src/east_py_datascience/__init__.py`

Add MAPIE registration:

```python
# Add to imports
from .mapie import mapie_impl

# Add to FUNCTION_REGISTRY
FUNCTION_REGISTRY = {
    # ... existing functions ...
    **mapie_impl,
}
```

#### 5. `src/index.ts`

Add MAPIE export:

```typescript
// Add to exports
export * from "./mapie/mapie.js";
```

#### 6. `src/mapie/mapie.spec.ts` (new file)

```typescript
import { describeEast, variant } from "@elaraai/east-node-std/testing";
import { Assert } from "@elaraai/east-node-std";
import { MAPIE } from "../index.js";

describeEast("MAPIE platform functions", (test) => {

    // ==========================================================================
    // Regression Tests
    // ==========================================================================

    test("trainConformalRegressor with XGBoost base model (split conformal)", $ => {
        // Training data
        const X_train = $.let([
            [1.0], [2.0], [3.0], [4.0], [5.0],
            [6.0], [7.0], [8.0], [9.0], [10.0],
        ]);
        const y_train = $.let([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5]);

        // Calibration data
        const X_calib = $.let([[2.5], [4.5], [6.5], [8.5]]);
        const y_calib = $.let([3.0, 5.0, 7.0, 9.0]);

        const config = $.let({
            base_model: variant('xgboost', {
                n_estimators: variant('some', 50n),
                max_depth: variant('some', 3n),
                learning_rate: variant('some', 0.1),
                min_child_weight: variant('none', null),
                subsample: variant('none', null),
                colsample_bytree: variant('none', null),
                reg_alpha: variant('none', null),
                reg_lambda: variant('none', null),
                gamma: variant('none', null),
                random_state: variant('some', 42n),
            }),
            method: variant('some', variant('split', null)),
            confidence_level: variant('some', 0.9),  // 90% coverage
            cv_folds: variant('none', null),
            random_state: variant('some', 42n),
        });

        const model = $.let(MAPIE.trainConformalRegressor(X_train, y_train, X_calib, y_calib, config));

        // Test prediction
        const X_test = $.let([[3.0], [5.0], [7.0]]);
        const result = $.let(MAPIE.predictInterval(model, X_test));

        // Check shapes
        $(Assert.equal(result.lower.size(), 3n));
        $(Assert.equal(result.pred.size(), 3n));
        $(Assert.equal(result.upper.size(), 3n));

        // Check interval ordering: lower <= pred <= upper
        $(Assert.lessEqual(result.lower.get(0n), result.pred.get(0n)));
        $(Assert.lessEqual(result.pred.get(0n), result.upper.get(0n)));
    });

    test("trainConformalRegressor with LightGBM base model", $ => {
        const X_train = $.let([
            [1.0], [2.0], [3.0], [4.0], [5.0],
            [6.0], [7.0], [8.0], [9.0], [10.0],
        ]);
        const y_train = $.let([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5]);
        const X_calib = $.let([[2.5], [4.5], [6.5], [8.5]]);
        const y_calib = $.let([3.0, 5.0, 7.0, 9.0]);

        const config = $.let({
            base_model: variant('lightgbm', {
                n_estimators: variant('some', 50n),
                max_depth: variant('some', 3n),
                learning_rate: variant('some', 0.1),
                num_leaves: variant('none', null),
                min_child_samples: variant('none', null),
                subsample: variant('none', null),
                colsample_bytree: variant('none', null),
                reg_alpha: variant('none', null),
                reg_lambda: variant('none', null),
                random_state: variant('some', 42n),
            }),
            method: variant('some', variant('split', null)),
            confidence_level: variant('some', 0.9),
            cv_folds: variant('none', null),
            random_state: variant('some', 42n),
        });

        const model = $.let(MAPIE.trainConformalRegressor(X_train, y_train, X_calib, y_calib, config));
        const X_test = $.let([[3.0], [5.0], [7.0]]);
        const result = $.let(MAPIE.predictInterval(model, X_test));

        $(Assert.equal(result.lower.size(), 3n));
        $(Assert.equal(result.pred.size(), 3n));
        $(Assert.equal(result.upper.size(), 3n));
    });

    test("trainConformalRegressor with cross method", $ => {
        const X_train = $.let([
            [1.0], [2.0], [3.0], [4.0], [5.0],
            [6.0], [7.0], [8.0], [9.0], [10.0],
        ]);
        const y_train = $.let([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5]);
        const X_calib = $.let([[2.5], [4.5], [6.5], [8.5]]);
        const y_calib = $.let([3.0, 5.0, 7.0, 9.0]);

        const config = $.let({
            base_model: variant('xgboost', {
                n_estimators: variant('some', 20n),
                max_depth: variant('some', 3n),
                learning_rate: variant('some', 0.1),
                min_child_weight: variant('none', null),
                subsample: variant('none', null),
                colsample_bytree: variant('none', null),
                reg_alpha: variant('none', null),
                reg_lambda: variant('none', null),
                gamma: variant('none', null),
                random_state: variant('some', 42n),
            }),
            method: variant('some', variant('cross', null)),
            confidence_level: variant('some', 0.9),
            cv_folds: variant('some', 3n),
            random_state: variant('some', 42n),
        });

        const model = $.let(MAPIE.trainConformalRegressor(X_train, y_train, X_calib, y_calib, config));
        const result = $.let(MAPIE.predictInterval(model, X_calib));

        $(Assert.equal(result.pred.size(), 4n));
    });

    test("trainCQR trains conformalized quantile regression", $ => {
        // Linear data with heteroscedastic noise (variance increases with x)
        const X_train = $.let([
            [1.0], [2.0], [3.0], [4.0], [5.0],
            [6.0], [7.0], [8.0], [9.0], [10.0],
        ]);
        const y_train = $.let([1.2, 2.1, 3.3, 4.0, 5.5, 6.2, 7.8, 8.1, 9.9, 10.2]);
        const X_calib = $.let([[2.5], [4.5], [6.5], [8.5]]);
        const y_calib = $.let([2.8, 4.6, 6.9, 8.7]);

        const config = $.let({
            xgboost_config: {
                n_estimators: variant('some', 50n),
                max_depth: variant('some', 3n),
                learning_rate: variant('some', 0.1),
                min_child_weight: variant('none', null),
                subsample: variant('none', null),
                colsample_bytree: variant('none', null),
                reg_alpha: variant('none', null),
                reg_lambda: variant('none', null),
                gamma: variant('none', null),
                random_state: variant('some', 42n),
            },
            confidence_level: variant('some', 0.9),
            random_state: variant('some', 42n),
        });

        const model = $.let(MAPIE.trainCQR(X_train, y_train, X_calib, y_calib, config));
        const X_test = $.let([[3.0], [5.0], [7.0]]);
        const result = $.let(MAPIE.predictInterval(model, X_test));

        $(Assert.equal(result.lower.size(), 3n));
        $(Assert.equal(result.pred.size(), 3n));
        $(Assert.equal(result.upper.size(), 3n));
    });

    test("error: X_train and y_train shape mismatch", $ => {
        const X_train = $.let([[1.0], [2.0], [3.0]]);
        const y_train = $.let([1.0, 2.0]);  // Mismatch!
        const X_calib = $.let([[1.5]]);
        const y_calib = $.let([1.5]);

        const config = $.let({
            base_model: variant('xgboost', {
                n_estimators: variant('some', 10n),
                max_depth: variant('none', null),
                learning_rate: variant('none', null),
                min_child_weight: variant('none', null),
                subsample: variant('none', null),
                colsample_bytree: variant('none', null),
                reg_alpha: variant('none', null),
                reg_lambda: variant('none', null),
                gamma: variant('none', null),
                random_state: variant('none', null),
            }),
            method: variant('none', null),
            confidence_level: variant('none', null),
            cv_folds: variant('none', null),
            random_state: variant('none', null),
        });

        $(Assert.throws(
            MAPIE.trainConformalRegressor(X_train, y_train, X_calib, y_calib, config),
            /X_train has 3 samples but y_train has 2 samples/
        ));
    });

    test("error: feature dimension mismatch between train and calib", $ => {
        const X_train = $.let([[1.0, 2.0], [3.0, 4.0]]);
        const y_train = $.let([1.0, 2.0]);
        const X_calib = $.let([[1.0]]);  // Different number of features!
        const y_calib = $.let([1.0]);

        const config = $.let({
            base_model: variant('xgboost', {
                n_estimators: variant('some', 10n),
                max_depth: variant('none', null),
                learning_rate: variant('none', null),
                min_child_weight: variant('none', null),
                subsample: variant('none', null),
                colsample_bytree: variant('none', null),
                reg_alpha: variant('none', null),
                reg_lambda: variant('none', null),
                gamma: variant('none', null),
                random_state: variant('none', null),
            }),
            method: variant('none', null),
            confidence_level: variant('none', null),
            cv_folds: variant('none', null),
            random_state: variant('none', null),
        });

        $(Assert.throws(
            MAPIE.trainConformalRegressor(X_train, y_train, X_calib, y_calib, config),
            /X_train has 2 features but X_calib has 1 features/
        ));
    });

    // ==========================================================================
    // Classification Tests
    // ==========================================================================

    test("trainConformalClassifier with XGBoost base model (LAC method)", $ => {
        // Binary classification data
        const X_train = $.let([
            [1.0, 0.0], [1.5, 0.5], [2.0, 1.0],  // Class 0
            [3.0, 3.0], [3.5, 3.5], [4.0, 4.0],  // Class 1
        ]);
        const y_train = $.let([0n, 0n, 0n, 1n, 1n, 1n]);

        // Calibration data
        const X_calib = $.let([[1.2, 0.2], [3.2, 3.2]]);
        const y_calib = $.let([0n, 1n]);

        const config = $.let({
            base_model: variant('xgboost', {
                n_estimators: variant('some', 50n),
                max_depth: variant('some', 3n),
                learning_rate: variant('some', 0.1),
                min_child_weight: variant('none', null),
                subsample: variant('none', null),
                colsample_bytree: variant('none', null),
                reg_alpha: variant('none', null),
                reg_lambda: variant('none', null),
                gamma: variant('none', null),
                random_state: variant('some', 42n),
            }),
            method: variant('some', variant('lac', null)),
            confidence_level: variant('some', 0.9),  // 90% coverage
            random_state: variant('some', 42n),
        });

        const model = $.let(MAPIE.trainConformalClassifier(X_train, y_train, X_calib, y_calib, config));

        // Test prediction
        const X_test = $.let([[1.0, 0.0], [4.0, 4.0]]);
        const result = $.let(MAPIE.predictSet(model, X_test));

        // Check shapes
        $(Assert.equal(result.pred.size(), 2n));
        $(Assert.equal(result.sets.size(), 2n));
        $(Assert.equal(result.probabilities.size(), 2n));
        $(Assert.equal(result.set_sizes.size(), 2n));

        // Each set should have at least one class
        $(Assert.greaterEqual(result.set_sizes.get(0n), 1n));
        $(Assert.greaterEqual(result.set_sizes.get(1n), 1n));
    });

    test("trainConformalClassifier with LightGBM base model", $ => {
        const X_train = $.let([
            [1.0, 0.0], [1.5, 0.5], [2.0, 1.0],
            [3.0, 3.0], [3.5, 3.5], [4.0, 4.0],
        ]);
        const y_train = $.let([0n, 0n, 0n, 1n, 1n, 1n]);
        const X_calib = $.let([[1.2, 0.2], [3.2, 3.2]]);
        const y_calib = $.let([0n, 1n]);

        const config = $.let({
            base_model: variant('lightgbm', {
                n_estimators: variant('some', 50n),
                max_depth: variant('some', 3n),
                learning_rate: variant('some', 0.1),
                num_leaves: variant('none', null),
                min_child_samples: variant('none', null),
                subsample: variant('none', null),
                colsample_bytree: variant('none', null),
                reg_alpha: variant('none', null),
                reg_lambda: variant('none', null),
                random_state: variant('some', 42n),
            }),
            method: variant('some', variant('lac', null)),
            confidence_level: variant('some', 0.9),
            random_state: variant('some', 42n),
        });

        const model = $.let(MAPIE.trainConformalClassifier(X_train, y_train, X_calib, y_calib, config));
        const X_test = $.let([[1.0, 0.0], [4.0, 4.0]]);
        const result = $.let(MAPIE.predictSet(model, X_test));

        $(Assert.equal(result.pred.size(), 2n));
        $(Assert.equal(result.set_sizes.size(), 2n));
    });

    test("trainConformalClassifier multiclass", $ => {
        // 3-class classification
        const X_train = $.let([
            [0.0, 0.0], [0.5, 0.5],  // Class 0
            [2.0, 0.0], [2.5, 0.5],  // Class 1
            [1.0, 2.0], [1.5, 2.5],  // Class 2
        ]);
        const y_train = $.let([0n, 0n, 1n, 1n, 2n, 2n]);
        const X_calib = $.let([[0.2, 0.2], [2.2, 0.2], [1.2, 2.2]]);
        const y_calib = $.let([0n, 1n, 2n]);

        const config = $.let({
            base_model: variant('xgboost', {
                n_estimators: variant('some', 30n),
                max_depth: variant('some', 3n),
                learning_rate: variant('some', 0.1),
                min_child_weight: variant('none', null),
                subsample: variant('none', null),
                colsample_bytree: variant('none', null),
                reg_alpha: variant('none', null),
                reg_lambda: variant('none', null),
                gamma: variant('none', null),
                random_state: variant('some', 42n),
            }),
            method: variant('none', null),  // Default LAC
            confidence_level: variant('some', 0.9),
            random_state: variant('some', 42n),
        });

        const model = $.let(MAPIE.trainConformalClassifier(X_train, y_train, X_calib, y_calib, config));
        const result = $.let(MAPIE.predictSet(model, X_calib));

        // Should have 3 probabilities per sample (3 classes)
        $(Assert.equal(result.probabilities.get(0n).size(), 3n));
    });

    test("error: classifier X_train and y_train shape mismatch", $ => {
        const X_train = $.let([[1.0, 0.0], [2.0, 1.0], [3.0, 2.0]]);
        const y_train = $.let([0n, 1n]);  // Mismatch!
        const X_calib = $.let([[1.5, 0.5]]);
        const y_calib = $.let([0n]);

        const config = $.let({
            base_model: variant('xgboost', {
                n_estimators: variant('some', 10n),
                max_depth: variant('none', null),
                learning_rate: variant('none', null),
                min_child_weight: variant('none', null),
                subsample: variant('none', null),
                colsample_bytree: variant('none', null),
                reg_alpha: variant('none', null),
                reg_lambda: variant('none', null),
                gamma: variant('none', null),
                random_state: variant('none', null),
            }),
            method: variant('none', null),
            confidence_level: variant('none', null),
            random_state: variant('none', null),
        });

        $(Assert.throws(
            MAPIE.trainConformalClassifier(X_train, y_train, X_calib, y_calib, config),
            /X_train has 3 samples but y_train has 2 samples/
        ));
    });

}, { exportOnly: true });
```

### Dependencies

Add to `pyproject.toml`:

```toml
[project.optional-dependencies]
mapie = ["mapie>=1.2.0"]
```

Or add as core dependency if MAPIE should always be available:

```toml
dependencies = [
    # ... existing ...
    "mapie>=1.2.0",
]
```

## Alternatives Considered

### Alternative A: Add to existing XGBoost/LightGBM modules

Add `trainConformalQuantile` and `predictConformalInterval` to XGBoost and LightGBM modules directly.

**Rejected**: MAPIE is a cross-cutting concern that works with multiple base models. A separate module is cleaner and makes the conformal prediction concept explicit.

### Alternative B: Wrapper-only approach

```typescript
const baseModel = XGBoost.trainRegressor(X, y, config);
const conformalModel = MAPIE.calibrate(baseModel, X_calib, y_calib);
```

**Rejected**: This requires passing model blobs between modules which is more complex. The integrated approach (train + calibrate in one step) is simpler for users.

### Alternative C: Support prediction at arbitrary confidence levels

The old MAPIE API allowed `predict(alpha=0.05)` at prediction time. MAPIE 1.2.0 requires specifying `confidence_level` at training time.

**Decision**: Match MAPIE 1.2.0 API exactly. If users need multiple confidence levels, they can specify them as a list during training: `confidence_level=[0.9, 0.95, 0.99]`.

## Testing Plan

### Regression Tests

1. **Split conformal with XGBoost** - basic training and prediction
2. **Split conformal with LightGBM** - alternative base model
3. **Cross method** - cross-validation based calibration
4. **CQR with XGBoost** - conformalized quantile regression
5. **Interval ordering** - verify lower <= pred <= upper
6. **Error: shape mismatch** - X and y dimensions
7. **Error: feature mismatch** - train vs calib features

### Classification Tests

8. **LAC method with XGBoost** - basic classification prediction sets
9. **LAC method with LightGBM** - alternative base classifier
10. **APS method** - adaptive prediction sets
11. **Multiclass classification** - more than 2 classes
12. **Prediction set sizes** - verify sets contain at least one class
13. **Error: classifier shape mismatch** - X and y dimensions

### Shared Tests

14. **Error: MAPIE not installed** - graceful error message

## Future Enhancements

1. **Multi-output support**: Extend to multi-target regression
2. **Time series**: Conformal prediction for sequential data
3. **Multiple confidence levels**: Support `confidence_level=[0.9, 0.95]` for multiple intervals
4. **CrossConformalClassifier**: Add cross-validation based classification
