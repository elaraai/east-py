# Module 3a: XGBoost Quantile Regression

## Purpose

Extend XGBoost with quantile regression for probabilistic predictions. Unlike point predictions, quantile regression provides prediction intervals and full predictive distributions, enabling uncertainty quantification without the performance overhead of probabilistic models like NGBoost.

## Motivation

- **Battle-tested**: XGBoost's quantile regression uses `objective='reg:quantileerror'` (pinball loss), available since XGBoost 1.3
- **Fast**: Same training speed as standard XGBoost regression
- **Flexible**: Train separate models per quantile or use multi-output for efficiency
- **Production-ready**: No additional dependencies beyond XGBoost

## Design Decisions

### Single vs Multi-Quantile Models

**Option A: One model per quantile** (Recommended)
- Train separate XGBRegressor for each quantile (e.g., 0.1, 0.5, 0.9)
- Pros: Simple, each quantile independently optimized, easy to add/remove quantiles
- Cons: Multiple models to store/manage, N times training cost

**Option B: Multi-output single model**
- Use XGBoost's multi-output regression
- Pros: Single model, potentially faster inference
- Cons: More complex, quantiles not independently optimized, crossing quantiles possible

**Decision**: Option A - separate models bundled in a single model blob. This is simpler, more robust, and aligns with best practices.

### Serialization

Use **cloudpickle** (not ONNX) for quantile models:
- ONNX conversion for quantile regression is less mature
- cloudpickle handles all XGBoost model variants
- Consistent with NGBoost and other probabilistic models

## Type Definitions (TypeScript)

### Config Type

```typescript
/**
 * Configuration for XGBoost quantile regression.
 * Extends base XGBoostConfigType with quantile-specific options.
 */
export const XGBoostQuantileConfigType = StructType({
    /** Quantiles to predict (e.g., [0.1, 0.5, 0.9] for 80% interval + median) */
    quantiles: VectorType,
    /** Number of boosting rounds (default 100) */
    n_estimators: OptionType(IntegerType),
    /** Maximum tree depth (default 6) */
    max_depth: OptionType(IntegerType),
    /** Learning rate (default 0.3) */
    learning_rate: OptionType(FloatType),
    /** Minimum sum of instance weight in child (default 1) */
    min_child_weight: OptionType(IntegerType),
    /** Subsample ratio of training instances (default 1.0) */
    subsample: OptionType(FloatType),
    /** Subsample ratio of columns (default 1.0) */
    colsample_bytree: OptionType(FloatType),
    /** L1 regularization (default 0) */
    reg_alpha: OptionType(FloatType),
    /** L2 regularization (default 1) */
    reg_lambda: OptionType(FloatType),
    /** Random seed */
    random_state: OptionType(IntegerType),
    /** Parallel threads (default -1 for all) */
    n_jobs: OptionType(IntegerType),
});
```

### Result Type

```typescript
/**
 * Result from quantile prediction.
 */
export const XGBoostQuantilePredictResultType = StructType({
    /** Quantile values that were predicted */
    quantiles: VectorType,
    /** Predictions matrix: (n_samples x n_quantiles) */
    predictions: MatrixType,
});
```

### Model Blob Type

Extend existing `XGBoostModelBlobType`:

```typescript
export const XGBoostModelBlobType = VariantType({
    // ... existing variants ...

    /** XGBoost quantile regressor (multiple models, one per quantile) */
    xgboost_quantile: StructType({
        /** Cloudpickle serialized dict of {quantile: model} */
        data: BlobType,
        /** Quantiles this model predicts */
        quantiles: VectorType,
        /** Number of input features */
        n_features: IntegerType,
    }),
});
```

## Platform Functions

### `xgboost_train_quantile`

Train XGBoost quantile regression models.

```python
PlatformFunction(
    name="xgboost_train_quantile",
    inputs=[MatrixType, VectorType, XGBoostQuantileConfigType],
    output=XGBoostModelBlobType,  # Returns "xgboost_quantile" variant
    type="sync",
    fn=xgboost_train_quantile_impl,
)
```

### `xgboost_predict_quantile`

Predict quantiles with trained model.

```python
PlatformFunction(
    name="xgboost_predict_quantile",
    inputs=[XGBoostModelBlobType, MatrixType],  # Expects "xgboost_quantile" variant
    output=XGBoostQuantilePredictResultType,
    type="sync",
    fn=xgboost_predict_quantile_impl,
)
```

## Python Implementation

```python
def xgboost_train_quantile_impl(
    X: EastArray,
    y: EastArray,
    config: EastStruct,
) -> EastVariant:
    """Train XGBoost quantile regressor models."""
    import xgboost as xgb
    import numpy as np

    X_np = east_matrix_to_numpy(X)
    y_np = east_vector_to_numpy(y)
    n_features = X_np.shape[1]

    # Get quantiles from config
    quantiles_arr = config.get("quantiles")
    quantiles = [float(q) for q in quantiles_arr]

    # Validate quantiles
    for q in quantiles:
        if not 0 < q < 1:
            raise ValueError(f"Quantiles must be in (0, 1), got {q}")

    # Extract common config
    base_params = {
        "n_estimators": int(_get_option(config.get("n_estimators"), 100)),
        "max_depth": int(_get_option(config.get("max_depth"), 6)),
        "learning_rate": float(_get_option(config.get("learning_rate"), 0.3)),
        "min_child_weight": int(_get_option(config.get("min_child_weight"), 1)),
        "subsample": float(_get_option(config.get("subsample"), 1.0)),
        "colsample_bytree": float(_get_option(config.get("colsample_bytree"), 1.0)),
        "reg_alpha": float(_get_option(config.get("reg_alpha"), 0.0)),
        "reg_lambda": float(_get_option(config.get("reg_lambda"), 1.0)),
        "n_jobs": int(_get_option(config.get("n_jobs"), -1)),
        "verbosity": 0,
    }

    random_state = _get_option(config.get("random_state"), None)
    if random_state is not None:
        base_params["random_state"] = int(random_state)

    # Train one model per quantile
    models = {}
    for q in quantiles:
        model = xgb.XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=q,
            **base_params,
        )
        model.fit(X_np, y_np)
        models[q] = model

    # Serialize all models together
    model_data = _serialize_model(models)

    return EastVariant(
        "xgboost_quantile",
        EastStruct({
            "data": model_data,
            "quantiles": numpy_to_east_vector(np.array(quantiles)),
            "n_features": n_features,
        }),
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

    X_np = east_matrix_to_numpy(X)
    n_samples = X_np.shape[0]

    # Deserialize models dict
    models = _deserialize_model(model_blob.value["data"])
    quantiles = east_vector_to_numpy(model_blob.value["quantiles"])

    # Predict each quantile
    predictions = np.zeros((n_samples, len(quantiles)))
    for i, q in enumerate(quantiles):
        predictions[:, i] = models[q].predict(X_np)

    return EastStruct({
        "quantiles": numpy_to_east_vector(quantiles),
        "predictions": numpy_to_east_matrix(predictions),
    })
```

## TypeScript API

```typescript
// In xgboost.ts

export const xgboost_train_quantile = East.platform(
    "xgboost_train_quantile",
    [MatrixType, VectorType, XGBoostQuantileConfigType],
    XGBoostModelBlobType
);

export const xgboost_predict_quantile = East.platform(
    "xgboost_predict_quantile",
    [XGBoostModelBlobType, MatrixType],
    XGBoostQuantilePredictResultType
);

// Update XGBoost export
export const XGBoost = {
    // ... existing ...
    /** Train quantile regressor */
    trainQuantile: xgboost_train_quantile,
    /** Predict quantiles */
    predictQuantile: xgboost_predict_quantile,
    // ...
};
```

## Usage Example

```typescript
import { East, variant } from "@elaraai/east";
import { XGBoost } from "@elaraai/east-py-datascience";

// Train quantile model for 80% prediction interval
const trainQuantile = East.function([], XGBoost.Types.ModelBlobType, $ => {
    const X = $.let([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]);
    const y = $.let([1.0, 2.0, 3.0, 4.0]);

    const config = $.let({
        quantiles: [0.1, 0.5, 0.9],  // 80% interval + median
        n_estimators: variant('some', 100n),
        max_depth: variant('some', 4n),
        learning_rate: variant('some', 0.1),
        min_child_weight: variant('none', null),
        subsample: variant('none', null),
        colsample_bytree: variant('none', null),
        reg_alpha: variant('none', null),
        reg_lambda: variant('none', null),
        random_state: variant('some', 42n),
        n_jobs: variant('none', null),
    });

    return $.return(XGBoost.trainQuantile(X, y, config));
});

// Make quantile predictions
const predictQuantile = East.function(
    [XGBoost.Types.ModelBlobType, MatrixType],
    XGBoost.Types.XGBoostQuantilePredictResultType,
    ($, model, X_new) => {
        return $.return(XGBoost.predictQuantile(model, X_new));
    }
);
```

## Helper Functions (Optional)

Consider adding convenience functions for common use cases:

### `xgboost_predict_interval`

Extract prediction interval from quantile predictions.

```typescript
export const XGBoostIntervalResultType = StructType({
    /** Point prediction (median, q=0.5) */
    prediction: VectorType,
    /** Lower bound */
    lower: VectorType,
    /** Upper bound */
    upper: VectorType,
    /** Confidence level (e.g., 0.8 for 80% interval) */
    confidence: FloatType,
});

export const xgboost_predict_interval = East.platform(
    "xgboost_predict_interval",
    [XGBoostModelBlobType, MatrixType],
    XGBoostIntervalResultType
);
```

This would validate that the model has symmetric quantiles (e.g., 0.1 and 0.9) and 0.5, then return a cleaner interface.

## Performance Considerations

1. **Training**: Linear in number of quantiles (N models = N times training)
2. **Inference**: Also linear but fast (XGBoost inference is highly optimized)
3. **Storage**: N models stored, but XGBoost models are compact

For typical use (3 quantiles: 0.1, 0.5, 0.9):
- Training: ~3x single model
- Inference: ~3x single prediction
- Storage: ~3x single model

## Testing

```python
def test_xgboost_quantile_regression():
    """Test XGBoost quantile regression."""
    import numpy as np

    # Generate synthetic data with heteroscedastic noise
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = X[:, 0] + 0.5 * X[:, 1] + np.random.randn(100) * (1 + 0.5 * np.abs(X[:, 0]))

    # Train model
    config = {
        "quantiles": [0.1, 0.5, 0.9],
        "n_estimators": {"type": "some", "value": 50},
        "max_depth": {"type": "some", "value": 3},
        # ... other options as "none"
    }

    model = xgboost_train_quantile_impl(
        numpy_to_east_matrix(X),
        numpy_to_east_vector(y),
        config_to_east_struct(config),
    )

    # Predict
    result = xgboost_predict_quantile_impl(model, numpy_to_east_matrix(X))

    predictions = east_matrix_to_numpy(result["predictions"])

    # Validate quantile ordering (lower < median < upper)
    assert np.all(predictions[:, 0] <= predictions[:, 1])
    assert np.all(predictions[:, 1] <= predictions[:, 2])

    # Check approximate coverage
    in_interval = (y >= predictions[:, 0]) & (y <= predictions[:, 2])
    coverage = np.mean(in_interval)
    assert 0.7 < coverage < 0.95  # Should be ~80% for [0.1, 0.9]
```

## Migration Path

This is a new feature addition - no breaking changes to existing XGBoost functions.

## Files to Modify

1. `src/xgboost/xgboost.ts` - Add TypeScript types and platform function declarations
2. `src/east_py_datascience/xgboost/xgboost_impl.py` - Add Python implementations
3. `src/east_py_datascience/types.py` - Add new types if not already present
4. `src/index.ts` - Export new types
5. `tests/` - Add test cases
6. `USAGE.md` - Document new functions
