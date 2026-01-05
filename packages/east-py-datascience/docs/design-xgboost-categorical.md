# Design: XGBoost Categorical Feature Support

## Purpose

Add native categorical feature support to XGBoost training functions. XGBoost 2.0+ handles categorical features natively via optimal split finding, which is more efficient than one-hot encoding and produces better splits for high-cardinality categoricals.

## Current State

- XGBoost version: 3.1.2 (dependency: `>=2.0.0`)
- All features are treated as numeric (float matrices)
- No way to indicate which columns are categorical

## Design

### Approach

Add an optional `categorical_features` parameter to `XGBoostConfigType` that specifies which column indices contain categorical values. The Python implementation will:
1. Convert the feature matrix to a pandas DataFrame
2. Cast specified columns to pandas `category` dtype
3. Pass `enable_categorical=True` to XGBoost

### File Changes

#### 1. `src/xgboost/xgboost.ts`

Add to `XGBoostConfigType`:

```typescript
export const XGBoostConfigType = StructType({
    // ... existing fields ...

    /** Column indices that contain categorical features (0-indexed) */
    categorical_features: OptionType(ArrayType(IntegerType)),
    /**
     * Max categories for one-hot encoding (default 4).
     * Features with more categories use partition-based splits.
     */
    max_cat_to_onehot: OptionType(IntegerType),
    /**
     * Max categories considered per split for partition-based method (default 64).
     * Higher values = better accuracy but slower training.
     */
    max_cat_threshold: OptionType(IntegerType),
});
```

Same changes for `XGBoostQuantileConfigType`.

#### 2. `src/east_py_datascience/xgboost/xgboost_impl.py`

Modify `xgboost_train_regressor_impl`, `xgboost_train_classifier_impl`, and `xgboost_train_quantile_impl`:

```python
def xgboost_train_regressor_impl(X, y, config):
    # ... existing code ...

    # Extract categorical features config
    categorical_features = _get_option(config.get("categorical_features"), None)

    if categorical_features is not None:
        cat_indices = [int(i) for i in categorical_features]
        # Validate indices
        for idx in cat_indices:
            if idx < 0 or idx >= X_np.shape[1]:
                raise RuntimeError(
                    f"xgboost_train_regressor: categorical_features index {idx} "
                    f"out of bounds for {X_np.shape[1]} features"
                )

        # Convert to DataFrame with categorical columns
        import pandas as pd
        df = pd.DataFrame(X_np)
        for idx in cat_indices:
            df[idx] = df[idx].astype('category')
        X_train = df
        enable_categorical = True
    else:
        X_train = X_np
        enable_categorical = False

    # Extract categorical config options
    max_cat_to_onehot = _get_option(config.get("max_cat_to_onehot"), None)
    max_cat_threshold = _get_option(config.get("max_cat_threshold"), None)

    model = xgb.XGBRegressor(
        # ... existing params ...
        enable_categorical=enable_categorical,
        max_cat_to_onehot=int(max_cat_to_onehot) if max_cat_to_onehot else 4,
        max_cat_threshold=int(max_cat_threshold) if max_cat_threshold else 64,
    )
    model.fit(X_train, y_np, sample_weight=sample_weight_np)
```

Also store `categorical_features` in model blob for prediction:

```python
return EastVariant(
    "xgboost_regressor",
    EastStruct({
        "data": model_data,
        "n_features": n_features,
        "categorical_features": (
            numpy_to_east_int_vector(np.array(cat_indices))
            if categorical_features is not None
            else None
        ),
    }),
)
```

Modify predict functions to handle categorical features:

```python
def xgboost_predict_impl(model_blob, X):
    # ... existing validation ...

    categorical_features = model_blob.value.get("categorical_features")
    if categorical_features is not None:
        cat_indices = east_int_vector_to_numpy(categorical_features)
        import pandas as pd
        df = pd.DataFrame(X_np)
        for idx in cat_indices:
            df[idx] = df[idx].astype('category')
        X_pred = df
    else:
        X_pred = X_np

    y_pred = model.predict(X_pred)
```

#### 3. `src/xgboost/xgboost.spec.ts`

Add tests:

```typescript
test("train_regressor with categorical features", $ => {
    // Feature 0: numeric, Feature 1: categorical (encoded as 0, 1, 2)
    const X = $.let([
        [1.0, 0.0],  // category A
        [2.0, 0.0],  // category A
        [3.0, 1.0],  // category B
        [4.0, 1.0],  // category B
        [5.0, 2.0],  // category C
        [6.0, 2.0],  // category C
    ]);
    const y = $.let([10.0, 11.0, 20.0, 21.0, 30.0, 31.0]);

    const config = $.let({
        n_estimators: variant('some', 50n),
        max_depth: variant('some', 3n),
        learning_rate: variant('some', 0.3),
        min_child_weight: variant('none', null),
        subsample: variant('none', null),
        colsample_bytree: variant('none', null),
        reg_alpha: variant('none', null),
        reg_lambda: variant('none', null),
        random_state: variant('some', 42n),
        n_jobs: variant('none', null),
        sample_weight: variant('none', null),
        categorical_features: variant('some', [1n]),  // Column 1 is categorical
    });

    const model = $.let(XGBoost.trainRegressor(X, y, config));
    const y_pred = $.let(XGBoost.predict(model, X));

    $(Assert.equal(y_pred.size(), 6n));
});

test("train_classifier with categorical features", $ => {
    // Similar test for classifier
});

test("error: categorical_features index out of bounds", $ => {
    const X = $.let([[1.0, 2.0], [3.0, 4.0]]);
    const y = $.let([1.0, 2.0]);

    const config = $.let({
        // ... other fields ...
        categorical_features: variant('some', [5n]),  // Invalid: only 2 features
        max_cat_to_onehot: variant('none', null),
        max_cat_threshold: variant('none', null),
    });

    $(Assert.throws(
        XGBoost.trainRegressor(X, y, config),
        /categorical_features index 5 out of bounds/
    ));
});

test("categorical with custom max_cat_to_onehot", $ => {
    // High-cardinality categorical (10 categories)
    const X = $.let([
        [1.0, 0.0], [2.0, 1.0], [3.0, 2.0], [4.0, 3.0], [5.0, 4.0],
        [6.0, 5.0], [7.0, 6.0], [8.0, 7.0], [9.0, 8.0], [10.0, 9.0],
    ]);
    const y = $.let([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);

    const config = $.let({
        // ... other fields ...
        categorical_features: variant('some', [1n]),
        max_cat_to_onehot: variant('some', 8n),  // Force one-hot for up to 8 categories
        max_cat_threshold: variant('some', 32n),
    });

    const model = $.let(XGBoost.trainRegressor(X, y, config));
    const y_pred = $.let(XGBoost.predict(model, X));
    $(Assert.equal(y_pred.size(), 10n));
});
```

### Model Blob Backward Compatibility

The `categorical_features` field in the model blob is optional (can be `None`). Models trained without categorical features will have `None`, and prediction functions check for this.

### Limitations

1. **Numeric encoding required**: Categorical values must be pre-encoded as whole numbers in float format (0.0, 1.0, 2.0, ...). The feature matrix is `MatrixType` (`ArrayType(ArrayType(FloatType))`), so categories are represented as floats.

2. **Consistent encoding**: The same numeric encoding must be used for training and prediction. Category 0.0 at training must mean the same thing at prediction.

3. **No automatic encoding**: Unlike sklearn's `OrdinalEncoder`, users must pre-encode categorical values.

## Alternatives Considered

### Alternative A: Separate `feature_types` array

Instead of `categorical_features: [1, 3]`, use `feature_types: ['q', 'c', 'q', 'c']` where 'q' = quantitative, 'c' = categorical.

**Rejected**: More verbose, requires specifying all features, and XGBoost's `enable_categorical` with pandas category dtype is simpler.

### Alternative B: New `trainRegressorCategorical` function

Create separate functions for categorical training.

**Rejected**: Adds API surface area unnecessarily. A config option is cleaner.

## Testing Plan

1. Train regressor with categorical features - verify predictions work
2. Train classifier with categorical features - verify predictions work
3. Train quantile regressor with categorical features - verify predictions work
4. Error: categorical index out of bounds
5. Mixed numeric + categorical features
6. Custom `max_cat_to_onehot` setting
7. Custom `max_cat_threshold` setting
8. High-cardinality categorical (verify partition-based splits work)
