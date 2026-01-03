# Design: Conditional Inference for Lightning (predict/encode with conditions)

## Problem Statement

The Lightning module supports conditional generation via `condition_dim` in temporal architectures (conv1d, sequential, transformer). Training already accepts conditions as the 6th parameter, and `decodeConditional` exists for conditional decoding from latent.

However, the current `predict` and `encode` functions do not support conditions:

```typescript
// Current signatures (no condition support)
export const lightning_predict = East.platform("lightning_predict",
    [LightningModelBlobType, MatrixType, OptionType(Tensor3DBoolType)],
    MatrixType
);

export const lightning_encode = East.platform("lightning_encode",
    [LightningModelBlobType, MatrixType],
    MatrixType
);
```

This causes issues when using Stage 1 embeddings as conditioning input for Stage 2 models:

```typescript
// This fails - predict doesn't accept 4th argument
const X_val_pred = $.let(Lightning.predict(output.model, X_val, variant("some", val_masks), variant("some", C_val)));
//                                                                                           ^^^^^^^^^^^^^^^^^ ERROR
```

### Use Case: Two-Stage Additive Model

1. **Stage 1: Binary autoencoder** - predicts which (additive, day) slots have any additive
   - Produces embeddings capturing "what additives, when"
   - No conditioning needed

2. **Stage 2: Amount predictor** - predicts binned amounts for each slot
   - Conditioned on Stage 1 embeddings (captures which slots are active)
   - Temporal architecture (conv1d) with `condition_dim = stage1_embedding_dim`
   - At inference, needs to pass Stage 1 embeddings as conditions

## Analysis: Which Functions Need Conditions?

### `encode` - Does NOT need conditions

Encoding extracts a latent representation from input X. This is independent of conditions:
- Encoder: X → z (latent)
- The condition only affects the decoder, not the encoder

**No change needed for `encode`.**

### `predict` - NEEDS conditions for conditional models

Prediction is a full forward pass: encode → decode. For conditional models:
- predict(X, condition) = decode(encode(X), condition)

If a model was trained with conditions and has `condition_dim` set, `predict` must accept conditions to produce correct outputs.

### `decodeConditional` - Already exists

Already supports conditional decoding from latent space.

## Proposed Changes

### TypeScript (`lightning.ts`)

Add optional conditions parameter to `lightning_predict`:

```typescript
/**
 * Predict using a Lightning model.
 *
 * @param model - Trained model blob
 * @param X - Input features matrix (n_samples, n_features)
 * @param masks - Optional 3D boolean masks for inference
 * @param conditions - Optional condition matrix for conditional models (n_samples, condition_dim)
 * @returns Predicted probabilities matrix (n_samples, output_dim)
 */
export const lightning_predict = East.platform(
    "lightning_predict",
    [
        LightningModelBlobType,
        MatrixType,
        OptionType(Tensor3DBoolType),
        OptionType(MatrixType),  // NEW: conditions
    ],
    MatrixType
);
```

Update `Lightning` namespace:

```typescript
export const Lightning = {
    // ...existing...

    /**
     * Predict using a Lightning model.
     *
     * Returns predictions from a trained model with optional mask and condition support.
     * For conditional models (trained with condition_dim), pass the condition vectors.
     */
    predict: lightning_predict,

    // ...existing...
};
```

### Python (`lightning_impl.py`)

Update `lightning_predict_impl`:

```python
def lightning_predict_impl(
    model_blob: EastVariant,
    X: EastArray,
    masks: EastVariant | None,
    conditions: EastVariant | None,  # NEW
) -> EastArray:
    """Predict using a Lightning model with optional conditions."""
    # Extract model data
    model_data = model_blob.value
    model_bytes = bytes(model_data.get("data"))
    model = _deserialize_model(model_bytes)
    model.eval()

    # Convert input
    X_np = east_matrix_to_numpy(X)
    X_tensor = torch.tensor(X_np, dtype=torch.float32)

    # Parse masks if provided
    masks_tensor = None
    if masks is not None and is_east_variant(masks) and masks.type == "some":
        masks_list = masks.value
        masks_np = np.array([[[bool(v) for v in row] for row in sample] for sample in masks_list])
        masks_tensor = torch.tensor(masks_np, dtype=torch.bool)

    # Parse conditions if provided
    condition_tensor = None
    if conditions is not None and is_east_variant(conditions) and conditions.type == "some":
        condition_np = east_matrix_to_numpy(conditions.value)
        condition_tensor = torch.tensor(condition_np, dtype=torch.float32)

        # Validate condition_dim matches model
        if not hasattr(model, 'condition_dim') or model.condition_dim is None:
            raise ValueError("Model has no condition_dim but conditions were provided")
        if condition_tensor.shape[1] != model.condition_dim:
            raise ValueError(
                f"Expected condition_dim={model.condition_dim}, got {condition_tensor.shape[1]}"
            )

    # Validate: if model expects conditions, they must be provided
    if hasattr(model, 'condition_dim') and model.condition_dim is not None:
        if condition_tensor is None:
            raise ValueError(
                f"Model requires condition_dim={model.condition_dim} but no conditions provided"
            )

    # Predict
    with torch.no_grad():
        if condition_tensor is not None:
            # Conditional forward pass
            probs = model.predict_probs_with_masks_conditional(
                X_tensor, masks_tensor, condition_tensor
            ).numpy()
        else:
            # Standard forward pass
            probs = model.predict_probs_with_masks(X_tensor, masks_tensor).numpy()

    return numpy_to_east_matrix(probs)
```

### Update `LightningMLP` class

Add method for conditional prediction:

```python
class LightningMLP(pl.LightningModule):
    # ...existing methods...

    def predict_probs_with_masks_conditional(
        self,
        X: torch.Tensor,
        masks: torch.Tensor | None,
        conditions: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass with conditions and apply output activation with masks."""
        # For temporal architectures with condition_dim
        if self.architecture_type in ("conv1d", "sequential", "transformer"):
            logits = self.net.forward(X, conditions)
        else:
            raise ValueError(f"Conditional predict not supported for {self.architecture_type}")

        probs = self.apply_output_activation(logits)

        if masks is not None:
            probs = probs * masks.float().view(probs.shape[0], -1)

        return probs
```

### Model Serialization

Ensure `condition_dim` is saved/loaded with the model:

```python
def _serialize_model(model: LightningMLP) -> bytes:
    """Serialize model to bytes."""
    state = {
        "state_dict": model.state_dict(),
        "hparams": model.hparams,
        "architecture_type": model.architecture_type,
        "output_type": model.output_type,
        "condition_dim": getattr(model, 'condition_dim', None),  # ENSURE THIS IS SAVED
    }
    buffer = io.BytesIO()
    torch.save(state, buffer)
    return buffer.getvalue()

def _deserialize_model(model_bytes: bytes) -> LightningMLP:
    """Deserialize model from bytes."""
    buffer = io.BytesIO(model_bytes)
    state = torch.load(buffer, weights_only=False)

    # Reconstruct model
    model = LightningMLP(**state["hparams"])
    model.load_state_dict(state["state_dict"])
    model.architecture_type = state["architecture_type"]
    model.output_type = state["output_type"]
    model.condition_dim = state.get("condition_dim")  # RESTORE THIS

    return model
```

## Tests

Add to `/home/crambelsoupy/src/east-py/packages/east-py-datascience/src/lightning/lightning.spec.ts`:

### Test 1: Conditional predict with conv1d

```typescript
test("conv1d: predict with conditions", $ => {
    // 2 channels x 3 time steps x 2 classes = 12 features
    const X = $.let([
        [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
    ]);

    // Condition: 3-dim feature vector per sample
    const conditions = $.let([
        [1.0, 0.0, 0.5],
        [0.0, 1.0, 0.8],
        [1.0, 0.0, 0.5],  // same as sample 0
        [0.5, 0.5, 0.3],
    ]);

    const config = $.let({
        architecture: variant('conv1d', {
            n_channels: 2n,
            sequence_length: 3n,
            conv_channels: [8n],
            kernel_size: 3n,
            latent_dim: 4n,
            condition_dim: variant('some', 3n),
        }),
        output: variant('multi_head', {
            n_heads: 6n,
            n_classes_per_head: 2n,
            class_weights: variant('none', null),
        }),
        learning_rate: variant('some', 0.01),
        max_epochs: variant('some', 100n),
        patience: variant('some', 20n),
        batch_size: variant('some', 2n),
        dropout: variant('some', 0.0),
        gradient_clip: variant('some', 1.0),
        weight_decay: variant('none', null),
        random_state: variant('some', 42n),
        epoch_callback: variant('none', null),
    });

    // Train with conditions
    const result = $.let(Lightning.train(
        X, X, config,
        variant('none', null),  // no masks
        variant('none', null),  // no group weights
        variant('some', conditions)  // conditions
    ));
    $(Assert.greaterEqual(result.best_epoch, 0n));

    // Predict with conditions (4th argument)
    const y_pred = $.let(Lightning.predict(
        result.model,
        X,
        variant('none', null),  // no masks
        variant('some', conditions)  // conditions
    ));

    $(Assert.equal(y_pred.size(), 4n));
    $(Assert.equal(y_pred.get(0n).size(), 12n));

    // Encode (no conditions needed)
    const z = $.let(Lightning.encode(result.model, X));
    $(Assert.equal(z.size(), 4n));
    $(Assert.equal(z.get(0n).size(), 4n));

    // decodeConditional with conditions
    const decoded = $.let(Lightning.decodeConditional(result.model, z, conditions));
    $(Assert.equal(decoded.size(), 4n));
    $(Assert.equal(decoded.get(0n).size(), 12n));
});
```

### Test 2: Error - predict without conditions on conditional model

```typescript
test("error: predict on conditional model without conditions", $ => {
    const X = $.let([
        [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
    ]);
    const conditions = $.let([[1.0, 0.0], [0.0, 1.0]]);

    const config = $.let({
        architecture: variant('conv1d', {
            n_channels: 2n,
            sequence_length: 3n,
            conv_channels: [8n],
            kernel_size: 3n,
            latent_dim: 4n,
            condition_dim: variant('some', 2n),  // requires conditions
        }),
        output: variant('multi_head', {
            n_heads: 6n,
            n_classes_per_head: 2n,
            class_weights: variant('none', null),
        }),
        learning_rate: variant('some', 0.01),
        max_epochs: variant('some', 20n),
        patience: variant('some', 5n),
        batch_size: variant('some', 2n),
        dropout: variant('none', null),
        gradient_clip: variant('none', null),
        weight_decay: variant('none', null),
        random_state: variant('some', 42n),
        epoch_callback: variant('none', null),
    });

    // Train with conditions
    const result = $.let(Lightning.train(
        X, X, config,
        variant('none', null),
        variant('none', null),
        variant('some', conditions)
    ));

    // Try to predict WITHOUT conditions - should fail
    $(Assert.throws(
        Lightning.predict(result.model, X, variant('none', null), variant('none', null)),
        /Model requires condition_dim=2 but no conditions provided/
    ));
});
```

### Test 3: Error - predict with wrong condition_dim

```typescript
test("error: predict with wrong condition_dim", $ => {
    const X = $.let([
        [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
    ]);
    const train_conditions = $.let([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5]]);

    const config = $.let({
        architecture: variant('conv1d', {
            n_channels: 2n,
            sequence_length: 3n,
            conv_channels: [8n],
            kernel_size: 3n,
            latent_dim: 4n,
            condition_dim: variant('some', 3n),  // expects 3 dims
        }),
        output: variant('multi_head', {
            n_heads: 6n,
            n_classes_per_head: 2n,
            class_weights: variant('none', null),
        }),
        learning_rate: variant('some', 0.01),
        max_epochs: variant('some', 20n),
        patience: variant('some', 5n),
        batch_size: variant('some', 2n),
        dropout: variant('none', null),
        gradient_clip: variant('none', null),
        weight_decay: variant('none', null),
        random_state: variant('some', 42n),
        epoch_callback: variant('none', null),
    });

    // Train with correct conditions
    const result = $.let(Lightning.train(
        X, X, config,
        variant('none', null),
        variant('none', null),
        variant('some', train_conditions)
    ));

    // Try to predict with WRONG condition_dim (2 instead of 3)
    const wrong_conditions = $.let([[1.0, 0.0], [0.0, 1.0]]);

    $(Assert.throws(
        Lightning.predict(result.model, X, variant('none', null), variant('some', wrong_conditions)),
        /Expected condition_dim=3, got 2/
    ));
});
```

### Test 4: Non-conditional model ignores conditions

```typescript
test("non-conditional model: predict ignores none conditions", $ => {
    const X = $.let([
        [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
    ]);

    const config = $.let({
        architecture: variant('conv1d', {
            n_channels: 2n,
            sequence_length: 3n,
            conv_channels: [8n],
            kernel_size: 3n,
            latent_dim: 4n,
            condition_dim: variant('none', null),  // no conditions
        }),
        output: variant('multi_head', {
            n_heads: 6n,
            n_classes_per_head: 2n,
            class_weights: variant('none', null),
        }),
        learning_rate: variant('some', 0.01),
        max_epochs: variant('some', 20n),
        patience: variant('some', 5n),
        batch_size: variant('some', 2n),
        dropout: variant('none', null),
        gradient_clip: variant('none', null),
        weight_decay: variant('none', null),
        random_state: variant('some', 42n),
        epoch_callback: variant('none', null),
    });

    // Train without conditions
    const result = $.let(Lightning.train(
        X, X, config,
        variant('none', null),
        variant('none', null),
        variant('none', null)  // no conditions
    ));

    // Predict without conditions (should work fine)
    const y_pred = $.let(Lightning.predict(
        result.model, X, variant('none', null), variant('none', null)
    ));

    $(Assert.equal(y_pred.size(), 2n));
    $(Assert.equal(y_pred.get(0n).size(), 12n));
});
```

### Test 5: Predict with masks AND conditions

```typescript
test("conv1d: predict with masks and conditions", $ => {
    const X = $.let([
        [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
    ]);

    // Masks: [n_samples, n_heads, n_classes] = [2, 6, 2]
    const masks = $.let([
        [[true, true], [true, true], [true, false], [true, true], [true, true], [true, true]],
        [[true, true], [true, true], [true, true], [true, true], [false, true], [true, true]],
    ]);

    const conditions = $.let([[1.0, 0.0], [0.0, 1.0]]);

    const config = $.let({
        architecture: variant('conv1d', {
            n_channels: 2n,
            sequence_length: 3n,
            conv_channels: [8n],
            kernel_size: 3n,
            latent_dim: 4n,
            condition_dim: variant('some', 2n),
        }),
        output: variant('multi_head', {
            n_heads: 6n,
            n_classes_per_head: 2n,
            class_weights: variant('none', null),
        }),
        learning_rate: variant('some', 0.01),
        max_epochs: variant('some', 50n),
        patience: variant('some', 10n),
        batch_size: variant('some', 2n),
        dropout: variant('some', 0.0),
        gradient_clip: variant('some', 1.0),
        weight_decay: variant('none', null),
        random_state: variant('some', 42n),
        epoch_callback: variant('none', null),
    });

    // Train with masks and conditions
    const result = $.let(Lightning.train(
        X, X, config,
        variant('some', masks),
        variant('none', null),
        variant('some', conditions)
    ));

    // Predict with both masks and conditions
    const y_pred = $.let(Lightning.predict(
        result.model, X,
        variant('some', masks),
        variant('some', conditions)
    ));

    $(Assert.equal(y_pred.size(), 2n));

    // Masked positions should have ~0 probability
    $(Assert.less(y_pred.get(0n).get(5n), East.value(0.001)));  // sample 0, head 2, class 1 masked
    $(Assert.less(y_pred.get(1n).get(8n), East.value(0.001)));  // sample 1, head 4, class 0 masked
});
```

### Test 6: Sequential with conditions

```typescript
test("sequential: predict with conditions", $ => {
    const X = $.let([
        [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
    ]);
    const conditions = $.let([[1.0, 0.0], [0.0, 1.0]]);

    const config = $.let({
        architecture: variant('sequential', {
            n_channels: 2n,
            sequence_length: 3n,
            hidden_size: 8n,
            n_layers: 1n,
            cell_type: variant('lstm', null),
            latent_dim: 4n,
            bidirectional: false,
            condition_dim: variant('some', 2n),
        }),
        output: variant('multi_head', {
            n_heads: 6n,
            n_classes_per_head: 2n,
            class_weights: variant('none', null),
        }),
        learning_rate: variant('some', 0.01),
        max_epochs: variant('some', 50n),
        patience: variant('some', 10n),
        batch_size: variant('some', 2n),
        dropout: variant('some', 0.0),
        gradient_clip: variant('some', 1.0),
        weight_decay: variant('none', null),
        random_state: variant('some', 42n),
        epoch_callback: variant('none', null),
    });

    const result = $.let(Lightning.train(
        X, X, config,
        variant('none', null),
        variant('none', null),
        variant('some', conditions)
    ));

    const y_pred = $.let(Lightning.predict(
        result.model, X, variant('none', null), variant('some', conditions)
    ));

    $(Assert.equal(y_pred.size(), 2n));
    $(Assert.equal(y_pred.get(0n).size(), 12n));
});
```

### Test 7: Transformer with conditions

```typescript
test("transformer: predict with conditions", $ => {
    const X = $.let([
        [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
    ]);
    const conditions = $.let([[1.0, 0.0], [0.0, 1.0]]);

    const config = $.let({
        architecture: variant('transformer', {
            n_channels: 2n,
            sequence_length: 3n,
            d_model: 8n,
            n_attention_heads: 2n,
            n_layers: 1n,
            d_ff: variant('none', null),
            latent_dim: 4n,
            condition_dim: variant('some', 2n),
        }),
        output: variant('multi_head', {
            n_heads: 6n,
            n_classes_per_head: 2n,
            class_weights: variant('none', null),
        }),
        learning_rate: variant('some', 0.01),
        max_epochs: variant('some', 50n),
        patience: variant('some', 10n),
        batch_size: variant('some', 2n),
        dropout: variant('some', 0.0),
        gradient_clip: variant('some', 1.0),
        weight_decay: variant('none', null),
        random_state: variant('some', 42n),
        epoch_callback: variant('none', null),
    });

    const result = $.let(Lightning.train(
        X, X, config,
        variant('none', null),
        variant('none', null),
        variant('some', conditions)
    ));

    const y_pred = $.let(Lightning.predict(
        result.model, X, variant('none', null), variant('some', conditions)
    ));

    $(Assert.equal(y_pred.size(), 2n));
    $(Assert.equal(y_pred.get(0n).size(), 12n));
});
```

## Implementation Checklist

### TypeScript (`lightning.ts`)

- [ ] Add 4th parameter `OptionType(MatrixType)` to `lightning_predict` for conditions
- [ ] Update JSDoc for `lightning_predict` to document conditions parameter
- [ ] Update `Lightning.predict` documentation

### Python (`lightning_impl.py`)

- [ ] Update `lightning_predict_impl` to accept 4th parameter (conditions)
- [ ] Add validation: if model has condition_dim, conditions must be provided
- [ ] Add validation: condition_dim must match model's expected dimension
- [ ] Add `predict_probs_with_masks_conditional` method to `LightningMLP`
- [ ] Ensure `condition_dim` is saved in model serialization
- [ ] Ensure `condition_dim` is restored in model deserialization

### Tests (`lightning.spec.ts`)

- [ ] Add conv1d predict with conditions test
- [ ] Add error test: predict without conditions on conditional model
- [ ] Add error test: predict with wrong condition_dim
- [ ] Add non-conditional model with none conditions test
- [ ] Add predict with masks AND conditions test
- [ ] Add sequential predict with conditions test
- [ ] Add transformer predict with conditions test

## Backward Compatibility

The change is fully backward compatible:
- The 4th `conditions` parameter is optional (`OptionType`)
- Existing code passing 3 arguments will continue to work
- Non-conditional models (no `condition_dim`) work exactly as before

## Summary

| Function | Before | After |
|----------|--------|-------|
| `predict` | 3 params (model, X, masks) | 4 params (model, X, masks, conditions) |
| `encode` | 2 params (model, X) | No change needed |
| `decode` | 2 params (model, z) | No change needed |
| `decodeConditional` | 3 params (model, z, conditions) | Already exists |
| `train` | 6 params (X, y, config, masks, group_weights, conditions) | Already supports conditions |

### API After Changes

```typescript
// Train with conditions
const result = Lightning.train(X, y, config, masks, group_weights, conditions);

// Encode (no conditions - encoding is unconditional)
const z = Lightning.encode(result.model, X);

// Predict with conditions
const y_pred = Lightning.predict(result.model, X, masks, conditions);

// Decode with conditions
const X_reconstructed = Lightning.decodeConditional(result.model, z, conditions);

// Decode without conditions (for non-conditional models)
const X_reconstructed = Lightning.decode(result.model, z);
```
