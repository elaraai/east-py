# Design: Group-Based Weights for Lightning (multi_head and binary)

## Problem Statement

The `multi_head` and `binary` output modes in Lightning currently accept global weights (`class_weights` for multi_head, `pos_weight` for binary). This works when class distributions are uniform across samples, but fails for sparse, heterogeneous data where different sample groups have different sparsity patterns.

### The Additive Model Use Case

Training an autoencoder on additive application patterns:
- 84 heads (6 additives × 14 days)
- 4 classes per head: [none, low, med, high]
- Data is **very sparse**: most slots are "none" (bin 0)

The sparsity varies by grape grade:
- Grade A: 98% "none" → rare events need very high weight
- Grade B: 85% "none" → moderate weight needed

With global class_weights (averaged across grades):
- Rare events in high-sparsity grades get under-weighted
- Model learns to predict "none" everywhere
- Embeddings fail to capture meaningful additive patterns

## Current API

```typescript
Lightning.train(X, y, {
    output: variant("multi_head", {
        n_heads: 84n,
        n_classes_per_head: 4n,
        class_weights: variant("some", weights),  // [n_heads][n_classes] - GLOBAL
    }),
}, variant("some", masks));
```

## Proposed Change: Group-Based Weights

Instead of per-sample weights (memory-intensive), use group-based weights where samples belong to discrete groups (e.g., grades).

### Memory Comparison

| Approach | Storage | 10K samples, 84 heads, 4 classes |
|----------|---------|----------------------------------|
| Per-sample | `[n_samples, n_heads, n_classes]` | ~13 MB |
| Group-based | `[n_groups, n_heads, n_classes] + [n_samples]` | ~14 KB (10 groups) |

### Type Changes

```typescript
// lightning.ts

/** Group-based weights - variant for different output types */
export const GroupWeightsType = StructType({
    /** Weights per group - shape depends on output type */
    weights: VariantType({
        /** For binary: pos_weight vector per group [n_groups][output_dim] */
        binary: ArrayType(ArrayType(FloatType)),
        /** For multi_head: class_weight matrix per group [n_groups][n_heads][n_classes] */
        multi_head: ArrayType(ArrayType(ArrayType(FloatType))),
    }),
    /** Group index per sample: [n_samples] */
    sample_groups: ArrayType(IntegerType),
});

// Train function signature (add 5th parameter)
export const lightning_train = East.platform(
    "lightning_train",
    [
        MatrixType,                              // X
        MatrixType,                              // y
        LightningConfigType,                     // config
        OptionType(Tensor3DBoolType),            // masks
        OptionType(GroupWeightsType),            // group_weights (NEW)
    ],
    LightningResultType
);
```

### Shape Summary

| Output Mode | weights variant | Shape | Interpretation |
|-------------|-----------------|-------|----------------|
| `binary` | `variant("binary", ...)` | `[n_groups][output_dim]` | pos_weight vector per group |
| `multi_head` | `variant("multi_head", ...)` | `[n_groups][n_heads][n_classes]` | class_weight matrix per group |

### Python Implementation

```python
# lightning_impl.py

def lightning_train(X, y, config, masks=None, group_weights=None):
    """
    Args:
        group_weights: Optional dict with:
            - weights: EastVariant with type "binary" or "multi_head"
                - binary: [n_groups][output_dim] pos_weight per group
                - multi_head: [n_groups][n_heads][n_classes] class_weight per group
            - sample_groups: [n_samples] of int group indices
    """
    output_type = config["output"].type

    # Validate: group_weights only supported for multi_head and binary
    if group_weights is not None:
        weights_variant = group_weights["weights"]
        weights_type = weights_variant.type  # "binary" or "multi_head"
        weights_data = weights_variant.value

        if output_type not in ("multi_head", "binary"):
            raise ValueError("group_weights only supported for multi_head and binary output")

        # Validate: weights variant matches output type
        if weights_type != output_type:
            raise ValueError(
                f"group_weights variant '{weights_type}' does not match output type '{output_type}'"
            )

        # Validate shape based on output type
        if output_type == "binary":
            expected_dim = y.shape[1]
            if len(weights_data[0]) != expected_dim:
                raise ValueError(
                    f"binary group_weights must have shape [n_groups][{expected_dim}], "
                    f"got [n_groups][{len(weights_data[0])}]"
                )
        elif output_type == "multi_head":
            n_heads = config["output"].value["n_heads"]
            n_classes = config["output"].value["n_classes_per_head"]
            if len(weights_data[0]) != n_heads or len(weights_data[0][0]) != n_classes:
                raise ValueError(
                    f"multi_head group_weights must have shape [n_groups][{n_heads}][{n_classes}]"
                )

    # Validate: group indices are in bounds and sample count matches
    if group_weights is not None:
        n_groups = len(weights_data)
        sample_groups = group_weights["sample_groups"]
        n_samples = X.shape[0]

        if len(sample_groups) != n_samples:
            raise ValueError(
                f"sample_groups length {len(sample_groups)} does not match X rows {n_samples}"
            )
        if len(sample_groups) == 0:
            raise ValueError("sample_groups cannot be empty")
        min_group = min(sample_groups)
        max_group = max(sample_groups)
        if min_group < 0:
            raise ValueError(f"sample_groups contains negative index {min_group}")
        if max_group >= n_groups:
            raise ValueError(
                f"sample_groups contains index {max_group} but only {n_groups} groups provided"
            )

    # Warn if both config weights and group_weights provided
    if group_weights is not None:
        if output_type == "multi_head":
            config_weights = config["output"].value.get("class_weights")
            if config_weights is not None and config_weights.type == "some":
                import warnings
                warnings.warn("group_weights provided; ignoring config class_weights")
        elif output_type == "binary":
            config_weights = config["output"].value.get("pos_weight")
            if config_weights is not None and config_weights.type == "some":
                import warnings
                warnings.warn("group_weights provided; ignoring config pos_weight")

    # Extract weights data for model (already validated above)
    group_weights_for_model = None
    if group_weights is not None:
        group_weights_for_model = {
            "weights": weights_data,  # The actual array data
            "weights_type": weights_type,  # "binary" or "multi_head"
        }

    model = LightningMLP(
        input_dim=X.shape[1],
        output_dim=y.shape[1],
        architecture=config["architecture"],
        output_config=config["output"],
        learning_rate=config.get("learning_rate", 1e-3),
        dropout=config.get("dropout", 0.1),
        group_weights=group_weights_for_model,
    )

    # Create dataset with group indices
    # When group_weights is provided, always use 4-element tuple: (x, y, masks, group_idx)
    # This avoids ambiguity in batch unpacking
    if group_weights is not None:
        sample_groups_tensor = torch.tensor(group_weights["sample_groups"], dtype=torch.long)
        n_samples = X.shape[0]

        # Create appropriate dummy masks based on output type
        if masks is not None:
            masks_tensor = torch.tensor(masks, dtype=torch.bool)
        elif output_type == "multi_head":
            n_heads = config["output"].value["n_heads"]
            n_classes = config["output"].value["n_classes_per_head"]
            masks_tensor = torch.ones((n_samples, n_heads, n_classes), dtype=torch.bool)
        elif output_type == "binary":
            output_dim = y.shape[1]
            masks_tensor = torch.ones((n_samples, 1, output_dim), dtype=torch.bool)

        dataset = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            masks_tensor,
            sample_groups_tensor,
        )
    else:
        dataset = TensorDataset(...)

    # ... rest of training


class LightningMLP(pl.LightningModule):
    def __init__(self, ..., group_weights=None):
        super().__init__()
        # ... existing init ...

        # Store whether we're using group weights (affects batch unpacking)
        self.use_group_weights = group_weights is not None
        self.group_weights_type = group_weights["weights_type"] if group_weights else None

        if group_weights is not None:
            self.register_buffer(
                'group_weights_tensor',
                torch.tensor(group_weights["weights"], dtype=torch.float32)
            )
        else:
            self.group_weights_tensor = None

    def training_step(self, batch):
        # Batch structure depends on whether group_weights was provided at init:
        # - With group_weights: (x, y, masks, group_idx) - always 4 elements
        # - Without: (x, y) or (x, y, masks) - 2 or 3 elements
        if self.use_group_weights:
            x, y, masks, group_idx = batch
        elif len(batch) == 3:
            x, y, masks = batch
            group_idx = None
        else:
            x, y = batch
            masks = None
            group_idx = None

        logits = self(x)
        loss = self._compute_loss(logits, y, masks, group_idx)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch):
        # Same unpacking logic as training_step
        if self.use_group_weights:
            x, y, masks, group_idx = batch
        elif len(batch) == 3:
            x, y, masks = batch
            group_idx = None
        else:
            x, y = batch
            masks = None
            group_idx = None

        logits = self(x)
        loss = self._compute_loss(logits, y, masks, group_idx)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def _compute_loss(self, logits, targets, masks=None, group_idx=None):
        """Compute loss based on output type."""
        if self.output_type == "regression":
            return F.mse_loss(logits, targets)
        elif self.output_type == "binary":
            return self._binary_loss(logits, targets, masks, group_idx)
        elif self.output_type == "multiclass":
            return self._multiclass_loss(logits, targets)
        elif self.output_type == "multi_head":
            return self._multi_head_loss(logits, targets, masks, group_idx)
        else:
            raise ValueError(f"Unknown output type: {self.output_type}")

    def _multi_head_loss(self, logits, targets, masks=None, group_idx=None):
        """Vectorized multi-head CE loss with group-based weights."""
        n_heads = self.output_config["n_heads"]
        n_classes = self.output_config["n_classes_per_head"]

        batch_size = logits.shape[0]
        logits = logits.view(batch_size, n_heads, n_classes)
        targets = targets.view(batch_size, n_heads, n_classes)
        target_indices = targets.argmax(dim=-1)  # [batch, n_heads]

        # Apply masks if provided
        if masks is not None:
            logits = logits.masked_fill(~masks, float('-inf'))

        # Compute log softmax (vectorized across all heads)
        log_probs = F.log_softmax(logits, dim=-1)  # [batch, n_heads, n_classes]

        # Gather log probs for target classes
        nll = -log_probs.gather(2, target_indices.unsqueeze(-1)).squeeze(-1)  # [batch, n_heads]

        # Apply weights
        if self.group_weights_tensor is not None and group_idx is not None:
            # Look up weights for each sample's group: [batch, n_heads, n_classes]
            batch_weights = self.group_weights_tensor[group_idx]
            # Gather weights for target classes
            sample_weights = batch_weights.gather(2, target_indices.unsqueeze(-1)).squeeze(-1)  # [batch, n_heads]
            weighted_nll = nll * sample_weights
            return weighted_nll.mean()
        elif self.output_config.get("class_weights") is not None:
            # Global weights fallback
            global_weights = torch.tensor(
                self.output_config["class_weights"],
                device=logits.device,
                dtype=torch.float32
            )  # [n_heads, n_classes]
            sample_weights = global_weights.gather(1, target_indices)  # [batch, n_heads]
            weighted_nll = nll * sample_weights
            return weighted_nll.mean()
        else:
            return nll.mean()

    def _binary_loss(self, logits, targets, masks=None, group_idx=None):
        """Binary cross-entropy loss with optional group-based pos_weights."""

        if self.group_weights_tensor is not None and group_idx is not None:
            # Per-sample pos_weights from group lookup: [batch, output_dim]
            batch_pos_weights = self.group_weights_tensor[group_idx]

            # Compute BCE with per-sample weights
            loss = F.binary_cross_entropy_with_logits(
                logits, targets, reduction='none'
            )  # [batch, output_dim]

            # Weight positive samples: loss * (target * pos_weight + (1 - target))
            # This matches PyTorch's pos_weight behavior
            weighted_loss = loss * (targets * batch_pos_weights + (1 - targets))

            # Apply masks if provided
            if masks is not None:
                if masks.dim() == 3:
                    masks = masks.squeeze(1)  # [batch, 1, output_dim] -> [batch, output_dim]
                weighted_loss = weighted_loss * masks.float()
                n_valid = masks.sum()
                if n_valid > 0:
                    return weighted_loss.sum() / n_valid
                return weighted_loss.sum()

            return weighted_loss.mean()

        else:
            # Existing global pos_weight path (unchanged)
            pos_weight = self.output_config.get("pos_weight")
            if pos_weight is not None:
                pos_weight = torch.tensor(pos_weight, dtype=torch.float32, device=logits.device)

            loss = F.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=pos_weight, reduction="none"
            )

            if masks is not None:
                if masks.dim() == 3:
                    masks = masks.squeeze(1)
                loss = loss * masks.float()
                n_valid = masks.sum()
                if n_valid > 0:
                    return loss.sum() / n_valid
                return loss.sum()

            return loss.mean()
```

### Validation Weights

The same `sample_groups` applies to validation data. During train/val split:

```python
def _create_dataloaders(self, dataset, val_split=0.1):
    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val

    train_dataset, val_dataset = random_split(dataset, [n_train, n_val])

    # Both splits retain their group indices from the original dataset
    # The group_idx tensor travels with each sample through the split
```

## Usage Examples

### multi_head Example

```typescript
// Compute per-grade class weights for multi_head
const unique_grades = $.let(grades.toArray());
const weights_per_group = $.let(unique_grades.map(($, g) =>
    likelihoods.grade_weights.get(g)  // [n_heads][n_classes] per grade
));
const sample_groups = $.let(train.map(($, sample) =>
    unique_grades.findIndex(($, g) => East.equal(g, sample.grade)).unwrap("some")
));

const result = $.let(Lightning.train(
    X_train,
    X_train,
    config,
    variant("some", train_masks),
    variant("some", {
        weights: variant("multi_head", weights_per_group),
        sample_groups,
    }),
));
```

### binary Example

```typescript
// Compute per-group pos_weights for binary
const weights_per_group = [
    Array(output_dim).fill(9.0),   // group 0: high pos_weight (very sparse)
    Array(output_dim).fill(3.0),   // group 1: medium pos_weight
    Array(output_dim).fill(1.0),   // group 2: low pos_weight (balanced)
];
const sample_groups = samples.map((_, i) => i % 3);  // assign to groups

const result = $.let(Lightning.train(
    X_train,
    y_train,
    config,
    variant("none", null),
    variant("some", {
        weights: variant("binary", weights_per_group),
        sample_groups,
    }),
));
```

## Tests

Tests to add to `lightning.spec.ts`:

### Test 1: multi_head with group weights

```typescript
test("multi_head with group weights", $ => {
    // 2 groups with different class distributions
    // Group 0: mostly class 0, Group 1: mostly class 1
    const X = $.let([
        [1.0, 0.0], [1.1, 0.1], [0.9, 0.1],  // group 0
        [0.0, 1.0], [0.1, 1.1], [0.1, 0.9],  // group 1
    ]);
    // 2 heads x 3 classes = 6 outputs
    const y = $.let([
        [1.0, 0.0, 0.0,  1.0, 0.0, 0.0],  // group 0: class 0
        [1.0, 0.0, 0.0,  1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0,  1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0,  0.0, 1.0, 0.0],  // group 1: class 1
        [0.0, 1.0, 0.0,  0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0,  0.0, 1.0, 0.0],
    ]);

    // Group weights: [n_groups][n_heads][n_classes]
    const group_weights = $.let({
        weights: variant('multi_head', [
            [[1.0, 2.0, 2.0], [1.0, 2.0, 2.0]],  // group 0: upweight rare classes
            [[2.0, 1.0, 2.0], [2.0, 1.0, 2.0]],  // group 1: upweight rare classes
        ]),
        sample_groups: [0n, 0n, 0n, 1n, 1n, 1n],
    });

    const config = $.let({
        architecture: variant('mlp', { hidden_layers: [16n] }),
        output: variant('multi_head', {
            n_heads: 2n,
            n_classes_per_head: 3n,
            class_weights: variant('none', null),
        }),
        learning_rate: variant('some', 0.01),
        max_epochs: variant('some', 100n),
        patience: variant('some', 20n),
        batch_size: variant('some', 3n),
        dropout: variant('some', 0.0),
        gradient_clip: variant('some', 1.0),
        weight_decay: variant('none', null),
        random_state: variant('some', 42n),
        epoch_callback: variant('none', null),
    });

    const result = $.let(Lightning.train(X, y, config, variant('none', null), variant('some', group_weights)));

    $(Assert.greaterEqual(result.best_epoch, 0n));

    const y_pred = $.let(Lightning.predict(result.model, X, variant('none', null)));
    $(Assert.equal(y_pred.size(), 6n));

    // Each head's probs should sum to ~1
    const h0_sum = $.let(y_pred.get(0n).get(0n).add(y_pred.get(0n).get(1n)).add(y_pred.get(0n).get(2n)));
    $(Assert.greater(h0_sum, East.value(0.99)));
    $(Assert.less(h0_sum, East.value(1.01)));
});
```

### Test 2: binary with group weights

```typescript
test("binary with group weights", $ => {
    // 2 groups with different sparsity
    const X = $.let([
        [1.0, 0.0], [1.1, 0.1], [0.9, 0.1],  // group 0
        [0.0, 1.0], [0.1, 1.1], [0.1, 0.9],  // group 1
    ]);
    // 4 binary outputs
    const y = $.let([
        [1.0, 0.0, 0.0, 0.0],  // group 0: sparse
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 0.0],  // group 1: denser
        [1.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 0.0],
    ]);

    // Group weights (pos_weight per group): [n_groups][output_dim]
    const group_weights = $.let({
        weights: variant('binary', [
            [5.0, 5.0, 5.0, 5.0],  // group 0: high pos_weight (sparse)
            [1.0, 1.0, 1.0, 1.0],  // group 1: low pos_weight (denser)
        ]),
        sample_groups: [0n, 0n, 0n, 1n, 1n, 1n],
    });

    const config = $.let({
        architecture: variant('mlp', { hidden_layers: [16n] }),
        output: variant('binary', { pos_weight: variant('none', null) }),
        learning_rate: variant('some', 0.01),
        max_epochs: variant('some', 100n),
        patience: variant('some', 20n),
        batch_size: variant('some', 3n),
        dropout: variant('some', 0.0),
        gradient_clip: variant('some', 1.0),
        weight_decay: variant('none', null),
        random_state: variant('some', 42n),
        epoch_callback: variant('none', null),
    });

    const result = $.let(Lightning.train(X, y, config, variant('none', null), variant('some', group_weights)));

    $(Assert.greaterEqual(result.best_epoch, 0n));

    const y_pred = $.let(Lightning.predict(result.model, X, variant('none', null)));
    $(Assert.equal(y_pred.size(), 6n));
    $(Assert.equal(y_pred.get(0n).size(), 4n));

    // Predictions should be between 0 and 1
    $(Assert.greaterEqual(y_pred.get(0n).get(0n), East.value(0.0)));
    $(Assert.lessEqual(y_pred.get(0n).get(0n), East.value(1.0)));
});
```

### Test 3: error - group_weights with regression output

```typescript
test("error: group_weights with regression output", $ => {
    const X = $.let([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]);
    const y = $.let([[1.0], [2.0], [3.0], [4.0]]);

    const group_weights = $.let({
        weights: variant('multi_head', [[[1.0]]]),
        sample_groups: [0n, 0n, 0n, 0n],
    });

    const config = $.let({
        architecture: variant('mlp', { hidden_layers: [8n] }),
        output: variant('regression', null),
        learning_rate: variant('none', null),
        max_epochs: variant('none', null),
        patience: variant('none', null),
        batch_size: variant('none', null),
        dropout: variant('none', null),
        gradient_clip: variant('none', null),
        weight_decay: variant('none', null),
        random_state: variant('none', null),
        epoch_callback: variant('none', null),
    });

    $(Assert.throws(
        Lightning.train(X, y, config, variant('none', null), variant('some', group_weights)),
        /group_weights only supported for multi_head and binary output/
    ));
});
```

### Test 4: error - weights variant mismatch

```typescript
test("error: weights variant does not match output type", $ => {
    const X = $.let([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]);
    const y = $.let([[1.0], [0.0], [1.0], [0.0]]);

    // Using multi_head variant with binary output
    const group_weights = $.let({
        weights: variant('multi_head', [[[1.0, 1.0]]]),
        sample_groups: [0n, 0n, 0n, 0n],
    });

    const config = $.let({
        architecture: variant('mlp', { hidden_layers: [8n] }),
        output: variant('binary', { pos_weight: variant('none', null) }),
        learning_rate: variant('none', null),
        max_epochs: variant('none', null),
        patience: variant('none', null),
        batch_size: variant('none', null),
        dropout: variant('none', null),
        gradient_clip: variant('none', null),
        weight_decay: variant('none', null),
        random_state: variant('none', null),
        epoch_callback: variant('none', null),
    });

    $(Assert.throws(
        Lightning.train(X, y, config, variant('none', null), variant('some', group_weights)),
        /group_weights variant 'multi_head' does not match output type 'binary'/
    ));
});
```

### Test 5: error - sample_groups out of bounds

```typescript
test("error: sample_groups index out of bounds", $ => {
    const X = $.let([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]);
    const y = $.let([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ]);

    // Only 1 group but sample_groups references index 1
    const group_weights = $.let({
        weights: variant('multi_head', [[[1.0, 1.0, 1.0]]]),  // 1 group
        sample_groups: [0n, 0n, 1n, 1n],  // ERROR: index 1 out of bounds
    });

    const config = $.let({
        architecture: variant('mlp', { hidden_layers: [8n] }),
        output: variant('multi_head', {
            n_heads: 1n,
            n_classes_per_head: 3n,
            class_weights: variant('none', null),
        }),
        learning_rate: variant('none', null),
        max_epochs: variant('none', null),
        patience: variant('none', null),
        batch_size: variant('none', null),
        dropout: variant('none', null),
        gradient_clip: variant('none', null),
        weight_decay: variant('none', null),
        random_state: variant('none', null),
        epoch_callback: variant('none', null),
    });

    $(Assert.throws(
        Lightning.train(X, y, config, variant('none', null), variant('some', group_weights)),
        /sample_groups contains index 1 but only 1 groups provided/
    ));
});
```

### Test 6: error - sample_groups length mismatch

```typescript
test("error: sample_groups length mismatch", $ => {
    const X = $.let([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]);  // 4 samples
    const y = $.let([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ]);

    const group_weights = $.let({
        weights: variant('multi_head', [[[1.0, 1.0, 1.0]]]),
        sample_groups: [0n, 0n],  // ERROR: only 2 indices for 4 samples
    });

    const config = $.let({
        architecture: variant('mlp', { hidden_layers: [8n] }),
        output: variant('multi_head', {
            n_heads: 1n,
            n_classes_per_head: 3n,
            class_weights: variant('none', null),
        }),
        learning_rate: variant('none', null),
        max_epochs: variant('none', null),
        patience: variant('none', null),
        batch_size: variant('none', null),
        dropout: variant('none', null),
        gradient_clip: variant('none', null),
        weight_decay: variant('none', null),
        random_state: variant('none', null),
        epoch_callback: variant('none', null),
    });

    $(Assert.throws(
        Lightning.train(X, y, config, variant('none', null), variant('some', group_weights)),
        /sample_groups length 2 does not match X rows 4/
    ));
});
```

## Migration

### additives_.ts

Before:
```typescript
// Complex per-grade lookup, then average
const avg_class_weights = computeAverage(train_class_weights);

const config = {
    output: variant("multi_head", {
        class_weights: variant("some", avg_class_weights),
    }),
};

Lightning.train(X, X, config, masks);
```

After:
```typescript
// Direct group-based weights
const unique_grades = $.let([...grades]);
const group_weights = $.let(unique_grades.map(($, g) =>
    likelihoods.grade_weights.get(g)
));
const sample_groups = $.let(train.map(($, s) =>
    unique_grades.indexOf(s.grade)
));

const config = {
    output: variant("multi_head", {
        n_heads: East.value(N_SLOTS),
        n_classes_per_head: n_bins,
        class_weights: variant("none", null),  // not needed
    }),
};

Lightning.train(X, X, config, masks, variant("some", {
    weights: variant("multi_head", group_weights),
    sample_groups,
}));
```

### Simplify AdditiveLikelihoodResultType

```typescript
// Before (legacy cruft)
export const AdditiveGradeConstraintConfigType = StructType({
    grade: IntegerType,                           // REDUNDANT (it's the key)
    class_weights: ArrayType(ArrayType(FloatType)),
    prior: ArrayType(FloatType),                  // UNUSED
});

export const AdditiveConstraintConfigType = StructType({
    n_bins: IntegerType,
    bin_boundaries: ArrayType(ArrayType(FloatType)),  // DUPLICATE
    bin_amounts: ArrayType(ArrayType(FloatType)),     // DUPLICATE
    grade_configs: DictType(IntegerType, AdditiveGradeConstraintConfigType),
    grades: SetType(IntegerType),
});

export const AdditiveLikelihoodResultType = StructType({
    likelihoods: AdditiveLikelihoodsType,
    bin_boundaries: ArrayType(ArrayType(FloatType)),
    bin_amounts: ArrayType(ArrayType(FloatType)),
    constraints: AdditiveConstraintConfigType,
});

// After (clean)
export const AdditiveLikelihoodResultType = StructType({
    likelihoods: AdditiveLikelihoodsType,
    bin_boundaries: ArrayType(ArrayType(FloatType)),  // [n_additives][n_bins-1]
    bin_amounts: ArrayType(ArrayType(FloatType)),     // [n_additives][n_bins]
    grade_weights: DictType(IntegerType, ArrayType(ArrayType(FloatType))),  // grade -> [n_heads][n_classes]
    n_bins: IntegerType,
});
```

## Implementation Notes

### 1. Vectorize existing global class_weights path

The current `_multi_head_loss` uses a loop over heads. As part of this change, vectorize
both the group weights AND the existing global class_weights path for consistency:

```python
def _multi_head_loss(self, logits, targets, masks=None, group_idx=None):
    """Vectorized multi-head CE loss."""
    # ... reshape to [batch, n_heads, n_classes] ...

    # Compute log softmax (vectorized)
    log_probs = F.log_softmax(logits, dim=-1)
    nll = -log_probs.gather(2, target_indices.unsqueeze(-1)).squeeze(-1)

    # Apply weights (vectorized for both cases)
    if self.group_weights_tensor is not None and group_idx is not None:
        # Group-based weights
        batch_weights = self.group_weights_tensor[group_idx]
        sample_weights = batch_weights.gather(2, target_indices.unsqueeze(-1)).squeeze(-1)
    elif self.class_weights is not None:
        # Global weights (also vectorized now)
        sample_weights = self.class_weights.gather(1, target_indices)
    else:
        sample_weights = None

    if sample_weights is not None:
        return (nll * sample_weights).mean()
    return nll.mean()
```

### 2. Hyperparameter serialization

The `use_group_weights` flag must be saved in hyperparameters for checkpoint loading:

```python
self.save_hyperparameters()  # includes use_group_weights
```

### 3. Group weights are training-only

Group weights only affect loss computation during training. They do NOT affect:
- `predict()` - returns probabilities based on model weights
- `encode()` - returns latent embeddings
- `decode()` - returns reconstructed output

This is correct behavior - weights are for training the model, not for inference.

## Summary

| Aspect | Before | After |
|--------|--------|-------|
| Supported output types | N/A | `multi_head` and `binary` |
| Weight storage (multi_head) | Global `[n_heads, n_classes]` | Group `[n_groups, n_heads, n_classes]` + indices |
| Weight storage (binary) | Global `[output_dim]` | Group `[n_groups, output_dim]` + indices |
| Memory (10K samples, 10 grades) | N/A | ~14 KB vs ~13 MB per-sample |
| Sparse data handling | Averaged, loses group info | Group-appropriate weights per sample |
| Loss computation | Loop over heads | Vectorized gather operations |
| API change | None | New optional 5th parameter with variant type |
| Backward compat | N/A | Fully compatible |
| Validation | N/A | Same group indices apply |
| Error handling | N/A | Validates group indices, output type, and variant match |
