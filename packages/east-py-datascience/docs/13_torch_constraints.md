# Design Document: east-py-datascience Enhancements

## 1. Overview

General enhancements to `east-py-datascience` for improved constraint handling and class imbalance in binary classification autoencoders.

**Note:** This design is intentionally general. It provides mechanisms (per-sample masks, per-output weights) without any domain-specific concepts. Callers build and pass these structures.

### Goals
- Per-output `pos_weight` arrays for class imbalance
- Per-sample masks for dynamic constraint application
- Static `data_mask` in constraint definitions
- Prior regularization loss
- Fix double-sigmoid issue with `bce_with_logits` + constraints

---

## 2. Type Changes (`types.py`)

### 2.1 PosWeightType

```python
PosWeightType = VariantType([
    ("scalar", FloatType),
    ("per_output", ArrayType(FloatType)),  # length = output_dim
])
```

### 2.2 PriorConfigType

```python
PriorConfigType = StructType([
    ("values", ArrayType(FloatType)),  # Prior probabilities per output
    ("weight", FloatType),              # Lambda weight for MSE regularization
])
```

### 2.3 RowConstraintType (add data_mask)

Static mask derived from data, combined with user `mask`.

```python
RowConstraintType = VariantType([
    ("binary", StructType([
        ("mask", OptionType(ArrayType(BooleanType))),       # User-specified mask
        ("data_mask", OptionType(ArrayType(BooleanType))),  # Data-derived static mask
    ])),
    ("mutex", StructType([
        ("mask", OptionType(ArrayType(BooleanType))),
        ("allow_none", OptionType(BooleanType)),
        ("data_mask", OptionType(ArrayType(BooleanType))),
    ])),
    ("at_most", StructType([
        ("max_count", IntegerType),
        ("mask", OptionType(ArrayType(BooleanType))),
        ("data_mask", OptionType(ArrayType(BooleanType))),
    ])),
])
```

### 2.4 Updated TorchTrainConfigType

```python
TorchTrainConfigType = StructType([
    # ... existing fields ...
    ("pos_weight", OptionType(PosWeightType)),
    ("prior", OptionType(PriorConfigType)),
    ("sample_masks", OptionType(ArrayType(ArrayType(ArrayType(BooleanType))))),  # (n_samples, n_rows, n_cols)
    ("sample_pos_weights", OptionType(ArrayType(ArrayType(FloatType)))),         # (n_samples, output_dim)
    ("sample_priors", OptionType(ArrayType(ArrayType(FloatType)))),              # (n_samples, output_dim)
])
```

---

## 3. Implementation Changes (`torch/torch_impl.py`)

### 3.1 ConstrainedOutputLayer: Add `return_logits` and `sample_masks`

```python
class ConstrainedOutputLayer:
    def __init__(self, row_constraints: list, n_rows: int, n_cols: int, return_logits: bool = False):
        """
        Args:
            row_constraints: List of (constraint_type, static_mask, params) tuples
            n_rows: Number of constraint rows
            n_cols: Number of columns per row
            return_logits: If True, return raw logits for binary constraints (for bce_with_logits)
        """
        self.return_logits = return_logits
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.row_constraints = row_constraints

    def __call__(self, x, sample_masks=None):
        """
        Args:
            x: (batch, n_rows * n_cols) - raw logits from decoder
            sample_masks: (batch, n_rows, n_cols) - per-sample boolean masks, or None
                          True = allowed, False = masked (output forced to 0/-inf)

        Returns:
            (batch, n_rows * n_cols) - constrained outputs
        """
        batch_size = x.shape[0]
        x = x.view(batch_size, self.n_rows, self.n_cols)

        outputs = []
        for row_idx, (ctype, static_mask, params) in enumerate(self.row_constraints):
            row_logits = x[:, row_idx, :]  # (batch, n_cols)

            # Apply static mask (from data_mask in constraint config)
            if static_mask is not None:
                row_logits = row_logits.masked_fill(~static_mask, float("-inf"))

            # Apply per-sample dynamic mask
            if sample_masks is not None:
                sample_row_mask = sample_masks[:, row_idx, :]  # (batch, n_cols)
                row_logits = row_logits.masked_fill(~sample_row_mask, float("-inf"))

            if ctype == "binary":
                if self.return_logits:
                    row_out = row_logits
                else:
                    row_out = torch.sigmoid(row_logits)

            elif ctype == "mutex":
                row_out = torch.softmax(row_logits, dim=-1)

            outputs.append(row_out)

        return torch.cat(outputs, dim=-1)
```

### 3.2 Training Setup

```python
# Determine if we need logits mode
use_logits_output = (loss_name == "bce_with_logits")

if output_constraints is not None:
    constrained_layer = ConstrainedOutputLayer(
        parsed_constraints, n_rows, n_cols,
        return_logits=use_logits_output
    )

# Parse sample_masks
sample_masks_config = _get_option(train_config.get("sample_masks"), None)
sample_masks_tensor = None
if sample_masks_config is not None:
    sample_masks_tensor = torch.tensor(sample_masks_config, dtype=torch.bool)
    # Shape: (n_samples, n_rows, n_cols)

# Parse sample_pos_weights
sample_pos_weights_config = _get_option(train_config.get("sample_pos_weights"), None)
sample_pos_weights_tensor = None
if sample_pos_weights_config is not None:
    sample_pos_weights_tensor = torch.tensor(sample_pos_weights_config, dtype=torch.float32)
    # Shape: (n_samples, output_dim)

# Parse sample_priors
sample_priors_config = _get_option(train_config.get("sample_priors"), None)
sample_priors_tensor = None
if sample_priors_config is not None:
    sample_priors_tensor = torch.tensor(sample_priors_config, dtype=torch.float32)
    # Shape: (n_samples, output_dim)
```

### 3.3 Per-Output pos_weight

```python
pos_weight_config = _get_option(train_config.get("pos_weight"), None)
pos_weight_tensor = None

if pos_weight_config is not None:
    if pos_weight_config.type == "scalar":
        pw = float(pos_weight_config.value)
        pos_weight_tensor = torch.full((output_dim,), pw, dtype=torch.float32)
    elif pos_weight_config.type == "per_output":
        pos_weight_tensor = torch.tensor(
            [float(w) for w in pos_weight_config.value],
            dtype=torch.float32
        )

# Loss creation
if loss_name == "bce_with_logits":
    # Note: pos_weight in BCEWithLogitsLoss is global, not per-sample
    # For per-sample weighting, use sample_pos_weights and reduction='none'
    criterion = nn.BCEWithLogitsLoss(reduction='none')
```

### 3.4 Training Loop with Per-Sample Masks and Weights

```python
def forward_batch(x_batch, batch_indices):
    decoder_output = model(x_batch)

    batch_masks = None
    if sample_masks_tensor is not None:
        batch_masks = sample_masks_tensor[batch_indices]

    outputs = constrained_layer(decoder_output, sample_masks=batch_masks)
    return outputs

def compute_loss(outputs, y_batch, batch_indices):
    # Base loss per element: (batch, output_dim)
    base_loss = criterion(outputs, y_batch)

    # Apply per-sample pos_weight if provided
    if sample_pos_weights_tensor is not None:
        batch_weights = sample_pos_weights_tensor[batch_indices]  # (batch, output_dim)
        # Weight positive samples only
        weighted_loss = base_loss * torch.where(
            y_batch == 1, batch_weights, torch.ones_like(batch_weights)
        )
        loss = weighted_loss.mean()
    elif pos_weight_tensor is not None:
        # Use global pos_weight
        weighted_loss = base_loss * torch.where(
            y_batch == 1, pos_weight_tensor, torch.ones_like(pos_weight_tensor)
        )
        loss = weighted_loss.mean()
    else:
        loss = base_loss.mean()

    # Add prior regularization if provided
    if sample_priors_tensor is not None:
        batch_priors = sample_priors_tensor[batch_indices]
        prior_loss = F.mse_loss(outputs, batch_priors)
        loss = loss + prior_weight * prior_loss
    elif prior_config is not None:
        prior_loss = F.mse_loss(outputs, prior_values.expand_as(outputs))
        loss = loss + prior_weight * prior_loss

    return loss
```

### 3.5 data_mask Handling in Constraint Parsing

```python
def _parse_row_constraints(constraints_config, n_rows: int, n_cols: int):
    parsed = []

    for i, constraint in enumerate(constraints_config):
        ctype = constraint.type
        cvalue = constraint.value

        mask = _parse_mask(cvalue.get("mask"))
        data_mask = _parse_mask(cvalue.get("data_mask"))

        # Combine: final_mask = mask AND data_mask
        if mask is not None and data_mask is not None:
            combined_mask = mask & data_mask
        elif data_mask is not None:
            combined_mask = data_mask
        else:
            combined_mask = mask

        parsed.append((ctype, combined_mask, cvalue))

    return parsed
```

### 3.6 Prediction with Optional Sample Masks

```python
def torch_mlp_predict_multi_impl(model_blob, X, sample_masks=None):
    """
    Args:
        model_blob: Trained model
        X: Input features (n_samples, input_dim)
        sample_masks: Optional (n_samples, n_rows, n_cols) boolean masks

    Returns:
        Predictions (n_samples, output_dim) with values in [0, 1]
    """
    # ... existing code ...

    if constrained_layer is not None:
        sample_masks_tensor = None
        if sample_masks is not None:
            sample_masks_tensor = torch.tensor(sample_masks, dtype=torch.bool)
        outputs = constrained_layer(decoder_output, sample_masks=sample_masks_tensor)

    # Apply sigmoid if model was trained with return_logits=True
    if model_metadata.get("return_logits_mode"):
        outputs = torch.sigmoid(outputs)

    return outputs
```

---

## 4. Utility Functions

### 4.1 compute_pos_weight_from_data

```python
def compute_pos_weight_from_data_impl(
    y: EastArray,
    smoothing: float = 1.0,
    cap: float = 20.0,
) -> EastArray:
    """Compute per-output pos_weight from binary target data.

    pos_weight[i] = min(cap, (n_negative[i] + smoothing) / (n_positive[i] + smoothing))
    """
    y_np = east_matrix_to_numpy(y)
    pos_counts = y_np.sum(axis=0)
    neg_counts = y_np.shape[0] - pos_counts
    pos_weights = (neg_counts + smoothing) / (pos_counts + smoothing)
    pos_weights = np.minimum(pos_weights, cap)
    return numpy_to_east_vector(pos_weights)
```

### 4.2 compute_data_mask

```python
def compute_data_mask_impl(
    likelihoods: EastArray,
    threshold: float = 0.0,
) -> EastArray:
    """Compute static mask from likelihood matrix.

    Positions where P <= threshold are masked (False).
    """
    lik_np = east_matrix_to_numpy(likelihoods)
    mask = lik_np > threshold
    return numpy_to_east_matrix(mask.astype(bool))
```

---

## 5. TypeScript Interface Updates

```typescript
// Torch.Types additions

export const PosWeightType = VariantType([
    variant("scalar", FloatType),
    variant("per_output", ArrayType(FloatType)),
]);

export const PriorConfigType = StructType({
    values: ArrayType(FloatType),
    weight: FloatType,
});

// Updated TorchTrainConfigType
export const TorchTrainConfigType = StructType({
    // ... existing fields ...
    pos_weight: OptionType(PosWeightType),
    prior: OptionType(PriorConfigType),
    sample_masks: OptionType(ArrayType(ArrayType(ArrayType(BooleanType)))),      // (n_samples, n_rows, n_cols)
    sample_pos_weights: OptionType(ArrayType(ArrayType(FloatType))),              // (n_samples, output_dim)
    sample_priors: OptionType(ArrayType(ArrayType(FloatType))),                   // (n_samples, output_dim)
});

// Updated constraint types
export const BinaryConstraintType = StructType({
    mask: OptionType(ArrayType(BooleanType)),
    data_mask: OptionType(ArrayType(BooleanType)),
});

export const MutexConstraintType = StructType({
    mask: OptionType(ArrayType(BooleanType)),
    allow_none: OptionType(BooleanType),
    data_mask: OptionType(ArrayType(BooleanType)),
});

// Updated predict function
declare function mlpPredictMulti(
    model: ModelBlobType,
    X: number[][],
    sample_masks?: boolean[][][],  // Optional (n_samples, n_rows, n_cols)
): number[][];
```

---

## 6. Required Tests

### 6.1 Per-Output pos_weight

| Test | Purpose | Expected |
|------|---------|----------|
| `test_per_output_pos_weight` | Per-output weights improve recall on rare outputs | Recall on rare outputs >= 10% higher than scalar |
| `test_no_double_sigmoid` | bce_with_logits + constraints works correctly | Predictions in [0,1], no NaN/Inf |
| `test_gradient_flow_logits_mode` | Gradients flow through return_logits=True | All gradients exist, no NaN/Inf |

### 6.2 Per-Sample Masks

| Test | Purpose | Expected |
|------|---------|----------|
| `test_sample_masks_applied` | Per-sample masks zero correct positions | Masked positions output exactly 0 |
| `test_sample_masks_with_data_mask` | Static + dynamic masks both apply | Both masks enforced |

### 6.3 Per-Sample Weights

| Test | Purpose | Expected |
|------|---------|----------|
| `test_sample_pos_weights` | Per-sample pos_weight applied | Different samples weighted differently |
| `test_sample_priors` | Per-sample prior regularization | Outputs biased toward sample-specific priors |

### 6.4 Integration

| Test | Purpose | Expected |
|------|---------|----------|
| `test_full_pipeline` | All features together | Model trains, masked positions=0, accuracy>80% |

---

## 7. Summary of Changes

| Component | Change |
|-----------|--------|
| `types.py` | Add `PosWeightType` (scalar/per_output) |
| `types.py` | Add `PriorConfigType` |
| `types.py` | Add `sample_masks`, `sample_pos_weights`, `sample_priors` to train config |
| `types.py` | Add `data_mask` to constraint types |
| `torch_impl.py` | Add `return_logits` to `ConstrainedOutputLayer.__init__` |
| `torch_impl.py` | Add `sample_masks` to `ConstrainedOutputLayer.__call__` |
| `torch_impl.py` | Combine `mask` and `data_mask` in parsing |
| `torch_impl.py` | Handle per-sample masks/weights/priors in training loop |
| `torch_impl.py` | Add `sample_masks` param to predict function |
| New function | `compute_pos_weight_from_data_impl` |
| New function | `compute_data_mask_impl` |

---

## 8. Implementation Order

1. `ConstrainedOutputLayer` changes (return_logits, sample_masks)
2. Per-output pos_weight
3. data_mask in constraint parsing
4. sample_masks in training loop
5. sample_pos_weights, sample_priors in training loop
6. Predict function with sample_masks
7. Utility functions
8. Tests
