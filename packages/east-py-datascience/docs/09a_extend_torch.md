# Module 9a: Extending PyTorch for Multi-Output and Autoencoders

## Motivation

The current `torch_mlp_train` function accepts `y: VectorType` (one target value per sample), limiting it to single-output regression/classification. To support:

1. **Multi-output regression** - Predicting multiple values per sample
2. **Autoencoders** - Where input equals target (X = y) for reconstruction learning

We need to extend the Torch module to accept `y: MatrixType`.

## Changes Required

### 1. Type Changes

**Current signatures:**
```python
# Training
inputs=[MatrixType, VectorType, TorchMLPConfigType, TorchTrainConfigType]
output=StructType([("model", ModelBlobType), ("result", TorchTrainResultType)])

# Prediction
inputs=[ModelBlobType, MatrixType]
output=VectorType
```

**New signatures:**
```python
# Training - y becomes MatrixType
inputs=[MatrixType, MatrixType, TorchMLPConfigType, TorchTrainConfigType]
output=StructType([("model", ModelBlobType), ("result", TorchTrainResultType)])

# Prediction - output becomes MatrixType
inputs=[ModelBlobType, MatrixType]
output=MatrixType
```

### 2. Implementation Changes

#### `torch_mlp_train_impl`

```python
def torch_mlp_train_impl(
    X: Matrix,
    y: Matrix,  # Changed from Vector
    mlp_config: EastStruct,
    train_config: EastStruct
) -> EastStruct:
    """Create and train PyTorch MLP model with multi-output support."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    X_np = east_matrix_to_numpy(X)
    y_np = east_matrix_to_numpy(y)  # Changed from east_vector_to_numpy

    n_features = X_np.shape[1]
    n_outputs = y_np.shape[1]  # Infer output dimension from y

    # MLP config
    hidden_layers_arr = mlp_config.get("hidden_layers")
    hidden_layers = [int(h) for h in hidden_layers_arr] if hidden_layers_arr else [64, 32]

    activation_variant = _get_option(mlp_config.get("activation"), None)
    activation_name = _get_enum_tag(activation_variant) if activation_variant else "relu"

    dropout = _get_option(mlp_config.get("dropout"), 0.0)

    # output_dim from config overrides inferred, but default to inferred
    output_dim = _get_option(mlp_config.get("output_dim"), n_outputs)

    # Build model
    activation_map = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "leaky_relu": nn.LeakyReLU,
    }
    activation_cls = activation_map.get(activation_name, nn.ReLU)

    layers = []
    prev_dim = n_features
    for hidden_dim in hidden_layers:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(activation_cls())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, output_dim))

    model = nn.Sequential(*layers)

    # Training config
    epochs = _get_option(train_config.get("epochs"), 100)
    batch_size = _get_option(train_config.get("batch_size"), 32)
    lr = _get_option(train_config.get("learning_rate"), 0.001)

    loss_variant = _get_option(train_config.get("loss"), None)
    loss_name = _get_enum_tag(loss_variant) if loss_variant else "mse"

    optimizer_variant = _get_option(train_config.get("optimizer"), None)
    optimizer_name = _get_enum_tag(optimizer_variant) if optimizer_variant else "adam"

    patience = _get_option(train_config.get("early_stopping"), 0)
    val_split = _get_option(train_config.get("validation_split"), 0.2)
    random_state = _get_option(train_config.get("random_state"), None)

    if random_state is not None:
        torch.manual_seed(random_state)

    # Convert to tensors - no unsqueeze needed, y is already 2D
    X_tensor = torch.FloatTensor(X_np)
    y_tensor = torch.FloatTensor(y_np)

    # Train/val split
    n = len(X_tensor)
    n_val = int(n * val_split)
    indices = torch.randperm(n)

    X_train = X_tensor[indices[n_val:]]
    y_train = y_tensor[indices[n_val:]]
    X_val = X_tensor[indices[:n_val]]
    y_val = y_tensor[indices[:n_val]]

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True
    )

    # Loss and optimizer
    loss_map = {
        "mse": nn.MSELoss,
        "mae": nn.L1Loss,
        "cross_entropy": nn.CrossEntropyLoss,
    }
    criterion = loss_map.get(loss_name, nn.MSELoss)()

    optimizer_map = {
        "adam": torch.optim.Adam,
        "sgd": torch.optim.SGD,
        "adamw": torch.optim.AdamW,
        "rmsprop": torch.optim.RMSprop,
    }
    optimizer = optimizer_map.get(optimizer_name, torch.optim.Adam)(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_epoch = 0
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        # Train
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        train_losses.append(epoch_loss / len(train_loader))

        # Validate
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val).item()
        val_losses.append(val_loss)

        # Early stopping with best model restoration
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        elif patience > 0:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Convert to ONNX
    onnx_data = _pytorch_to_onnx(model, n_features)

    model_blob = EastVariant("torch_mlp", EastStruct({
        "onnx": onnx_data,
        "n_features": n_features,
        "hidden_layers": EastArray(IntegerType, hidden_layers),
        "output_dim": output_dim,
    }))

    train_result = EastStruct({
        "train_losses": EastArray(FloatType, train_losses),
        "val_losses": EastArray(FloatType, val_losses),
        "best_epoch": best_epoch,
    })

    return EastStruct({
        "model": model_blob,
        "result": train_result,
    })
```

#### `torch_mlp_predict_impl`

```python
def torch_mlp_predict_impl(
    model_blob: ModelBlob,
    X: Matrix
) -> Matrix:  # Changed from Vector
    """Make predictions with PyTorch MLP (multi-output)."""
    if model_blob.type != "torch_mlp":
        raise ValueError(f"Expected torch_mlp, got {model_blob.type}")

    onnx_blob = model_blob.value["onnx"]
    return _onnx_predict_matrix(onnx_blob, X)  # Returns matrix instead of vector
```

### 3. Helper Function for ONNX Matrix Prediction

```python
def _onnx_predict_matrix(onnx_blob: bytes, X: Matrix) -> Matrix:
    """Run ONNX inference returning matrix output."""
    import onnxruntime as ort
    import numpy as np

    X_np = east_matrix_to_numpy(X)

    session = ort.InferenceSession(onnx_blob)
    input_name = session.get_inputs()[0].name

    outputs = session.run(None, {input_name: X_np.astype(np.float32)})[0]

    return numpy_to_east_matrix(outputs)
```

## Backward Compatibility

To maintain backward compatibility with existing code that passes `y` as a vector:

```python
def torch_mlp_train_impl(X: Matrix, y, mlp_config, train_config):
    # Handle both vector and matrix y
    if isinstance(y, EastArray) and not _is_matrix(y):
        # Convert vector to single-column matrix
        y_np = east_vector_to_numpy(y).reshape(-1, 1)
    else:
        y_np = east_matrix_to_numpy(y)
    # ... rest of implementation
```

## Usage Examples

### Multi-Output Regression

```typescript
import { East, variant } from "@elaraai/east";
import { Torch } from "@elaraai/east-py-datascience";

const train = East.function([], Torch.Types.TorchTrainOutputType, $ => {
    // X: 4 samples, 2 features
    const X = $.let([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]);
    // y: 4 samples, 3 outputs
    const y = $.let([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 6.0, 9.0], [4.0, 8.0, 12.0]]);

    const mlp_config = $.let({
        hidden_layers: [32n, 16n],
        activation: variant('some', variant('relu', {})),
        dropout: variant('some', 0.1),
        output_dim: variant('none', null),  // Inferred from y: 3
    });

    const train_config = $.let({
        epochs: variant('some', 100n),
        batch_size: variant('some', 4n),
        learning_rate: variant('some', 0.001),
        loss: variant('some', variant('mse', {})),
        optimizer: variant('some', variant('adam', {})),
        early_stopping: variant('some', 15n),
        validation_split: variant('some', 0.2),
        random_state: variant('some', 42n),
    });

    return $.return(Torch.mlpTrain(X, y, mlp_config, train_config));
});
```

### Autoencoder (X = y)

```typescript
import { East, variant } from "@elaraai/east";
import { Torch } from "@elaraai/east-py-datascience";

const trainAutoencoder = East.function([], Torch.Types.TorchTrainOutputType, $ => {
    // Origin proportions: 4 samples, 10 origins
    const X = $.let([
        [0.5, 0.3, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.4, 0.4, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.3, 0.3, 0.2, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]);

    // Autoencoder: y = X (reconstruct input)
    const y = $.let(X);

    // Bottleneck architecture: 10 -> 64 -> 10 (embedding) -> 32 -> 10
    const mlp_config = $.let({
        hidden_layers: [64n, 10n, 32n],  // Middle layer (10) is the bottleneck/embedding
        activation: variant('some', variant('relu', {})),
        dropout: variant('some', 0.2),
        output_dim: variant('none', null),  // Inferred: 10 (same as input)
    });

    const train_config = $.let({
        epochs: variant('some', 200n),
        batch_size: variant('some', 32n),
        learning_rate: variant('some', 0.001),
        loss: variant('some', variant('mse', {})),
        optimizer: variant('some', variant('adam', {})),
        early_stopping: variant('some', 20n),
        validation_split: variant('some', 0.2),
        random_state: variant('some', 42n),
    });

    return $.return(Torch.mlpTrain(X, y, mlp_config, train_config));
});
```

## TypeScript Type Updates

Update `torch.d.ts`:

```typescript
// Training function - y becomes MatrixType
export declare const torch_mlp_train: PlatformDefinition<
    [MatrixType, MatrixType, TorchMLPConfigType, TorchTrainConfigType],
    TorchTrainOutputType
>;

// Prediction function - returns MatrixType
export declare const torch_mlp_predict: PlatformDefinition<
    [TorchModelBlobType, MatrixType],
    MatrixType
>;
```

## Notes

1. **Output dimension inference**: When `output_dim` is not specified in config, it's inferred from `y.shape[1]`

2. **Softmax for autoencoders**: For origin proportion reconstruction where outputs should sum to 1, consider adding a `softmax` output activation option to the config

3. **Embedding extraction**: To extract bottleneck activations as embeddings, a separate `torch_mlp_get_activations` function could be added in the future

4. **Loss functions**: MSE works well for reconstruction. For probability distributions (like origin proportions), KL divergence could be added as a loss option
