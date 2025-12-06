# Module 9: PyTorch (`torch_impl.py`)

## Purpose

Neural network models using PyTorch, exported to ONNX for inference.

## Config Types

```python
TorchMLPConfigType = StructType([
    ("hidden_layers", ArrayType(IntegerType)),          # e.g., [64, 32]
    ("activation", OptionType(ActivationFunctionType)), # default relu
    ("dropout", OptionType(FloatType)),                 # default 0.0
    ("output_dim", OptionType(IntegerType)),            # default 1 (regression)
])

TorchTrainConfigType = StructType([
    ("epochs", OptionType(IntegerType)),            # default 100
    ("batch_size", OptionType(IntegerType)),        # default 32
    ("learning_rate", OptionType(FloatType)),       # default 0.001
    ("loss", OptionType(LossFunctionType)),         # default mse
    ("optimizer", OptionType(OptimizerType)),       # default adam
    ("early_stopping", OptionType(IntegerType)),    # patience, 0 = disabled
    ("validation_split", OptionType(FloatType)),    # default 0.2
])
```

## Platform Functions

### `torch_mlp_train`

Create and train Multi-Layer Perceptron model.

```python
PlatformFunction(
    name="torch_mlp_train",
    inputs=[MatrixType, VectorType, TorchMLPConfigType, TorchTrainConfigType],
    output=StructType([
        ("model", ModelBlobType),  # Returns "torch_mlp" variant
        ("result", TorchTrainResultType),
    ]),
    type="sync",
    fn=torch_mlp_train_impl,
)

def torch_mlp_train_impl(
    X: Matrix,
    y: Vector,
    mlp_config: EastStruct,
    train_config: EastStruct
) -> EastStruct:
    """Create and train PyTorch MLP model."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    X_np = east_matrix_to_numpy(X)
    y_np = east_vector_to_numpy(y)
    n_features = X_np.shape[1]

    # MLP config
    hidden_layers_arr = mlp_config.get("hidden_layers")
    hidden_layers = [int(h) for h in hidden_layers_arr] if hidden_layers_arr else [64, 32]

    activation_variant = _get_option(mlp_config.get("activation"), None)
    activation_name = _get_enum_tag(activation_variant) if activation_variant else "relu"

    dropout = _get_option(mlp_config.get("dropout"), 0.0)
    output_dim = _get_option(mlp_config.get("output_dim"), 1)

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

    # Convert to tensors
    X_tensor = torch.FloatTensor(X_np)
    y_tensor = torch.FloatTensor(y_np).unsqueeze(1)

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
    criterion = nn.MSELoss() if loss_name == "mse" else nn.L1Loss()

    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_epoch = 0
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

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
        elif patience > 0:
            patience_counter += 1
            if patience_counter >= patience:
                break

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

### `torch_mlp_predict`

Make predictions with trained PyTorch MLP.

```python
PlatformFunction(
    name="torch_mlp_predict",
    inputs=[ModelBlobType, MatrixType],  # Expects "torch_mlp" variant
    output=VectorType,
    type="sync",
    fn=torch_mlp_predict_impl,
)

def torch_mlp_predict_impl(
    model_blob: ModelBlob,
    X: Matrix
) -> Vector:
    """Make predictions with PyTorch MLP."""
    if model_blob.type != "torch_mlp":
        raise ValueError(f"Expected torch_mlp, got {model_blob.type}")

    onnx_blob = model_blob.value["onnx"]
    return _onnx_predict_regression(onnx_blob, X)
```
