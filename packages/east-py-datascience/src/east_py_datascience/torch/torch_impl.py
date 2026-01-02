#
# Copyright (c) 2025 Elara AI Pty Ltd
# Licensed under the Business Source License 1.1. See LICENSE.md for details.
#
"""PyTorch platform functions for East.

Provides neural network models using PyTorch.
Uses cloudpickle for model serialization.
"""

import warnings

import numpy as np

from east.runtime.platform import PlatformFunction
from east.types.types import (
    ArrayType,
    BlobType,
    FloatType,
    IntegerType,
    StructType,
    VariantType,
)
from east.types.values import EastArray, EastBlob, EastStruct, EastVariant

from east_py_datascience.types import (
    MatrixType,
    VectorType,
    TorchMLPConfigType,
    TorchTrainConfigType,
    TorchTrainResultType,
    ModelBlobType,
    _get_option,
    _get_enum_tag,
    east_matrix_to_numpy,
    east_vector_to_numpy,
    numpy_to_east_vector,
    numpy_to_east_matrix,
)


# ============================================================================
# Serialization Helpers
# ============================================================================


def _serialize_model(model) -> EastBlob:
    """Serialize PyTorch model using cloudpickle."""
    try:
        import cloudpickle
    except ImportError as e:
        raise RuntimeError(
            f"_serialize_model: cloudpickle not installed - install with 'pip install cloudpickle' - {e}"
        )

    try:
        return EastBlob(cloudpickle.dumps(model))
    except Exception as e:
        raise RuntimeError(f"_serialize_model: Failed to serialize model - {e}")


def _deserialize_model(blob: EastBlob):
    """Deserialize PyTorch model using cloudpickle."""
    try:
        import cloudpickle
    except ImportError as e:
        raise RuntimeError(
            f"_deserialize_model: cloudpickle not installed - install with 'pip install cloudpickle' - {e}"
        )

    try:
        return cloudpickle.loads(bytes(blob))
    except Exception as e:
        raise RuntimeError(f"_deserialize_model: Failed to deserialize model - {e}")




# ============================================================================
# Internal Training Helper
# ============================================================================


def _torch_mlp_train_internal(
    X_np: np.ndarray,
    y_np: np.ndarray,
    mlp_config: EastStruct,
    train_config: EastStruct,
    is_multi_output: bool,
) -> EastStruct:
    """Internal training logic shared between single and multi-output training.

    Args:
        X_np: Input features as numpy array (n_samples, n_features)
        y_np: Targets as numpy array - 1D for single output, 2D for multi output
        mlp_config: MLP configuration struct
        train_config: Training configuration struct
        is_multi_output: Whether this is multi-output training

    Returns:
        EastStruct with model and training result
    """
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as e:
        raise RuntimeError(
            f"torch_mlp_train: PyTorch not installed - install with 'pip install torch' - {e}"
        )

    # Validate shapes
    if X_np.shape[0] != y_np.shape[0]:
        raise RuntimeError(
            f"torch_mlp_train: X and y have different sample counts - X: {X_np.shape[0]}, y: {y_np.shape[0]}"
        )

    n_features = X_np.shape[1]

    # Determine output dimension
    if is_multi_output:
        n_outputs = y_np.shape[1]
    else:
        n_outputs = 1

    # MLP config
    hidden_layers_arr = mlp_config.get("hidden_layers")
    hidden_layers = (
        [int(h) for h in hidden_layers_arr] if hidden_layers_arr else [64, 32]
    )

    activation_variant = _get_option(mlp_config.get("activation"), None)
    activation_name = (
        _get_enum_tag(activation_variant) if activation_variant else "relu"
    )

    dropout = _get_option(mlp_config.get("dropout"), 0.0)

    # Output activation (applied to final layer only)
    output_activation_variant = _get_option(mlp_config.get("output_activation"), None)
    output_activation_name = (
        _get_enum_tag(output_activation_variant)
        if output_activation_variant
        else "none"
    )

    # Parse loss
    loss_variant = _get_option(train_config.get("loss"), None)
    loss_name = _get_enum_tag(loss_variant) if loss_variant else "mse"

    # output_dim from config overrides inferred, but default to inferred n_outputs
    output_dim = _get_option(mlp_config.get("output_dim"), n_outputs)
    if output_dim is not None:
        output_dim = int(output_dim)
    else:
        output_dim = n_outputs

    # Build model
    try:
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

        # Add simple output activation if specified
        if output_activation_name == "softmax":
            layers.append(nn.Softmax(dim=-1))
        elif output_activation_name == "sigmoid":
            layers.append(nn.Sigmoid())
        # "none" = no activation (linear output) - default

        model = nn.Sequential(*layers)
    except Exception as e:
        raise RuntimeError(f"torch_mlp_train: Failed to build model - {e}")

    # Training config
    epochs = _get_option(train_config.get("epochs"), 100)
    if epochs is not None:
        epochs = int(epochs)
    else:
        epochs = 100

    batch_size = _get_option(train_config.get("batch_size"), 32)
    if batch_size is not None:
        batch_size = int(batch_size)
    else:
        batch_size = 32

    lr = _get_option(train_config.get("learning_rate"), 0.001)
    if lr is not None:
        lr = float(lr)
    else:
        lr = 0.001

    optimizer_variant = _get_option(train_config.get("optimizer"), None)
    optimizer_name = _get_enum_tag(optimizer_variant) if optimizer_variant else "adam"

    patience = _get_option(train_config.get("early_stopping"), 0)
    if patience is not None:
        patience = int(patience)
    else:
        patience = 0

    val_split = _get_option(train_config.get("validation_split"), 0.2)
    if val_split is not None:
        val_split = float(val_split)
    else:
        val_split = 0.2

    random_state = _get_option(train_config.get("random_state"), None)
    if random_state is not None:
        random_state = int(random_state)
        torch.manual_seed(random_state)
        np.random.seed(random_state)

    # Convert to tensors and prepare data
    try:
        X_tensor = torch.FloatTensor(X_np)
        y_tensor = torch.FloatTensor(y_np)

        # For single output, unsqueeze to 2D
        if not is_multi_output:
            y_tensor = y_tensor.unsqueeze(1)

        # Train/val split
        n = len(X_tensor)
        n_val = int(n * val_split)
        n_val = max(1, n_val)  # At least 1 validation sample

        indices = torch.randperm(n)
        train_indices = indices[n_val:]
        val_indices = indices[:n_val]

        X_train = X_tensor[train_indices]
        y_train = y_tensor[train_indices]
        X_val = X_tensor[val_indices]
        y_val = y_tensor[val_indices]

        # Create data loader
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Loss and optimizer
        if loss_name == "bce_with_logits":
            if output_activation_name == "sigmoid":
                raise RuntimeError(
                    "torch_mlp_train: bce_with_logits loss applies sigmoid internally. "
                    "Do not use with sigmoid output_activation - use 'none' instead."
                )
            criterion = nn.BCEWithLogitsLoss()
        elif loss_name == "bce":
            criterion = nn.BCELoss()
        else:
            loss_map = {
                "mse": nn.MSELoss,
                "mae": nn.L1Loss,
                "cross_entropy": nn.CrossEntropyLoss,
                "kl_div": lambda: nn.KLDivLoss(reduction="batchmean"),
            }
            criterion = loss_map.get(loss_name, nn.MSELoss)()

        # KL divergence requires log probabilities as input
        use_log_for_kl = loss_name == "kl_div"

        if optimizer_name == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        elif optimizer_name == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        elif optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        else:  # rmsprop
            optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    except Exception as e:
        raise RuntimeError(
            f"torch_mlp_train: Failed to prepare training data - X shape: {X_np.shape} - {e}"
        )

    # Training loop
    try:
        # Suppress PyTorch warnings during training
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=Warning)
            train_losses = []
            val_losses = []
            best_val_loss = float("inf")
            best_epoch = 0
            patience_counter = 0

            for epoch in range(epochs):
                # Train
                model.train()
                epoch_loss = 0.0
                for X_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    outputs = model(X_batch)

                    # KL divergence requires log probabilities as input
                    if use_log_for_kl:
                        # Clamp to avoid log(0)
                        outputs = torch.log(outputs.clamp(min=1e-10))

                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                train_losses.append(epoch_loss / len(train_loader))

                # Validate
                model.eval()
                with torch.no_grad():
                    val_pred = model(X_val)

                    if use_log_for_kl:
                        val_pred = torch.log(val_pred.clamp(min=1e-10))

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
    except Exception as e:
        raise RuntimeError(
            f"torch_mlp_train: Training failed - X shape: {X_np.shape} - {e}"
        )

    # Serialize model
    model_data = _serialize_model(model)

    model_blob = EastVariant(
        "torch_mlp",
        EastStruct(
            {
                "data": model_data,
                "n_features": n_features,
                "hidden_layers": EastArray(IntegerType, hidden_layers),
                "output_dim": output_dim,
            }
        ),
    )

    train_result = EastStruct(
        {
            "train_losses": EastArray(FloatType, train_losses),
            "val_losses": EastArray(FloatType, val_losses),
            "best_epoch": best_epoch,
        }
    )

    return EastStruct(
        {
            "model": model_blob,
            "result": train_result,
        }
    )


# ============================================================================
# Platform Function Implementations
# ============================================================================


def torch_mlp_train_impl(
    X: EastArray,
    y: EastArray,
    mlp_config: EastStruct,
    train_config: EastStruct,
) -> EastStruct:
    """Create and train PyTorch MLP model (single output)."""
    try:
        X_np = east_matrix_to_numpy(X)
        y_np = east_vector_to_numpy(y)
    except Exception as e:
        raise RuntimeError(f"torch_mlp_train: Invalid input data - {e}")

    return _torch_mlp_train_internal(
        X_np, y_np, mlp_config, train_config, is_multi_output=False
    )


def torch_mlp_train_multi_impl(
    X: EastArray,
    y: EastArray,
    mlp_config: EastStruct,
    train_config: EastStruct,
) -> EastStruct:
    """Create and train PyTorch MLP model (multi-output).

    Supports multi-output regression and autoencoders where y is a matrix.
    Output dimension is inferred from y.shape[1] unless overridden in config.
    """
    try:
        X_np = east_matrix_to_numpy(X)
        y_np = east_matrix_to_numpy(y)
    except Exception as e:
        raise RuntimeError(f"torch_mlp_train_multi: Invalid input data - {e}")

    return _torch_mlp_train_internal(
        X_np, y_np, mlp_config, train_config, is_multi_output=True
    )


def torch_mlp_predict_impl(
    model_blob: EastVariant,
    X: EastArray,
) -> EastArray:
    """Make predictions with PyTorch MLP (single output)."""
    try:
        import torch
    except ImportError as e:
        raise RuntimeError(
            f"torch_mlp_predict: PyTorch not installed - install with 'pip install torch' - {e}"
        )

    # Validate model type
    if model_blob.type != "torch_mlp":
        raise RuntimeError(
            f"torch_mlp_predict: Expected torch_mlp model, got {model_blob.type}"
        )

    try:
        model = _deserialize_model(model_blob.value["data"])
    except Exception as e:
        raise RuntimeError(f"torch_mlp_predict: Failed to deserialize model - {e}")

    try:
        X_np = east_matrix_to_numpy(X)
    except Exception as e:
        raise RuntimeError(f"torch_mlp_predict: Invalid input data - {e}")

    # Make predictions
    try:
        # Suppress warnings during prediction
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=Warning)
            # Set model to eval mode
            model.eval()

            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_np)
                predictions = model(X_tensor).numpy()

        # Flatten if single output
        if predictions.ndim > 1 and predictions.shape[1] == 1:
            predictions = predictions.flatten()

        return numpy_to_east_vector(predictions)
    except Exception as e:
        raise RuntimeError(
            f"torch_mlp_predict: Prediction failed - X shape: {X_np.shape} - {e}"
        )


def torch_mlp_predict_multi_impl(
    model_blob: EastVariant,
    X: EastArray,
) -> EastArray:
    """Make predictions with PyTorch MLP (multi-output).

    Returns a matrix where each row contains the predicted outputs for a sample.

    Args:
        model_blob: Trained MLP model blob
        X: Input features (n_samples x n_features)
    """
    try:
        import torch
    except ImportError as e:
        raise RuntimeError(
            f"torch_mlp_predict_multi: PyTorch not installed - install with 'pip install torch' - {e}"
        )

    # Validate model type
    if model_blob.type != "torch_mlp":
        raise RuntimeError(
            f"torch_mlp_predict_multi: Expected torch_mlp model, got {model_blob.type}"
        )

    try:
        model = _deserialize_model(model_blob.value["data"])
    except Exception as e:
        raise RuntimeError(
            f"torch_mlp_predict_multi: Failed to deserialize model - {e}"
        )

    try:
        X_np = east_matrix_to_numpy(X)
    except Exception as e:
        raise RuntimeError(f"torch_mlp_predict_multi: Invalid input data - {e}")

    # Make predictions
    try:
        # Suppress warnings during prediction
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=Warning)
            # Set model to eval mode
            model.eval()

            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_np)
                predictions = model(X_tensor).numpy()

        # Ensure 2D output
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)

        return numpy_to_east_matrix(predictions)
    except Exception as e:
        raise RuntimeError(
            f"torch_mlp_predict_multi: Prediction failed - X shape: {X_np.shape} - {e}"
        )


def torch_mlp_encode_impl(
    model_blob: EastVariant,
    X: EastArray,
    layer_index: int,
) -> EastArray:
    """Extract intermediate layer activations (embeddings) from MLP.

    For autoencoders, this allows extracting the bottleneck representation.
    The layer_index specifies which layer's output to return (0-indexed).

    For an autoencoder with architecture [input -> 8 -> 2 -> 8 -> output]:
    - layer_index=0: output after first Linear+Activation (8 features)
    - layer_index=1: output after second Linear+Activation (2 features) <- bottleneck
    - layer_index=2: output after third Linear+Activation (8 features)

    Note: Each "layer" in hidden_layers corresponds to Linear+Activation(+Dropout),
    so layer_index=1 means after the 2nd hidden layer block.

    Args:
        model_blob: Trained MLP model blob
        X: Input feature matrix (n_samples x n_features)
        layer_index: Which hidden layer's output to return (0-indexed)

    Returns:
        Matrix of intermediate activations (n_samples x hidden_dim at that layer)
    """
    try:
        import torch
    except ImportError as e:
        raise RuntimeError(
            f"torch_mlp_encode: PyTorch not installed - install with 'pip install torch' - {e}"
        )

    # Validate model type
    if model_blob.type != "torch_mlp":
        raise RuntimeError(
            f"torch_mlp_encode: Expected torch_mlp model, got {model_blob.type}"
        )

    try:
        model = _deserialize_model(model_blob.value["data"])
    except Exception as e:
        raise RuntimeError(f"torch_mlp_encode: Failed to deserialize model - {e}")

    try:
        X_np = east_matrix_to_numpy(X)
    except Exception as e:
        raise RuntimeError(f"torch_mlp_encode: Invalid input data - {e}")

    # Get hidden layer dimensions from model metadata
    hidden_layers = list(model_blob.value["hidden_layers"])
    n_hidden = len(hidden_layers)

    if layer_index < 0 or layer_index >= n_hidden:
        raise RuntimeError(
            f"torch_mlp_encode: layer_index {layer_index} out of range. "
            f"Model has {n_hidden} hidden layers (0 to {n_hidden - 1})."
        )

    # Extract activations up to the specified layer
    try:
        model.eval()

        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_np)

            # Run through model layers up to and including the target layer
            # Model structure: [Linear, Activation, (Dropout), Linear, Activation, (Dropout), ..., Linear]
            # We need to count how many "blocks" we've passed
            current_hidden_layer = -1
            x = X_tensor

            for i, layer in enumerate(model):
                x = layer(x)

                # Check if this is a Linear layer (start of a new block)
                if isinstance(layer, torch.nn.Linear):
                    # Check if this is NOT the final output layer
                    # Final layer is followed by nothing or is the last layer
                    if i < len(model) - 1:
                        current_hidden_layer += 1

                        # If we've reached our target layer, we need to apply
                        # the activation (and optionally dropout) before returning
                        if current_hidden_layer == layer_index:
                            # Apply subsequent non-linear layers until next Linear
                            for j in range(i + 1, len(model)):
                                next_layer = model[j]
                                if isinstance(next_layer, torch.nn.Linear):
                                    break
                                x = next_layer(x)
                            break

            embeddings = x.numpy()

        # Ensure 2D output
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(-1, 1)

        return numpy_to_east_matrix(embeddings)
    except Exception as e:
        raise RuntimeError(
            f"torch_mlp_encode: Encoding failed - X shape: {X_np.shape}, "
            f"layer_index: {layer_index} - {e}"
        )


def torch_mlp_decode_impl(
    model_blob: EastVariant,
    embeddings: EastArray,
    layer_index: int,
) -> EastArray:
    """Decode embeddings back through the decoder portion of an MLP.

    For autoencoders, this takes bottleneck activations and runs them through
    the decoder to reconstruct the output. This is the complement to mlpEncode.

    For an autoencoder with architecture [input -> 8 -> 2 -> 8 -> output]:
    - layer_index=1: Start from the 2-dim bottleneck, run through layers 2+ to output
    - layer_index=0: Start from the 8-dim first layer, run through layers 1+ to output

    Use case: Compute weighted average of origin embeddings, then decode to
    get the reconstructed blend weight distribution.

    Args:
        model_blob: Trained MLP model blob
        embeddings: Embedding matrix (n_samples x hidden_dim at layer_index)
        layer_index: Which hidden layer the embeddings come from (0-indexed)

    Returns:
        Decoded output matrix (n_samples x output_dim)
    """
    try:
        import torch
    except ImportError as e:
        raise RuntimeError(
            f"torch_mlp_decode: PyTorch not installed - install with 'pip install torch' - {e}"
        )

    # Validate model type
    if model_blob.type != "torch_mlp":
        raise RuntimeError(
            f"torch_mlp_decode: Expected torch_mlp model, got {model_blob.type}"
        )

    try:
        model = _deserialize_model(model_blob.value["data"])
    except Exception as e:
        raise RuntimeError(f"torch_mlp_decode: Failed to deserialize model - {e}")

    try:
        emb_np = east_matrix_to_numpy(embeddings)
    except Exception as e:
        raise RuntimeError(f"torch_mlp_decode: Invalid embeddings data - {e}")

    # Get hidden layer dimensions from model metadata
    hidden_layers = list(model_blob.value["hidden_layers"])
    n_hidden = len(hidden_layers)

    if layer_index < 0 or layer_index >= n_hidden:
        raise RuntimeError(
            f"torch_mlp_decode: layer_index {layer_index} out of range. "
            f"Model has {n_hidden} hidden layers (0 to {n_hidden - 1})."
        )

    # Validate embedding dimensions match expected layer size
    expected_dim = int(hidden_layers[layer_index])
    actual_dim = emb_np.shape[1] if emb_np.ndim > 1 else 1
    if actual_dim != expected_dim:
        raise RuntimeError(
            f"torch_mlp_decode: Embedding dimension {actual_dim} doesn't match "
            f"expected dimension {expected_dim} for layer {layer_index}."
        )

    # Run embeddings through decoder (layers after layer_index)
    try:
        model.eval()

        with torch.no_grad():
            x = torch.FloatTensor(emb_np)

            # Find where to start in the model
            # We need to skip: (layer_index + 1) hidden layer blocks
            # Each block is: Linear + Activation + (optional Dropout)
            current_hidden_layer = -1
            start_from_next = False

            for i, layer in enumerate(model):
                if start_from_next:
                    # We're past the target layer, apply remaining layers
                    x = layer(x)
                elif isinstance(layer, torch.nn.Linear):
                    if i < len(model) - 1:  # Not the final output layer
                        current_hidden_layer += 1
                        if current_hidden_layer == layer_index:
                            # Found target layer - skip it and its activation/dropout
                            # Start applying from the NEXT Linear layer
                            start_from_next = True
                            # Skip activation and dropout that follow this Linear
                            continue
                    else:
                        # This is the final output layer
                        x = layer(x)

            output = x.numpy()

        # Ensure 2D output
        if output.ndim == 1:
            output = output.reshape(-1, 1)

        return numpy_to_east_matrix(output)
    except Exception as e:
        raise RuntimeError(
            f"torch_mlp_decode: Decoding failed - embeddings shape: {emb_np.shape}, "
            f"layer_index: {layer_index} - {e}"
        )


# ============================================================================
# Result Type for Training Output
# ============================================================================

# Combined output type: model + training result
TorchTrainOutputType = StructType(
    [
        (
            "model",
            VariantType(
                [
                    (
                        "torch_mlp",
                        StructType(
                            [
                                ("data", BlobType),
                                ("n_features", IntegerType),
                                ("hidden_layers", ArrayType(IntegerType)),
                                ("output_dim", IntegerType),
                            ]
                        ),
                    ),
                ]
            ),
        ),
        ("result", TorchTrainResultType),
    ]
)


# ============================================================================
# Platform Function Registration
# ============================================================================

torch_impl = [
    # Single-output functions (original)
    PlatformFunction(
        name="torch_mlp_train",
        inputs=[MatrixType, VectorType, TorchMLPConfigType, TorchTrainConfigType],
        output=TorchTrainOutputType,
        type="sync",
        fn=torch_mlp_train_impl,
    ),
    PlatformFunction(
        name="torch_mlp_predict",
        inputs=[ModelBlobType, MatrixType],
        output=VectorType,
        type="sync",
        fn=torch_mlp_predict_impl,
    ),
    # Multi-output functions (for multi-output regression and autoencoders)
    PlatformFunction(
        name="torch_mlp_train_multi",
        inputs=[MatrixType, MatrixType, TorchMLPConfigType, TorchTrainConfigType],
        output=TorchTrainOutputType,
        type="sync",
        fn=torch_mlp_train_multi_impl,
    ),
    PlatformFunction(
        name="torch_mlp_predict_multi",
        inputs=[ModelBlobType, MatrixType],
        output=MatrixType,
        type="sync",
        fn=torch_mlp_predict_multi_impl,
    ),
    # Encoding function (for extracting intermediate layer activations / embeddings)
    PlatformFunction(
        name="torch_mlp_encode",
        inputs=[ModelBlobType, MatrixType, IntegerType],
        output=MatrixType,
        type="sync",
        fn=torch_mlp_encode_impl,
    ),
    # Decoding function (for reconstructing from embeddings)
    PlatformFunction(
        name="torch_mlp_decode",
        inputs=[ModelBlobType, MatrixType, IntegerType],
        output=MatrixType,
        type="sync",
        fn=torch_mlp_decode_impl,
    ),
]

__all__ = [
    "torch_impl",
]
