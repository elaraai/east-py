#
# Copyright (c) 2025 Elara AI Pty Ltd
# Licensed under the Business Source License 1.1. See LICENSE.md for details.
#
"""PyTorch platform functions for East.

Provides neural network models using PyTorch.
Uses cloudpickle for model serialization.
"""

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

    loss_variant = _get_option(train_config.get("loss"), None)
    loss_name = _get_enum_tag(loss_variant) if loss_variant else "mse"

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

        X_train = X_tensor[indices[n_val:]]
        y_train = y_tensor[indices[n_val:]]
        X_val = X_tensor[indices[:n_val]]
        y_val = y_tensor[indices[:n_val]]

        train_loader = DataLoader(
            TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True
        )

        # Loss and optimizer
        loss_map = {
            "mse": nn.MSELoss,
            "mae": nn.L1Loss,
            "cross_entropy": nn.CrossEntropyLoss,
        }
        criterion = loss_map.get(loss_name, nn.MSELoss)()

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
]

__all__ = [
    "torch_impl",
]
