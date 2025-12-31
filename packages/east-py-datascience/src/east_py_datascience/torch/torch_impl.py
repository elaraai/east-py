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
    BooleanType,
    FloatType,
    IntegerType,
    OptionType,
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
    PosWeightType,
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


def _deserialize_model_package(blob: EastBlob):
    """Deserialize model package and extract model + constrained layer + metadata.

    Handles both old format (just model) and new format (dict with model + constrained_layer).

    Returns:
        tuple: (model, constrained_layer, return_logits_mode) where constrained_layer may be None
    """
    deserialized = _deserialize_model(blob)

    # Check if it's the new package format
    if isinstance(deserialized, dict) and "model" in deserialized:
        model = deserialized["model"]
        constrained_layer = deserialized.get("constrained_layer", None)
        return_logits_mode = deserialized.get("return_logits_mode", False)
    else:
        # Old format: just the model
        model = deserialized
        constrained_layer = None
        return_logits_mode = False

    return model, constrained_layer, return_logits_mode


# ============================================================================
# Constrained Output Layer
# ============================================================================


def _parse_row_constraints(constraints_config, n_rows: int, n_cols: int):
    """Parse row constraints from config into a list of constraint specs.

    Returns list of tuples: (constraint_type, mask_tensor_or_none, extra_params)

    The mask is the combination of user-specified 'mask' and data-derived 'data_mask'.
    Both are AND-ed together: final_mask = mask AND data_mask
    """
    import torch

    row_constraints_arr = constraints_config.get("row_constraints")
    if row_constraints_arr is None or len(row_constraints_arr) != n_rows:
        raise RuntimeError(
            f"output_constraints.row_constraints must have exactly {n_rows} entries, "
            f"got {len(row_constraints_arr) if row_constraints_arr else 0}"
        )

    parsed = []
    for i, constraint in enumerate(row_constraints_arr):
        ctype = constraint.type  # "binary", "mutex", or "at_most"
        cvalue = constraint.value  # The struct with mask, allow_none, etc.

        # Parse user mask
        mask_option = _get_option(cvalue.get("mask"), None)
        if mask_option is not None:
            mask_list = [bool(m) for m in mask_option]
            if len(mask_list) != n_cols:
                raise RuntimeError(
                    f"Row {i} mask length {len(mask_list)} != output columns {n_cols}"
                )
            mask = torch.tensor(mask_list, dtype=torch.bool)
        else:
            mask = None

        # Parse data_mask (static mask derived from data)
        data_mask_option = _get_option(cvalue.get("data_mask"), None)
        if data_mask_option is not None:
            data_mask_list = [bool(m) for m in data_mask_option]
            if len(data_mask_list) != n_cols:
                raise RuntimeError(
                    f"Row {i} data_mask length {len(data_mask_list)} != output columns {n_cols}"
                )
            data_mask = torch.tensor(data_mask_list, dtype=torch.bool)
        else:
            data_mask = None

        # Combine masks: final_mask = mask AND data_mask
        if mask is not None and data_mask is not None:
            combined_mask = mask & data_mask
        elif data_mask is not None:
            combined_mask = data_mask
        else:
            combined_mask = mask

        # Parse extra params based on type
        if ctype == "binary":
            parsed.append(("binary", combined_mask, {}))
        elif ctype == "mutex":
            allow_none = _get_option(cvalue.get("allow_none"), False)
            parsed.append(("mutex", combined_mask, {"allow_none": allow_none}))
        elif ctype == "at_most":
            max_count = int(cvalue.get("max_count"))
            parsed.append(("at_most", combined_mask, {"max_count": max_count}))
        else:
            raise RuntimeError(f"Unknown constraint type: {ctype}")

    return parsed


class ConstrainedOutputLayer:
    """Custom output layer that applies per-row constraints.

    This is a callable class (not nn.Module) to avoid issues with cloudpickle
    and to make it work with the existing sequential model structure.
    """

    def __init__(
        self,
        row_constraints: list,
        n_rows: int,
        n_cols: int,
        return_logits: bool = False,
    ):
        """
        Args:
            row_constraints: List of (type, mask, params) tuples from _parse_row_constraints
            n_rows: Number of output rows (e.g., number of tasks)
            n_cols: Number of output columns (e.g., number of days)
            return_logits: If True, return raw logits for binary constraints (for bce_with_logits)
        """

        self.row_constraints = row_constraints
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.return_logits = return_logits

        # Pre-compute mask tensors for efficiency
        self.masks = []
        for ctype, mask, params in row_constraints:
            self.masks.append(mask)

    def __call__(self, x, sample_masks=None):
        """Apply constrained activations to logits.

        Args:
            x: Tensor of shape (batch, n_rows * n_cols) - flat logits
            sample_masks: Optional tensor of shape (batch, n_rows, n_cols) - per-sample boolean masks
                          True = allowed, False = masked (output forced to 0/-inf)

        Returns:
            Tensor of shape (batch, n_rows * n_cols) - activated with constraints
        """
        import torch
        import torch.nn.functional as F

        batch_size = x.shape[0]
        # Reshape to (batch, n_rows, n_cols)
        x = x.view(batch_size, self.n_rows, self.n_cols)

        outputs = []
        for row_idx, (ctype, mask, params) in enumerate(self.row_constraints):
            row_logits = x[:, row_idx, :]  # (batch, n_cols)

            # Apply static mask (from data_mask in constraint config, combined with user mask)
            if mask is not None:
                mask_tensor = mask.to(row_logits.device)
                row_logits = row_logits.masked_fill(~mask_tensor, float("-inf"))

            # Apply per-sample dynamic mask
            if sample_masks is not None:
                sample_row_mask = sample_masks[:, row_idx, :]  # (batch, n_cols)
                row_logits = row_logits.masked_fill(~sample_row_mask, float("-inf"))

            if ctype == "binary":
                if self.return_logits:
                    # Return raw logits for bce_with_logits loss
                    row_out = row_logits
                else:
                    # Independent sigmoid per position
                    row_out = torch.sigmoid(row_logits)
                    # Masked positions will be sigmoid(-inf) = 0

            elif ctype == "mutex":
                # Softmax: exactly one position active (mutually exclusive)
                if params.get("allow_none", False):
                    # Add a "none" option by concatenating a zero logit
                    # Then softmax over n_cols+1, but only return first n_cols
                    none_logit = torch.zeros(batch_size, 1, device=row_logits.device)
                    extended = torch.cat([row_logits, none_logit], dim=-1)
                    probs = F.softmax(extended, dim=-1)
                    row_out = probs[:, : self.n_cols]  # Drop the "none" probability
                else:
                    row_out = F.softmax(row_logits, dim=-1)
                # Masked positions will be softmax(-inf) = 0

            elif ctype == "at_most":
                # At most N positions active
                # Strategy: sigmoid, then zero out all but top-k by probability
                max_count = params["max_count"]
                probs = torch.sigmoid(row_logits)

                if mask is not None:
                    # Masked positions are already 0 from sigmoid(-inf)
                    pass

                # Keep only top-k probabilities, zero out the rest
                if max_count < self.n_cols:
                    # Get top-k indices
                    _, topk_indices = torch.topk(probs, max_count, dim=-1)
                    # Create mask for top-k
                    topk_mask = torch.zeros_like(probs, dtype=torch.bool)
                    topk_mask.scatter_(1, topk_indices, True)
                    row_out = probs * topk_mask.float()
                else:
                    row_out = probs
            else:
                raise RuntimeError(f"Unknown constraint type: {ctype}")

            outputs.append(row_out)

        # Stack back to (batch, n_rows, n_cols) then flatten
        result = torch.stack(outputs, dim=1)
        return result.view(batch_size, -1)


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

    # Output constraints (overrides output_activation if set)
    output_constraints = _get_option(mlp_config.get("output_constraints"), None)
    constrained_layer = None

    # Parse loss early (needed for constrained layer configuration)
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

        # Handle output: either constrained layer or simple activation
        if output_constraints is not None:
            # Parse constraints and create constrained output layer
            row_constraints_arr = output_constraints.get("row_constraints")
            n_rows = len(row_constraints_arr) if row_constraints_arr else 0
            if n_rows == 0:
                raise RuntimeError("output_constraints.row_constraints cannot be empty")
            if output_dim % n_rows != 0:
                raise RuntimeError(
                    f"output_dim ({output_dim}) must be divisible by number of row constraints ({n_rows})"
                )
            n_cols = output_dim // n_rows

            parsed_constraints = _parse_row_constraints(
                output_constraints, n_rows, n_cols
            )

            # Determine if we need logits mode (for bce_with_logits with constraints)
            use_logits_output = loss_name == "bce_with_logits"

            constrained_layer = ConstrainedOutputLayer(
                parsed_constraints, n_rows, n_cols, return_logits=use_logits_output
            )
            # Don't add to sequential - we'll apply it separately
        else:
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

    # loss_name already parsed earlier for constrained layer configuration

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

    # Parse pos_weight (now a variant: scalar or per_output)
    pos_weight_config = _get_option(train_config.get("pos_weight"), None)
    pos_weight_scalar = None
    pos_weight_per_output = None
    if pos_weight_config is not None:
        if pos_weight_config.type == "scalar":
            pos_weight_scalar = float(pos_weight_config.value)
        elif pos_weight_config.type == "per_output":
            pos_weight_per_output = [float(w) for w in pos_weight_config.value]

    # Parse prior config (global prior regularization)
    prior_config = _get_option(train_config.get("prior"), None)
    prior_values = None
    prior_weight = 0.0
    if prior_config is not None:
        prior_values = [float(v) for v in prior_config.get("values")]
        prior_weight = float(prior_config.get("weight"))

    # Parse sample_constraints (per-sample masks, pos_weights, priors)
    sample_constraints = _get_option(train_config.get("sample_constraints"), None)
    sample_masks_list = None
    sample_pos_weights_list = None
    sample_priors_list = None
    if sample_constraints is not None:
        sample_masks_list = _get_option(sample_constraints.get("masks"), None)
        sample_pos_weights_list = _get_option(
            sample_constraints.get("pos_weights"), None
        )
        sample_priors_list = _get_option(sample_constraints.get("priors"), None)

    # Convert to tensors and prepare data
    try:
        X_tensor = torch.FloatTensor(X_np)
        y_tensor = torch.FloatTensor(y_np)

        # For single output, unsqueeze to 2D
        if not is_multi_output:
            y_tensor = y_tensor.unsqueeze(1)

        # Convert sample_constraints to tensors
        sample_masks_tensor = None
        sample_pos_weights_tensor = None
        sample_priors_tensor = None

        if sample_masks_list is not None:
            # Convert nested list to tensor: (n_samples, n_rows, n_cols)
            sample_masks_tensor = torch.tensor(
                [
                    [[bool(v) for v in row] for row in sample]
                    for sample in sample_masks_list
                ],
                dtype=torch.bool,
            )

        if sample_pos_weights_list is not None:
            # Convert nested list to tensor: (n_samples, output_dim)
            sample_pos_weights_tensor = torch.tensor(
                [[float(v) for v in sample] for sample in sample_pos_weights_list],
                dtype=torch.float32,
            )

        if sample_priors_list is not None:
            # Convert nested list to tensor: (n_samples, output_dim)
            sample_priors_tensor = torch.tensor(
                [[float(v) for v in sample] for sample in sample_priors_list],
                dtype=torch.float32,
            )

        # Convert global prior values to tensor
        prior_values_tensor = None
        if prior_values is not None:
            prior_values_tensor = torch.tensor(prior_values, dtype=torch.float32)

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

        # Split sample_constraints tensors too
        sample_masks_train = None
        sample_masks_val = None
        sample_pos_weights_train = None
        sample_priors_train = None

        if sample_masks_tensor is not None:
            sample_masks_train = sample_masks_tensor[train_indices]
            sample_masks_val = sample_masks_tensor[val_indices]

        if sample_pos_weights_tensor is not None:
            sample_pos_weights_train = sample_pos_weights_tensor[train_indices]
            # Note: sample_pos_weights_val not used (validation uses base loss)

        if sample_priors_tensor is not None:
            sample_priors_train = sample_priors_tensor[train_indices]
            # Note: sample_priors_val not used (validation uses base loss)

        # Create data loader with indices for sample-specific data
        train_dataset = TensorDataset(
            X_train, y_train, torch.arange(len(train_indices))
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Loss and optimizer
        # Note: For bce_with_logits, the loss applies sigmoid internally,
        # so model should NOT have sigmoid output_activation
        pos_weight_tensor = None

        if loss_name == "bce_with_logits":
            if output_activation_name == "sigmoid":
                raise RuntimeError(
                    "torch_mlp_train: bce_with_logits loss applies sigmoid internally. "
                    "Do not use with sigmoid output_activation - use 'none' instead."
                )
            # Build pos_weight tensor from config
            if pos_weight_per_output is not None:
                pos_weight_tensor = torch.tensor(
                    pos_weight_per_output, dtype=torch.float32
                )
            elif pos_weight_scalar is not None:
                pos_weight_tensor = torch.full((output_dim,), pos_weight_scalar)

            # Use reduction='none' when we have per-sample weights/priors
            # This allows us to apply custom weighting
            use_manual_reduction = (
                sample_pos_weights_train is not None
                or sample_priors_train is not None
                or prior_values_tensor is not None
            )

            if use_manual_reduction:
                criterion = nn.BCEWithLogitsLoss(reduction="none")
            elif pos_weight_tensor is not None:
                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
            else:
                criterion = nn.BCEWithLogitsLoss()
        elif loss_name == "bce":
            if pos_weight_scalar is not None or pos_weight_per_output is not None:
                # BCELoss doesn't support pos_weight directly
                warnings.warn(
                    "pos_weight is not supported with 'bce' loss. "
                    "Use 'bce_with_logits' instead for class weighting."
                )
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
        import torch.nn.functional as F

        # Suppress PyTorch warnings during training
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=Warning)
            train_losses = []
            val_losses = []
            best_val_loss = float("inf")
            best_epoch = 0
            patience_counter = 0

            # Check if we need manual loss reduction
            use_manual_reduction = (
                sample_pos_weights_train is not None
                or sample_priors_train is not None
                or prior_values_tensor is not None
            )

            for epoch in range(epochs):
                # Train
                model.train()
                epoch_loss = 0.0
                for X_batch, y_batch, batch_indices in train_loader:
                    optimizer.zero_grad()
                    outputs = model(X_batch)

                    # Get per-sample masks for this batch if available
                    batch_masks = None
                    if constrained_layer is not None and sample_masks_train is not None:
                        batch_masks = sample_masks_train[batch_indices]

                    # Apply constrained output layer if set
                    if constrained_layer is not None:
                        outputs = constrained_layer(outputs, sample_masks=batch_masks)

                    # KL divergence requires log probabilities as input
                    if use_log_for_kl:
                        # Clamp to avoid log(0)
                        outputs = torch.log(outputs.clamp(min=1e-10))

                    # Compute base loss
                    base_loss = criterion(outputs, y_batch)

                    # Apply per-sample weighting if using manual reduction
                    if use_manual_reduction and base_loss.dim() > 0:
                        # Apply per-sample pos_weight if provided
                        if sample_pos_weights_train is not None:
                            batch_weights = sample_pos_weights_train[batch_indices]
                            # Weight positive samples only
                            weighted_loss = base_loss * torch.where(
                                y_batch == 1,
                                batch_weights,
                                torch.ones_like(batch_weights),
                            )
                        elif pos_weight_tensor is not None:
                            # Use global pos_weight
                            weighted_loss = base_loss * torch.where(
                                y_batch == 1,
                                pos_weight_tensor,
                                torch.ones_like(pos_weight_tensor),
                            )
                        else:
                            weighted_loss = base_loss

                        loss = weighted_loss.mean()

                        # Add prior regularization if provided
                        if sample_priors_train is not None:
                            batch_priors = sample_priors_train[batch_indices]
                            prior_loss = F.mse_loss(
                                torch.sigmoid(outputs)
                                if loss_name == "bce_with_logits"
                                else outputs,
                                batch_priors,
                            )
                            loss = loss + prior_weight * prior_loss
                        elif prior_values_tensor is not None:
                            prior_target = prior_values_tensor.expand_as(outputs)
                            prior_loss = F.mse_loss(
                                torch.sigmoid(outputs)
                                if loss_name == "bce_with_logits"
                                else outputs,
                                prior_target,
                            )
                            loss = loss + prior_weight * prior_loss
                    else:
                        loss = base_loss

                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                train_losses.append(epoch_loss / len(train_loader))

                # Validate
                model.eval()
                with torch.no_grad():
                    val_pred = model(X_val)

                    # Get per-sample masks for validation if available
                    if constrained_layer is not None and sample_masks_val is not None:
                        val_pred = constrained_layer(
                            val_pred, sample_masks=sample_masks_val
                        )
                    elif constrained_layer is not None:
                        val_pred = constrained_layer(val_pred)

                    if use_log_for_kl:
                        val_pred = torch.log(val_pred.clamp(min=1e-10))

                    val_loss_tensor = criterion(val_pred, y_val)
                    if val_loss_tensor.dim() > 0:
                        val_loss = val_loss_tensor.mean().item()
                    else:
                        val_loss = val_loss_tensor.item()

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

    # Serialize model (include constrained layer and metadata)
    model_package = {
        "model": model,
        "constrained_layer": constrained_layer,
        "return_logits_mode": constrained_layer.return_logits
        if constrained_layer
        else False,
    }
    model_data = _serialize_model(model_package)

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
        model, constrained_layer, return_logits_mode = _deserialize_model_package(
            model_blob.value["data"]
        )
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
                predictions = model(X_tensor)
                # Apply constrained layer if present
                if constrained_layer is not None:
                    predictions = constrained_layer(predictions)
                # Apply sigmoid if model was trained with return_logits=True
                if return_logits_mode:
                    predictions = torch.sigmoid(predictions)
                predictions = predictions.numpy()

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
    sample_masks_opt: EastVariant | None = None,
) -> EastArray:
    """Make predictions with PyTorch MLP (multi-output).

    Returns a matrix where each row contains the predicted outputs for a sample.

    Args:
        model_blob: Trained MLP model blob
        X: Input features (n_samples x n_features)
        sample_masks_opt: Optional per-sample boolean masks (n_samples x n_rows x n_cols)
                         True = allowed, False = masked (output forced to 0)
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
        model, constrained_layer, return_logits_mode = _deserialize_model_package(
            model_blob.value["data"]
        )
    except Exception as e:
        raise RuntimeError(
            f"torch_mlp_predict_multi: Failed to deserialize model - {e}"
        )

    try:
        X_np = east_matrix_to_numpy(X)
    except Exception as e:
        raise RuntimeError(f"torch_mlp_predict_multi: Invalid input data - {e}")

    # Unwrap optional sample_masks
    sample_masks = _get_option(sample_masks_opt, None)

    # Parse sample_masks if provided
    sample_masks_tensor = None
    if sample_masks is not None:
        try:
            sample_masks_tensor = torch.tensor(
                [[[bool(v) for v in row] for row in sample] for sample in sample_masks],
                dtype=torch.bool,
            )
        except Exception as e:
            raise RuntimeError(f"torch_mlp_predict_multi: Invalid sample_masks - {e}")

    # Make predictions
    try:
        # Suppress warnings during prediction
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=Warning)
            # Set model to eval mode
            model.eval()

            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_np)
                predictions = model(X_tensor)
                # Apply constrained layer if present
                if constrained_layer is not None:
                    predictions = constrained_layer(
                        predictions, sample_masks=sample_masks_tensor
                    )
                # Apply sigmoid if model was trained with return_logits=True
                if return_logits_mode:
                    predictions = torch.sigmoid(predictions)
                predictions = predictions.numpy()

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
        model, _, _ = _deserialize_model_package(model_blob.value["data"])
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
        model, constrained_layer, return_logits_mode = _deserialize_model_package(
            model_blob.value["data"]
        )
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

            # Apply constrained layer if present
            if constrained_layer is not None:
                x = constrained_layer(x)

            # Apply sigmoid if model was trained with return_logits=True
            if return_logits_mode:
                x = torch.sigmoid(x)

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

# ============================================================================
# Utility Functions
# ============================================================================


def torch_compute_pos_weight_impl(
    y: EastArray, per_output: bool = False
) -> EastVariant:
    """Compute pos_weight from target data for class imbalance handling.

    For binary classification with imbalanced classes, pos_weight compensates
    by weighting positive samples more heavily.

    Formula: pos_weight = min(cap, (n_neg + smoothing) / (n_pos + smoothing))

    Args:
        y: Target matrix (n_samples x output_dim) with binary values (0/1)
        per_output: If True, compute per-output weights; otherwise compute scalar

    Returns:
        PosWeightType variant: either scalar(float) or per_output(array of floats)
    """
    import numpy as np

    smoothing = 1.0
    cap = 20.0

    try:
        y_np = east_matrix_to_numpy(y)
    except Exception as e:
        raise RuntimeError(f"torch_compute_pos_weight: Invalid input data - {e}")

    if per_output:
        # Compute per-output pos_weight
        n_samples = y_np.shape[0]
        pos_weights = []
        for col in range(y_np.shape[1]):
            n_pos = np.sum(y_np[:, col] == 1)
            n_neg = n_samples - n_pos
            weight = (n_neg + smoothing) / (n_pos + smoothing)
            pos_weights.append(float(min(weight, cap)))
        return EastVariant("per_output", EastArray(FloatType, pos_weights))
    else:
        # Compute scalar pos_weight across all outputs
        n_pos = np.sum(y_np == 1)
        n_total = y_np.size
        n_neg = n_total - n_pos
        weight = (n_neg + smoothing) / (n_pos + smoothing)
        return EastVariant("scalar", float(min(weight, cap)))


def torch_compute_data_mask_impl(y: EastArray, threshold: float = 0.0) -> EastArray:
    """Compute data_mask from target data for constraint configuration.

    Identifies which output positions have any non-zero values across samples.
    Used to create static masks that exclude positions that are never active.

    Args:
        y: Target matrix (n_samples x output_dim) with values
        threshold: Values > threshold are considered active (default 0.0)

    Returns:
        Boolean array (output_dim,): True for positions with any active values
    """
    import numpy as np

    try:
        y_np = east_matrix_to_numpy(y)
    except Exception as e:
        raise RuntimeError(f"torch_compute_data_mask: Invalid input data - {e}")

    # Check which columns have any value > threshold
    has_value = np.any(y_np > threshold, axis=0)
    return EastArray(BooleanType, [bool(v) for v in has_value])


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
        inputs=[
            ModelBlobType,
            MatrixType,
            OptionType(ArrayType(ArrayType(ArrayType(BooleanType)))),
        ],
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
    # Utility functions for constraint configuration
    PlatformFunction(
        name="torch_compute_pos_weight",
        inputs=[MatrixType, BooleanType],
        output=PosWeightType,
        type="sync",
        fn=torch_compute_pos_weight_impl,
    ),
    PlatformFunction(
        name="torch_compute_data_mask",
        inputs=[MatrixType, FloatType],
        output=ArrayType(BooleanType),
        type="sync",
        fn=torch_compute_data_mask_impl,
    ),
]

__all__ = [
    "torch_impl",
]
