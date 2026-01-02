#
# Copyright (c) 2025 Elara AI Pty Ltd
# Licensed under the Business Source License 1.1. See LICENSE.md for details.
#
"""Lightning-based neural network platform functions for East Data Science.

Provides a production-grade neural network training module using PyTorch Lightning.
Supports regression, binary classification, multiclass classification, and
multi-head categorical outputs.
"""

import pickle
import tempfile
import shutil
from typing import Callable

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

from east.runtime.platform import PlatformFunction
from east.types.values import EastArray, EastBlob, EastStruct, EastVariant, is_east_variant


from east_py_datascience.types import (
    MatrixType,
    LightningConfigType,
    LightningResultType,
    ModelBlobType,
    GroupWeightsType,
    _get_option,
    east_matrix_to_numpy,
    numpy_to_east_matrix,
)

class EpochCallback(pl.Callback):
    """Callback that invokes user-provided East function each epoch."""

    def __init__(self, fn: Callable):
        self.fn = fn

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        epoch = trainer.current_epoch
        train_loss = float(trainer.callback_metrics.get("train_loss", 0.0))
        val_loss = float(trainer.callback_metrics.get("val_loss", 0.0))
        # Call the East function directly - it's a compiled Python callable
        self.fn(epoch, train_loss, val_loss)


# ============================================================================
# Lightning Model
# ============================================================================


class LightningMLP(pl.LightningModule):
    """Production-grade MLP/Autoencoder using PyTorch Lightning."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        architecture_type: str,
        architecture_config: dict,
        output_type: str,
        output_config: dict,
        learning_rate: float = 1e-3,
        dropout: float = 0.1,
        weight_decay: float = 0.0,
        group_weights: dict | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.architecture_type = architecture_type
        self.output_type = output_type
        self.output_config = output_config
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Group weights support
        self.use_group_weights = group_weights is not None
        self.group_weights_type = group_weights["weights_type"] if group_weights else None
        if group_weights is not None:
            self.register_buffer(
                "group_weights_tensor",
                torch.tensor(group_weights["weights"], dtype=torch.float32)
            )
        else:
            self.group_weights_tensor = None

        # Build network based on architecture
        if architecture_type == "autoencoder":
            encoder_layers = architecture_config["encoder_layers"]
            latent_dim = architecture_config["latent_dim"]
            decoder_layers = architecture_config["decoder_layers"]

            self.encoder = self._build_mlp(input_dim, encoder_layers, latent_dim, dropout)
            self.decoder = self._build_mlp(latent_dim, decoder_layers, output_dim, dropout)
            self.latent_dim = latent_dim
        else:
            hidden_layers = architecture_config["hidden_layers"]
            self.net = self._build_mlp(input_dim, hidden_layers, output_dim, dropout)
            self.latent_dim = None

        # Store class weights as buffers if provided (only used when group_weights not provided)
        if output_type == "multi_head" and output_config.get("class_weights") is not None:
            self.register_buffer(
                "class_weights", torch.tensor(output_config["class_weights"], dtype=torch.float32)
            )
        else:
            self.class_weights = None

    def _build_mlp(
        self, input_dim: int, hidden_layers: list[int], output_dim: int, dropout: float
    ) -> nn.Sequential:
        """Build an MLP with LayerNorm and dropout."""
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass - returns raw logits."""
        if self.architecture_type == "autoencoder":
            return self.decoder(self.encoder(x))
        return self.net(x)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space (autoencoder only)."""
        if self.architecture_type != "autoencoder":
            raise ValueError("encode() only available for autoencoder architecture")
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to output (autoencoder only)."""
        if self.architecture_type != "autoencoder":
            raise ValueError("decode() only available for autoencoder architecture")
        return self.decoder(z)

    def training_step(self, batch, batch_idx):
        # Batch structure depends on whether group_weights was provided:
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
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
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
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def _compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        masks: torch.Tensor | None = None,
        group_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute loss based on output type."""
        if self.output_type == "regression":
            return F.mse_loss(logits, targets)

        elif self.output_type == "binary":
            return self._binary_loss(logits, targets, masks, group_idx)

        elif self.output_type == "multiclass":
            class_weights = self.output_config.get("class_weights")
            if class_weights is not None:
                class_weights = torch.tensor(class_weights, dtype=torch.float32, device=logits.device)
            target_indices = targets.argmax(dim=-1)
            return F.cross_entropy(logits, target_indices, weight=class_weights)

        elif self.output_type == "multi_head":
            return self._multi_head_loss(logits, targets, masks, group_idx)

        else:
            raise ValueError(f"Unknown output type: {self.output_type}")

    def _binary_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        masks: torch.Tensor | None = None,
        group_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute binary cross-entropy loss with optional group-based pos_weights."""
        if self.group_weights_tensor is not None and group_idx is not None:
            # Per-sample pos_weights from group lookup: [batch, output_dim]
            batch_pos_weights = self.group_weights_tensor[group_idx]

            # Compute BCE with per-sample weights
            loss = F.binary_cross_entropy_with_logits(
                logits, targets, reduction="none"
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
            # Existing global pos_weight path
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

    def _multi_head_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        masks: torch.Tensor | None = None,
        group_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Vectorized multi-head CE loss with group-based weights."""
        n_heads = self.output_config["n_heads"]
        n_classes = self.output_config["n_classes_per_head"]

        batch_size = logits.shape[0]
        logits = logits.view(batch_size, n_heads, n_classes)
        targets = targets.view(batch_size, n_heads, n_classes)
        target_indices = targets.argmax(dim=-1)  # [batch, n_heads]

        # Apply masks if provided
        if masks is not None:
            # masks: (batch, n_heads, n_classes) - True = valid
            logits = logits.masked_fill(~masks, float("-inf"))

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
        elif self.class_weights is not None:
            # Global weights (also vectorized)
            # class_weights: [n_heads, n_classes], target_indices: [batch, n_heads]
            # Expand to [batch, n_heads, n_classes] then gather
            batch_size = target_indices.shape[0]
            expanded_weights = self.class_weights.unsqueeze(0).expand(batch_size, -1, -1)
            sample_weights = expanded_weights.gather(2, target_indices.unsqueeze(-1)).squeeze(-1)  # [batch, n_heads]
            weighted_nll = nll * sample_weights
            return weighted_nll.mean()
        else:
            return nll.mean()

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

    def predict_probs(self, x: torch.Tensor) -> torch.Tensor:
        """Get output probabilities (applies appropriate activation)."""
        return self.predict_probs_with_masks(x, None)

    def apply_output_activation(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply output activation to logits (without running forward pass).

        Used by decode() to convert decoder output to probabilities.
        """
        if self.output_type == "regression":
            return logits
        elif self.output_type == "binary":
            return torch.sigmoid(logits)
        elif self.output_type == "multiclass":
            return F.softmax(logits, dim=-1)
        elif self.output_type == "multi_head":
            n_heads = self.output_config["n_heads"]
            n_classes = self.output_config["n_classes_per_head"]
            batch_size = logits.shape[0]
            logits = logits.view(batch_size, n_heads, n_classes)
            probs = F.softmax(logits, dim=-1)
            return probs.view(batch_size, -1)
        else:
            return logits

    def predict_probs_with_masks(
        self, x: torch.Tensor, masks: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Get output probabilities with optional masking for binary/multi-head outputs."""
        logits = self(x)

        if self.output_type == "regression":
            return logits
        elif self.output_type == "binary":
            probs = torch.sigmoid(logits)
            # Apply masks if provided: set masked positions to 0
            if masks is not None:
                if masks.dim() == 3:
                    masks = masks.squeeze(1)
                probs = probs * masks.float()
            return probs
        elif self.output_type == "multiclass":
            return F.softmax(logits, dim=-1)
        elif self.output_type == "multi_head":
            n_heads = self.output_config["n_heads"]
            n_classes = self.output_config["n_classes_per_head"]
            batch_size = logits.shape[0]
            logits = logits.view(batch_size, n_heads, n_classes)

            # Apply masks if provided: set masked positions to -inf before softmax
            if masks is not None:
                # masks: (batch, n_heads, n_classes) - True = valid
                logits = logits.masked_fill(~masks, float("-inf"))

            probs = F.softmax(logits, dim=-1)
            return probs.view(batch_size, -1)
        else:
            return logits


# ============================================================================
# Serialization Helpers
# ============================================================================


def _serialize_model(model: LightningMLP) -> bytes:
    """Serialize model state_dict + hyperparameters."""
    data = {
        "state_dict": model.state_dict(),
        "hparams": dict(model.hparams),
    }
    return pickle.dumps(data)


def _deserialize_model(blob: bytes) -> LightningMLP:
    """Deserialize model from state_dict + hyperparameters."""
    data = pickle.loads(blob)
    hparams = data["hparams"]

    model = LightningMLP(
        input_dim=hparams["input_dim"],
        output_dim=hparams["output_dim"],
        architecture_type=hparams["architecture_type"],
        architecture_config=hparams["architecture_config"],
        output_type=hparams["output_type"],
        output_config=hparams["output_config"],
        learning_rate=hparams.get("learning_rate", 1e-3),
        dropout=hparams.get("dropout", 0.1),
        weight_decay=hparams.get("weight_decay", 0.0),
    )
    # Use strict=False to allow loading models trained with group_weights
    # (which have group_weights_tensor buffer) into models without it.
    # Group weights are training-only and not needed for inference.
    model.load_state_dict(data["state_dict"], strict=False)
    return model


# ============================================================================
# Platform Function Implementations
# ============================================================================


def lightning_train_impl(
    X: EastArray,
    y: EastArray,
    config: EastStruct,
    masks: EastVariant | None,
    group_weights: EastVariant | None,
) -> EastStruct:
    """Train a Lightning model."""
    import warnings

    # Convert inputs
    X_np = east_matrix_to_numpy(X)
    y_np = east_matrix_to_numpy(y)

    n_samples = X_np.shape[0]
    input_dim = X_np.shape[1]
    output_dim = y_np.shape[1]

    # Parse architecture config
    arch = config.get("architecture")
    architecture_type = arch.type
    if architecture_type == "autoencoder":
        architecture_config = {
            "encoder_layers": [int(x) for x in arch.value.get("encoder_layers")],
            "latent_dim": int(arch.value.get("latent_dim")),
            "decoder_layers": [int(x) for x in arch.value.get("decoder_layers")],
        }
        latent_dim = architecture_config["latent_dim"]
    else:
        architecture_config = {
            "hidden_layers": [int(x) for x in arch.value.get("hidden_layers")],
        }
        latent_dim = None

    # Parse output config
    output = config.get("output")
    output_type = output.type
    if output_type == "binary":
        pos_weight = _get_option(output.value.get("pos_weight"), None)
        output_config = {
            "pos_weight": list(pos_weight) if pos_weight is not None else None,
        }
    elif output_type == "multiclass":
        class_weights = _get_option(output.value.get("class_weights"), None)
        output_config = {
            "n_classes": int(output.value.get("n_classes")),
            "class_weights": list(class_weights) if class_weights is not None else None,
        }
    elif output_type == "multi_head":
        class_weights = _get_option(output.value.get("class_weights"), None)
        output_config = {
            "n_heads": int(output.value.get("n_heads")),
            "n_classes_per_head": int(output.value.get("n_classes_per_head")),
            "class_weights": east_matrix_to_numpy(class_weights) if class_weights is not None else None,
        }
    else:
        output_config = {}

    # Parse and validate group_weights
    group_weights_for_model = None
    sample_groups_list = None

    if group_weights is not None and is_east_variant(group_weights) and group_weights.type == "some":
        gw_struct = group_weights.value
        weights_variant = gw_struct.get("weights")
        weights_type = weights_variant.type  # "binary" or "multi_head"
        weights_data = list(weights_variant.value)  # Convert to list

        # Validate: group_weights only supported for multi_head and binary
        if output_type not in ("multi_head", "binary"):
            raise ValueError("group_weights only supported for multi_head and binary output")

        # Validate: weights variant matches output type
        if weights_type != output_type:
            raise ValueError(
                f"group_weights variant '{weights_type}' does not match output type '{output_type}'"
            )

        # Convert weights to nested lists
        if weights_type == "binary":
            weights_data = [[float(v) for v in group] for group in weights_data]
            expected_dim = output_dim
            if len(weights_data[0]) != expected_dim:
                raise ValueError(
                    f"binary group_weights must have shape [n_groups][{expected_dim}], "
                    f"got [n_groups][{len(weights_data[0])}]"
                )
        else:  # multi_head
            weights_data = [[[float(v) for v in cls] for cls in head] for head in weights_data]
            n_heads = output_config["n_heads"]
            n_classes = output_config["n_classes_per_head"]
            if len(weights_data[0]) != n_heads or len(weights_data[0][0]) != n_classes:
                raise ValueError(
                    f"multi_head group_weights must have shape [n_groups][{n_heads}][{n_classes}]"
                )

        # Validate group indices
        sample_groups_list = [int(g) for g in gw_struct.get("sample_groups")]
        n_groups = len(weights_data)

        if len(sample_groups_list) != n_samples:
            raise ValueError(
                f"sample_groups length {len(sample_groups_list)} does not match X rows {n_samples}"
            )
        if len(sample_groups_list) == 0:
            raise ValueError("sample_groups cannot be empty")
        min_group = min(sample_groups_list)
        max_group = max(sample_groups_list)
        if min_group < 0:
            raise ValueError(f"sample_groups contains negative index {min_group}")
        if max_group >= n_groups:
            raise ValueError(
                f"sample_groups contains index {max_group} but only {n_groups} groups provided"
            )

        # Warn if both config weights and group_weights provided
        if output_type == "multi_head" and output_config.get("class_weights") is not None:
            warnings.warn("group_weights provided; ignoring config class_weights")
        elif output_type == "binary" and output_config.get("pos_weight") is not None:
            warnings.warn("group_weights provided; ignoring config pos_weight")

        group_weights_for_model = {
            "weights": weights_data,
            "weights_type": weights_type,
        }

    # Training params with defaults
    learning_rate = float(_get_option(config.get("learning_rate"), 1e-3))
    max_epochs = int(_get_option(config.get("max_epochs"), 100))
    patience = int(_get_option(config.get("patience"), 10))
    batch_size = int(_get_option(config.get("batch_size"), 32))
    dropout = float(_get_option(config.get("dropout"), 0.1))
    gradient_clip = float(_get_option(config.get("gradient_clip"), 1.0))
    weight_decay = float(_get_option(config.get("weight_decay"), 0.0))
    random_state = _get_option(config.get("random_state"), None)
    epoch_callback_fn = _get_option(config.get("epoch_callback"), None)

    if random_state is not None:
        pl.seed_everything(int(random_state), workers=True)

    # Create model
    model = LightningMLP(
        input_dim=input_dim,
        output_dim=output_dim,
        architecture_type=architecture_type,
        architecture_config=architecture_config,
        output_type=output_type,
        output_config=output_config,
        learning_rate=learning_rate,
        dropout=dropout,
        weight_decay=weight_decay,
        group_weights=group_weights_for_model,
    )

    # Prepare data
    X_tensor = torch.tensor(X_np, dtype=torch.float32)
    y_tensor = torch.tensor(y_np, dtype=torch.float32)

    # Handle masks
    masks_tensor = None
    if masks is not None and is_east_variant(masks) and masks.type == "some":
        masks_list = masks.value
        # Convert 3D masks to tensor
        masks_np = np.array([[[bool(v) for v in row] for row in sample] for sample in masks_list])
        masks_tensor = torch.tensor(masks_np, dtype=torch.bool)

    # Create dataset with group indices when group_weights is provided
    if group_weights_for_model is not None:
        sample_groups_tensor = torch.tensor(sample_groups_list, dtype=torch.long)

        # Create appropriate dummy masks if not provided
        if masks_tensor is None:
            if output_type == "multi_head":
                n_heads = output_config["n_heads"]
                n_classes = output_config["n_classes_per_head"]
                masks_tensor = torch.ones((n_samples, n_heads, n_classes), dtype=torch.bool)
            elif output_type == "binary":
                masks_tensor = torch.ones((n_samples, 1, output_dim), dtype=torch.bool)

        dataset = TensorDataset(X_tensor, y_tensor, masks_tensor, sample_groups_tensor)
    elif masks_tensor is not None:
        dataset = TensorDataset(X_tensor, y_tensor, masks_tensor)
    else:
        dataset = TensorDataset(X_tensor, y_tensor)

    # Split data - ensure at least 1 validation sample when possible
    val_size = max(1, int(0.1 * n_samples)) if n_samples >= 2 else 0
    train_size = n_samples - val_size

    generator = torch.Generator()
    if random_state is not None:
        generator.manual_seed(int(random_state))

    if val_size > 0:
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size], generator=generator
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        monitor_metric = "val_loss"
    else:
        train_dataset = dataset
        val_loader = None
        monitor_metric = "train_loss"

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Use temp directory for checkpoints (cleaned up after training)
    checkpoint_dir = tempfile.mkdtemp(prefix="lightning_ckpt_")
    try:
        # Train
        callbacks = [
            pl.callbacks.EarlyStopping(monitor=monitor_metric, patience=patience, mode="min"),
            pl.callbacks.ModelCheckpoint(
                dirpath=checkpoint_dir,
                monitor=monitor_metric,
                mode="min",
                save_top_k=1,
            ),
        ]

        # Add user epoch callback if provided
        if epoch_callback_fn is not None:
            callbacks.append(EpochCallback(epoch_callback_fn))

        trainer = pl.Trainer(
            max_epochs=max_epochs,
            callbacks=callbacks,
            gradient_clip_val=gradient_clip,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            accelerator="auto",
            deterministic=random_state is not None,
        )

        trainer.fit(model, train_loader, val_loader)

        # Get best model
        best_model_path = trainer.checkpoint_callback.best_model_path
        if best_model_path:
            # weights_only=False needed for PyTorch 2.6+ (we trust our own checkpoints)
            model = LightningMLP.load_from_checkpoint(best_model_path, weights_only=False)
    finally:
        # Clean up temp checkpoint directory
        shutil.rmtree(checkpoint_dir, ignore_errors=True)

    # Get final metrics
    train_loss = float(trainer.callback_metrics.get("train_loss", 0.0))
    val_loss = float(trainer.callback_metrics.get("val_loss", 0.0))
    best_epoch = trainer.current_epoch

    # Serialize model
    model_blob = EastBlob(_serialize_model(model))

    # Create result with model blob variant
    result_model = EastVariant(
        "lightning",
        EastStruct(
            {
                "data": model_blob,
                "n_features": input_dim,
                "output_dim": output_dim,
                "architecture_type": architecture_type,
                "output_type": output_type,
                "latent_dim": EastVariant("some", latent_dim) if latent_dim else EastVariant("none", None),
            }
        ),
    )

    return EastStruct(
        {
            "model": result_model,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_epoch": best_epoch,
        }
    )


def lightning_predict_impl(
    model_blob: EastVariant,
    X: EastArray,
    masks: EastVariant | None,
) -> EastArray:
    """Predict using a Lightning model."""
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

    # Predict
    with torch.no_grad():
        probs = model.predict_probs_with_masks(X_tensor, masks_tensor).numpy()

    return numpy_to_east_matrix(probs)


def lightning_encode_impl(
    model_blob: EastVariant,
    X: EastArray,
) -> EastArray:
    """Encode input to latent space (autoencoder only)."""
    model_data = model_blob.value
    model_bytes = bytes(model_data.get("data"))
    model = _deserialize_model(model_bytes)
    model.eval()

    if model.architecture_type != "autoencoder":
        raise ValueError("encode() only available for autoencoder architecture")

    X_np = east_matrix_to_numpy(X)
    X_tensor = torch.tensor(X_np, dtype=torch.float32)

    with torch.no_grad():
        embeddings = model.encode(X_tensor).numpy()

    return numpy_to_east_matrix(embeddings)


def lightning_decode_impl(
    model_blob: EastVariant,
    z: EastArray,
) -> EastArray:
    """Decode latent to output (autoencoder only)."""
    model_data = model_blob.value
    model_bytes = bytes(model_data.get("data"))
    model = _deserialize_model(model_bytes)
    model.eval()

    if model.architecture_type != "autoencoder":
        raise ValueError("decode() only available for autoencoder architecture")

    z_np = east_matrix_to_numpy(z)
    z_tensor = torch.tensor(z_np, dtype=torch.float32)

    with torch.no_grad():
        logits = model.decode(z_tensor)
        probs = model.apply_output_activation(logits)
        output = probs.numpy()

    return numpy_to_east_matrix(output)


# ============================================================================
# Platform Function Registration
# ============================================================================

# 3D tensor type for masks
Tensor3DType = EastArray  # ArrayType(ArrayType(ArrayType(BooleanType)))

lightning_impl = [
    PlatformFunction(
        name="lightning_train",
        inputs=[MatrixType, MatrixType, LightningConfigType, Tensor3DType, GroupWeightsType],
        output=LightningResultType,
        type="sync",
        fn=lightning_train_impl,
    ),
    PlatformFunction(
        name="lightning_predict",
        inputs=[ModelBlobType, MatrixType, Tensor3DType],
        output=MatrixType,
        type="sync",
        fn=lightning_predict_impl,
    ),
    PlatformFunction(
        name="lightning_encode",
        inputs=[ModelBlobType, MatrixType],
        output=MatrixType,
        type="sync",
        fn=lightning_encode_impl,
    ),
    PlatformFunction(
        name="lightning_decode",
        inputs=[ModelBlobType, MatrixType],
        output=MatrixType,
        type="sync",
        fn=lightning_decode_impl,
    ),
]
