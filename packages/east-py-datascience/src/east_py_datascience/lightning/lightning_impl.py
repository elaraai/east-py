#
# Copyright (c) 2025 Elara AI Pty Ltd
# Licensed under the Business Source License 1.1. See LICENSE.md for details.
#
"""Lightning-based neural network platform functions for East Data Science.

Provides a production-grade neural network training module using PyTorch Lightning.
Supports regression, binary classification, multiclass classification, and
multi-head categorical outputs.
"""

import logging
import os
import pickle
import tempfile
import shutil
import warnings
from typing import Callable

# Suppress PyTorch Lightning logging - rely on exceptions for errors
os.environ["PYTORCH_LIGHTNING_DISABLE_POSSIBLE_USER_WARNINGS"] = "1"
os.environ["LT_DISABLE_STATUS_BAR"] = "1"
warnings.filterwarnings("ignore", module="torch")
warnings.filterwarnings("ignore", module="pytorch_lightning")
warnings.filterwarnings("ignore", module="lightning")

import numpy as np  # noqa: E402
import pytorch_lightning as pl  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset, random_split  # noqa: E402

# Suppress loggers AFTER importing - loggers are created during import
logging.getLogger("pytorch_lightning.utilities.rank_zero").setLevel(logging.CRITICAL)
logging.getLogger("lightning_fabric.utilities.seed").setLevel(logging.CRITICAL)
logging.getLogger("pytorch_lightning").setLevel(logging.CRITICAL)
logging.getLogger("lightning").setLevel(logging.CRITICAL)
logging.getLogger("lightning_fabric").setLevel(logging.CRITICAL)

from east.runtime.platform import PlatformFunction  # noqa: E402
from east.types.values import EastArray, EastBlob, EastStruct, EastVariant, is_east_variant  # noqa: E402


from east_py_datascience.types import (  # noqa: E402
    MatrixType,
    LightningConfigType,
    LightningResultType,
    LightningGenerateConfigType,
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
# Temporal Architecture Classes
# ============================================================================


class Conv1DAutoencoder(nn.Module):
    """1D Convolutional autoencoder for temporal patterns."""

    def __init__(
        self,
        n_channels: int,
        sequence_length: int,
        n_classes: int,
        conv_channels: list[int],
        kernel_size: int,
        latent_dim: int,
        condition_dim: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.sequence_length = sequence_length
        self.n_classes = n_classes
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim

        # Input: [batch, n_channels, sequence_length, n_classes]
        # Reshape to: [batch, n_channels * n_classes, sequence_length]
        in_channels = n_channels * n_classes

        # Encoder: conv layers over sequence dimension
        encoder_layers = []
        prev_channels = in_channels
        for out_channels in conv_channels:
            encoder_layers.extend([
                nn.Conv1d(prev_channels, out_channels, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_channels = out_channels

        self.encoder_conv = nn.Sequential(*encoder_layers)

        # Flatten and project to latent
        encoder_output_size = conv_channels[-1] * sequence_length
        self.encoder_fc = nn.Linear(encoder_output_size, latent_dim)

        # Decoder: project and reshape
        # Decoder input: latent + condition (if provided)
        decoder_input_dim = latent_dim + (condition_dim or 0)
        self.decoder_fc = nn.Linear(decoder_input_dim, encoder_output_size)

        # Decoder: transposed conv layers (mirror of encoder)
        decoder_layers = []
        prev_channels = conv_channels[-1]  # Start from encoder's final output
        for out_channels in reversed(conv_channels[:-1]):
            decoder_layers.extend([
                nn.ConvTranspose1d(prev_channels, out_channels, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_channels = out_channels

        # Final layer to original channels
        decoder_layers.append(
            nn.ConvTranspose1d(prev_channels, in_channels, kernel_size, padding=kernel_size // 2)
        )
        self.decoder_conv = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode to latent space."""
        batch_size = x.shape[0]
        # x: [batch, n_channels * sequence_length * n_classes]
        x = x.view(batch_size, self.n_channels, self.sequence_length, self.n_classes)
        # -> [batch, n_channels * n_classes, sequence_length]
        x = x.permute(0, 1, 3, 2).reshape(batch_size, -1, self.sequence_length)
        x = self.encoder_conv(x)
        x = x.flatten(1)
        return self.encoder_fc(x)

    def decode(self, z: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
        """Decode from latent space."""
        if self.condition_dim is not None:
            if condition is None:
                raise ValueError("Model requires condition vector but none provided")
            if condition.shape[1] != self.condition_dim:
                raise ValueError(f"Expected condition_dim={self.condition_dim}, got {condition.shape[1]}")
            z = torch.cat([z, condition], dim=1)
        elif condition is not None:
            raise ValueError("Model has no condition_dim but condition was provided")

        batch_size = z.shape[0]
        x = self.decoder_fc(z)
        x = x.view(batch_size, -1, self.sequence_length)
        x = self.decoder_conv(x)
        # -> [batch, n_channels, n_classes, sequence_length]
        x = x.view(batch_size, self.n_channels, self.n_classes, self.sequence_length)
        # -> [batch, n_channels, sequence_length, n_classes]
        x = x.permute(0, 1, 3, 2)
        # -> [batch, n_channels * sequence_length * n_classes]
        return x.reshape(batch_size, -1)

    def forward(self, x: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
        return self.decode(self.encode(x), condition)


class SequentialAutoencoder(nn.Module):
    """LSTM/GRU autoencoder for sequential dependencies."""

    def __init__(
        self,
        n_channels: int,
        sequence_length: int,
        n_classes: int,
        hidden_size: int,
        n_layers: int,
        cell_type: str,  # "lstm" or "gru" (from variant tag)
        latent_dim: int,
        bidirectional: bool = False,
        condition_dim: int | None = None,
        dropout: float = 0.1,
    ):
        if cell_type not in ("lstm", "gru"):
            raise ValueError(f"cell_type must be 'lstm' or 'gru', got '{cell_type}'")
        super().__init__()
        self.n_channels = n_channels
        self.sequence_length = sequence_length
        self.n_classes = n_classes
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.bidirectional = bidirectional
        self.cell_type = cell_type
        self.condition_dim = condition_dim
        self.n_layers = n_layers

        input_size = n_channels * n_classes
        num_directions = 2 if bidirectional else 1

        RNNClass = nn.LSTM if cell_type == "lstm" else nn.GRU
        self.encoder_rnn = RNNClass(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if n_layers > 1 else 0,
        )

        # Project final hidden state to latent
        encoder_output_size = hidden_size * num_directions * n_layers
        self.encoder_fc = nn.Linear(encoder_output_size, latent_dim)

        # Decoder: always unidirectional (bidirectional requires full sequence upfront)
        # Project latent to decoder's initial hidden state
        # Decoder input: latent + condition (if provided)
        decoder_input_dim = latent_dim + (condition_dim or 0)
        decoder_hidden_size = hidden_size * n_layers  # unidirectional
        self.decoder_fc = nn.Linear(decoder_input_dim, decoder_hidden_size)

        # Decoder input size includes condition_dim for autoregressive generation
        # (condition is concatenated to input at each timestep)
        decoder_rnn_input_size = input_size + (condition_dim or 0)
        self.decoder_rnn = RNNClass(
            input_size=decoder_rnn_input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=False,  # Always unidirectional for generation
            dropout=dropout if n_layers > 1 else 0,
        )

        # Output projection (decoder is always unidirectional)
        self.output_fc = nn.Linear(hidden_size, input_size)

    def init_hidden(self, batch_size: int, device: torch.device | None = None) -> torch.Tensor | tuple:
        """Initialize hidden state for autoregressive generation.

        Args:
            batch_size: Number of sequences to generate
            device: Device to create tensors on

        Returns:
            For LSTM: (h_0, c_0) tuple
            For GRU: h_0 tensor
        """
        if device is None:
            device = next(self.parameters()).device
        zeros = torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device)
        if self.cell_type == "lstm":
            return (zeros, zeros.clone())
        return zeros

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode sequence to latent."""
        batch_size = x.shape[0]
        # x: [batch, n_channels * sequence_length * n_classes]
        x = x.view(batch_size, self.n_channels, self.sequence_length, self.n_classes)
        # -> [batch, sequence_length, n_channels * n_classes]
        x = x.permute(0, 2, 1, 3).reshape(batch_size, self.sequence_length, -1)

        _, hidden = self.encoder_rnn(x)
        if self.cell_type == "lstm":
            hidden = hidden[0]  # Take h, not c
        # hidden: [n_layers * num_directions, batch, hidden_size]
        hidden = hidden.permute(1, 0, 2).reshape(batch_size, -1)
        return self.encoder_fc(hidden)

    def decode(self, z: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
        """Decode latent to sequence (parallel, not autoregressive)."""
        if self.condition_dim is not None:
            if condition is None:
                raise ValueError("Model requires condition vector but none provided")
            if condition.shape[1] != self.condition_dim:
                raise ValueError(f"Expected condition_dim={self.condition_dim}, got {condition.shape[1]}")
            z = torch.cat([z, condition], dim=1)
        elif condition is not None:
            raise ValueError("Model has no condition_dim but condition was provided")

        batch_size = z.shape[0]
        hidden = self.decoder_fc(z)
        # Reshape to RNN hidden state format (decoder is always unidirectional)
        hidden = hidden.view(batch_size, self.n_layers, self.hidden_size)
        hidden = hidden.permute(1, 0, 2).contiguous()

        if self.cell_type == "lstm":
            hidden = (hidden, torch.zeros_like(hidden))

        # Parallel decoding: use zeros as input, hidden state provides context
        # This is NOT autoregressive - all positions decoded simultaneously
        decoder_input = torch.zeros(
            batch_size, self.sequence_length, self.n_channels * self.n_classes,
            device=z.device
        )
        # If using conditions, concatenate condition to each timestep
        if self.condition_dim is not None and condition is not None:
            # condition: (batch, condition_dim) -> (batch, sequence_length, condition_dim)
            cond_expanded = condition.unsqueeze(1).expand(-1, self.sequence_length, -1)
            decoder_input = torch.cat([decoder_input, cond_expanded], dim=-1)
        output, _ = self.decoder_rnn(decoder_input, hidden)
        output = self.output_fc(output)

        # -> [batch, sequence_length, n_channels, n_classes]
        output = output.view(batch_size, self.sequence_length, self.n_channels, self.n_classes)
        # -> [batch, n_channels, sequence_length, n_classes]
        output = output.permute(0, 2, 1, 3)
        return output.reshape(batch_size, -1)

    def forward(self, x: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
        return self.decode(self.encode(x), condition)


class TransformerAutoencoder(nn.Module):
    """Transformer autoencoder with positional encoding.

    Note: The decoder uses self-attention with the decoded sequence serving as both
    query and memory. This is intentional - at decode time we don't have encoder
    outputs to cross-attend to (unlike seq2seq), so we rely on the latent projection
    and positional encoding to provide the necessary context.
    """

    def __init__(
        self,
        n_channels: int,
        sequence_length: int,
        n_classes: int,
        d_model: int,
        n_attention_heads: int,
        n_layers: int,
        d_ff: int | None,
        latent_dim: int,
        condition_dim: int | None = None,
        dropout: float = 0.1,
    ):
        if d_model % n_attention_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_attention_heads ({n_attention_heads})"
            )
        super().__init__()
        self.n_channels = n_channels
        self.sequence_length = sequence_length
        self.n_classes = n_classes
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim

        input_size = n_channels * n_classes
        d_ff = d_ff or (4 * d_model)

        # Input projection
        self.input_proj = nn.Linear(input_size, d_model)

        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, sequence_length, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_attention_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Latent projection (mean pool over sequence)
        self.encoder_fc = nn.Linear(d_model, latent_dim)

        # Decoder projection
        # Decoder input: latent + condition (if provided)
        decoder_input_dim = latent_dim + (condition_dim or 0)
        self.decoder_fc = nn.Linear(decoder_input_dim, d_model * sequence_length)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_attention_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # Output projection
        self.output_proj = nn.Linear(d_model, input_size)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode to latent space via attention."""
        batch_size = x.shape[0]
        # x: [batch, n_channels * sequence_length * n_classes]
        x = x.view(batch_size, self.n_channels, self.sequence_length, self.n_classes)
        # -> [batch, sequence_length, n_channels * n_classes]
        x = x.permute(0, 2, 1, 3).reshape(batch_size, self.sequence_length, -1)

        x = self.input_proj(x) + self.pos_encoding
        x = self.transformer_encoder(x)
        # Mean pool over sequence
        x = x.mean(dim=1)
        return self.encoder_fc(x)

    def decode(self, z: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
        """Decode from latent via attention."""
        if self.condition_dim is not None:
            if condition is None:
                raise ValueError("Model requires condition vector but none provided")
            if condition.shape[1] != self.condition_dim:
                raise ValueError(f"Expected condition_dim={self.condition_dim}, got {condition.shape[1]}")
            z = torch.cat([z, condition], dim=1)
        elif condition is not None:
            raise ValueError("Model has no condition_dim but condition was provided")

        batch_size = z.shape[0]
        # Project and reshape to sequence
        x = self.decoder_fc(z)
        x = x.view(batch_size, self.sequence_length, self.d_model)

        # Use positional encoding as memory for cross-attention
        memory = x + self.pos_encoding
        x = self.transformer_decoder(x + self.pos_encoding, memory)
        x = self.output_proj(x)

        # -> [batch, sequence_length, n_channels, n_classes]
        x = x.view(batch_size, self.sequence_length, self.n_channels, self.n_classes)
        # -> [batch, n_channels, sequence_length, n_classes]
        x = x.permute(0, 2, 1, 3)
        return x.reshape(batch_size, -1)

    def forward(self, x: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
        return self.decode(self.encode(x), condition)


# ============================================================================
# Decision Transformer
# ============================================================================


class DecisionTransformerNet(nn.Module):
    """
    Decision Transformer for return-conditioned action prediction.

    Architecture (global return mode):
    1. Token sequence: [R, s_0, a_0, s_1, a_1, ..., s_{T-1}, a_{T-1}]
    2. Total tokens: 1 + 2*T (single return token + T state-action pairs)
    3. Apply causal transformer
    4. Extract action predictions from state token positions

    When predicting a_t from position s_t:
    - Can see: R, s_0, a_0, ..., s_{t-1}, a_{t-1}, s_t
    - Cannot see: a_t, s_{t+1}, ... (causally masked)
    """

    def __init__(
        self,
        sequence_length: int,
        state_dim: int,
        action_dim: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int | None = None,
        dropout: float = 0.1,
        return_embedding: str = "global",  # "global" or "per_timestep"
    ):
        super().__init__()
        self.sequence_length = sequence_length
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.d_model = d_model
        self.return_embedding = return_embedding

        d_ff = d_ff if d_ff is not None else 4 * d_model

        # Embedding layers
        self.return_embed = nn.Linear(1, d_model)
        self.state_embed = nn.Linear(state_dim, d_model)
        self.action_embed = nn.Linear(action_dim, d_model)

        # Token count: 1 (return) + 2*T (state, action pairs) for global mode
        # Or 3*T for per_timestep mode
        n_tokens = 1 + 2 * sequence_length if return_embedding == "global" else 3 * sequence_length
        self.pos_embed = nn.Embedding(n_tokens, d_model)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Action prediction head
        self.action_head = nn.Linear(d_model, action_dim)

        # Causal mask
        self._register_causal_mask(n_tokens)

    def _register_causal_mask(self, size: int):
        """
        Register causal attention mask.

        PyTorch nn.TransformerEncoder expects mask where True = block attention.
        This creates upper triangular mask: position i can only attend to positions <= i.
        """
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)

    def forward(
        self,
        returns: torch.Tensor,      # (batch, 1) for global → embedded to (batch, 1, d_model)
        states: torch.Tensor,       # (batch, seq_len, state_dim)
        actions: torch.Tensor,      # (batch, seq_len, action_dim) - ground truth, NOT shifted
        temporal_mask: torch.Tensor | None = None,  # (batch, seq_len) - valid timesteps
    ) -> torch.Tensor:
        """
        Forward pass with correct causal token layout.

        Token layout (global mode): [R, s_0, a_0, s_1, a_1, ..., s_{T-1}, a_{T-1}]
        Token count: 1 + 2*T

        When predicting a_t from state position s_t:
        - Can see: R, s_0, a_0, s_1, a_1, ..., s_{t-1}, a_{t-1}, s_t
        - Cannot see: a_t, s_{t+1}, ... (future positions, causally masked)

        The model sees all previous actions when predicting current action.
        """
        batch_size = states.shape[0]
        seq_len = states.shape[1]
        device = states.device

        # Validate input shapes
        if states.shape[-1] != self.state_dim:
            raise ValueError(
                f"Expected state_dim={self.state_dim}, got {states.shape[-1]}"
            )
        if actions.shape[-1] != self.action_dim:
            raise ValueError(
                f"Expected action_dim={self.action_dim}, got {actions.shape[-1]}"
            )
        if seq_len != self.sequence_length:
            raise ValueError(
                f"Expected sequence_length={self.sequence_length}, got {seq_len}"
            )
        if self.return_embedding == "global":
            if returns.shape != (batch_size, 1):
                raise ValueError(
                    f"Global mode expects returns shape (batch, 1), got {returns.shape}"
                )
        else:
            if returns.shape != (batch_size, seq_len, 1):
                raise ValueError(
                    f"Per-timestep mode expects returns shape (batch, seq_len, 1), "
                    f"got {returns.shape}"
                )

        # Embed inputs
        state_emb = self.state_embed(states)     # (batch, seq_len, d_model)
        action_emb = self.action_embed(actions)  # (batch, seq_len, d_model)

        if self.return_embedding == "global":
            # Single return token at start
            ret_emb = self.return_embed(returns)  # (batch, d_model)
            ret_emb = ret_emb.unsqueeze(1)        # (batch, 1, d_model)

            # Interleave states and actions: [s_0, a_0, s_1, a_1, ...]
            sa_interleaved = torch.stack([state_emb, action_emb], dim=2)  # (batch, T, 2, d_model)
            sa_interleaved = sa_interleaved.reshape(batch_size, 2 * seq_len, self.d_model)

            # Concatenate: [R] + [s_0, a_0, s_1, a_1, ...]
            tokens = torch.cat([ret_emb, sa_interleaved], dim=1)  # (batch, 1+2T, d_model)
            n_tokens = 1 + 2 * seq_len

            # State positions: 1, 3, 5, ... (after return token, then every 2)
            state_positions = torch.arange(1, n_tokens, 2, device=device)
        else:
            # Per-timestep return: [R_0, s_0, a_0, R_1, s_1, a_1, ...]
            ret_emb = self.return_embed(returns)  # (batch, seq_len, d_model)
            tokens = torch.stack([ret_emb, state_emb, action_emb], dim=2)
            tokens = tokens.reshape(batch_size, 3 * seq_len, self.d_model)
            n_tokens = 3 * seq_len

            # State positions: 1, 4, 7, ...
            state_positions = torch.arange(1, n_tokens, 3, device=device)

        # Add positional embeddings
        positions = torch.arange(n_tokens, device=device)
        tokens = tokens + self.pos_embed(positions)

        # Apply transformer with causal mask
        hidden = self.transformer(tokens, mask=self.causal_mask[:n_tokens, :n_tokens])

        # Extract hidden states at STATE positions for action prediction
        # When at s_t, causal mask allows seeing [R, s_0, a_0, ..., s_{t-1}, a_{t-1}, s_t]
        state_hidden = hidden[:, state_positions, :]  # (batch, seq_len, d_model)

        # Predict actions
        action_preds = self.action_head(state_hidden)  # (batch, seq_len, action_dim)

        return action_preds

    def generate(
        self,
        returns: torch.Tensor,              # (batch, 1) - target return
        states: torch.Tensor,               # (batch, seq_len, state_dim)
        temperature: float = 0.0,
        constraints: torch.Tensor | None = None,  # (seq_len, action_dim) - FALSE disables
        temporal_mask: torch.Tensor | None = None,  # (seq_len,) - valid timesteps
        head_configs: list[dict] | None = None,  # head type info for multi_head_mixed
    ) -> torch.Tensor:
        """
        Autoregressive action generation.

        Complexity: O(T²) due to full forward pass per step. Acceptable for T < 50.
        For longer sequences, implement KV-caching to reduce to O(T).

        For each timestep t:
        1. Build action history with generated actions a_0, ..., a_{t-1}
        2. Forward pass (model sees R, s_0, a_0, ..., s_{t-1}, a_{t-1}, s_t)
        3. Apply constraints if provided
        4. Sample or argmax based on temperature

        Args:
            head_configs: List of head configurations for multi_head_mixed output.
                Each dict has 'head_type' (variant with 'binary' or 'multiclass').
                Binary heads use 1 logit, multiclass heads use n_classes logits.
                If None, all outputs are treated as independent binary (1 logit each).

        Note:
            - Only global return embedding is supported for generation.
            - Batch generation requires all samples to have the same valid sequence length.
            - Future timestep positions embed zeros through action_embed.
        """
        if self.return_embedding != "global":
            raise NotImplementedError("Generation only supports global return embedding")

        batch_size = states.shape[0]
        seq_len = states.shape[1]
        device = states.device

        generated_actions = torch.zeros(batch_size, seq_len, self.action_dim, device=device)

        valid_len = seq_len
        if temporal_mask is not None:
            valid_len = int(temporal_mask.sum().item())

        for t in range(valid_len):
            # Build action history with generated actions so far
            action_history = torch.zeros(batch_size, seq_len, self.action_dim, device=device)
            action_history[:, :t, :] = generated_actions[:, :t, :]

            # Forward pass
            action_preds = self.forward(returns, states, action_history)

            # Get prediction for timestep t
            logits_t = action_preds[:, t, :]  # (batch, action_dim)

            # Apply constraints if provided
            if constraints is not None:
                logits_t = logits_t.masked_fill(~constraints[t], float('-inf'))

            # Sample based on head types
            if head_configs is None:
                # Default: treat all outputs as independent binary (1 logit each)
                if temperature > 0:
                    probs = torch.sigmoid(logits_t / temperature)
                    sampled = torch.bernoulli(probs)
                else:
                    sampled = (torch.sigmoid(logits_t) > 0.5).float()
            else:
                # Handle each head according to its type
                sampled = torch.zeros_like(logits_t)
                idx = 0
                for head in head_configs:
                    head_type = head['head_type']

                    if head_type['type'] == 'binary':
                        # Binary: single logit
                        head_logits = logits_t[:, idx:idx+1]
                        if temperature > 0:
                            probs = torch.sigmoid(head_logits / temperature)
                            sampled[:, idx:idx+1] = torch.bernoulli(probs)
                        else:
                            sampled[:, idx:idx+1] = (torch.sigmoid(head_logits) > 0.5).float()
                        idx += 1
                    else:  # multiclass
                        n_classes = head_type['multiclass']['n_classes']
                        head_logits = logits_t[:, idx:idx+n_classes]

                        # Validate constraints don't mask all classes (would cause NaN)
                        if constraints is not None:
                            head_constraints = constraints[t, idx:idx+n_classes]
                            if not head_constraints.any():
                                raise ValueError(
                                    f"All classes masked for multiclass head at timestep {t}. "
                                    "At least one class must be allowed."
                                )

                        if temperature > 0:
                            probs = F.softmax(head_logits / temperature, dim=-1)
                            choices = torch.multinomial(probs, 1).squeeze(-1)
                        else:
                            choices = head_logits.argmax(dim=-1)
                        # One-hot encode the choice
                        sampled[:, idx:idx+n_classes] = F.one_hot(choices, n_classes).float()
                        idx += n_classes

            generated_actions[:, t, :] = sampled

        return generated_actions


def head_to_logit_index(head_configs: list[dict], head_idx: int) -> int:
    """
    Convert head index to starting logit index.

    Head indices are 0-based (head 0, head 1, ...).
    Logit indices depend on head types: binary=1, multiclass=n_classes.

    Example:
        Head 0: binary (1 logit)     → logit index 0
        Head 1: binary (1 logit)     → logit index 1
        Head 2: multiclass(4)        → logit index 2
        Head 3: binary (1 logit)     → logit index 6
    """
    idx = 0
    for i, head in enumerate(head_configs):
        if i == head_idx:
            return idx
        head_type = head['head_type']
        if head_type['type'] == 'binary':
            idx += 1
        else:
            idx += head_type['multiclass']['n_classes']
    raise ValueError(f"Invalid head index: {head_idx}")


def compute_multi_head_mixed_loss(
    preds: torch.Tensor,      # (batch, seq_len, action_dim)
    targets: torch.Tensor,    # (batch, seq_len, action_dim)
    head_configs: list[dict],
    masks: torch.Tensor,      # (batch, seq_len)
    validate_onehot: bool = False,  # Enable in debug mode
) -> torch.Tensor:
    """
    Compute loss for multi_head_mixed output.

    Binary heads: BCE with logits (1 logit per head)
    Multiclass heads: Cross-entropy (targets are one-hot, converted to indices)
    Conditional heads: Loss only computed when referenced head's target is 1

    Note: conditional_on is a HEAD INDEX, not a logit index. The function
    converts to logit index internally using head_to_logit_index().
    """
    batch_size, seq_len, _ = preds.shape
    total_loss = torch.zeros(batch_size, seq_len, device=preds.device)

    idx = 0
    for head in head_configs:
        head_type = head['head_type']
        conditional_on = head.get('conditional_on')

        if head_type['type'] == 'binary':
            # Single logit, BCE loss
            head_pred = preds[:, :, idx]           # (batch, seq_len)
            head_target = targets[:, :, idx]       # (batch, seq_len)
            head_loss = F.binary_cross_entropy_with_logits(
                head_pred, head_target, reduction='none'
            )
            idx += 1
        else:  # multiclass
            n_classes = head_type['multiclass']['n_classes']
            head_pred = preds[:, :, idx:idx+n_classes]      # (batch, seq_len, n_classes)
            head_target = targets[:, :, idx:idx+n_classes]  # (batch, seq_len, n_classes) one-hot

            # Validate one-hot encoding if requested
            if validate_onehot:
                sums = head_target.sum(dim=-1)
                valid = ((sums == 1) | (sums == 0)).all()
                if not valid:
                    raise ValueError("Multiclass targets must be one-hot encoded")

            # Convert one-hot to indices
            head_target_idx = head_target.argmax(dim=-1)    # (batch, seq_len)

            # Reshape for cross_entropy
            head_loss = F.cross_entropy(
                head_pred.reshape(-1, n_classes),
                head_target_idx.reshape(-1),
                reduction='none'
            ).reshape(batch_size, seq_len)

            idx += n_classes

        # Apply conditional mask if specified (using HEAD index, converted to logit index)
        if conditional_on is not None:
            # Validate that conditional_on references a binary head
            ref_head = head_configs[conditional_on]
            if ref_head['head_type']['type'] != 'binary':
                raise ValueError(
                    f"conditional_on must reference a binary head, "
                    f"but head {conditional_on} is {ref_head['head_type']['type']}"
                )
            logit_idx = head_to_logit_index(head_configs, conditional_on)
            occurrence_mask = targets[:, :, logit_idx] == 1  # (batch, seq_len)
            head_loss = head_loss * occurrence_mask

        total_loss += head_loss

    # Apply temporal mask
    masked_loss = total_loss * masks
    n_valid = masks.sum()
    if n_valid > 0:
        loss = masked_loss.sum() / n_valid
    else:
        loss = masked_loss.sum()

    return loss


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
        self.architecture_config = architecture_config
        self.output_type = output_type
        self.output_config = output_config
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Extract condition_dim for temporal architectures
        self.condition_dim = architecture_config.get("condition_dim")

        # Validate temporal architecture config
        self._validate_temporal_config()

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
        elif architecture_type == "conv1d":
            n_classes = self._get_n_classes()
            self.net = Conv1DAutoencoder(
                n_channels=architecture_config["n_channels"],
                sequence_length=architecture_config["sequence_length"],
                n_classes=n_classes,
                conv_channels=architecture_config["conv_channels"],
                kernel_size=architecture_config["kernel_size"],
                latent_dim=architecture_config["latent_dim"],
                condition_dim=self.condition_dim,
                dropout=dropout,
            )
            self.latent_dim = architecture_config["latent_dim"]
        elif architecture_type == "sequential":
            n_classes = self._get_n_classes()
            cell_type = architecture_config["cell_type"]
            self.net = SequentialAutoencoder(
                n_channels=architecture_config["n_channels"],
                sequence_length=architecture_config["sequence_length"],
                n_classes=n_classes,
                hidden_size=architecture_config["hidden_size"],
                n_layers=architecture_config["n_layers"],
                cell_type=cell_type,
                latent_dim=architecture_config["latent_dim"],
                bidirectional=architecture_config.get("bidirectional", False),
                condition_dim=self.condition_dim,
                dropout=dropout,
            )
            self.latent_dim = architecture_config["latent_dim"]
        elif architecture_type == "transformer":
            n_classes = self._get_n_classes()
            self.net = TransformerAutoencoder(
                n_channels=architecture_config["n_channels"],
                sequence_length=architecture_config["sequence_length"],
                n_classes=n_classes,
                d_model=architecture_config["d_model"],
                n_attention_heads=architecture_config["n_attention_heads"],
                n_layers=architecture_config["n_layers"],
                d_ff=architecture_config.get("d_ff"),
                latent_dim=architecture_config["latent_dim"],
                condition_dim=self.condition_dim,
                dropout=dropout,
            )
            self.latent_dim = architecture_config["latent_dim"]
        else:  # mlp
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

    def _get_n_classes(self) -> int:
        """Get number of classes per position for temporal architectures."""
        if self.output_type == "multi_head":
            return self.output_config["n_classes_per_head"]
        elif self.output_type == "binary":
            return 1
        elif self.output_type == "multiclass":
            return self.output_config["n_classes"]
        else:  # regression
            return 1

    def _validate_temporal_config(self):
        """Validate temporal architecture config matches output config."""
        if self.architecture_type not in ("conv1d", "sequential", "transformer"):
            return

        # Temporal architectures require multi_head or binary output
        if self.output_type not in ("multi_head", "binary"):
            raise ValueError(
                f"Temporal architecture '{self.architecture_type}' requires multi_head or binary output, "
                f"got '{self.output_type}'"
            )

        # Binary output: no additional validation needed here
        # (output_dim validation happens in forward pass)
        if self.output_type == "binary":
            return

        # n_heads must equal n_channels * sequence_length
        n_channels = self.architecture_config["n_channels"]
        sequence_length = self.architecture_config["sequence_length"]
        expected_heads = n_channels * sequence_length
        actual_heads = self.output_config["n_heads"]

        if actual_heads != expected_heads:
            raise ValueError(
                f"n_heads ({actual_heads}) must equal n_channels * sequence_length "
                f"({n_channels} * {sequence_length} = {expected_heads})"
            )

        # Conv1D-specific validation
        if self.architecture_type == "conv1d":
            conv_channels = self.architecture_config["conv_channels"]
            if not conv_channels:
                raise ValueError("conv_channels must not be empty")

            kernel_size = self.architecture_config["kernel_size"]
            if kernel_size < 1:
                raise ValueError(f"kernel_size must be >= 1, got {kernel_size}")
            if kernel_size % 2 == 0:
                raise ValueError(f"kernel_size must be odd for symmetric padding, got {kernel_size}")

        # Transformer-specific validation
        if self.architecture_type == "transformer":
            d_model = self.architecture_config["d_model"]
            n_attention_heads = self.architecture_config["n_attention_heads"]
            if d_model % n_attention_heads != 0:
                raise ValueError(
                    f"d_model ({d_model}) must be divisible by n_attention_heads ({n_attention_heads})"
                )

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

    def forward(self, x: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass - returns raw logits."""
        if self.architecture_type == "autoencoder":
            return self.decoder(self.encoder(x))
        elif self.architecture_type in ("conv1d", "sequential", "transformer"):
            return self.net(x, condition)
        return self.net(x)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space (autoencoder and temporal architectures)."""
        if self.architecture_type == "autoencoder":
            return self.encoder(x)
        elif self.architecture_type in ("conv1d", "sequential", "transformer"):
            return self.net.encode(x)
        raise ValueError(f"encode() not available for {self.architecture_type} architecture")

    def decode(self, z: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
        """Decode latent to output (autoencoder and temporal architectures)."""
        if self.architecture_type == "autoencoder":
            return self.decoder(z)
        elif self.architecture_type in ("conv1d", "sequential", "transformer"):
            return self.net.decode(z, condition)
        raise ValueError(f"decode() not available for {self.architecture_type} architecture")

    def training_step(self, batch, batch_idx):
        # Batch structure depends on features used:
        # Base: (x, y)
        # With masks: (x, y, masks)
        # With group_weights: (x, y, masks, group_idx)
        # With conditions: adds condition as last element
        x, y, masks, group_idx, condition = self._unpack_batch(batch)

        logits = self(x, condition)
        loss = self._compute_loss(logits, y, masks, group_idx)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, masks, group_idx, condition = self._unpack_batch(batch)

        logits = self(x, condition)
        loss = self._compute_loss(logits, y, masks, group_idx)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def _unpack_batch(self, batch):
        """Unpack batch based on what features are present."""
        # Batch structure:
        # (x, y) - basic
        # (x, y, masks) - with masks
        # (x, y, masks, group_idx) - with group_weights
        # (x, y, masks, group_idx, condition) - with conditions
        # Note: conditions require masks and group_idx slots (possibly None tensors)

        batch_len = len(batch)
        condition = None
        group_idx = None
        masks = None

        if batch_len == 2:
            x, y = batch
        elif batch_len == 3:
            x, y, masks = batch
        elif batch_len == 4:
            x, y, masks, group_idx = batch
        elif batch_len == 5:
            x, y, masks, group_idx, condition = batch
        else:
            raise ValueError(f"Unexpected batch length: {batch_len}")

        return x, y, masks, group_idx, condition

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

        # Track which targets are valid (target position is unmasked)
        valid_targets = None
        if masks is not None:
            # masks: (batch, n_heads, n_classes) - True = valid
            # Check if the TARGET position is valid (not just any position)
            valid_targets = masks.gather(2, target_indices.unsqueeze(-1)).squeeze(-1)  # [batch, n_heads]
            logits = logits.masked_fill(~masks, float("-inf"))

        # Compute log softmax (vectorized across all heads)
        log_probs = F.log_softmax(logits, dim=-1)  # [batch, n_heads, n_classes]

        # Gather log probs for target classes
        nll = -log_probs.gather(2, target_indices.unsqueeze(-1)).squeeze(-1)  # [batch, n_heads]

        # Zero out loss where target is masked (would be inf)
        if valid_targets is not None:
            nll = nll.masked_fill(~valid_targets, 0.0)

        # Apply weights
        if self.group_weights_tensor is not None and group_idx is not None:
            # Look up weights for each sample's group: [batch, n_heads, n_classes]
            batch_weights = self.group_weights_tensor[group_idx]
            # Gather weights for target classes
            sample_weights = batch_weights.gather(2, target_indices.unsqueeze(-1)).squeeze(-1)  # [batch, n_heads]
            weighted_nll = nll * sample_weights

            # Average only over valid targets
            if valid_targets is not None:
                n_valid = valid_targets.sum()
                if n_valid > 0:
                    return weighted_nll.sum() / n_valid
                return torch.tensor(0.0, device=logits.device)
            return weighted_nll.mean()

        elif self.class_weights is not None:
            # Global weights (also vectorized)
            batch_size = target_indices.shape[0]
            expanded_weights = self.class_weights.unsqueeze(0).expand(batch_size, -1, -1)
            sample_weights = expanded_weights.gather(2, target_indices.unsqueeze(-1)).squeeze(-1)  # [batch, n_heads]
            weighted_nll = nll * sample_weights

            # Average only over valid targets
            if valid_targets is not None:
                n_valid = valid_targets.sum()
                if n_valid > 0:
                    return weighted_nll.sum() / n_valid
                return torch.tensor(0.0, device=logits.device)
            return weighted_nll.mean()

        else:
            # Average only over valid targets
            if valid_targets is not None:
                n_valid = valid_targets.sum()
                if n_valid > 0:
                    return nll.sum() / n_valid
                return torch.tensor(0.0, device=logits.device)
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

    def predict_probs_with_masks_conditional(
        self,
        x: torch.Tensor,
        masks: torch.Tensor | None,
        conditions: torch.Tensor,
    ) -> torch.Tensor:
        """Get output probabilities with conditions and optional masking."""
        # For temporal architectures with condition_dim
        if self.architecture_type not in ("conv1d", "sequential", "transformer"):
            raise ValueError(f"Conditional predict not supported for {self.architecture_type}")

        logits = self.net.forward(x, conditions)
        probs = self.apply_output_activation(logits)

        # Apply masks if provided
        if masks is not None:
            if self.output_type == "multi_head":
                n_heads = self.output_config["n_heads"]
                n_classes = self.output_config["n_classes_per_head"]
                batch_size = probs.shape[0]
                probs = probs.view(batch_size, n_heads, n_classes)
                probs = probs.masked_fill(~masks, 0.0)
                probs = probs.view(batch_size, -1)
            else:
                probs = probs * masks.float().view(probs.shape[0], -1)

        return probs

    def generate_autoregressive(
        self,
        prefix: torch.Tensor | None,
        condition: torch.Tensor | None,
        n_steps: int,
        temperature: float = 1.0,
        return_probs: bool = False,
    ) -> torch.Tensor:
        """Generate sequence autoregressively.

        Args:
            prefix: Optional prefix sequence (n_prefix_steps, n_channels) to continue from.
                   If None or empty, starts from zeros.
            condition: Optional condition vector (1, condition_dim) to condition generation.
            n_steps: Number of steps to generate.
            temperature: Sampling temperature. 0.0 = argmax, > 0 = scaled sampling.
            return_probs: If True, return probabilities. If False, return samples.

        Returns:
            Generated sequence (n_steps, n_channels) - does NOT include prefix.
        """
        if self.architecture_type != "sequential":
            raise ValueError(
                f"generate_autoregressive requires sequential architecture, got {self.architecture_type}"
            )

        if self.output_type not in ("binary", "regression"):
            raise ValueError(
                f"generate_autoregressive requires binary or regression output, got {self.output_type}"
            )

        net = self.net
        n_channels = net.n_channels

        # Initialize hidden state
        hidden = net.init_hidden(batch_size=1)

        # Determine input size: n_channels (+ condition_dim if using conditions)
        input_size = n_channels
        if condition is not None and net.condition_dim:
            input_size += net.condition_dim

        # Process prefix to establish hidden state
        if prefix is not None and prefix.shape[0] > 0:
            # prefix: (n_prefix_steps, n_channels)
            prefix_seq = prefix.unsqueeze(0)  # (1, n_prefix_steps, n_channels)

            if condition is not None and net.condition_dim:
                # Concatenate condition to each prefix timestep
                cond_expanded = condition.expand(prefix.shape[0], -1)  # (n_prefix_steps, condition_dim)
                prefix_with_cond = torch.cat([prefix, cond_expanded], dim=-1)  # (n_prefix_steps, input_size)
                prefix_seq = prefix_with_cond.unsqueeze(0)  # (1, n_prefix_steps, input_size)

            _, hidden = net.decoder_rnn(prefix_seq, hidden)

        # Initial input for generation
        if prefix is not None and prefix.shape[0] > 0:
            x_t = prefix[-1:].clone()  # (1, n_channels) - last prefix step
        else:
            x_t = torch.zeros(1, n_channels, device=next(net.parameters()).device)

        # Autoregressive generation loop
        outputs = []
        for _ in range(n_steps):
            # Concatenate condition to input at each step
            if condition is not None and net.condition_dim:
                x_t_with_cond = torch.cat([x_t, condition], dim=-1).unsqueeze(1)  # (1, 1, input_size)
            else:
                x_t_with_cond = x_t.unsqueeze(1)  # (1, 1, n_channels)

            # Decoder step
            out_t, hidden = net.decoder_rnn(x_t_with_cond, hidden)
            logits_t = net.output_fc(out_t.squeeze(1))  # (1, n_channels)
            probs_t = self.apply_output_activation(logits_t)

            if return_probs:
                outputs.append(probs_t)
            else:
                # Sample or argmax
                if temperature > 0:
                    samples_t = self._sample_from_probs(probs_t, temperature)
                else:
                    samples_t = self._argmax_from_probs(probs_t)
                outputs.append(samples_t)
                x_t = samples_t  # Next input

        return torch.cat(outputs, dim=0)  # (n_steps, n_channels)

    def _sample_from_probs(self, probs: torch.Tensor, temperature: float) -> torch.Tensor:
        """Sample from probabilities with temperature scaling."""
        if self.output_type == "binary":
            # Scale logits by temperature, then sample
            # Avoid log(0) by clamping
            logits = torch.logit(probs.clamp(1e-7, 1 - 1e-7))
            scaled_probs = torch.sigmoid(logits / temperature)
            return torch.bernoulli(scaled_probs)
        else:
            # Regression: add scaled noise
            noise = torch.randn_like(probs) * temperature
            return probs + noise

    def _argmax_from_probs(self, probs: torch.Tensor) -> torch.Tensor:
        """Deterministic output from probabilities."""
        if self.output_type == "binary":
            return (probs > 0.5).float()
        else:
            return probs


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
    conditions: EastVariant | None,
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
    elif architecture_type == "conv1d":
        condition_dim = _get_option(arch.value.get("condition_dim"), None)
        architecture_config = {
            "n_channels": int(arch.value.get("n_channels")),
            "sequence_length": int(arch.value.get("sequence_length")),
            "conv_channels": [int(x) for x in arch.value.get("conv_channels")],
            "kernel_size": int(arch.value.get("kernel_size")),
            "latent_dim": int(arch.value.get("latent_dim")),
            "condition_dim": int(condition_dim) if condition_dim is not None else None,
        }
        latent_dim = architecture_config["latent_dim"]
    elif architecture_type == "sequential":
        condition_dim = _get_option(arch.value.get("condition_dim"), None)
        cell_type_variant = arch.value.get("cell_type")
        cell_type = cell_type_variant.type  # "lstm" or "gru"
        architecture_config = {
            "n_channels": int(arch.value.get("n_channels")),
            "sequence_length": int(arch.value.get("sequence_length")),
            "hidden_size": int(arch.value.get("hidden_size")),
            "n_layers": int(arch.value.get("n_layers")),
            "cell_type": cell_type,
            "latent_dim": int(arch.value.get("latent_dim")),
            "bidirectional": bool(arch.value.get("bidirectional")),
            "condition_dim": int(condition_dim) if condition_dim is not None else None,
        }
        latent_dim = architecture_config["latent_dim"]
    elif architecture_type == "transformer":
        condition_dim = _get_option(arch.value.get("condition_dim"), None)
        d_ff = _get_option(arch.value.get("d_ff"), None)
        architecture_config = {
            "n_channels": int(arch.value.get("n_channels")),
            "sequence_length": int(arch.value.get("sequence_length")),
            "d_model": int(arch.value.get("d_model")),
            "n_attention_heads": int(arch.value.get("n_attention_heads")),
            "n_layers": int(arch.value.get("n_layers")),
            "d_ff": int(d_ff) if d_ff is not None else None,
            "latent_dim": int(arch.value.get("latent_dim")),
            "condition_dim": int(condition_dim) if condition_dim is not None else None,
        }
        latent_dim = architecture_config["latent_dim"]
    else:  # mlp
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

    # Handle conditions (6th parameter)
    conditions_tensor = None
    has_condition_dim = architecture_config.get("condition_dim") is not None
    if conditions is not None and is_east_variant(conditions) and conditions.type == "some":
        conditions_np = east_matrix_to_numpy(conditions.value)
        conditions_tensor = torch.tensor(conditions_np, dtype=torch.float32)
        if not has_condition_dim:
            raise ValueError("conditions provided but architecture has no condition_dim set")
        if conditions_np.shape[0] != n_samples:
            raise ValueError(
                f"conditions rows {conditions_np.shape[0]} does not match X rows {n_samples}"
            )
        if conditions_np.shape[1] != architecture_config["condition_dim"]:
            raise ValueError(
                f"conditions columns {conditions_np.shape[1]} does not match condition_dim "
                f"{architecture_config['condition_dim']}"
            )
    elif has_condition_dim:
        raise ValueError("architecture has condition_dim set but no conditions provided")

    # Create dataset based on what features are present
    # Order: (x, y, masks, group_idx, condition)
    # We use dummy tensors for unused slots when conditions are present
    if conditions_tensor is not None:
        # Need all 5 elements
        if masks_tensor is None:
            if output_type == "multi_head":
                n_heads = output_config["n_heads"]
                n_classes = output_config["n_classes_per_head"]
                masks_tensor = torch.ones((n_samples, n_heads, n_classes), dtype=torch.bool)
            else:
                masks_tensor = torch.ones((n_samples, 1, output_dim), dtype=torch.bool)

        if group_weights_for_model is not None:
            sample_groups_tensor = torch.tensor(sample_groups_list, dtype=torch.long)
        else:
            # Dummy tensor - all samples in group 0
            sample_groups_tensor = torch.zeros(n_samples, dtype=torch.long)

        dataset = TensorDataset(X_tensor, y_tensor, masks_tensor, sample_groups_tensor, conditions_tensor)
    elif group_weights_for_model is not None:
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
        val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0)
        monitor_metric = "val_loss"
    else:
        train_dataset = dataset
        val_loader = None
        monitor_metric = "train_loss"

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

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
    conditions: EastVariant | None,
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
        if model.condition_dim is None:
            raise ValueError("Model has no condition_dim but conditions were provided")
        if condition_tensor.shape[1] != model.condition_dim:
            raise ValueError(
                f"Expected condition_dim={model.condition_dim}, got {condition_tensor.shape[1]}"
            )

    # Validate: if model expects conditions, they must be provided
    if model.condition_dim is not None and condition_tensor is None:
        raise ValueError(
            f"Model requires condition_dim={model.condition_dim} but no conditions provided"
        )

    # Predict
    with torch.no_grad():
        if condition_tensor is not None:
            probs = model.predict_probs_with_masks_conditional(
                X_tensor, masks_tensor, condition_tensor
            ).numpy()
        else:
            probs = model.predict_probs_with_masks(X_tensor, masks_tensor).numpy()

    return numpy_to_east_matrix(probs)


def lightning_encode_impl(
    model_blob: EastVariant,
    X: EastArray,
) -> EastArray:
    """Encode input to latent space (autoencoder and temporal architectures)."""
    model_data = model_blob.value
    model_bytes = bytes(model_data.get("data"))
    model = _deserialize_model(model_bytes)
    model.eval()

    if model.architecture_type not in ("autoencoder", "conv1d", "sequential", "transformer"):
        raise ValueError(f"encode() not available for {model.architecture_type} architecture")

    X_np = east_matrix_to_numpy(X)
    X_tensor = torch.tensor(X_np, dtype=torch.float32)

    with torch.no_grad():
        embeddings = model.encode(X_tensor).numpy()

    return numpy_to_east_matrix(embeddings)


def lightning_decode_impl(
    model_blob: EastVariant,
    z: EastArray,
) -> EastArray:
    """Decode latent to output (autoencoder and temporal architectures without condition)."""
    model_data = model_blob.value
    model_bytes = bytes(model_data.get("data"))
    model = _deserialize_model(model_bytes)
    model.eval()

    if model.architecture_type not in ("autoencoder", "conv1d", "sequential", "transformer"):
        raise ValueError(f"decode() not available for {model.architecture_type} architecture")

    # Check if model requires condition
    if model.condition_dim is not None:
        raise ValueError(
            f"Model has condition_dim={model.condition_dim} but decode() was called without condition. "
            "Use decodeConditional() instead."
        )

    z_np = east_matrix_to_numpy(z)
    z_tensor = torch.tensor(z_np, dtype=torch.float32)

    with torch.no_grad():
        logits = model.decode(z_tensor)
        probs = model.apply_output_activation(logits)
        output = probs.numpy()

    return numpy_to_east_matrix(output)


def lightning_decode_conditional_impl(
    model_blob: EastVariant,
    z: EastArray,
    condition: EastArray,
) -> EastArray:
    """Decode latent to output with condition vector (temporal architectures with condition_dim)."""
    model_data = model_blob.value
    model_bytes = bytes(model_data.get("data"))
    model = _deserialize_model(model_bytes)
    model.eval()

    if model.architecture_type not in ("conv1d", "sequential", "transformer"):
        raise ValueError(
            f"decodeConditional() requires temporal architecture (conv1d, sequential, transformer), "
            f"got {model.architecture_type}"
        )

    if model.condition_dim is None:
        raise ValueError("Model has no condition_dim but condition was provided")

    z_np = east_matrix_to_numpy(z)
    condition_np = east_matrix_to_numpy(condition)

    if condition_np.shape[1] != model.condition_dim:
        raise ValueError(
            f"Expected condition_dim={model.condition_dim}, got {condition_np.shape[1]}"
        )

    z_tensor = torch.tensor(z_np, dtype=torch.float32)
    condition_tensor = torch.tensor(condition_np, dtype=torch.float32)

    with torch.no_grad():
        logits = model.decode(z_tensor, condition_tensor)
        probs = model.apply_output_activation(logits)
        output = probs.numpy()

    return numpy_to_east_matrix(output)


def lightning_generate_sequence_impl(
    model_blob: EastVariant,
    prefix: EastArray,
    condition: EastVariant | None,
    config: EastStruct,
) -> EastArray:
    """Generate sequence autoregressively with optional prefix and condition.

    Args:
        model_blob: Trained sequential model blob
        prefix: Prefix matrix (n_prefix_steps, n_channels), can be empty []
        condition: Optional condition matrix (1, condition_dim)
        config: Generation config with n_steps, temperature, return_probs

    Returns:
        Generated sequence matrix (n_steps, n_channels)
    """
    model_data = model_blob.value
    model_bytes = bytes(model_data.get("data"))
    model = _deserialize_model(model_bytes)
    model.eval()

    # Validate architecture
    arch_type = model_data.get("architecture_type")
    if arch_type != "sequential":
        raise ValueError(
            f"generateSequence requires sequential architecture, got {arch_type}"
        )

    # Parse config
    n_steps = int(config.get("n_steps"))
    temperature = float(config.get("temperature"))
    return_probs = bool(config.get("return_probs"))

    # Parse prefix
    prefix_np = east_matrix_to_numpy(prefix)
    if prefix_np.shape[0] > 0:
        prefix_tensor = torch.tensor(prefix_np, dtype=torch.float32)
    else:
        prefix_tensor = None

    # Parse condition
    condition_tensor = None
    if condition is not None and is_east_variant(condition) and condition.type == "some":
        condition_np = east_matrix_to_numpy(condition.value)
        condition_tensor = torch.tensor(condition_np, dtype=torch.float32)

        # Validate condition_dim
        if model.condition_dim is None:
            raise ValueError("Model has no condition_dim but condition was provided")
        if condition_np.shape[1] != model.condition_dim:
            raise ValueError(
                f"Expected condition_dim={model.condition_dim}, got {condition_np.shape[1]}"
            )

    # Validate: if model expects conditions, they must be provided
    if model.condition_dim is not None and condition_tensor is None:
        raise ValueError(
            f"Model requires condition_dim={model.condition_dim} but no condition provided"
        )

    with torch.no_grad():
        generated = model.generate_autoregressive(
            prefix=prefix_tensor,
            condition=condition_tensor,
            n_steps=n_steps,
            temperature=temperature,
            return_probs=return_probs,
        )

    return numpy_to_east_matrix(generated.numpy())


# ============================================================================
# Decision Transformer Platform Functions
# ============================================================================


def _parse_head_configs(heads_list: list) -> list[dict]:
    """Parse head configs from East structures to Python dicts."""
    result = []
    for head in heads_list:
        head_type_variant = head.get("head_type")
        class_weights = _get_option(head.get("class_weights"), None)
        conditional_on = _get_option(head.get("conditional_on"), None)

        head_type_tag = head_type_variant.type
        if head_type_tag == "binary":
            head_type = {"type": "binary"}
        else:  # multiclass
            n_classes = int(head_type_variant.value.get("n_classes"))
            head_type = {"type": "multiclass", "multiclass": {"n_classes": n_classes}}

        result.append({
            "head_type": head_type,
            "class_weights": list(class_weights) if class_weights else None,
            "conditional_on": int(conditional_on) if conditional_on is not None else None,
        })
    return result


def lightning_train_trajectory_impl(
    returns: EastArray,
    states: EastArray,
    actions: EastArray,
    masks: EastArray,
    config: EastStruct,
) -> EastStruct:
    """
    Train a Decision Transformer with trajectory data.

    Args:
        returns: Return per sample (n_samples,) - actual outcome achieved
        states: List of state matrices (n_samples × (seq_len, state_dim))
        actions: List of action matrices (n_samples × (seq_len, action_dim))
        masks: List of temporal masks (n_samples × (seq_len,)) - valid timesteps
        config: Training configuration with decision_transformer architecture

    Returns:
        Training result with model blob and metrics
    """
    # Convert inputs
    returns_np = np.array([float(r) for r in returns], dtype=np.float32)
    states_list = [east_matrix_to_numpy(s) for s in states]
    actions_list = [east_matrix_to_numpy(a) for a in actions]
    masks_list = [np.array([float(m) for m in mask], dtype=np.float32) for mask in masks]

    n_samples = len(returns_np)

    # Stack to tensors
    returns_tensor = torch.tensor(returns_np, dtype=torch.float32).unsqueeze(-1)
    states_tensor = torch.stack([torch.tensor(s, dtype=torch.float32) for s in states_list])
    actions_tensor = torch.stack([torch.tensor(a, dtype=torch.float32) for a in actions_list])
    masks_tensor = torch.stack([torch.tensor(m, dtype=torch.float32) for m in masks_list])

    # Parse architecture config
    arch = config.get("architecture")
    architecture_type = arch.type
    if architecture_type != "decision_transformer":
        raise ValueError(
            f"trainTrajectory requires decision_transformer architecture, got {architecture_type}"
        )

    d_ff = _get_option(arch.value.get("d_ff"), None)
    dropout = _get_option(arch.value.get("dropout"), 0.1)
    return_embedding_variant = arch.value.get("return_embedding")
    return_embedding = return_embedding_variant.type  # "global" or "per_timestep"

    # Validate return embedding mode
    if return_embedding == "per_timestep":
        raise NotImplementedError(
            "per_timestep return embedding is not yet supported in training. "
            "Use global return embedding instead."
        )

    architecture_config = {
        "sequence_length": int(arch.value.get("sequence_length")),
        "state_dim": int(arch.value.get("state_dim")),
        "action_dim": int(arch.value.get("action_dim")),
        "d_model": int(arch.value.get("d_model")),
        "n_attention_heads": int(arch.value.get("n_attention_heads")),
        "n_layers": int(arch.value.get("n_layers")),
        "d_ff": int(d_ff) if d_ff is not None else None,
        "dropout": float(dropout) if dropout is not None else 0.1,
        "return_embedding": return_embedding,
    }

    # Parse output config
    output = config.get("output")
    output_type = output.type

    head_configs = None
    if output_type == "multi_head_mixed":
        heads_list = list(output.value.get("heads"))
        head_configs = _parse_head_configs(heads_list)
    elif output_type == "multi_head":
        # Convert uniform multi_head to head_configs format
        n_heads = int(output.value.get("n_heads"))
        n_classes_per_head = int(output.value.get("n_classes_per_head"))
        head_configs = [
            {
                "head_type": {"type": "multiclass", "multiclass": {"n_classes": n_classes_per_head}},
                "conditional_on": None,
            }
            for _ in range(n_heads)
        ]
    elif output_type == "binary":
        # Treat each action dim as independent binary
        pass
    else:
        raise ValueError(
            f"trainTrajectory requires binary, multi_head, or multi_head_mixed output, got {output_type}"
        )

    # Training params with defaults
    learning_rate = float(_get_option(config.get("learning_rate"), 1e-3))
    max_epochs = int(_get_option(config.get("max_epochs"), 100))
    patience = int(_get_option(config.get("patience"), 10))
    batch_size = int(_get_option(config.get("batch_size"), 32))
    gradient_clip = float(_get_option(config.get("gradient_clip"), 1.0))
    weight_decay = float(_get_option(config.get("weight_decay"), 0.0))
    random_state = _get_option(config.get("random_state"), None)
    epoch_callback_fn = _get_option(config.get("epoch_callback"), None)

    if random_state is not None:
        pl.seed_everything(int(random_state), workers=True)

    # Create model
    model = DecisionTransformerNet(
        sequence_length=architecture_config["sequence_length"],
        state_dim=architecture_config["state_dim"],
        action_dim=architecture_config["action_dim"],
        d_model=architecture_config["d_model"],
        n_heads=architecture_config["n_attention_heads"],
        n_layers=architecture_config["n_layers"],
        d_ff=architecture_config["d_ff"],
        dropout=architecture_config["dropout"],
        return_embedding=architecture_config["return_embedding"],
    )

    # Create dataset
    dataset = TensorDataset(returns_tensor, states_tensor, actions_tensor, masks_tensor)

    # Split data
    val_size = max(1, int(0.1 * n_samples)) if n_samples >= 2 else 0
    train_size = n_samples - val_size

    generator = torch.Generator()
    if random_state is not None:
        generator.manual_seed(int(random_state))

    if val_size > 0:
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size], generator=generator
        )
    else:
        train_dataset = dataset
        val_dataset = None

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0) if val_dataset else None

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Training loop
    best_val_loss = float('inf')
    best_epoch = 0
    best_state_dict = None
    epochs_without_improvement = 0

    for epoch in range(max_epochs):
        # Training
        model.train()
        train_losses = []
        for batch in train_loader:
            batch_returns, batch_states, batch_actions, batch_masks = batch

            optimizer.zero_grad()

            # Forward pass
            action_preds = model(batch_returns, batch_states, batch_actions)

            # Compute loss
            if head_configs is not None:
                loss = compute_multi_head_mixed_loss(
                    action_preds, batch_actions, head_configs, batch_masks
                )
            else:
                # Binary loss
                loss = F.binary_cross_entropy_with_logits(
                    action_preds, batch_actions, reduction='none'
                )
                # Apply temporal mask
                loss = loss * batch_masks.unsqueeze(-1)
                n_valid = batch_masks.sum() * action_preds.shape[-1]
                if n_valid > 0:
                    loss = loss.sum() / n_valid
                else:
                    loss = loss.sum()

            # Gradient clipping
            loss.backward()
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()

            train_losses.append(loss.item())

        train_loss = np.mean(train_losses)

        # Validation
        val_loss = train_loss  # Default if no validation
        if val_loader is not None:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    batch_returns, batch_states, batch_actions, batch_masks = batch
                    action_preds = model(batch_returns, batch_states, batch_actions)

                    if head_configs is not None:
                        loss = compute_multi_head_mixed_loss(
                            action_preds, batch_actions, head_configs, batch_masks
                        )
                    else:
                        loss = F.binary_cross_entropy_with_logits(
                            action_preds, batch_actions, reduction='none'
                        )
                        loss = loss * batch_masks.unsqueeze(-1)
                        n_valid = batch_masks.sum() * action_preds.shape[-1]
                        if n_valid > 0:
                            loss = loss.sum() / n_valid
                        else:
                            loss = loss.sum()

                    val_losses.append(loss.item())

            val_loss = np.mean(val_losses)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Epoch callback
        if epoch_callback_fn is not None:
            epoch_callback_fn(epoch, train_loss, val_loss)

        if epochs_without_improvement >= patience:
            break

    # Restore best model
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # Serialize model
    model_data = {
        "state_dict": model.state_dict(),
        "architecture_config": architecture_config,
        "output_type": output_type,
        "head_configs": head_configs,
    }
    model_blob_bytes = pickle.dumps(model_data)

    # Create result
    action_dim = architecture_config["action_dim"]
    state_dim = architecture_config["state_dim"]

    result_model = EastVariant(
        "lightning",
        EastStruct({
            "data": EastBlob(model_blob_bytes),
            "n_features": state_dim,
            "output_dim": action_dim,
            "architecture_type": "decision_transformer",
            "output_type": output_type,
            "latent_dim": EastVariant("none", None),
        }),
    )

    return EastStruct({
        "model": result_model,
        "train_loss": float(train_loss),
        "val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
    })


def lightning_generate_trajectory_impl(
    model_blob: EastVariant,
    states: EastArray,
    target_returns: EastArray,
    config: EastStruct,
) -> EastArray:
    """
    Generate action sequences autoregressively from trajectory model.

    Args:
        model_blob: Trained model from trainTrajectory
        states: List of state matrices (n_samples × (seq_len, state_dim))
        target_returns: Target returns (n_samples,)
        config: Generation configuration

    Returns:
        List of generated action matrices (n_samples × (seq_len, action_dim))
    """
    # Extract model data
    model_data = model_blob.value
    arch_type = model_data.get("architecture_type")

    if arch_type != "decision_transformer":
        raise ValueError(
            f"generateTrajectory requires decision_transformer architecture, got {arch_type}"
        )

    model_bytes = bytes(model_data.get("data"))
    saved_data = pickle.loads(model_bytes)

    # Reconstruct model
    arch_config = saved_data["architecture_config"]
    model = DecisionTransformerNet(
        sequence_length=arch_config["sequence_length"],
        state_dim=arch_config["state_dim"],
        action_dim=arch_config["action_dim"],
        d_model=arch_config["d_model"],
        n_heads=arch_config["n_attention_heads"],
        n_layers=arch_config["n_layers"],
        d_ff=arch_config.get("d_ff"),
        dropout=arch_config.get("dropout", 0.1),
        return_embedding=arch_config["return_embedding"],
    )
    model.load_state_dict(saved_data["state_dict"])
    model.eval()

    # Get head configs
    head_configs = saved_data.get("head_configs")

    # Parse inputs
    states_list = [east_matrix_to_numpy(s) for s in states]
    returns_np = np.array([float(r) for r in target_returns], dtype=np.float32)

    states_tensor = torch.stack([torch.tensor(s, dtype=torch.float32) for s in states_list])
    returns_tensor = torch.tensor(returns_np, dtype=torch.float32).unsqueeze(-1)

    # Parse config
    temperature = float(config.get("temperature"))
    # return_probs = bool(config.get("return_probs"))  # TODO: implement if needed

    # Parse optional constraints
    action_constraints = _get_option(config.get("action_constraints"), None)
    temporal_mask = _get_option(config.get("temporal_mask"), None)

    constraints_tensor = None
    if action_constraints is not None:
        constraints_np = east_matrix_to_numpy(action_constraints)
        constraints_tensor = torch.tensor(constraints_np, dtype=torch.bool)

    temporal_mask_tensor = None
    if temporal_mask is not None:
        temporal_mask_np = np.array([float(m) for m in temporal_mask], dtype=np.float32)
        temporal_mask_tensor = torch.tensor(temporal_mask_np, dtype=torch.bool)

    # Parse head configs from config (overrides saved if provided)
    config_head_configs = _get_option(config.get("head_configs"), None)
    if config_head_configs is not None:
        head_configs = _parse_head_configs(list(config_head_configs))

    # Generate
    with torch.no_grad():
        generated_actions = model.generate(
            returns_tensor,
            states_tensor,
            temperature=temperature,
            constraints=constraints_tensor,
            temporal_mask=temporal_mask_tensor,
            head_configs=head_configs,
        )

    # Convert back to East array of matrices
    result = []
    for i in range(generated_actions.shape[0]):
        action_matrix = generated_actions[i].numpy()
        result.append(numpy_to_east_matrix(action_matrix))

    return EastArray(MatrixType, result)


# ============================================================================
# Platform Function Registration
# ============================================================================

# 3D tensor type for masks
Tensor3DType = EastArray  # ArrayType(ArrayType(ArrayType(BooleanType)))

lightning_impl = [
    PlatformFunction(
        name="lightning_train",
        inputs=[MatrixType, MatrixType, LightningConfigType, Tensor3DType, GroupWeightsType, MatrixType],
        output=LightningResultType,
        type="sync",
        fn=lightning_train_impl,
    ),
    PlatformFunction(
        name="lightning_predict",
        inputs=[ModelBlobType, MatrixType, Tensor3DType, MatrixType],
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
    PlatformFunction(
        name="lightning_decode_conditional",
        inputs=[ModelBlobType, MatrixType, MatrixType],
        output=MatrixType,
        type="sync",
        fn=lightning_decode_conditional_impl,
    ),
    PlatformFunction(
        name="lightning_generate_sequence",
        inputs=[ModelBlobType, MatrixType, MatrixType, LightningGenerateConfigType],
        output=MatrixType,
        type="sync",
        fn=lightning_generate_sequence_impl,
    ),
    PlatformFunction(
        name="lightning_train_trajectory",
        inputs=[
            EastArray,  # returns: VectorType
            EastArray,  # states: ArrayType(MatrixType)
            EastArray,  # actions: ArrayType(MatrixType)
            EastArray,  # masks: ArrayType(VectorType)
            LightningConfigType,
        ],
        output=LightningResultType,
        type="sync",
        fn=lightning_train_trajectory_impl,
    ),
    PlatformFunction(
        name="lightning_generate_trajectory",
        inputs=[
            ModelBlobType,
            EastArray,  # states: ArrayType(MatrixType)
            EastArray,  # target_returns: VectorType
            EastArray,  # TrajectoryGenerateConfigType (as struct)
        ],
        output=EastArray,  # ArrayType(MatrixType)
        type="sync",
        fn=lightning_generate_trajectory_impl,
    ),
]
