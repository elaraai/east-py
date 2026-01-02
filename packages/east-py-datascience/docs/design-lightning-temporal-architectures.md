# Design: Temporal Architectures for Lightning (Conv1D, Sequential, Transformer)

## Problem Statement

The current Lightning module only supports dense MLP architectures. For sequential/temporal data where adjacent positions are correlated (e.g., additive amounts across fermentation days), flat MLPs cannot capture these dependencies effectively.

### The Additive Model Use Case

Training an autoencoder on additive application patterns:
- 7 additive types x 14 days = 98 time slots
- 4 bins per slot (none, low, med, high)
- **Temporal dependencies**: adding acid on day 1 affects whether acid is needed on day 2

With a flat MLP autoencoder:
- Each (additive, day, bin) position is treated independently
- Model cannot learn "front-loaded vs gradual" patterns
- Embeddings fail to capture meaningful temporal structure

### Binary + Amount Two-Stage Approach

For sparse data with temporal dependencies, a two-stage approach works best:

1. **Stage 1: Binary autoencoder** - predict whether each slot has any additive
   - Uses existing `binary` output with `pos_weight` for class imbalance
   - Produces embeddings capturing "what additives, when"

2. **Stage 2: Amount predictor** - predict bin amounts for positive slots
   - Uses temporal architecture (conv1d/sequential/transformer)
   - Conditioned on binary mask from Stage 1
   - Models day-to-day dependencies

This design adds temporal architectures for Stage 2 (and general use).

## Current Architecture Types

```typescript
export const LightningArchitectureType = VariantType({
    mlp: StructType({
        hidden_layers: ArrayType(IntegerType),
    }),
    autoencoder: StructType({
        encoder_layers: ArrayType(IntegerType),
        latent_dim: IntegerType,
        decoder_layers: ArrayType(IntegerType),
    }),
});
```

## Proposed Changes

### New Architecture Variants

```typescript
export const LightningArchitectureType = VariantType({
    // Existing
    mlp: StructType({
        hidden_layers: ArrayType(IntegerType),
    }),
    autoencoder: StructType({
        encoder_layers: ArrayType(IntegerType),
        latent_dim: IntegerType,
        decoder_layers: ArrayType(IntegerType),
    }),

    // NEW: Conv1D for local temporal patterns
    conv1d: StructType({
        /** Number of channels (e.g., additive types) */
        n_channels: IntegerType,
        /** Sequence length (e.g., days) */
        sequence_length: IntegerType,
        /** Conv layer channel sizes */
        conv_channels: ArrayType(IntegerType),
        /** Kernel size for convolutions (default: 3) */
        kernel_size: IntegerType,
        /** Latent dimension after flattening */
        latent_dim: IntegerType,
    }),

    // NEW: Sequential (LSTM/GRU) for long-range dependencies
    sequential: StructType({
        /** Number of channels (e.g., additive types) */
        n_channels: IntegerType,
        /** Sequence length (e.g., days) */
        sequence_length: IntegerType,
        /** RNN hidden size */
        hidden_size: IntegerType,
        /** Number of RNN layers */
        n_layers: IntegerType,
        /** Cell type */
        cell_type: VariantType({ lstm: NullType, gru: NullType }),
        /** Latent dimension (from final hidden state) */
        latent_dim: IntegerType,
        /** Bidirectional (default: false) */
        bidirectional: BooleanType,
    }),

    // NEW: Transformer for attention-based patterns
    transformer: StructType({
        /** Number of channels (e.g., additive types) */
        n_channels: IntegerType,
        /** Sequence length (e.g., days) */
        sequence_length: IntegerType,
        /** Model dimension */
        d_model: IntegerType,
        /** Number of attention heads (must divide d_model evenly) */
        n_attention_heads: IntegerType,
        /** Number of transformer layers */
        n_layers: IntegerType,
        /** Feedforward dimension (default: 4 * d_model) */
        d_ff: OptionType(IntegerType),
        /** Latent dimension (mean pooled output) */
        latent_dim: IntegerType,
    }),
});
```

### Input Reshaping

All temporal architectures expect input as a flat vector `[n_channels * sequence_length * n_classes]` and internally reshape:

```
Input:  [batch, n_channels * sequence_length * n_classes]
        ↓ reshape
        [batch, n_channels, sequence_length, n_classes]
        ↓ temporal encoding
        [batch, latent_dim]
        ↓ temporal decoding
        [batch, n_channels, sequence_length, n_classes]
        ↓ reshape
Output: [batch, n_channels * sequence_length * n_classes]
```

This maintains compatibility with existing training/prediction APIs.

### Data Layout

The flat input vector is ordered as: **channel-major, then time, then class**.

For `n_channels=2, sequence_length=3, n_classes=2`:
```
Index:  0    1    2    3    4    5    6    7    8    9   10   11
        |----ch0----|----ch0----|----ch0----|----ch1----|----ch1----|----ch1----|
        |--t0--|    |--t1--|    |--t2--|    |--t0--|    |--t1--|    |--t2--|
        c0  c1      c0  c1      c0  c1      c0  c1      c0  c1      c0  c1
```

After reshape to `[batch, n_channels, sequence_length, n_classes]`:
- `data[b, ch, t, c]` = original flat index `ch * (sequence_length * n_classes) + t * n_classes + c`

### Kernel Size Constraint

**`kernel_size` must be odd** for symmetric padding. With odd kernel sizes, `padding = kernel_size // 2` preserves sequence length:
- `kernel_size=3`: padding=1, output_length = input_length
- `kernel_size=5`: padding=2, output_length = input_length

Even kernel sizes would cause asymmetric padding and length changes. Validation will reject even values.

## Python Implementation

### Conv1D Autoencoder

```python
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
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.sequence_length = sequence_length
        self.n_classes = n_classes
        self.latent_dim = latent_dim

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
        self.decoder_fc = nn.Linear(latent_dim, encoder_output_size)

        # Decoder: transposed conv layers (mirror of encoder)
        # Reset prev_channels explicitly for clarity
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

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space."""
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))
```

### Sequential (LSTM/GRU) Autoencoder

```python
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
        decoder_hidden_size = hidden_size * n_layers  # unidirectional
        self.decoder_fc = nn.Linear(latent_dim, decoder_hidden_size)

        self.decoder_rnn = RNNClass(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=False,  # Always unidirectional for generation
            dropout=dropout if n_layers > 1 else 0,
        )

        # Output projection (decoder is always unidirectional)
        self.output_fc = nn.Linear(hidden_size, input_size)

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

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to sequence (parallel, not autoregressive)."""
        batch_size = z.shape[0]
        hidden = self.decoder_fc(z)
        # Reshape to RNN hidden state format (decoder is always unidirectional)
        n_layers = self.decoder_rnn.num_layers
        hidden = hidden.view(batch_size, n_layers, self.hidden_size)
        hidden = hidden.permute(1, 0, 2).contiguous()

        if self.cell_type == "lstm":
            hidden = (hidden, torch.zeros_like(hidden))

        # Parallel decoding: use zeros as input, hidden state provides context
        # This is NOT autoregressive - all positions decoded simultaneously
        decoder_input = torch.zeros(
            batch_size, self.sequence_length, self.n_channels * self.n_classes,
            device=z.device
        )
        output, _ = self.decoder_rnn(decoder_input, hidden)
        output = self.output_fc(output)

        # -> [batch, sequence_length, n_channels, n_classes]
        output = output.view(batch_size, self.sequence_length, self.n_channels, self.n_classes)
        # -> [batch, n_channels, sequence_length, n_classes]
        output = output.permute(0, 2, 1, 3)
        return output.reshape(batch_size, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))
```

### Transformer Autoencoder

```python
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
        self.decoder_fc = nn.Linear(latent_dim, d_model * sequence_length)

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

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent via attention."""
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))
```

### Integration with LightningMLP

Update `LightningMLP.__init__` to handle new architecture types:

```python
class LightningMLP(pl.LightningModule):
    def __init__(self, ...):
        # ... existing code ...

        if architecture_type == "autoencoder":
            # ... existing autoencoder code ...
        elif architecture_type == "conv1d":
            self.net = Conv1DAutoencoder(
                n_channels=architecture_config["n_channels"],
                sequence_length=architecture_config["sequence_length"],
                n_classes=self._get_n_classes(),
                conv_channels=architecture_config["conv_channels"],
                kernel_size=architecture_config["kernel_size"],
                latent_dim=architecture_config["latent_dim"],
                dropout=dropout,
            )
            self.latent_dim = architecture_config["latent_dim"]
        elif architecture_type == "sequential":
            # Extract cell_type from variant tag
            cell_type_variant = architecture_config["cell_type"]
            cell_type = cell_type_variant.type  # "lstm" or "gru"
            self.net = SequentialAutoencoder(
                n_channels=architecture_config["n_channels"],
                sequence_length=architecture_config["sequence_length"],
                n_classes=self._get_n_classes(),
                hidden_size=architecture_config["hidden_size"],
                n_layers=architecture_config["n_layers"],
                cell_type=cell_type,
                latent_dim=architecture_config["latent_dim"],
                bidirectional=architecture_config.get("bidirectional", False),
                dropout=dropout,
            )
            self.latent_dim = architecture_config["latent_dim"]
        elif architecture_type == "transformer":
            self.net = TransformerAutoencoder(
                n_channels=architecture_config["n_channels"],
                sequence_length=architecture_config["sequence_length"],
                n_classes=self._get_n_classes(),
                d_model=architecture_config["d_model"],
                n_attention_heads=architecture_config["n_attention_heads"],
                n_layers=architecture_config["n_layers"],
                d_ff=architecture_config.get("d_ff"),
                latent_dim=architecture_config["latent_dim"],
                dropout=dropout,
            )
            self.latent_dim = architecture_config["latent_dim"]
        else:  # mlp
            # ... existing mlp code ...

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

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode to latent space (autoencoder and temporal architectures)."""
        if self.architecture_type == "autoencoder":
            return self.encoder(x)
        elif self.architecture_type in ("conv1d", "sequential", "transformer"):
            return self.net.encode(x)
        raise ValueError(f"encode() not available for {self.architecture_type}")

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent (autoencoder and temporal architectures)."""
        if self.architecture_type == "autoencoder":
            return self.decoder(z)
        elif self.architecture_type in ("conv1d", "sequential", "transformer"):
            return self.net.decode(z)
        raise ValueError(f"decode() not available for {self.architecture_type}")
```

### Output Type Compatibility

Temporal architectures require `multi_head` output (they reshape flat input to `[n_channels, sequence_length, n_classes]`):

| Architecture | regression | binary | multiclass | multi_head |
|--------------|------------|--------|------------|------------|
| mlp          | ✓          | ✓      | ✓          | ✓          |
| autoencoder  | ✓          | ✓      | ✓          | ✓          |
| conv1d       | ✗          | ✗      | ✗          | ✓          |
| sequential   | ✗          | ✗      | ✗          | ✓          |
| transformer  | ✗          | ✗      | ✗          | ✓          |

### Validation

Add validation for temporal architecture configurations:

```python
def _validate_temporal_config(self):
    """Validate temporal architecture config matches output config."""
    if self.architecture_type not in ("conv1d", "sequential", "transformer"):
        return

    # Temporal architectures require multi_head output
    if self.output_type != "multi_head":
        raise ValueError(
            f"Temporal architecture '{self.architecture_type}' requires multi_head output, "
            f"got '{self.output_type}'"
        )

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
```

## Usage Examples

### Conv1D for Additive Amounts (Stage 2)

```typescript
// After Stage 1 binary prediction, use conv1d for amount prediction
const config = $.let({
    architecture: variant('conv1d', {
        n_channels: 7n,           // 7 additive types
        sequence_length: 14n,     // 14 fermentation days
        conv_channels: [32n, 64n],
        kernel_size: 3n,          // look at 3 adjacent days
        latent_dim: 16n,
    }),
    output: variant('multi_head', {
        n_heads: East.value(7n * 14n),  // 98 slots
        n_classes_per_head: 4n,          // 4 bins
        class_weights: variant('none', null),
    }),
    learning_rate: variant('some', 0.001),
    max_epochs: variant('some', 200n),
    patience: variant('some', 20n),
    batch_size: variant('some', 32n),
    dropout: variant('some', 0.1),
    random_state: variant('some', 42n),
    epoch_callback: variant('none', null),
});

// Train on samples where binary prediction is positive
const result = $.let(Lightning.train(X_positive, y_amounts, config, variant('some', masks), variant('none', null)));

// Get embeddings
const embeddings = $.let(Lightning.encode(result.model, X_positive));
```

### Sequential (LSTM) for Long-Range Dependencies

```typescript
const config = $.let({
    architecture: variant('sequential', {
        n_channels: 7n,
        sequence_length: 14n,
        hidden_size: 64n,
        n_layers: 2n,
        cell_type: variant('lstm', null),
        latent_dim: 16n,
        bidirectional: true,
    }),
    output: variant('multi_head', {
        n_heads: East.value(7n * 14n),
        n_classes_per_head: 4n,
        class_weights: variant('none', null),
    }),
    // ... other config ...
});
```

### Transformer for Attention Patterns

```typescript
const config = $.let({
    architecture: variant('transformer', {
        n_channels: 7n,
        sequence_length: 14n,
        d_model: 64n,
        n_attention_heads: 4n,  // must divide d_model evenly
        n_layers: 2n,
        d_ff: variant('none', null),  // defaults to 4 * d_model
        latent_dim: 16n,
    }),
    output: variant('multi_head', {
        n_heads: East.value(7n * 14n),
        n_classes_per_head: 4n,
        class_weights: variant('none', null),
    }),
    // ... other config ...
});
```

## Tests

Add to `/home/crambelsoupy/src/east-py/packages/east-py-datascience/src/lightning/lightning.spec.ts`:

### Test 1: Conv1D autoencoder trains and produces embeddings

```typescript
test("conv1d: train, encode, decode works", $ => {
    // Simulated temporal data: 2 channels x 4 time steps x 3 classes = 24 features
    const X = $.let([
        // Channel patterns across time
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0,  // ch0: pattern A
         0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0], // ch1: pattern B
        [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,  // ch0: pattern C
         1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0], // ch1: pattern D
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0,  // same as sample 0
         0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0,  // ch0: pattern E
         0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0], // ch1: pattern F
    ]);

    const config = $.let({
        architecture: variant('conv1d', {
            n_channels: 2n,
            sequence_length: 4n,
            conv_channels: [8n, 16n],
            kernel_size: 3n,
            latent_dim: 4n,
        }),
        output: variant('multi_head', {
            n_heads: 8n,  // 2 channels x 4 time steps
            n_classes_per_head: 3n,
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

    const result = $.let(Lightning.train(X, X, config, variant('none', null), variant('none', null)));
    $(Assert.greaterEqual(result.best_epoch, 0n));

    // Encode to latent
    const z = $.let(Lightning.encode(result.model, X));
    $(Assert.equal(z.size(), 4n));
    $(Assert.equal(z.get(0n).size(), 4n));

    // Similar patterns should have similar embeddings (samples 0 and 2 are identical)
    const emb0 = $.let(z.get(0n));
    const emb2 = $.let(z.get(2n));
    const dist_same = $.let(
        emb0.get(0n).subtract(emb2.get(0n)).abs()
            .add(emb0.get(1n).subtract(emb2.get(1n)).abs())
    );

    const emb1 = $.let(z.get(1n));
    const dist_diff = $.let(
        emb0.get(0n).subtract(emb1.get(0n)).abs()
            .add(emb0.get(1n).subtract(emb1.get(1n)).abs())
    );

    $(Assert.less(dist_same, dist_diff));

    // Decode should produce valid output
    const X_decoded = $.let(Lightning.decode(result.model, z));
    $(Assert.equal(X_decoded.size(), 4n));
    $(Assert.equal(X_decoded.get(0n).size(), 24n));
});
```

### Test 2: Sequential (LSTM) autoencoder

```typescript
test("sequential: LSTM train, encode, decode works", $ => {
    // Same data structure as conv1d test
    const X = $.let([
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
         0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
         1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
         0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0,
         0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
    ]);

    const config = $.let({
        architecture: variant('sequential', {
            n_channels: 2n,
            sequence_length: 4n,
            hidden_size: 16n,
            n_layers: 1n,
            cell_type: variant('lstm', null),
            latent_dim: 4n,
            bidirectional: false,
        }),
        output: variant('multi_head', {
            n_heads: 8n,
            n_classes_per_head: 3n,
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

    const result = $.let(Lightning.train(X, X, config, variant('none', null), variant('none', null)));
    $(Assert.greaterEqual(result.best_epoch, 0n));

    const z = $.let(Lightning.encode(result.model, X));
    $(Assert.equal(z.size(), 4n));
    $(Assert.equal(z.get(0n).size(), 4n));

    const X_decoded = $.let(Lightning.decode(result.model, z));
    $(Assert.equal(X_decoded.size(), 4n));
    $(Assert.equal(X_decoded.get(0n).size(), 24n));
});
```

### Test 3: Sequential (GRU) with bidirectional

```typescript
test("sequential: GRU bidirectional works", $ => {
    const X = $.let([
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
         0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
         1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ]);

    const config = $.let({
        architecture: variant('sequential', {
            n_channels: 2n,
            sequence_length: 4n,
            hidden_size: 8n,
            n_layers: 2n,
            cell_type: variant('gru', null),
            latent_dim: 4n,
            bidirectional: true,
        }),
        output: variant('multi_head', {
            n_heads: 8n,
            n_classes_per_head: 3n,
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

    const result = $.let(Lightning.train(X, X, config, variant('none', null), variant('none', null)));
    $(Assert.greaterEqual(result.best_epoch, 0n));

    const z = $.let(Lightning.encode(result.model, X));
    $(Assert.equal(z.size(), 2n));
    $(Assert.equal(z.get(0n).size(), 4n));
});
```

### Test 4: Transformer autoencoder

```typescript
test("transformer: train, encode, decode works", $ => {
    const X = $.let([
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
         0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
         1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
         0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0,
         0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
    ]);

    const config = $.let({
        architecture: variant('transformer', {
            n_channels: 2n,
            sequence_length: 4n,
            d_model: 16n,
            n_attention_heads: 2n,  // 16 / 2 = 8 per head
            n_layers: 1n,
            d_ff: variant('none', null),
            latent_dim: 4n,
        }),
        output: variant('multi_head', {
            n_heads: 8n,
            n_classes_per_head: 3n,
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

    const result = $.let(Lightning.train(X, X, config, variant('none', null), variant('none', null)));
    $(Assert.greaterEqual(result.best_epoch, 0n));

    const z = $.let(Lightning.encode(result.model, X));
    $(Assert.equal(z.size(), 4n));
    $(Assert.equal(z.get(0n).size(), 4n));

    // Similar patterns have similar embeddings
    const emb0 = $.let(z.get(0n));
    const emb2 = $.let(z.get(2n));
    const dist_same = $.let(
        emb0.get(0n).subtract(emb2.get(0n)).abs()
            .add(emb0.get(1n).subtract(emb2.get(1n)).abs())
    );

    const emb1 = $.let(z.get(1n));
    const dist_diff = $.let(
        emb0.get(0n).subtract(emb1.get(0n)).abs()
            .add(emb0.get(1n).subtract(emb1.get(1n)).abs())
    );

    $(Assert.less(dist_same, dist_diff));

    const X_decoded = $.let(Lightning.decode(result.model, z));
    $(Assert.equal(X_decoded.size(), 4n));
    $(Assert.equal(X_decoded.get(0n).size(), 24n));
});
```

### Test 5: Temporal architectures with masks and group weights

```typescript
test("conv1d with masks and group weights", $ => {
    const X = $.let([
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
         0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
         1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
         0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0,
         0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
    ]);

    // Masks: [n_samples, n_heads, n_classes] = [4, 8, 3]
    const masks = $.let([
        [[true, true, true], [true, true, true], [true, true, true], [true, true, true],
         [true, true, true], [true, true, true], [true, true, true], [true, true, true]],
        [[true, true, false], [true, true, true], [true, true, true], [true, true, true],
         [true, true, true], [true, true, true], [true, true, true], [true, true, true]],
        [[true, true, true], [true, true, true], [true, true, true], [true, true, true],
         [true, true, true], [true, true, true], [true, true, true], [true, true, true]],
        [[true, true, true], [true, true, true], [true, true, true], [true, true, true],
         [true, true, true], [true, true, true], [true, true, true], [false, true, true]],
    ]);

    // Group weights: 2 groups x 8 heads x 3 classes
    const group_weights = $.let({
        weights: variant('multi_head', [
            [[1.0, 2.0, 2.0], [1.0, 2.0, 2.0], [1.0, 2.0, 2.0], [1.0, 2.0, 2.0],
             [1.0, 2.0, 2.0], [1.0, 2.0, 2.0], [1.0, 2.0, 2.0], [1.0, 2.0, 2.0]],
            [[2.0, 1.0, 2.0], [2.0, 1.0, 2.0], [2.0, 1.0, 2.0], [2.0, 1.0, 2.0],
             [2.0, 1.0, 2.0], [2.0, 1.0, 2.0], [2.0, 1.0, 2.0], [2.0, 1.0, 2.0]],
        ]),
        sample_groups: [0n, 0n, 1n, 1n],
    });

    const config = $.let({
        architecture: variant('conv1d', {
            n_channels: 2n,
            sequence_length: 4n,
            conv_channels: [8n],
            kernel_size: 3n,
            latent_dim: 4n,
        }),
        output: variant('multi_head', {
            n_heads: 8n,
            n_classes_per_head: 3n,
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

    const result = $.let(Lightning.train(X, X, config, variant('some', masks), variant('some', group_weights)));
    $(Assert.greaterEqual(result.best_epoch, 0n));

    // Predict with masks
    const y_pred = $.let(Lightning.predict(result.model, X, variant('some', masks)));
    $(Assert.equal(y_pred.size(), 4n));

    // Masked positions should have ~0 probability
    $(Assert.less(y_pred.get(1n).get(2n), East.value(0.001)));  // sample 1, head 0, class 2
    $(Assert.less(y_pred.get(3n).get(21n), East.value(0.001))); // sample 3, head 7, class 0
});
```

### Test 6: Error - encode on mlp architecture

```typescript
test("error: encode on mlp architecture", $ => {
    const X = $.let([[1.0, 2.0], [3.0, 4.0]]);
    const y = $.let([[1.0], [2.0]]);

    const config = $.let({
        architecture: variant('mlp', { hidden_layers: [8n] }),
        output: variant('regression', null),
        learning_rate: variant('some', 0.01),
        max_epochs: variant('some', 10n),
        patience: variant('some', 5n),
        batch_size: variant('some', 2n),
        dropout: variant('none', null),
        gradient_clip: variant('none', null),
        weight_decay: variant('none', null),
        random_state: variant('some', 42n),
        epoch_callback: variant('none', null),
    });

    const result = $.let(Lightning.train(X, y, config, variant('none', null), variant('none', null)));

    $(Assert.throws(
        Lightning.encode(result.model, X),
        /encode\(\) not available for mlp/
    ));
});
```

### Test 7: Error - temporal architecture with non-multi_head output

```typescript
test("error: conv1d requires multi_head output", $ => {
    const X = $.let([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]);
    const y = $.let([[1.0]]);  // regression output

    const config = $.let({
        architecture: variant('conv1d', {
            n_channels: 2n,
            sequence_length: 3n,
            conv_channels: [8n],
            kernel_size: 3n,
            latent_dim: 4n,
        }),
        output: variant('regression', null),  // ERROR: should be multi_head
        learning_rate: variant('some', 0.01),
        max_epochs: variant('some', 10n),
        patience: variant('some', 5n),
        batch_size: variant('some', 2n),
        dropout: variant('none', null),
        gradient_clip: variant('none', null),
        weight_decay: variant('none', null),
        random_state: variant('some', 42n),
        epoch_callback: variant('none', null),
    });

    $(Assert.throws(
        Lightning.train(X, y, config, variant('none', null), variant('none', null)),
        /Temporal architecture 'conv1d' requires multi_head output/
    ));
});
```

### Test 8: Error - n_heads mismatch with n_channels * sequence_length

```typescript
test("error: n_heads must equal n_channels * sequence_length", $ => {
    const X = $.let([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0]]);

    const config = $.let({
        architecture: variant('conv1d', {
            n_channels: 2n,
            sequence_length: 3n,  // 2 * 3 = 6 expected heads
            conv_channels: [8n],
            kernel_size: 3n,
            latent_dim: 4n,
        }),
        output: variant('multi_head', {
            n_heads: 4n,  // ERROR: should be 6
            n_classes_per_head: 2n,
            class_weights: variant('none', null),
        }),
        learning_rate: variant('some', 0.01),
        max_epochs: variant('some', 10n),
        patience: variant('some', 5n),
        batch_size: variant('some', 2n),
        dropout: variant('none', null),
        gradient_clip: variant('none', null),
        weight_decay: variant('none', null),
        random_state: variant('some', 42n),
        epoch_callback: variant('none', null),
    });

    $(Assert.throws(
        Lightning.train(X, X, config, variant('none', null), variant('none', null)),
        /n_heads \(4\) must equal n_channels \* sequence_length \(2 \* 3 = 6\)/
    ));
});
```

## Implementation Checklist

### TypeScript (`lightning.ts`)

- [ ] Add `conv1d` variant to `LightningArchitectureType`
- [ ] Add `sequential` variant to `LightningArchitectureType` (with `cell_type` as variant)
- [ ] Add `transformer` variant to `LightningArchitectureType` (with `n_attention_heads`)
- [ ] Update `LightningModelBlobType` to include `architecture_type` for new types

### Python (`lightning_impl.py`)

- [ ] Implement `Conv1DAutoencoder` class
- [ ] Implement `SequentialAutoencoder` class (with cell_type validation)
- [ ] Implement `TransformerAutoencoder` class (with d_model/n_attention_heads validation)
- [ ] Update `LightningMLP.__init__` to handle new architecture types
- [ ] Update `LightningMLP.encode()` to support temporal architectures
- [ ] Update `LightningMLP.decode()` to support temporal architectures
- [ ] Implement `_validate_temporal_config()` with all validations:
  - [ ] Temporal architectures require multi_head output
  - [ ] n_heads == n_channels * sequence_length
  - [ ] conv_channels non-empty
  - [ ] kernel_size odd and >= 1
  - [ ] d_model divisible by n_attention_heads
- [ ] Update model serialization to save/load new architecture types

### Tests (`lightning.spec.ts`)

- [ ] Add conv1d train/encode/decode test
- [ ] Add sequential LSTM test
- [ ] Add sequential GRU bidirectional test
- [ ] Add transformer test
- [ ] Add temporal with masks and group weights test
- [ ] Add error test for encode on mlp
- [ ] Add error test for temporal with non-multi_head output
- [ ] Add error test for n_heads mismatch

## Performance Considerations

### Architecture Choice

| Architecture | Best For | Complexity | Memory |
|--------------|----------|------------|--------|
| Conv1D | Local patterns (adjacent days) | Low | Low |
| LSTM/GRU | Long-range dependencies | Medium | Medium |
| Transformer | Complex attention patterns | High | High |

For 14-day fermentation data with ~1000 samples:
- **Conv1D recommended** - short sequences, local patterns matter most
- LSTM/GRU viable but may be overkill
- Transformer likely overkill (needs more data)

### Kernel Size for Conv1D

- `kernel_size=3`: looks at day-1, day, day+1 (recommended for fermentation)
- `kernel_size=5`: looks at 5-day window
- `kernel_size=7`: looks at week-long window

## Conditional Generation

Temporal architectures enable generating coherent plans given batch characteristics (grade, variety, target metrics). The decoder learns temporal dependencies, so generated plans respect "if acid on day 1, skip day 2" patterns.

### Conditioning Approach

Add optional `condition_dim` to temporal architectures. When provided:
- Encoder: `encode(x)` → latent (unchanged)
- Decoder: `decode(z, condition)` → plan (condition concatenated to latent)
- Training: `forward(x, condition)` → reconstructed (decoder uses condition during training)

At inference:
- Provide batch features as condition
- Sample or interpolate latent from similar batches
- Decode to generate coherent plan

### Training with Conditions

**Critical**: The decoder must see conditions during training, otherwise it cannot learn to use them. This requires a 6th parameter to `lightning_train`:

```typescript
export const lightning_train = East.platform(
    "lightning_train",
    [
        MatrixType,                      // X features
        MatrixType,                      // y targets
        LightningConfigType,             // config
        OptionType(MaskType),            // masks
        OptionType(GroupWeightsType),    // group_weights
        OptionType(MatrixType),          // NEW: conditions (n_samples, condition_dim)
    ],
    LightningResultType
);
```

When `conditions` is provided and `condition_dim` is set in the architecture config:
- Training uses `forward(x, condition)` which calls `decode(encode(x), condition)`
- Decoder learns to incorporate condition features into reconstruction
- At inference, different conditions produce different outputs from the same latent

### Type Changes

Add `condition_dim` to each temporal architecture variant:

```typescript
conv1d: StructType({
    n_channels: IntegerType,
    sequence_length: IntegerType,
    conv_channels: ArrayType(IntegerType),
    kernel_size: IntegerType,
    latent_dim: IntegerType,
    condition_dim: OptionType(IntegerType),  // NEW: dimension of condition vector
}),

sequential: StructType({
    n_channels: IntegerType,
    sequence_length: IntegerType,
    hidden_size: IntegerType,
    n_layers: IntegerType,
    cell_type: StringType,
    latent_dim: IntegerType,
    bidirectional: BooleanType,
    condition_dim: OptionType(IntegerType),  // NEW
}),

transformer: StructType({
    n_channels: IntegerType,
    sequence_length: IntegerType,
    d_model: IntegerType,
    n_heads: IntegerType,
    n_layers: IntegerType,
    d_ff: OptionType(IntegerType),
    latent_dim: IntegerType,
    condition_dim: OptionType(IntegerType),  // NEW
}),
```

Add new platform function for conditional decoding:

```typescript
/**
 * Decode latent with condition vector (temporal architectures with condition_dim).
 *
 * @param model - Trained model with condition_dim set
 * @param z - Latent embeddings matrix (n_samples, latent_dim)
 * @param condition - Condition vectors (n_samples, condition_dim)
 * @returns Decoded output matrix (n_samples, output_dim)
 */
export const lightning_decode_conditional = East.platform(
    "lightning_decode_conditional",
    [LightningModelBlobType, MatrixType, MatrixType],
    MatrixType
);
```

### Python Implementation

Update temporal autoencoder classes to accept optional condition:

```python
class Conv1DAutoencoder(nn.Module):
    def __init__(
        self,
        n_channels: int,
        sequence_length: int,
        n_classes: int,
        conv_channels: list[int],
        kernel_size: int,
        latent_dim: int,
        condition_dim: int | None = None,  # NEW
        dropout: float = 0.1,
    ):
        super().__init__()
        # ... existing init ...
        self.condition_dim = condition_dim

        # Decoder input: latent + condition (if provided)
        decoder_input_dim = latent_dim + (condition_dim or 0)
        self.decoder_fc = nn.Linear(decoder_input_dim, encoder_output_size)

    def decode(self, z: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
        """Decode from latent, optionally conditioned."""
        if self.condition_dim is not None:
            if condition is None:
                raise ValueError("Model requires condition vector but none provided")
            if condition.shape[1] != self.condition_dim:
                raise ValueError(f"Expected condition_dim={self.condition_dim}, got {condition.shape[1]}")
            z = torch.cat([z, condition], dim=1)
        elif condition is not None:
            raise ValueError("Model has no condition_dim but condition was provided")

        # ... rest of decode unchanged ...

    def forward(self, x: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
        return self.decode(self.encode(x), condition)
```

Add platform function implementation:

```python
@PlatformFunction("lightning_decode_conditional")
def lightning_decode_conditional_impl(
    model_blob: EastVariant,
    z: list[list[float]],
    condition: list[list[float]],
) -> list[list[float]]:
    """Decode latent with condition vector."""
    model = _load_model(model_blob)
    z_tensor = torch.tensor(z, dtype=torch.float32)
    condition_tensor = torch.tensor(condition, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        if hasattr(model, 'net') and hasattr(model.net, 'decode'):
            # Temporal architecture
            output = model.net.decode(z_tensor, condition_tensor)
        else:
            raise ValueError("decode_conditional requires temporal architecture with condition_dim")

        output = model.apply_output_activation(output)

    return numpy_to_east_matrix(output.numpy())
```

### Usage Example: Generate Plan for New Batch

```typescript
// Condition features: [grade_onehot(3), target_alcohol, target_ph, variety_onehot(5)]
// Total condition_dim = 10

// Train conditional model
const config = $.let({
    architecture: variant('conv1d', {
        n_channels: 7n,
        sequence_length: 14n,
        conv_channels: [32n, 64n],
        kernel_size: 3n,
        latent_dim: 16n,
        condition_dim: variant('some', 10n),  // batch features
    }),
    output: variant('multi_head', {
        n_heads: East.value(7n * 14n),
        n_classes_per_head: 4n,
        class_weights: variant('none', null),
    }),
    // ... other config ...
});

// Training: condition = batch features for each sample
const conditions = $.let(train_samples.map(($, s) => [
    ...grade_onehot(s.grade),           // 3 dims
    s.target_alcohol,                    // 1 dim
    s.target_ph,                         // 1 dim
    ...variety_onehot(s.variety),       // 5 dims
]));

// Note: Training passes condition through forward()
// Need to modify train() to accept optional condition matrix
const result = $.let(Lightning.train(X, X, config, variant('none', null), variant('none', null)));

// Generation: new batch
const new_batch_condition = $.let([[
    0.0, 1.0, 0.0,    // Grade B (one-hot)
    13.5,              // target alcohol
    3.4,               // target pH
    1.0, 0.0, 0.0, 0.0, 0.0,  // Shiraz (one-hot)
]]);

// Get latent from similar batches (e.g., average of Grade B Shiraz embeddings)
const similar_latent = $.let(compute_mean_latent_for_grade_variety(embeddings, 'B', 'Shiraz'));

// Generate plan
const generated_plan = $.let(Lightning.decodeConditional(
    result.model,
    similar_latent,
    new_batch_condition
));
// generated_plan: [1, 98 * 4] = probabilities for each (additive, day, bin)
```

### Two-Stage Generation Pipeline

For production use, combine binary + amount stages:

```typescript
// Stage 1: Binary model (which additives, which days) - no conditioning
const binary_config = $.let({
    architecture: variant('autoencoder', {
        encoder_layers: [64n],
        latent_dim: 16n,
        decoder_layers: [64n],
    }),
    output: variant('binary', { pos_weight: variant('some', pos_weights) }),
    // ...
});
// Binary model trained without conditions (uses decode, not decodeConditional)
const binary_model = $.let(Lightning.train(
    X_binary, X_binary, binary_config, masks, group_weights, variant('none', null)
));

// Stage 2: Amount model (how much, given which slots are active) - with conditioning
const amount_config = $.let({
    architecture: variant('conv1d', {
        n_channels: 7n,
        sequence_length: 14n,
        conv_channels: [32n],
        kernel_size: 3n,
        latent_dim: 8n,
        condition_dim: variant('some', 10n),  // batch features
    }),
    output: variant('multi_head', {
        n_heads: East.value(7n * 14n),
        n_classes_per_head: 4n,
        class_weights: variant('none', null),
    }),
    // ...
});
// Train with conditions on samples where binary=1
const amount_model = $.let(Lightning.train(
    X_amounts, X_amounts, amount_config, masks, group_weights, variant('some', amount_conditions)
));

// Generation pipeline
const generate_plan = East.function(
    [BatchFeaturesType],  // grade, variety, targets
    GeneratedPlanType,
    ($, batch_features) => {
        // 1. Encode batch features to condition vector
        const condition = $.let(encode_batch_features(batch_features));

        // 2. Find similar batches, get mean latent
        const binary_latent = $.let(find_similar_binary_latent(batch_features));
        const amount_latent = $.let(find_similar_amount_latent(batch_features));

        // 3. Generate binary mask (which slots have additives)
        const binary_probs = $.let(Lightning.decode(binary_model.model, binary_latent));
        const binary_mask = $.let(binary_probs.map(($, p) => p.greaterThan(0.5)));

        // 4. Generate amounts for active slots
        const amount_probs = $.let(Lightning.decodeConditional(
            amount_model.model,
            amount_latent,
            condition
        ));

        // 5. Combine: use amounts where binary=1, else bin 0
        const plan = $.let(combine_binary_and_amounts(binary_mask, amount_probs));

        return $.return(plan);
    }
);
```

### Tests for Conditional Generation

Add to `lightning.spec.ts`:

```typescript
test("conv1d conditional: train and decode with condition", $ => {
    // 2 channels x 3 time steps x 2 classes = 12 features
    const X = $.let([
        [1.0, 0.0, 0.0, 1.0, 0.0, 1.0,  0.0, 1.0, 1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 0.0, 1.0, 0.0,  1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 1.0, 0.0, 1.0,  0.0, 1.0, 1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0, 1.0, 0.0,  1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
    ]);

    // Condition: 3-dim feature vector per sample
    const conditions = $.let([
        [1.0, 0.0, 0.5],  // condition A
        [0.0, 1.0, 0.8],  // condition B
        [1.0, 0.0, 0.5],  // condition A (same as sample 0)
        [0.5, 0.5, 0.3],  // condition C
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

    // Train with conditions (6th parameter)
    const result = $.let(Lightning.train(X, X, config, variant('none', null), variant('none', null), variant('some', conditions)));
    $(Assert.greaterEqual(result.best_epoch, 0n));

    // Encode (condition not needed for encoding)
    const z = $.let(Lightning.encode(result.model, X));
    $(Assert.equal(z.size(), 4n));
    $(Assert.equal(z.get(0n).size(), 4n));

    // Decode with condition
    const decoded = $.let(Lightning.decodeConditional(result.model, z, conditions));
    $(Assert.equal(decoded.size(), 4n));
    $(Assert.equal(decoded.get(0n).size(), 12n));

    // Same latent + same condition should give similar output
    // Samples 0 and 2 have same condition
    const out0 = $.let(decoded.get(0n));
    const out2 = $.let(decoded.get(2n));
    // They should be more similar than out0 vs out1 (different condition)
});

test("error: decodeConditional on model without condition_dim", $ => {
    const X = $.let([
        [1.0, 0.0, 0.0, 1.0, 0.0, 1.0,  0.0, 1.0, 1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 0.0, 1.0, 0.0,  1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
    ]);

    const config = $.let({
        architecture: variant('conv1d', {
            n_channels: 2n,
            sequence_length: 3n,
            conv_channels: [8n],
            kernel_size: 3n,
            latent_dim: 4n,
            condition_dim: variant('none', null),  // no conditioning
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

    // Train without conditions (6th param is none)
    const result = $.let(Lightning.train(X, X, config, variant('none', null), variant('none', null), variant('none', null)));
    const z = $.let(Lightning.encode(result.model, X));
    const conditions = $.let([[1.0, 0.0], [0.0, 1.0]]);

    $(Assert.throws(
        Lightning.decodeConditional(result.model, z, conditions),
        /Model has no condition_dim but condition was provided/
    ));
});

test("error: decodeConditional with wrong condition_dim", $ => {
    const X = $.let([
        [1.0, 0.0, 0.0, 1.0, 0.0, 1.0,  0.0, 1.0, 1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 0.0, 1.0, 0.0,  1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
    ]);

    // Training conditions (3 dims)
    const train_conditions = $.let([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5]]);

    const config = $.let({
        architecture: variant('conv1d', {
            n_channels: 2n,
            sequence_length: 3n,
            conv_channels: [8n],
            kernel_size: 3n,
            latent_dim: 4n,
            condition_dim: variant('some', 3n),  // expects 3
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
    const result = $.let(Lightning.train(X, X, config, variant('none', null), variant('none', null), variant('some', train_conditions)));
    const z = $.let(Lightning.encode(result.model, X));

    // Try to decode with wrong condition dim (2 instead of 3)
    const wrong_conditions = $.let([[1.0, 0.0], [0.0, 1.0]]);

    $(Assert.throws(
        Lightning.decodeConditional(result.model, z, wrong_conditions),
        /Expected condition_dim=3, got 2/
    ));
});

test("sequential conditional: LSTM with condition", $ => {
    const X = $.let([
        [1.0, 0.0, 0.0, 1.0, 0.0, 1.0,  0.0, 1.0, 1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 0.0, 1.0, 0.0,  1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
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

    // Train with conditions
    const result = $.let(Lightning.train(X, X, config, variant('none', null), variant('none', null), variant('some', conditions)));
    $(Assert.greaterEqual(result.best_epoch, 0n));

    const z = $.let(Lightning.encode(result.model, X));
    const decoded = $.let(Lightning.decodeConditional(result.model, z, conditions));
    $(Assert.equal(decoded.size(), 2n));
    $(Assert.equal(decoded.get(0n).size(), 12n));
});

test("transformer conditional: with condition", $ => {
    const X = $.let([
        [1.0, 0.0, 0.0, 1.0, 0.0, 1.0,  0.0, 1.0, 1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 0.0, 1.0, 0.0,  1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
    ]);
    const conditions = $.let([[1.0, 0.0], [0.0, 1.0]]);

    const config = $.let({
        architecture: variant('transformer', {
            n_channels: 2n,
            sequence_length: 3n,
            d_model: 8n,
            n_attention_heads: 2n,  // 8 / 2 = 4 per head
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

    // Train with conditions
    const result = $.let(Lightning.train(X, X, config, variant('none', null), variant('none', null), variant('some', conditions)));
    $(Assert.greaterEqual(result.best_epoch, 0n));

    const z = $.let(Lightning.encode(result.model, X));
    const decoded = $.let(Lightning.decodeConditional(result.model, z, conditions));
    $(Assert.equal(decoded.size(), 2n));
    $(Assert.equal(decoded.get(0n).size(), 12n));
});
```

### Implementation Checklist (Conditional Generation)

#### TypeScript (`lightning.ts`)

- [ ] Add `condition_dim: OptionType(IntegerType)` to `conv1d` variant
- [ ] Add `condition_dim: OptionType(IntegerType)` to `sequential` variant
- [ ] Add `condition_dim: OptionType(IntegerType)` to `transformer` variant
- [ ] Add 6th parameter `OptionType(MatrixType)` to `lightning_train` for conditions
- [ ] Add `lightning_decode_conditional` platform function
- [ ] Add `decodeConditional` to `Lightning` namespace

#### Python (`lightning_impl.py`)

- [ ] Update `lightning_train_impl` to accept 6th parameter (conditions)
- [ ] Update training loop to pass conditions to `forward()` when provided
- [ ] Update `Conv1DAutoencoder.__init__` to accept `condition_dim`
- [ ] Update `Conv1DAutoencoder.decode` and `forward` to accept optional condition
- [ ] Update `SequentialAutoencoder.__init__` to accept `condition_dim`
- [ ] Update `SequentialAutoencoder.decode` and `forward` to accept optional condition
- [ ] Update `TransformerAutoencoder.__init__` to accept `condition_dim`
- [ ] Update `TransformerAutoencoder.decode` and `forward` to accept optional condition
- [ ] Implement `lightning_decode_conditional_impl` platform function
- [ ] Update model serialization to save `condition_dim`
- [ ] Add validation: if condition_dim set, conditions must be provided during training

#### Tests (`lightning.spec.ts`)

- [ ] Add conv1d conditional train/decode test
- [ ] Add sequential conditional test
- [ ] Add transformer conditional test
- [ ] Add error test: decodeConditional on model without condition_dim
- [ ] Add error test: wrong condition_dim size
- [ ] Add error test: condition_dim set but no conditions provided during training

## Summary

| Aspect | Before | After |
|--------|--------|-------|
| Architecture types | `mlp`, `autoencoder` | + `conv1d`, `sequential`, `transformer` |
| Temporal modeling | Not supported | Supported via reshape + temporal layers |
| Local patterns | Must learn via dense layers | Conv1D captures directly |
| Long-range | Must learn via dense layers | LSTM/GRU/Transformer captures directly |
| encode/decode | Only autoencoder | All temporal architectures |
| Conditional generation | Not supported | `condition_dim` + 6th train param + `decodeConditional` |
| `cell_type` | N/A | Variant type (`lstm` or `gru`) |
| `n_attention_heads` | N/A | Replaces `n_heads` in transformer (avoids collision) |
| Validation | Basic | Comprehensive (output type, shape, kernel_size, d_model) |
| API compatibility | N/A | Fully backward compatible (6th param is optional) |
