/**
 * Copyright (c) 2025 Elara AI Pty Ltd
 * Dual-licensed under AGPL-3.0 and commercial license. See LICENSE for details.
 */

/**
 * Lightning platform functions for East.
 *
 * Provides production-grade neural network training using PyTorch Lightning.
 * Supports regression, binary classification, multiclass classification,
 * and multi-head categorical outputs.
 *
 * @packageDocumentation
 */

import {
    East,
    StructType,
    VariantType,
    OptionType,
    IntegerType,
    FloatType,
    BlobType,
    ArrayType,
    NullType,
    BooleanType,
    FunctionType,
    StringType,
} from "@elaraai/east";
import { VectorType, MatrixType } from "../types.js";

// Re-export shared types
export { VectorType, MatrixType } from "../types.js";

// ===========================================
// Type Definitions
// ===========================================

/**
 * Lightning output mode - determines loss function and output activation.
 */
export const LightningOutputType = VariantType({
    /** Regression: MSE loss, no activation */
    regression: NullType,
    /** Binary: BCE loss, sigmoid activation */
    binary: StructType({
        /** Optional per-position pos_weights for class imbalance [output_dim] */
        pos_weight: OptionType(VectorType),
    }),
    /** Multiclass: CrossEntropy loss, softmax activation */
    multiclass: StructType({
        /** Number of classes */
        n_classes: IntegerType,
        /** Optional per-class weights */
        class_weights: OptionType(VectorType),
    }),
    /** Multi-head categorical: N independent CrossEntropy heads */
    multi_head: StructType({
        /** Number of heads (e.g., 84 time slots) */
        n_heads: IntegerType,
        /** Classes per head (e.g., 4 bins) */
        n_classes_per_head: IntegerType,
        /** Optional class weights matrix (n_heads, n_classes) */
        class_weights: OptionType(MatrixType),
    }),
});

/**
 * Cell type for sequential architectures.
 */
export const CellType = VariantType({
    lstm: NullType,
    gru: NullType,
});

/**
 * Lightning architecture type.
 */
export const LightningArchitectureType = VariantType({
    /** Simple MLP: input → hidden → output */
    mlp: StructType({
        /** Hidden layer sizes */
        hidden_layers: ArrayType(IntegerType),
    }),
    /** Autoencoder: input → encoder → latent → decoder → output */
    autoencoder: StructType({
        /** Encoder hidden layer sizes */
        encoder_layers: ArrayType(IntegerType),
        /** Latent dimension (bottleneck) */
        latent_dim: IntegerType,
        /** Decoder hidden layer sizes */
        decoder_layers: ArrayType(IntegerType),
    }),
    /** Conv1D: 1D convolutional autoencoder for temporal patterns */
    conv1d: StructType({
        /** Number of channels (e.g., additive types) */
        n_channels: IntegerType,
        /** Sequence length (e.g., days) */
        sequence_length: IntegerType,
        /** Conv layer channel sizes */
        conv_channels: ArrayType(IntegerType),
        /** Kernel size for convolutions (must be odd) */
        kernel_size: IntegerType,
        /** Latent dimension after flattening */
        latent_dim: IntegerType,
        /** Optional condition dimension for conditional generation */
        condition_dim: OptionType(IntegerType),
    }),
    /** Sequential: LSTM/GRU autoencoder for long-range dependencies */
    sequential: StructType({
        /** Number of channels (e.g., additive types) */
        n_channels: IntegerType,
        /** Sequence length (e.g., days) */
        sequence_length: IntegerType,
        /** RNN hidden size */
        hidden_size: IntegerType,
        /** Number of RNN layers */
        n_layers: IntegerType,
        /** Cell type: lstm or gru */
        cell_type: CellType,
        /** Latent dimension (from final hidden state) */
        latent_dim: IntegerType,
        /** Bidirectional encoder (decoder is always unidirectional) */
        bidirectional: BooleanType,
        /** Optional condition dimension for conditional generation */
        condition_dim: OptionType(IntegerType),
    }),
    /** Transformer: attention-based autoencoder for complex patterns */
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
        /** Optional condition dimension for conditional generation */
        condition_dim: OptionType(IntegerType),
    }),
});

/**
 * Epoch callback function type: (epoch, train_loss, val_loss) -> void
 */
export const LightningEpochCallbackType = FunctionType(
    [IntegerType, FloatType, FloatType],
    NullType
);

/**
 * Lightning training configuration.
 */
export const LightningConfigType = StructType({
    /** Model architecture */
    architecture: LightningArchitectureType,
    /** Output mode (determines loss function) */
    output: LightningOutputType,
    /** Learning rate (default: 1e-3) */
    learning_rate: OptionType(FloatType),
    /** Maximum epochs (default: 100) */
    max_epochs: OptionType(IntegerType),
    /** Early stopping patience (default: 10) */
    patience: OptionType(IntegerType),
    /** Batch size (default: 32) */
    batch_size: OptionType(IntegerType),
    /** Dropout rate (default: 0.1) */
    dropout: OptionType(FloatType),
    /** Gradient clipping value (default: 1.0) */
    gradient_clip: OptionType(FloatType),
    /** L2 regularization weight decay (default: 0) */
    weight_decay: OptionType(FloatType),
    /** Random seed for reproducibility */
    random_state: OptionType(IntegerType),
    /** Optional callback called each epoch */
    epoch_callback: OptionType(LightningEpochCallbackType),
});

/**
 * Lightning model blob structure.
 */
export const LightningModelBlobType = VariantType({
    lightning: StructType({
        /** Serialized model data (state_dict + hparams) */
        data: BlobType,
        /** Input dimension */
        n_features: IntegerType,
        /** Output dimension */
        output_dim: IntegerType,
        /** Architecture type */
        architecture_type: StringType,
        /** Output type */
        output_type: StringType,
        /** Latent dimension (autoencoder only) */
        latent_dim: OptionType(IntegerType),
    }),
});

/**
 * Lightning training result.
 */
export const LightningResultType = StructType({
    /** Trained model blob */
    model: LightningModelBlobType,
    /** Final training loss */
    train_loss: FloatType,
    /** Final validation loss */
    val_loss: FloatType,
    /** Best epoch (for early stopping) */
    best_epoch: IntegerType,
});

/**
 * 3D boolean tensor for masks: (n_samples, n_heads, n_classes)
 */
export const Tensor3DBoolType = ArrayType(ArrayType(ArrayType(BooleanType)));

/**
 * Group-based weights for per-sample class weighting.
 *
 * Instead of per-sample weights (memory-intensive), samples belong to discrete
 * groups (e.g., grades) with different weight configurations per group.
 */
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

// ===========================================
// Platform Functions
// ===========================================

/**
 * Train a Lightning model.
 *
 * @param X - Input features matrix (n_samples, n_features)
 * @param y - Target matrix (n_samples, output_dim)
 * @param config - Training configuration
 * @param masks - Optional 3D boolean masks (n_samples, n_heads, n_classes)
 * @param group_weights - Optional group-based weights for per-sample weighting
 * @param conditions - Optional condition matrix for conditional generation (n_samples, condition_dim)
 * @returns Training result with model blob and metrics
 */
export const lightning_train = East.platform(
    "lightning_train",
    [MatrixType, MatrixType, LightningConfigType, OptionType(Tensor3DBoolType), OptionType(GroupWeightsType), OptionType(MatrixType)],
    LightningResultType
);

/**
 * Predict using a Lightning model.
 *
 * @param model - Trained model blob
 * @param X - Input features matrix (n_samples, n_features)
 * @param masks - Optional 3D boolean masks for inference
 * @param conditions - Optional condition matrix for conditional models (n_samples, condition_dim)
 * @returns Predicted probabilities matrix (n_samples, output_dim)
 */
export const lightning_predict = East.platform(
    "lightning_predict",
    [LightningModelBlobType, MatrixType, OptionType(Tensor3DBoolType), OptionType(MatrixType)],
    MatrixType
);

/**
 * Encode input to latent space (autoencoder only).
 *
 * @param model - Trained autoencoder model blob
 * @param X - Input features matrix (n_samples, n_features)
 * @returns Latent embeddings matrix (n_samples, latent_dim)
 */
export const lightning_encode = East.platform(
    "lightning_encode",
    [LightningModelBlobType, MatrixType],
    MatrixType
);

/**
 * Decode latent to output (autoencoder only).
 *
 * @param model - Trained autoencoder model blob
 * @param z - Latent embeddings matrix (n_samples, latent_dim)
 * @returns Decoded output matrix (n_samples, output_dim)
 */
export const lightning_decode = East.platform(
    "lightning_decode",
    [LightningModelBlobType, MatrixType],
    MatrixType
);

/**
 * Decode latent to output with condition (temporal architectures with condition_dim).
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

// ===========================================
// Grouped Export
// ===========================================

/**
 * Lightning types namespace.
 */
export const LightningTypes = {
    OutputType: LightningOutputType,
    ArchitectureType: LightningArchitectureType,
    CellType,
    EpochCallbackType: LightningEpochCallbackType,
    ConfigType: LightningConfigType,
    ResultType: LightningResultType,
    ModelBlobType: LightningModelBlobType,
    Tensor3DBoolType,
    GroupWeightsType,
} as const;

/**
 * Lightning platform functions namespace.
 *
 * Provides production-grade neural network training using PyTorch Lightning.
 *
 * @example
 * ```typescript
 * const result = Lightning.train(X, y, {
 *     architecture: variant("autoencoder", {
 *         encoder_layers: [64n],
 *         latent_dim: 16n,
 *         decoder_layers: [64n],
 *     }),
 *     output: variant("multi_head", {
 *         n_heads: 84n,
 *         n_classes_per_head: 4n,
 *         class_weights: variant("none", null),
 *     }),
 * }, variant("none", null));
 *
 * const embeddings = Lightning.encode(result.model, X);
 * const predictions = Lightning.predict(result.model, X, variant("none", null));
 * ```
 */
export const Lightning = {
    /**
     * Train a Lightning model.
     *
     * Trains a neural network using PyTorch Lightning with early stopping,
     * gradient clipping, and optional epoch callbacks.
     */
    train: lightning_train,

    /**
     * Predict using a Lightning model.
     *
     * Returns predictions from a trained model with optional mask support
     * for multi-head outputs.
     */
    predict: lightning_predict,

    /**
     * Encode inputs to latent space.
     *
     * Extracts latent embeddings from an autoencoder model.
     */
    encode: lightning_encode,

    /**
     * Decode latent embeddings to output space.
     *
     * Reconstructs outputs from latent embeddings using an autoencoder model.
     */
    decode: lightning_decode,

    /**
     * Decode latent embeddings with condition vector.
     *
     * Reconstructs outputs from latent embeddings and condition vectors
     * using a temporal architecture model with condition_dim set.
     */
    decodeConditional: lightning_decode_conditional,

    /**
     * Type definitions for Lightning functions.
     */
    Types: LightningTypes,
} as const;
