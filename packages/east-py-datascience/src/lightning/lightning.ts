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
// Decision Transformer Types
// ===========================================

/**
 * Return embedding mode for Decision Transformer.
 */
export const ReturnEmbeddingType = VariantType({
    /** Single return value for entire sequence */
    global: NullType,
    /** Return-to-go at each timestep */
    per_timestep: NullType,
});

/**
 * Per-head output configuration for multi_head_mixed.
 */
export const HeadConfigType = StructType({
    /** Output type: binary (1 logit, sigmoid, BCE) or multiclass (n_classes logits, softmax, CE) */
    head_type: VariantType({
        /** Single binary output: 1 logit, sigmoid, BCE loss */
        binary: NullType,
        /** Multi-class output: n_classes logits, softmax, CE loss */
        multiclass: StructType({
            n_classes: IntegerType,
        }),
    }),
    /** Optional class weights for this head */
    class_weights: OptionType(VectorType),
    /** Optional: index of head this depends on (loss only computed when that head is 1) */
    conditional_on: OptionType(IntegerType),
});

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
    /**
     * Mixed output types per head.
     * For Decision Transformer: combines binary (1 logit) and multiclass (n_classes logits) heads.
     * Binary heads: 1 logit → sigmoid → BCE loss
     * Multiclass heads: n_classes logits → softmax → CE loss
     * Action vectors use one-hot encoding for multiclass heads.
     */
    multi_head_mixed: StructType({
        /** Array of head configurations */
        heads: ArrayType(HeadConfigType),
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
    /**
     * Decision Transformer: return-conditioned sequence generation.
     * Token layout: [R, s_0, a_0, s_1, a_1, ..., s_{T-1}, a_{T-1}]
     * Predicts actions conditioned on desired return and state history.
     */
    decision_transformer: StructType({
        /** Sequence length (timesteps) */
        sequence_length: IntegerType,
        /** State dimension per timestep */
        state_dim: IntegerType,
        /** Action dimension per timestep */
        action_dim: IntegerType,
        /** Model dimension (transformer hidden size) */
        d_model: IntegerType,
        /** Number of attention heads */
        n_attention_heads: IntegerType,
        /** Number of transformer layers */
        n_layers: IntegerType,
        /** Feedforward dimension (default: 4 * d_model) */
        d_ff: OptionType(IntegerType),
        /** Dropout rate */
        dropout: OptionType(FloatType),
        /** Whether return is per-timestep or global */
        return_embedding: ReturnEmbeddingType,
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

/**
 * Configuration for autoregressive sequence generation.
 */
export const LightningGenerateConfigType = StructType({
    /** Number of steps to generate */
    n_steps: IntegerType,
    /** Sampling temperature: 0.0 = argmax, > 0 = scaled sampling */
    temperature: FloatType,
    /** If true, return probabilities. If false, return samples. */
    return_probs: BooleanType,
});

/**
 * Generate sequence autoregressively from a sequential model.
 *
 * Shapes:
 * - prefix: (n_prefix_steps, n_channels) - partial history to continue from, can be empty []
 * - condition: (1, condition_dim) - conditioning features, or none
 * - returns: (n_steps, n_channels) - generated timesteps only (not including prefix)
 *
 * @param model - Trained sequential model blob
 * @param prefix - Partial history to continue from
 * @param condition - Optional conditioning features
 * @param config - Generation configuration
 * @returns Generated sequence matrix
 */
export const lightning_generate_sequence = East.platform(
    "lightning_generate_sequence",
    [LightningModelBlobType, MatrixType, OptionType(MatrixType), LightningGenerateConfigType],
    MatrixType
);

// ===========================================
// Decision Transformer Platform Functions
// ===========================================

/**
 * Configuration for Decision Transformer trajectory generation.
 */
export const TrajectoryGenerateConfigType = StructType({
    /** Sampling temperature (0.0 = argmax, > 0 = stochastic) */
    temperature: FloatType,
    /** Whether to return probabilities or samples */
    return_probs: BooleanType,
    /** Optional constraint mask: (seq_len, action_dim) - FALSE disables action */
    action_constraints: OptionType(MatrixType),
    /** Optional temporal mask: (seq_len,) - FALSE marks invalid timesteps */
    temporal_mask: OptionType(VectorType),
    /** Optional head configs for multi_head_mixed output (enables proper multiclass sampling) */
    head_configs: OptionType(ArrayType(HeadConfigType)),
    /** Optional action prefix: (seq_len, action_dim) - known actions for timesteps 0..start_timestep-1 */
    action_prefix: OptionType(MatrixType),
    /** Timestep to start generation from (0 = generate all, 5 = use prefix for 0-4, generate 5+) */
    start_timestep: OptionType(IntegerType),
});

/**
 * Train with trajectory data for return-conditioned sequence generation.
 *
 * Use with decision_transformer architecture.
 *
 * @param returns - Return per sample (n_samples,) - actual outcome achieved
 * @param states - State matrices: n_samples × (seq_len, state_dim)
 * @param actions - Action matrices: n_samples × (seq_len, action_dim)
 * @param masks - Temporal masks: n_samples × (seq_len,) - valid timesteps
 * @param config - Training configuration with decision_transformer architecture
 * @returns Training result with model blob and metrics
 */
export const lightning_train_trajectory = East.platform(
    "lightning_train_trajectory",
    [
        VectorType,              // returns: (n_samples,)
        ArrayType(MatrixType),   // states: n_samples × (seq_len, state_dim)
        ArrayType(MatrixType),   // actions: n_samples × (seq_len, action_dim)
        ArrayType(VectorType),   // masks: n_samples × (seq_len,)
        LightningConfigType,     // config with decision_transformer architecture
    ],
    LightningResultType
);

/**
 * Generate action sequences autoregressively from trajectory model.
 *
 * Use with models trained via trainTrajectory.
 *
 * @param model - Trained model from trainTrajectory
 * @param states - State matrices: n_samples × (seq_len, state_dim)
 * @param target_returns - Target returns: (n_samples,)
 * @param config - Generation configuration
 * @returns Generated actions: n_samples × (seq_len, action_dim)
 */
export const lightning_generate_trajectory = East.platform(
    "lightning_generate_trajectory",
    [
        LightningModelBlobType,  // Trained model from trainTrajectory
        ArrayType(MatrixType),   // states: n_samples × (seq_len, state_dim)
        VectorType,              // target_returns: (n_samples,)
        TrajectoryGenerateConfigType,
    ],
    ArrayType(MatrixType)  // Generated actions: n_samples × (seq_len, action_dim)
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
    GenerateConfigType: LightningGenerateConfigType,
    // Decision Transformer types
    ReturnEmbeddingType,
    HeadConfigType,
    TrajectoryGenerateConfigType,
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
     * Generate sequence autoregressively.
     *
     * Generates a sequence from a trained sequential model, optionally
     * continuing from a prefix and conditioned on input features.
     */
    generateSequence: lightning_generate_sequence,

    /**
     * Train a Decision Transformer with trajectory data.
     *
     * Trains a return-conditioned sequence generation model that learns
     * to predict actions given states and desired returns.
     *
     * @example
     * ```typescript
     * const result = Lightning.trainTrajectory(
     *     returns, states, actions, masks,
     *     {
     *         architecture: variant("decision_transformer", {
     *             sequence_length: 14n,
     *             state_dim: 8n,
     *             action_dim: 11n,
     *             d_model: 64n,
     *             n_attention_heads: 4n,
     *             n_layers: 3n,
     *             d_ff: variant("none", null),
     *             dropout: variant("some", 0.1),
     *             return_embedding: variant("global", null),
     *         }),
     *         output: variant("multi_head_mixed", { heads: [...] }),
     *         ...
     *     }
     * );
     * ```
     */
    trainTrajectory: lightning_train_trajectory,

    /**
     * Generate action sequences from a Decision Transformer.
     *
     * Autoregressively generates actions conditioned on target returns
     * and state sequences.
     */
    generateTrajectory: lightning_generate_trajectory,

    /**
     * Type definitions for Lightning functions.
     */
    Types: LightningTypes,
} as const;
