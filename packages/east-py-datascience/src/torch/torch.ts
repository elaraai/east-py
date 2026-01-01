/**
 * Copyright (c) 2025 Elara AI Pty Ltd
 * Dual-licensed under AGPL-3.0 and commercial license. See LICENSE for details.
 */

/**
 * PyTorch platform functions for East.
 *
 * Provides neural network models using PyTorch.
 * Uses cloudpickle for model serialization.
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
} from "@elaraai/east";
import { VectorType, MatrixType } from "../types.js";

// Re-export shared types for convenience
export { VectorType, MatrixType } from "../types.js";

// ============================================================================
// Enum Types
// ============================================================================

/**
 * Activation function type for hidden layers.
 */
export const TorchActivationType = VariantType({
    /** Rectified Linear Unit */
    relu: NullType,
    /** Hyperbolic tangent */
    tanh: NullType,
    /** Sigmoid function */
    sigmoid: NullType,
    /** Leaky ReLU */
    leaky_relu: NullType,
});

/**
 * Loss function type for training.
 */
export const TorchLossType = VariantType({
    /** Mean Squared Error (regression) */
    mse: NullType,
    /** Mean Absolute Error (regression) */
    mae: NullType,
    /** Cross Entropy (multi-class classification with integer targets) */
    cross_entropy: NullType,
    /** KL Divergence (distribution matching, use with softmax output) */
    kl_div: NullType,
    /** Binary Cross Entropy (multi-label binary, requires sigmoid output) */
    bce: NullType,
    /** Binary Cross Entropy with Logits (more stable, applies sigmoid internally - do NOT use with sigmoid output_activation) */
    bce_with_logits: NullType,
});

/**
 * Optimizer type for training.
 */
export const TorchOptimizerType = VariantType({
    /** Adam optimizer */
    adam: NullType,
    /** Stochastic Gradient Descent */
    sgd: NullType,
    /** AdamW with weight decay */
    adamw: NullType,
    /** RMSprop optimizer */
    rmsprop: NullType,
});

/**
 * Output activation function type for the final layer.
 * Applied only to the output layer, not hidden layers.
 */
export const TorchOutputActivationType = VariantType({
    /** No activation (linear output) - default */
    none: NullType,
    /** Softmax (outputs sum to 1, for probability distributions) */
    softmax: NullType,
    /** Sigmoid (each output independently in [0,1]) */
    sigmoid: NullType,
});

// ============================================================================
// Output Constraint Types
// ============================================================================

/**
 * Per-row output constraint for structured/constrained outputs.
 *
 * Each row of the output matrix can have its own constraint type,
 * enabling architecturally-enforced constraints like mutual exclusivity
 * or position masking.
 *
 * @example
 * ```ts
 * // Binary output with some positions masked (impossible)
 * variant("binary", { mask: variant("some", [true, true, false, true]), data_mask: variant("none", null) })
 *
 * // Mutually exclusive - exactly one position active (softmax)
 * variant("mutex", { mask: variant("none", null), allow_none: variant("some", true), data_mask: variant("none", null) })
 *
 * // At most 2 positions active
 * variant("at_most", { max_count: 2n, mask: variant("none", null), data_mask: variant("none", null) })
 * ```
 */
export const RowConstraintType = VariantType({
    /** Independent binary outputs (sigmoid), optionally masked */
    binary: StructType({
        /** Which positions are valid (true = valid). None = all valid */
        mask: OptionType(ArrayType(BooleanType)),
        /** Data-derived static mask. None = not applied. Combined with mask via AND. */
        data_mask: OptionType(ArrayType(BooleanType)),
    }),

    /** Mutually exclusive - at most one position active (softmax) */
    mutex: StructType({
        /** Which positions are valid. None = all valid */
        mask: OptionType(ArrayType(BooleanType)),
        /** Allow "none selected" by outputting all zeros when no position dominates */
        allow_none: OptionType(BooleanType),
        /** Data-derived static mask. None = not applied. Combined with mask via AND. */
        data_mask: OptionType(ArrayType(BooleanType)),
        /** Per-class weights for handling class imbalance. Applied via weighted cross-entropy loss. */
        class_weights: OptionType(ArrayType(FloatType)),
    }),

    /** At most N positions active (top-k selection with sigmoid) */
    at_most: StructType({
        /** Maximum number of active positions */
        max_count: IntegerType,
        /** Which positions are valid. None = all valid */
        mask: OptionType(ArrayType(BooleanType)),
        /** Data-derived static mask. None = not applied. Combined with mask via AND. */
        data_mask: OptionType(ArrayType(BooleanType)),
    }),
});

/**
 * Configuration for constrained multi-output.
 *
 * Specifies per-row constraints that are architecturally enforced
 * (not just penalized in the loss). This guarantees constraint
 * satisfaction at inference time.
 */
export const ConstrainedOutputConfigType = StructType({
    /** Constraint for each output row. Length must match output dimension. */
    row_constraints: ArrayType(RowConstraintType),
});

// ============================================================================
// Positive Weight and Prior Types
// ============================================================================

/**
 * Positive weight type for class imbalance handling.
 *
 * Used with BCE losses to weight positive samples more heavily.
 * Computed as: n_negative / n_positive
 */
export const PosWeightType = VariantType({
    /** Single weight applied to all outputs */
    scalar: FloatType,
    /** Per-output weights (length = output_dim) */
    per_output: ArrayType(FloatType),
});

/**
 * Prior regularization configuration.
 *
 * Adds MSE regularization term to push outputs towards prior probabilities.
 */
export const PriorConfigType = StructType({
    /** Prior probabilities per output position */
    values: ArrayType(FloatType),
    /** Lambda weight for the MSE regularization term */
    weight: FloatType,
});

/**
 * Per-sample constraints configuration.
 *
 * Allows specifying masks, weights, and priors that vary per sample
 * rather than being fixed for all samples.
 */
export const SampleConstraintsConfigType = StructType({
    /**
     * Per-sample boolean masks: (n_samples, n_rows, n_cols)
     * True = allowed, False = masked (output forced to 0/-inf)
     */
    masks: OptionType(ArrayType(ArrayType(ArrayType(BooleanType)))),
    /**
     * Per-sample positive weights: (n_samples, output_dim)
     * Weights positive samples more heavily during training
     */
    pos_weights: OptionType(ArrayType(ArrayType(FloatType))),
    /**
     * Per-sample prior values: (n_samples, output_dim)
     * Target probabilities for prior regularization
     */
    priors: OptionType(ArrayType(ArrayType(FloatType))),
    /**
     * Per-sample class weights for mutex rows: (n_samples, n_mutex_rows, n_classes)
     * Overrides static class_weights in mutex constraints for per-sample variation.
     * Applied via weighted cross-entropy loss. Only used for mutex row constraints.
     */
    mutex_class_weights: OptionType(ArrayType(ArrayType(ArrayType(FloatType)))),
});

// ============================================================================
// Config Types
// ============================================================================

/**
 * Configuration for MLP architecture.
 */
export const TorchMLPConfigType = StructType({
    /** Hidden layer sizes, e.g., [64, 32] */
    hidden_layers: ArrayType(IntegerType),
    /** Activation function for hidden layers (default relu) */
    activation: OptionType(TorchActivationType),
    /** Output activation function (default none/linear). Ignored if output_constraints is set. */
    output_activation: OptionType(TorchOutputActivationType),
    /** Dropout rate (default 0.0) */
    dropout: OptionType(FloatType),
    /** Output dimension (default 1) */
    output_dim: OptionType(IntegerType),
    /**
     * Per-row output constraints for structured outputs.
     * When set, overrides output_activation with per-row constraint handling.
     * Each constraint specifies how that row of the output is activated/constrained.
     */
    output_constraints: OptionType(ConstrainedOutputConfigType),
});

/**
 * Configuration for training.
 */
export const TorchTrainConfigType = StructType({
    /** Number of epochs (default 100) */
    epochs: OptionType(IntegerType),
    /** Batch size (default 32) */
    batch_size: OptionType(IntegerType),
    /** Learning rate (default 0.001) */
    learning_rate: OptionType(FloatType),
    /** Loss function (default mse) */
    loss: OptionType(TorchLossType),
    /** Optimizer (default adam) */
    optimizer: OptionType(TorchOptimizerType),
    /** Early stopping patience, 0 = disabled */
    early_stopping: OptionType(IntegerType),
    /** Validation split fraction (default 0.2) */
    validation_split: OptionType(FloatType),
    /** Random seed for reproducibility */
    random_state: OptionType(IntegerType),
    /** Positive class weight for BCE losses (scalar or per-output) */
    pos_weight: OptionType(PosWeightType),
    /** Prior regularization configuration (global) */
    prior: OptionType(PriorConfigType),
    /** Per-sample constraints (masks, pos_weights, priors) */
    sample_constraints: OptionType(SampleConstraintsConfigType),
});

// ============================================================================
// Result Types
// ============================================================================

/**
 * Result type for training.
 */
export const TorchTrainResultType = StructType({
    /** Training loss per epoch */
    train_losses: VectorType,
    /** Validation loss per epoch */
    val_losses: VectorType,
    /** Best epoch (for early stopping) */
    best_epoch: IntegerType,
});

/**
 * Combined result from training (model + metrics).
 */
export const TorchTrainOutputType = StructType({
    /** Trained model blob */
    model: VariantType({
        torch_mlp: StructType({
            data: BlobType,
            n_features: IntegerType,
            hidden_layers: ArrayType(IntegerType),
            output_dim: IntegerType,
        }),
    }),
    /** Training result with losses */
    result: TorchTrainResultType,
});

// ============================================================================
// Model Blob Types
// ============================================================================

/**
 * Model blob type for serialized PyTorch models.
 */
export const TorchModelBlobType = VariantType({
    /** PyTorch MLP model */
    torch_mlp: StructType({
        /** Cloudpickle serialized model */
        data: BlobType,
        /** Number of input features */
        n_features: IntegerType,
        /** Hidden layer sizes */
        hidden_layers: ArrayType(IntegerType),
        /** Output dimension */
        output_dim: IntegerType,
    }),
});

// ============================================================================
// Platform Functions
// ============================================================================

/**
 * Train a PyTorch MLP model.
 *
 * @param X - Feature matrix
 * @param y - Target vector
 * @param mlp_config - MLP architecture configuration
 * @param train_config - Training configuration
 * @returns Model blob and training result
 */
export const torch_mlp_train = East.platform(
    "torch_mlp_train",
    [MatrixType, VectorType, TorchMLPConfigType, TorchTrainConfigType],
    TorchTrainOutputType
);

/**
 * Make predictions with a trained PyTorch MLP.
 *
 * @param model - Trained MLP model blob
 * @param X - Feature matrix
 * @returns Predicted values
 */
export const torch_mlp_predict = East.platform(
    "torch_mlp_predict",
    [TorchModelBlobType, MatrixType],
    VectorType
);

/**
 * Train a PyTorch MLP model with multi-output support.
 *
 * Supports multi-output regression (predicting multiple values per sample)
 * and autoencoders (where input equals target for reconstruction learning).
 * Output dimension is inferred from y.shape[1] unless overridden in config.
 *
 * @param X - Feature matrix (n_samples x n_features)
 * @param y - Target matrix (n_samples x n_outputs)
 * @param mlp_config - MLP architecture configuration
 * @param train_config - Training configuration
 * @returns Model blob and training result
 */
export const torch_mlp_train_multi = East.platform(
    "torch_mlp_train_multi",
    [MatrixType, MatrixType, TorchMLPConfigType, TorchTrainConfigType],
    TorchTrainOutputType
);

/**
 * Make predictions with a trained PyTorch MLP (multi-output).
 *
 * Returns a matrix where each row contains the predicted outputs for a sample.
 *
 * @param model - Trained MLP model blob
 * @param X - Feature matrix (n_samples x n_features)
 * @param sample_masks - Optional per-sample boolean masks (n_samples x n_rows x n_cols)
 * @returns Predicted matrix (n_samples x n_outputs)
 */
export const torch_mlp_predict_multi = East.platform(
    "torch_mlp_predict_multi",
    [TorchModelBlobType, MatrixType, OptionType(ArrayType(ArrayType(ArrayType(BooleanType))))],
    MatrixType
);

/**
 * Extract intermediate layer activations (embeddings) from a trained MLP.
 *
 * For autoencoders, this allows extracting the bottleneck representation.
 * The layer_index specifies which hidden layer's output to return (0-indexed).
 *
 * For an autoencoder with architecture [input -> 8 -> 2 -> 8 -> output]
 * (hidden_layers: [8, 2, 8]):
 * - layer_index=0: output after first hidden layer (8 features)
 * - layer_index=1: output after second hidden layer (2 features) <- bottleneck
 * - layer_index=2: output after third hidden layer (8 features)
 *
 * @param model - Trained MLP model blob
 * @param X - Feature matrix (n_samples x n_features)
 * @param layer_index - Which hidden layer's output to return (0-indexed)
 * @returns Embedding matrix (n_samples x hidden_dim at that layer)
 *
 * @example
 * ```ts
 * // Train autoencoder: 4 features -> 8 -> 2 (bottleneck) -> 8 -> 4 features
 * const mlp_config = $.let({
 *     hidden_layers: [8n, 2n, 8n],
 *     activation: variant('some', variant('relu', {})),
 *     dropout: variant('none', null),
 *     output_dim: variant('none', null),
 * });
 * const output = $.let(Torch.mlpTrainMulti(X, X, mlp_config, train_config));
 *
 * // Extract bottleneck embeddings (layer_index=1 for the 2-dim bottleneck)
 * const embeddings = $.let(Torch.mlpEncode(output.model, X, 1n));
 * // embeddings is now (n_samples x 2)
 * ```
 */
export const torch_mlp_encode = East.platform(
    "torch_mlp_encode",
    [TorchModelBlobType, MatrixType, IntegerType],
    MatrixType
);

/**
 * Decode embeddings back through the decoder portion of an MLP.
 *
 * For autoencoders, this takes bottleneck activations and runs them through
 * the decoder to reconstruct the output. This is the complement to mlpEncode.
 *
 * For an autoencoder with architecture [input -> 8 -> 2 -> 8 -> output]
 * (hidden_layers: [8, 2, 8]):
 * - layer_index=1: Start from the 2-dim bottleneck, run through layers 2+ to output
 * - layer_index=0: Start from the 8-dim first layer, run through layers 1+ to output
 *
 * Use case: Compute weighted average of origin embeddings, then decode to
 * get the reconstructed blend weight distribution.
 *
 * @param model - Trained MLP model blob
 * @param embeddings - Embedding matrix (n_samples x hidden_dim at layer_index)
 * @param layer_index - Which hidden layer the embeddings come from (0-indexed)
 * @returns Decoded output matrix (n_samples x output_dim)
 *
 * @example
 * ```ts
 * // After training autoencoder and extracting embeddings...
 * const origin_embeddings = $.let(Torch.mlpEncode(output.model, X_onehot, 1n));
 *
 * // Compute weighted blend embedding (e.g., 50% origin A + 50% origin B)
 * const blend_embedding = $.let(...); // weighted average of origin embeddings
 *
 * // Decode back to weight distribution
 * const reconstructed = $.let(Torch.mlpDecode(output.model, blend_embedding, 1n));
 * ```
 */
export const torch_mlp_decode = East.platform(
    "torch_mlp_decode",
    [TorchModelBlobType, MatrixType, IntegerType],
    MatrixType
);

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Compute pos_weight from target data for class imbalance handling.
 *
 * For binary classification with imbalanced classes, pos_weight compensates
 * by weighting positive samples more heavily. Computed as: n_negative / n_positive
 *
 * @param y - Target matrix (n_samples x output_dim) with binary values (0/1)
 * @param per_output - If true, compute per-output weights; otherwise compute scalar
 * @returns PosWeightType variant: either scalar(float) or per_output(array of floats)
 */
export const torch_compute_pos_weight = East.platform(
    "torch_compute_pos_weight",
    [MatrixType, BooleanType],
    PosWeightType
);

/**
 * Compute data_mask from target data for constraint configuration.
 *
 * Identifies which output positions have any non-zero values across samples.
 * Used to create static masks that exclude positions that are never active.
 *
 * @param y - Target matrix (n_samples x output_dim) with values
 * @param threshold - Values > threshold are considered active (default 0.0)
 * @returns Boolean array (output_dim,): True for positions with any active values
 */
export const torch_compute_data_mask = East.platform(
    "torch_compute_data_mask",
    [MatrixType, FloatType],
    ArrayType(BooleanType)
);

/**
 * Compute class weights for mutex rows based on class frequencies.
 *
 * For each mutex row, computes inverse frequency weights that help handle
 * class imbalance. Classes with fewer samples get higher weights.
 *
 * Formula: weight[c] = min(cap, (n_total / n_classes) / (n_class_c + smoothing))
 *
 * @param y - Target matrix (n_samples x output_dim) with one-hot encoded values
 * @param n_rows - Number of constraint rows (used to determine n_cols = output_dim / n_rows)
 * @param mutex_row_indices - Indices of mutex rows to compute weights for (0-indexed)
 * @returns Array of weight arrays: (n_mutex_rows x n_classes)
 */
export const torch_compute_mutex_class_weights = East.platform(
    "torch_compute_mutex_class_weights",
    [MatrixType, IntegerType, ArrayType(IntegerType)],
    ArrayType(ArrayType(FloatType))
);

// ============================================================================
// Grouped Export
// ============================================================================

/**
 * Type definitions for PyTorch functions.
 */
export const TorchTypes = {
    /** Vector type (array of floats) */
    VectorType,
    /** Matrix type (2D array of floats) */
    MatrixType,
    /** Activation function type for hidden layers */
    TorchActivationType,
    /** Output activation function type */
    TorchOutputActivationType,
    /** Loss function type */
    TorchLossType,
    /** Optimizer type */
    TorchOptimizerType,
    /** Per-row output constraint type */
    RowConstraintType,
    /** Constrained output configuration type */
    ConstrainedOutputConfigType,
    /** Positive weight type (scalar or per-output) */
    PosWeightType,
    /** Prior regularization configuration type */
    PriorConfigType,
    /** Per-sample constraints configuration type */
    SampleConstraintsConfigType,
    /** MLP configuration type */
    TorchMLPConfigType,
    /** Training configuration type */
    TorchTrainConfigType,
    /** Training result type */
    TorchTrainResultType,
    /** Training output type (model + result) */
    TorchTrainOutputType,
    /** Model blob type for PyTorch models */
    ModelBlobType: TorchModelBlobType,
} as const;

/**
 * PyTorch neural network models.
 *
 * Provides MLP training and inference using PyTorch.
 *
 * @example
 * ```ts
 * import { East, variant } from "@elaraai/east";
 * import { Torch } from "@elaraai/east-py-datascience";
 *
 * const train = East.function([], Torch.Types.TorchTrainOutputType, $ => {
 *     const X = $.let([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]);
 *     const y = $.let([1.0, 2.0, 3.0, 4.0]);
 *     const mlp_config = $.let({
 *         hidden_layers: [32n, 16n],
 *         activation: variant('none', null),
 *         dropout: variant('none', null),
 *         output_dim: variant('none', null),
 *     });
 *     const train_config = $.let({
 *         epochs: variant('some', 50n),
 *         batch_size: variant('some', 4n),
 *         learning_rate: variant('some', 0.01),
 *         loss: variant('none', null),
 *         optimizer: variant('none', null),
 *         early_stopping: variant('none', null),
 *         validation_split: variant('some', 0.2),
 *         random_state: variant('some', 42n),
 *     });
 *     return $.return(Torch.mlpTrain(X, y, mlp_config, train_config));
 * });
 * ```
 */
export const Torch = {
    /** Train MLP model (single output) */
    mlpTrain: torch_mlp_train,
    /** Make predictions with MLP (single output) */
    mlpPredict: torch_mlp_predict,
    /** Train MLP model (multi-output) */
    mlpTrainMulti: torch_mlp_train_multi,
    /** Make predictions with MLP (multi-output, with optional sample_masks) */
    mlpPredictMulti: torch_mlp_predict_multi,
    /** Extract intermediate layer activations (embeddings) from MLP */
    mlpEncode: torch_mlp_encode,
    /** Decode embeddings back through decoder portion of MLP */
    mlpDecode: torch_mlp_decode,
    /** Compute pos_weight from target data for class imbalance */
    computePosWeight: torch_compute_pos_weight,
    /** Compute data_mask from target data for constraint configuration */
    computeDataMask: torch_compute_data_mask,
    /** Compute class weights for mutex rows based on class frequencies */
    computeMutexClassWeights: torch_compute_mutex_class_weights,
    /** Type definitions */
    Types: TorchTypes,
} as const;
