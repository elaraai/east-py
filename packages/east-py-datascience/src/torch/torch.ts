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
} from "@elaraai/east";
import { VectorType, MatrixType } from "../types.js";

// Re-export shared types for convenience
export { VectorType, MatrixType } from "../types.js";

// ============================================================================
// Enum Types
// ============================================================================

/**
 * Activation function type for neural networks.
 */
export const TorchActivationType = VariantType({
    /** Rectified Linear Unit */
    relu: StructType({}),
    /** Hyperbolic tangent */
    tanh: StructType({}),
    /** Sigmoid function */
    sigmoid: StructType({}),
    /** Leaky ReLU */
    leaky_relu: StructType({}),
});

/**
 * Loss function type for training.
 */
export const TorchLossType = VariantType({
    /** Mean Squared Error (regression) */
    mse: StructType({}),
    /** Mean Absolute Error (regression) */
    mae: StructType({}),
    /** Cross Entropy (classification) */
    cross_entropy: StructType({}),
});

/**
 * Optimizer type for training.
 */
export const TorchOptimizerType = VariantType({
    /** Adam optimizer */
    adam: StructType({}),
    /** Stochastic Gradient Descent */
    sgd: StructType({}),
    /** AdamW with weight decay */
    adamw: StructType({}),
    /** RMSprop optimizer */
    rmsprop: StructType({}),
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
    /** Activation function (default relu) */
    activation: OptionType(TorchActivationType),
    /** Dropout rate (default 0.0) */
    dropout: OptionType(FloatType),
    /** Output dimension (default 1) */
    output_dim: OptionType(IntegerType),
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
 * @returns Predicted matrix (n_samples x n_outputs)
 */
export const torch_mlp_predict_multi = East.platform(
    "torch_mlp_predict_multi",
    [TorchModelBlobType, MatrixType],
    MatrixType
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
    /** Activation function type */
    TorchActivationType,
    /** Loss function type */
    TorchLossType,
    /** Optimizer type */
    TorchOptimizerType,
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
    /** Make predictions with MLP (multi-output) */
    mlpPredictMulti: torch_mlp_predict_multi,
    /** Type definitions */
    Types: TorchTypes,
} as const;
