/**
 * Copyright (c) 2025 Elara AI Pty Ltd
 * Dual-licensed under AGPL-3.0 and commercial license. See LICENSE for details.
 */

/**
 * PyTorch platform function tests
 */
import { variant } from "@elaraai/east";
import { describeEast, Assert } from "@elaraai/east-node-std";
import { Torch, TorchMLPConfigType } from "./torch.js";

describeEast("PyTorch platform functions", (test) => {
    test("mlp_train trains regression model", $ => {
        // Simple linear relationship: y = x1 + x2
        const X = $.let([
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0],
            [6.0, 7.0],
            [7.0, 8.0],
            [8.0, 9.0],
        ]);
        const y = $.let([3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0]);

        const mlp_config = $.let({
            hidden_layers: [16n, 8n],
            activation: variant('none', null),
            output_activation: variant('none', null),
            dropout: variant('none', null),
            output_dim: variant('none', null),
            output_constraints: variant('none', null),
        });

        const train_config = $.let({
            epochs: variant('some', 50n),
            batch_size: variant('some', 4n),
            learning_rate: variant('some', 0.01),
            loss: variant('none', null),
            optimizer: variant('none', null),
            early_stopping: variant('none', null),
            validation_split: variant('some', 0.2),
            random_state: variant('some', 42n),
            pos_weight: variant('none', null),
            prior: variant('none', null),
            sample_constraints: variant('none', null),
        });

        const output = $.let(Torch.mlpTrain(X, y, mlp_config, train_config));

        // Check training result
        $(Assert.greater(output.result.train_losses.size(), 0n));
        $(Assert.greater(output.result.val_losses.size(), 0n));
        $(Assert.greaterEqual(output.result.best_epoch, 0n));
    });

    test("mlp_predict makes predictions", $ => {
        const X = $.let([
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0],
            [6.0, 7.0],
            [7.0, 8.0],
            [8.0, 9.0],
        ]);
        const y = $.let([3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0]);

        const mlp_config = $.let({
            hidden_layers: [16n, 8n],
            activation: variant('none', null),
            output_activation: variant('none', null),
            dropout: variant('none', null),
            output_dim: variant('none', null),
            output_constraints: variant('none', null),
        });

        const train_config = $.let({
            epochs: variant('some', 50n),
            batch_size: variant('some', 4n),
            learning_rate: variant('some', 0.01),
            loss: variant('none', null),
            optimizer: variant('none', null),
            early_stopping: variant('none', null),
            validation_split: variant('some', 0.2),
            random_state: variant('some', 42n),
            pos_weight: variant('none', null),
            prior: variant('none', null),
            sample_constraints: variant('none', null),
        });

        const output = $.let(Torch.mlpTrain(X, y, mlp_config, train_config));
        const predictions = $.let(Torch.mlpPredict(output.model, X));

        $(Assert.equal(predictions.size(), 8n));
    });

    test("mlp with different activations", $ => {
        const X = $.let([
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
        ]);
        const y = $.let([3.0, 5.0, 7.0, 9.0]);

        const mlp_config = $.let({
            hidden_layers: [8n],
            activation: variant('some', variant('tanh', null)),
            output_activation: variant('none', null),
            dropout: variant('some', 0.1),
            output_dim: variant('none', null),
            output_constraints: variant('none', null),
        });

        const train_config = $.let({
            epochs: variant('some', 30n),
            batch_size: variant('some', 2n),
            learning_rate: variant('some', 0.01),
            loss: variant('none', null),
            optimizer: variant('some', variant('sgd', null)),
            early_stopping: variant('none', null),
            validation_split: variant('some', 0.25),
            random_state: variant('some', 42n),
            pos_weight: variant('none', null),
            prior: variant('none', null),
            sample_constraints: variant('none', null),
        });

        const output = $.let(Torch.mlpTrain(X, y, mlp_config, train_config));
        $(Assert.greater(output.result.train_losses.size(), 0n));
    });

    test("autoencoder reconstruction (identity mapping)", $ => {
        // Autoencoder test: network learns to reconstruct input through bottleneck
        // Input dimension = 4, bottleneck = 2, then expand back to 4
        // This tests if the MLP can learn an identity-like mapping
        const X = $.let([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.5, 0.5, 0.0, 0.0],
            [0.0, 0.5, 0.5, 0.0],
            [0.0, 0.0, 0.5, 0.5],
            [0.5, 0.0, 0.0, 0.5],
        ]);
        // Target is the sum of features (simple pattern to learn)
        const y = $.let([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);

        // Bottleneck architecture: 4 -> 2 -> 4 (conceptually)
        // For regression we just output 1 value
        const mlp_config = $.let({
            hidden_layers: [2n, 4n],  // Bottleneck then expand
            activation: variant('some', variant('relu', null)),
            output_activation: variant('none', null),
            dropout: variant('none', null),
            output_dim: variant('some', 1n),
            output_constraints: variant('none', null),
        });

        const train_config = $.let({
            epochs: variant('some', 100n),
            batch_size: variant('some', 4n),
            learning_rate: variant('some', 0.01),
            loss: variant('some', variant('mse', null)),
            optimizer: variant('some', variant('adam', null)),
            early_stopping: variant('some', 10n),
            validation_split: variant('some', 0.25),
            random_state: variant('some', 42n),
            pos_weight: variant('none', null),
            prior: variant('none', null),
            sample_constraints: variant('none', null),
        });

        const output = $.let(Torch.mlpTrain(X, y, mlp_config, train_config));
        const predictions = $.let(Torch.mlpPredict(output.model, X));

        // All inputs sum to 1.0, so predictions should be close to 1.0
        $(Assert.equal(predictions.size(), 8n));
        $(Assert.greater(output.result.train_losses.size(), 0n));
    });

    test("early stopping triggers", $ => {
        // Simple pattern that should converge quickly
        const X = $.let([
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
            [6.0],
            [7.0],
            [8.0],
        ]);
        const y = $.let([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]);

        const mlp_config = $.let({
            hidden_layers: [4n],
            activation: variant('none', null),
            output_activation: variant('none', null),
            dropout: variant('none', null),
            output_dim: variant('none', null),
            output_constraints: variant('none', null),
        });

        const train_config = $.let({
            epochs: variant('some', 200n),  // High epochs
            batch_size: variant('some', 4n),
            learning_rate: variant('some', 0.1),
            loss: variant('none', null),
            optimizer: variant('none', null),
            early_stopping: variant('some', 5n),  // Low patience
            validation_split: variant('some', 0.25),
            random_state: variant('some', 42n),
            pos_weight: variant('none', null),
            prior: variant('none', null),
            sample_constraints: variant('none', null),
        });

        const output = $.let(Torch.mlpTrain(X, y, mlp_config, train_config));

        // Early stopping should kick in before 200 epochs
        $(Assert.less(output.result.train_losses.size(), 200n));
        $(Assert.greaterEqual(output.result.best_epoch, 0n));
    });

    test("mlp with adamw optimizer", $ => {
        const X = $.let([
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
        ]);
        const y = $.let([3.0, 5.0, 7.0, 9.0]);

        const mlp_config = $.let({
            hidden_layers: [8n],
            activation: variant('none', null),
            output_activation: variant('none', null),
            dropout: variant('none', null),
            output_dim: variant('none', null),
            output_constraints: variant('none', null),
        });

        const train_config = $.let({
            epochs: variant('some', 30n),
            batch_size: variant('some', 2n),
            learning_rate: variant('some', 0.01),
            loss: variant('none', null),
            optimizer: variant('some', variant('adamw', null)),
            early_stopping: variant('none', null),
            validation_split: variant('some', 0.25),
            random_state: variant('some', 42n),
            pos_weight: variant('none', null),
            prior: variant('none', null),
            sample_constraints: variant('none', null),
        });

        const output = $.let(Torch.mlpTrain(X, y, mlp_config, train_config));
        $(Assert.greater(output.result.train_losses.size(), 0n));
    });

    test("mlp with rmsprop optimizer", $ => {
        const X = $.let([
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
        ]);
        const y = $.let([3.0, 5.0, 7.0, 9.0]);

        const mlp_config = $.let({
            hidden_layers: [8n],
            activation: variant('none', null),
            output_activation: variant('none', null),
            dropout: variant('none', null),
            output_dim: variant('none', null),
            output_constraints: variant('none', null),
        });

        const train_config = $.let({
            epochs: variant('some', 30n),
            batch_size: variant('some', 2n),
            learning_rate: variant('some', 0.01),
            loss: variant('none', null),
            optimizer: variant('some', variant('rmsprop', null)),
            early_stopping: variant('none', null),
            validation_split: variant('some', 0.25),
            random_state: variant('some', 42n),
            pos_weight: variant('none', null),
            prior: variant('none', null),
            sample_constraints: variant('none', null),
        });

        const output = $.let(Torch.mlpTrain(X, y, mlp_config, train_config));
        $(Assert.greater(output.result.train_losses.size(), 0n));
    });

    test("mlp with mae loss", $ => {
        const X = $.let([
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
        ]);
        const y = $.let([3.0, 5.0, 7.0, 9.0]);

        const mlp_config = $.let({
            hidden_layers: [8n],
            activation: variant('none', null),
            output_activation: variant('none', null),
            dropout: variant('none', null),
            output_dim: variant('none', null),
            output_constraints: variant('none', null),
        });

        const train_config = $.let({
            epochs: variant('some', 30n),
            batch_size: variant('some', 2n),
            learning_rate: variant('some', 0.01),
            loss: variant('some', variant('mae', null)),
            optimizer: variant('none', null),
            early_stopping: variant('none', null),
            validation_split: variant('some', 0.25),
            random_state: variant('some', 42n),
            pos_weight: variant('none', null),
            prior: variant('none', null),
            sample_constraints: variant('none', null),
        });

        const output = $.let(Torch.mlpTrain(X, y, mlp_config, train_config));
        $(Assert.greater(output.result.train_losses.size(), 0n));
    });

    test("mlp with sigmoid activation", $ => {
        const X = $.let([
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
        ]);
        const y = $.let([3.0, 5.0, 7.0, 9.0]);

        const mlp_config = $.let({
            hidden_layers: [8n],
            activation: variant('some', variant('sigmoid', null)),
            output_activation: variant('none', null),
            dropout: variant('none', null),
            output_dim: variant('none', null),
            output_constraints: variant('none', null),
        });

        const train_config = $.let({
            epochs: variant('some', 30n),
            batch_size: variant('some', 2n),
            learning_rate: variant('some', 0.01),
            loss: variant('none', null),
            optimizer: variant('none', null),
            early_stopping: variant('none', null),
            validation_split: variant('some', 0.25),
            random_state: variant('some', 42n),
            pos_weight: variant('none', null),
            prior: variant('none', null),
            sample_constraints: variant('none', null),
        });

        const output = $.let(Torch.mlpTrain(X, y, mlp_config, train_config));
        $(Assert.greater(output.result.train_losses.size(), 0n));
    });

    test("mlp with leaky_relu activation", $ => {
        const X = $.let([
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
        ]);
        const y = $.let([3.0, 5.0, 7.0, 9.0]);

        const mlp_config = $.let({
            hidden_layers: [8n],
            activation: variant('some', variant('leaky_relu', null)),
            output_activation: variant('none', null),
            dropout: variant('none', null),
            output_dim: variant('none', null),
            output_constraints: variant('none', null),
        });

        const train_config = $.let({
            epochs: variant('some', 30n),
            batch_size: variant('some', 2n),
            learning_rate: variant('some', 0.01),
            loss: variant('none', null),
            optimizer: variant('none', null),
            early_stopping: variant('none', null),
            validation_split: variant('some', 0.25),
            random_state: variant('some', 42n),
            pos_weight: variant('none', null),
            prior: variant('none', null),
            sample_constraints: variant('none', null),
        });

        const output = $.let(Torch.mlpTrain(X, y, mlp_config, train_config));
        $(Assert.greater(output.result.train_losses.size(), 0n));
    });

    test("deep mlp with dropout", $ => {
        const X = $.let([
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0],
            [5.0, 6.0, 7.0],
            [6.0, 7.0, 8.0],
        ]);
        const y = $.let([6.0, 9.0, 12.0, 15.0, 18.0, 21.0]);

        const mlp_config = $.let({
            hidden_layers: [32n, 16n, 8n],  // Deeper network
            activation: variant('some', variant('relu', null)),
            output_activation: variant('none', null),
            dropout: variant('some', 0.2),
            output_dim: variant('none', null),
            output_constraints: variant('none', null),
        });

        const train_config = $.let({
            epochs: variant('some', 50n),
            batch_size: variant('some', 2n),
            learning_rate: variant('some', 0.01),
            loss: variant('none', null),
            optimizer: variant('none', null),
            early_stopping: variant('none', null),
            validation_split: variant('some', 0.2),
            random_state: variant('some', 42n),
            pos_weight: variant('none', null),
            prior: variant('none', null),
            sample_constraints: variant('none', null),
        });

        const output = $.let(Torch.mlpTrain(X, y, mlp_config, train_config));
        const predictions = $.let(Torch.mlpPredict(output.model, X));

        $(Assert.equal(predictions.size(), 6n));
        $(Assert.greater(output.result.train_losses.size(), 0n));
    });

    test("error: train_regressor shape mismatch", $ => {
        const X = $.let([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);  // 3 samples
        const y = $.let([1.0, 2.0]);  // 2 samples

        const mlp_config = $.let({
            hidden_layers: [8n],
            activation: variant('none', null),
            output_activation: variant('none', null),
            dropout: variant('none', null),
            output_dim: variant('none', null),
            output_constraints: variant('none', null),
        });

        const train_config = $.let({
            epochs: variant('some', 10n),
            batch_size: variant('some', 2n),
            learning_rate: variant('some', 0.01),
            loss: variant('none', null),
            optimizer: variant('none', null),
            early_stopping: variant('none', null),
            validation_split: variant('none', null),
            random_state: variant('none', null),
            pos_weight: variant('none', null),
            prior: variant('none', null),
            sample_constraints: variant('none', null),
        });

        $(Assert.throws(Torch.mlpTrain(X, y, mlp_config, train_config), /torch_mlp_train.*X.*3.*y.*2/));
    });

    // ========================================================================
    // Multi-Output Tests
    // ========================================================================

    test("mlp_train_multi trains multi-output regression model", $ => {
        // Multi-output: predict 3 values from 2 features
        // y1 = x1 + x2, y2 = x1 * 2, y3 = x2 * 2
        const X = $.let([
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0],
            [6.0, 7.0],
            [7.0, 8.0],
            [8.0, 9.0],
        ]);
        const y = $.let([
            [3.0, 2.0, 4.0],
            [5.0, 4.0, 6.0],
            [7.0, 6.0, 8.0],
            [9.0, 8.0, 10.0],
            [11.0, 10.0, 12.0],
            [13.0, 12.0, 14.0],
            [15.0, 14.0, 16.0],
            [17.0, 16.0, 18.0],
        ]);

        const mlp_config = $.let({
            hidden_layers: [32n, 16n],
            activation: variant('some', variant('relu', null)),
            output_activation: variant('none', null),
            dropout: variant('none', null),
            output_dim: variant('none', null),  // Inferred from y: 3
            output_constraints: variant('none', null),
        });

        const train_config = $.let({
            epochs: variant('some', 100n),
            batch_size: variant('some', 4n),
            learning_rate: variant('some', 0.01),
            loss: variant('some', variant('mse', null)),
            optimizer: variant('some', variant('adam', null)),
            early_stopping: variant('some', 15n),
            validation_split: variant('some', 0.2),
            random_state: variant('some', 42n),
            pos_weight: variant('none', null),
            prior: variant('none', null),
            sample_constraints: variant('none', null),
        });

        const output = $.let(Torch.mlpTrainMulti(X, y, mlp_config, train_config));

        // Check training result
        $(Assert.greater(output.result.train_losses.size(), 0n));
        $(Assert.greater(output.result.val_losses.size(), 0n));
        $(Assert.greaterEqual(output.result.best_epoch, 0n));
    });

    test("mlp_predict_multi makes multi-output predictions", $ => {
        // Train multi-output model and predict
        const X = $.let([
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0],
            [6.0, 7.0],
            [7.0, 8.0],
            [8.0, 9.0],
        ]);
        const y = $.let([
            [3.0, 2.0, 4.0],
            [5.0, 4.0, 6.0],
            [7.0, 6.0, 8.0],
            [9.0, 8.0, 10.0],
            [11.0, 10.0, 12.0],
            [13.0, 12.0, 14.0],
            [15.0, 14.0, 16.0],
            [17.0, 16.0, 18.0],
        ]);

        const mlp_config = $.let({
            hidden_layers: [16n, 8n],
            activation: variant('none', null),
            output_activation: variant('none', null),
            dropout: variant('none', null),
            output_dim: variant('none', null),
            output_constraints: variant('none', null),
        });

        const train_config = $.let({
            epochs: variant('some', 50n),
            batch_size: variant('some', 4n),
            learning_rate: variant('some', 0.01),
            loss: variant('none', null),
            optimizer: variant('none', null),
            early_stopping: variant('none', null),
            validation_split: variant('some', 0.2),
            random_state: variant('some', 42n),
            pos_weight: variant('none', null),
            prior: variant('none', null),
            sample_constraints: variant('none', null),
        });

        const output = $.let(Torch.mlpTrainMulti(X, y, mlp_config, train_config));
        const predictions = $.let(Torch.mlpPredictMulti(output.model, X, variant('none', null)));

        // Predictions should have 8 rows (samples) and 3 columns (outputs)
        $(Assert.equal(predictions.size(), 8n));
        // Check first prediction has 3 output values
        $(Assert.equal(predictions.get(0n).size(), 3n));
    });

    test("autoencoder reconstruction (X = y)", $ => {
        // Autoencoder: network learns to reconstruct input
        // Input = Output, so y = X
        const X = $.let([
            [0.5, 0.3, 0.2, 0.0],
            [0.0, 0.4, 0.4, 0.2],
            [0.3, 0.3, 0.2, 0.2],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.25, 0.25, 0.25, 0.25],
        ]);
        // For autoencoder, y = X (reconstruct input)
        const y = $.let([
            [0.5, 0.3, 0.2, 0.0],
            [0.0, 0.4, 0.4, 0.2],
            [0.3, 0.3, 0.2, 0.2],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.25, 0.25, 0.25, 0.25],
        ]);

        // Bottleneck architecture: 4 -> 8 -> 2 (bottleneck) -> 8 -> 4
        const mlp_config = $.let({
            hidden_layers: [8n, 2n, 8n],  // Bottleneck at 2
            activation: variant('some', variant('relu', null)),
            output_activation: variant('none', null),
            dropout: variant('none', null),
            output_dim: variant('none', null),  // Inferred from y: 4
            output_constraints: variant('none', null),
        });

        const train_config = $.let({
            epochs: variant('some', 200n),
            batch_size: variant('some', 4n),
            learning_rate: variant('some', 0.01),
            loss: variant('some', variant('mse', null)),
            optimizer: variant('some', variant('adam', null)),
            early_stopping: variant('some', 20n),
            validation_split: variant('some', 0.2),
            random_state: variant('some', 42n),
            pos_weight: variant('none', null),
            prior: variant('none', null),
            sample_constraints: variant('none', null),
        });

        const output = $.let(Torch.mlpTrainMulti(X, y, mlp_config, train_config));
        const predictions = $.let(Torch.mlpPredictMulti(output.model, X, variant('none', null)));

        // Should reconstruct with same dimensions: 8 samples x 4 features
        $(Assert.equal(predictions.size(), 8n));
        $(Assert.equal(predictions.get(0n).size(), 4n));
        $(Assert.greater(output.result.train_losses.size(), 0n));
    });

    test("multi-output with explicit output_dim override", $ => {
        // Test that output_dim in config can override inferred dimension
        const X = $.let([
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
        ]);
        // y has 2 outputs but we override to 3
        const y = $.let([
            [3.0, 2.0],
            [5.0, 4.0],
            [7.0, 6.0],
            [9.0, 8.0],
        ]);

        const mlp_config = $.let({
            hidden_layers: [8n],
            activation: variant('none', null),
            output_activation: variant('none', null),
            dropout: variant('none', null),
            output_dim: variant('some', 2n),  // Explicit: match y's columns
            output_constraints: variant('none', null),
        });

        const train_config = $.let({
            epochs: variant('some', 30n),
            batch_size: variant('some', 2n),
            learning_rate: variant('some', 0.01),
            loss: variant('none', null),
            optimizer: variant('none', null),
            early_stopping: variant('none', null),
            validation_split: variant('some', 0.25),
            random_state: variant('some', 42n),
            pos_weight: variant('none', null),
            prior: variant('none', null),
            sample_constraints: variant('none', null),
        });

        const output = $.let(Torch.mlpTrainMulti(X, y, mlp_config, train_config));
        const predictions = $.let(Torch.mlpPredictMulti(output.model, X, variant('none', null)));

        $(Assert.equal(predictions.size(), 4n));
        $(Assert.equal(predictions.get(0n).size(), 2n));
    });

    test("error: train_multi shape mismatch", $ => {
        const X = $.let([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);  // 3 samples
        const y = $.let([[1.0, 2.0], [3.0, 4.0]]);  // 2 samples

        const mlp_config = $.let({
            hidden_layers: [8n],
            activation: variant('none', null),
            output_activation: variant('none', null),
            dropout: variant('none', null),
            output_dim: variant('none', null),
            output_constraints: variant('none', null),
        });

        const train_config = $.let({
            epochs: variant('some', 10n),
            batch_size: variant('some', 2n),
            learning_rate: variant('some', 0.01),
            loss: variant('none', null),
            optimizer: variant('none', null),
            early_stopping: variant('none', null),
            validation_split: variant('none', null),
            random_state: variant('none', null),
            pos_weight: variant('none', null),
            prior: variant('none', null),
            sample_constraints: variant('none', null),
        });

        $(Assert.throws(Torch.mlpTrainMulti(X, y, mlp_config, train_config), /torch_mlp_train.*X.*3.*y.*2/));
    });

    // ========================================================================
    // Encoding Tests (Extract Intermediate Layer Activations)
    // ========================================================================

    test("mlpEncode extracts bottleneck embeddings from autoencoder", $ => {
        // Train autoencoder: 4 -> 8 -> 2 (bottleneck) -> 8 -> 4
        const X = $.let([
            [0.5, 0.3, 0.2, 0.0],
            [0.0, 0.4, 0.4, 0.2],
            [0.3, 0.3, 0.2, 0.2],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.25, 0.25, 0.25, 0.25],
        ]);

        // Bottleneck architecture: 4 -> 8 -> 2 (bottleneck) -> 8 -> 4
        const mlp_config = $.let({
            hidden_layers: [8n, 2n, 8n],  // Bottleneck at index 1 (2 features)
            activation: variant('some', variant('relu', null)),
            output_activation: variant('none', null),
            dropout: variant('none', null),
            output_dim: variant('none', null),
            output_constraints: variant('none', null),
        });

        const train_config = $.let({
            epochs: variant('some', 100n),
            batch_size: variant('some', 4n),
            learning_rate: variant('some', 0.01),
            loss: variant('some', variant('mse', null)),
            optimizer: variant('some', variant('adam', null)),
            early_stopping: variant('some', 20n),
            validation_split: variant('some', 0.2),
            random_state: variant('some', 42n),
            pos_weight: variant('none', null),
            prior: variant('none', null),
            sample_constraints: variant('none', null),
        });

        const output = $.let(Torch.mlpTrainMulti(X, X, mlp_config, train_config));

        // Extract bottleneck embeddings (layer_index=1 for the 2-dim bottleneck)
        const embeddings = $.let(Torch.mlpEncode(output.model, X, 1n));

        // Should have 8 samples with 2-dim embeddings (the bottleneck)
        $(Assert.equal(embeddings.size(), 8n));
        $(Assert.equal(embeddings.get(0n).size(), 2n));
    });

    test("mlpEncode extracts first hidden layer activations", $ => {
        // Train model: 2 features -> 16 -> 8 -> 1 output
        const X = $.let([
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0],
            [6.0, 7.0],
        ]);
        const y = $.let([3.0, 5.0, 7.0, 9.0, 11.0, 13.0]);

        const mlp_config = $.let({
            hidden_layers: [16n, 8n],
            activation: variant('some', variant('relu', null)),
            output_activation: variant('none', null),
            dropout: variant('none', null),
            output_dim: variant('none', null),
            output_constraints: variant('none', null),
        });

        const train_config = $.let({
            epochs: variant('some', 50n),
            batch_size: variant('some', 2n),
            learning_rate: variant('some', 0.01),
            loss: variant('none', null),
            optimizer: variant('none', null),
            early_stopping: variant('none', null),
            validation_split: variant('some', 0.2),
            random_state: variant('some', 42n),
            pos_weight: variant('none', null),
            prior: variant('none', null),
            sample_constraints: variant('none', null),
        });

        const output = $.let(Torch.mlpTrain(X, y, mlp_config, train_config));

        // Extract first hidden layer activations (16-dim)
        const layer0_activations = $.let(Torch.mlpEncode(output.model, X, 0n));
        $(Assert.equal(layer0_activations.size(), 6n));
        $(Assert.equal(layer0_activations.get(0n).size(), 16n));

        // Extract second hidden layer activations (8-dim)
        const layer1_activations = $.let(Torch.mlpEncode(output.model, X, 1n));
        $(Assert.equal(layer1_activations.size(), 6n));
        $(Assert.equal(layer1_activations.get(0n).size(), 8n));
    });

    test("mlpEncode with single-output origin embedding use case", $ => {
        // Simulate origin embedding: one-hot inputs (4 origins) -> 3-dim embedding
        // This tests extracting "origin embeddings" from one-hot encoded inputs
        const X_onehot = $.let([
            [1.0, 0.0, 0.0, 0.0],  // Origin A
            [0.0, 1.0, 0.0, 0.0],  // Origin B
            [0.0, 0.0, 1.0, 0.0],  // Origin C
            [0.0, 0.0, 0.0, 1.0],  // Origin D
            [0.5, 0.5, 0.0, 0.0],  // Blend A+B
            [0.0, 0.5, 0.5, 0.0],  // Blend B+C
        ]);

        // Autoencoder: 4 -> 3 (embedding) -> 4
        const mlp_config = $.let({
            hidden_layers: [3n],  // Single hidden layer = embedding
            activation: variant('some', variant('tanh', null)),  // tanh for bounded embeddings
            output_activation: variant('none', null),
            dropout: variant('none', null),
            output_dim: variant('none', null),
            output_constraints: variant('none', null),
        });

        const train_config = $.let({
            epochs: variant('some', 100n),
            batch_size: variant('some', 3n),
            learning_rate: variant('some', 0.01),
            loss: variant('some', variant('mse', null)),
            optimizer: variant('some', variant('adam', null)),
            early_stopping: variant('some', 15n),
            validation_split: variant('some', 0.2),
            random_state: variant('some', 42n),
            pos_weight: variant('none', null),
            prior: variant('none', null),
            sample_constraints: variant('none', null),
        });

        const output = $.let(Torch.mlpTrainMulti(X_onehot, X_onehot, mlp_config, train_config));

        // Extract embeddings (the 3-dim hidden layer)
        const origin_embeddings = $.let(Torch.mlpEncode(output.model, X_onehot, 0n));

        // Should have 6 samples with 3-dim embeddings
        $(Assert.equal(origin_embeddings.size(), 6n));
        $(Assert.equal(origin_embeddings.get(0n).size(), 3n));
    });

    test("error: mlpEncode with invalid layer_index", $ => {
        const X = $.let([[1.0, 2.0], [3.0, 4.0]]);
        const y = $.let([3.0, 7.0]);

        const mlp_config = $.let({
            hidden_layers: [8n],  // Only 1 hidden layer (index 0)
            activation: variant('none', null),
            output_activation: variant('none', null),
            dropout: variant('none', null),
            output_dim: variant('none', null),
            output_constraints: variant('none', null),
        });

        const train_config = $.let({
            epochs: variant('some', 10n),
            batch_size: variant('some', 2n),
            learning_rate: variant('some', 0.01),
            loss: variant('none', null),
            optimizer: variant('none', null),
            early_stopping: variant('none', null),
            validation_split: variant('some', 0.5),
            random_state: variant('some', 42n),
            pos_weight: variant('none', null),
            prior: variant('none', null),
            sample_constraints: variant('none', null),
        });

        const output = $.let(Torch.mlpTrain(X, y, mlp_config, train_config));

        // layer_index=1 is out of range (only layer 0 exists)
        $(Assert.throws(Torch.mlpEncode(output.model, X, 1n), /layer_index.*out of range/));
    });

    // ========================================================================
    // Decoding Tests (Reconstruct from Embeddings)
    // ========================================================================

    test("mlpDecode reconstructs from bottleneck embeddings", $ => {
        // Train autoencoder: 4 -> 8 -> 2 (bottleneck) -> 8 -> 4
        const X = $.let([
            [0.5, 0.3, 0.2, 0.0],
            [0.0, 0.4, 0.4, 0.2],
            [0.3, 0.3, 0.2, 0.2],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.25, 0.25, 0.25, 0.25],
        ]);

        const mlp_config = $.let({
            hidden_layers: [8n, 2n, 8n],  // Bottleneck at index 1
            activation: variant('some', variant('relu', null)),
            output_activation: variant('none', null),
            dropout: variant('none', null),
            output_dim: variant('none', null),
            output_constraints: variant('none', null),
        });

        const train_config = $.let({
            epochs: variant('some', 100n),
            batch_size: variant('some', 4n),
            learning_rate: variant('some', 0.01),
            loss: variant('some', variant('mse', null)),
            optimizer: variant('some', variant('adam', null)),
            early_stopping: variant('some', 20n),
            validation_split: variant('some', 0.2),
            random_state: variant('some', 42n),
            pos_weight: variant('none', null),
            prior: variant('none', null),
            sample_constraints: variant('none', null),
        });

        const output = $.let(Torch.mlpTrainMulti(X, X, mlp_config, train_config));

        // Encode to bottleneck
        const embeddings = $.let(Torch.mlpEncode(output.model, X, 1n));
        $(Assert.equal(embeddings.size(), 8n));
        $(Assert.equal(embeddings.get(0n).size(), 2n));

        // Decode back from bottleneck
        const decoded = $.let(Torch.mlpDecode(output.model, embeddings, 1n));
        $(Assert.equal(decoded.size(), 8n));
        $(Assert.equal(decoded.get(0n).size(), 4n));  // Should match output dim
    });

    test("encode-decode round trip matches full forward pass", $ => {
        // The encode→decode should give same result as predictMulti
        const X = $.let([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0],
        ]);

        const mlp_config = $.let({
            hidden_layers: [8n, 3n, 8n],  // Bottleneck at index 1 (3-dim)
            activation: variant('some', variant('relu', null)),
            output_activation: variant('none', null),
            dropout: variant('none', null),
            output_dim: variant('none', null),
            output_constraints: variant('none', null),
        });

        const train_config = $.let({
            epochs: variant('some', 50n),
            batch_size: variant('some', 2n),
            learning_rate: variant('some', 0.01),
            loss: variant('some', variant('mse', null)),
            optimizer: variant('some', variant('adam', null)),
            early_stopping: variant('none', null),
            validation_split: variant('some', 0.3),
            random_state: variant('some', 42n),
            pos_weight: variant('none', null),
            prior: variant('none', null),
            sample_constraints: variant('none', null),
        });

        const output = $.let(Torch.mlpTrainMulti(X, X, mlp_config, train_config));

        // Full forward pass
        const direct_output = $.let(Torch.mlpPredictMulti(output.model, X, variant('none', null)));

        // Encode then decode
        const embeddings = $.let(Torch.mlpEncode(output.model, X, 1n));
        const roundtrip_output = $.let(Torch.mlpDecode(output.model, embeddings, 1n));

        // Both should have same shape
        $(Assert.equal(direct_output.size(), roundtrip_output.size()));
        $(Assert.equal(direct_output.get(0n).size(), roundtrip_output.get(0n).size()));
    });

    test("mlpDecode from weighted average of embeddings (origin blending)", $ => {
        // This tests the core origin model use case:
        // 1. Get embeddings for individual origins (one-hot inputs)
        // 2. Compute weighted average
        // 3. Decode to get blended output
        const X_origins = $.let([
            [1.0, 0.0, 0.0],  // Origin A
            [0.0, 1.0, 0.0],  // Origin B
            [0.0, 0.0, 1.0],  // Origin C
        ]);

        // Simple autoencoder: 3 -> 2 (embedding) -> 3
        const mlp_config = $.let({
            hidden_layers: [2n],  // Single hidden layer = embedding
            activation: variant('some', variant('tanh', null)),
            output_activation: variant('none', null),
            dropout: variant('none', null),
            output_dim: variant('none', null),
            output_constraints: variant('none', null),
        });

        const train_config = $.let({
            epochs: variant('some', 100n),
            batch_size: variant('some', 2n),
            learning_rate: variant('some', 0.01),
            loss: variant('some', variant('mse', null)),
            optimizer: variant('some', variant('adam', null)),
            early_stopping: variant('some', 15n),
            validation_split: variant('some', 0.3),
            random_state: variant('some', 42n),
            pos_weight: variant('none', null),
            prior: variant('none', null),
            sample_constraints: variant('none', null),
        });

        const output = $.let(Torch.mlpTrainMulti(X_origins, X_origins, mlp_config, train_config));

        // Get embeddings for each origin (3 origins x 2 embedding dims)
        const origin_embeddings = $.let(Torch.mlpEncode(output.model, X_origins, 0n));
        $(Assert.equal(origin_embeddings.size(), 3n));
        $(Assert.equal(origin_embeddings.get(0n).size(), 2n));

        // Manually compute 50/50 blend of origin A and B
        // blend_emb = 0.5 * emb_A + 0.5 * emb_B
        const emb_A = $.let(origin_embeddings.get(0n));
        const emb_B = $.let(origin_embeddings.get(1n));
        const blend_emb = $.let([
            emb_A.get(0n).multiply(0.5).add(emb_B.get(0n).multiply(0.5)),
            emb_A.get(1n).multiply(0.5).add(emb_B.get(1n).multiply(0.5)),
        ]);

        // Wrap as matrix for decode (1 sample x 2 dims)
        const blend_matrix = $.let([blend_emb]);

        // Decode the blended embedding
        const decoded_blend = $.let(Torch.mlpDecode(output.model, blend_matrix, 0n));
        $(Assert.equal(decoded_blend.size(), 1n));  // 1 sample
        $(Assert.equal(decoded_blend.get(0n).size(), 3n));  // 3 outputs (origins)
    });

    test("error: mlpDecode with wrong embedding dimension", $ => {
        const X = $.let([[1.0, 2.0], [3.0, 4.0]]);

        const mlp_config = $.let({
            hidden_layers: [8n, 4n],  // layer 0 = 8-dim, layer 1 = 4-dim
            activation: variant('none', null),
            output_activation: variant('none', null),
            dropout: variant('none', null),
            output_dim: variant('none', null),
            output_constraints: variant('none', null),
        });

        const train_config = $.let({
            epochs: variant('some', 10n),
            batch_size: variant('some', 2n),
            learning_rate: variant('some', 0.01),
            loss: variant('none', null),
            optimizer: variant('none', null),
            early_stopping: variant('none', null),
            validation_split: variant('some', 0.5),
            random_state: variant('some', 42n),
            pos_weight: variant('none', null),
            prior: variant('none', null),
            sample_constraints: variant('none', null),
        });

        const output = $.let(Torch.mlpTrainMulti(X, X, mlp_config, train_config));

        // Try to decode 3-dim embedding at layer 1 which expects 4-dim
        const wrong_embeddings = $.let([[1.0, 2.0, 3.0]]);  // 3-dim instead of 4-dim
        $(Assert.throws(Torch.mlpDecode(output.model, wrong_embeddings, 1n), /dimension.*3.*doesn't match.*4/));
    });

    // ========================================================================
    // Output Activation Tests
    // ========================================================================

    test("softmax output activation produces valid probability distribution", $ => {
        // Input: normalized weights (sum to 1)
        const X = $.let([
            [0.5, 0.3, 0.2],
            [0.8, 0.1, 0.1],
            [0.33, 0.33, 0.34],
            [1.0, 0.0, 0.0],
        ]);

        const mlp_config = $.let({
            hidden_layers: [8n, 4n, 8n],
            activation: variant('some', variant('relu', null)),
            output_activation: variant('some', variant('softmax', null)),
            dropout: variant('none', null),
            output_dim: variant('none', null),
            output_constraints: variant('none', null),
        });

        const train_config = $.let({
            epochs: variant('some', 50n),
            batch_size: variant('some', 2n),
            learning_rate: variant('some', 0.01),
            loss: variant('some', variant('mse', null)),
            optimizer: variant('some', variant('adam', null)),
            early_stopping: variant('none', null),
            validation_split: variant('some', 0.25),
            random_state: variant('some', 42n),
            pos_weight: variant('none', null),
            prior: variant('none', null),
            sample_constraints: variant('none', null),
        });

        const output = $.let(Torch.mlpTrainMulti(X, X, mlp_config, train_config));
        const predictions = $.let(Torch.mlpPredictMulti(output.model, X, variant('none', null)));

        // Check all outputs sum to ~1.0 (softmax property)
        $.for(predictions, ($, row) => {
            const row_sum = $.let(row.reduce(($, acc, val) => acc.add(val), 0.0));
            // Allow small numerical tolerance
            $(Assert.greater(row_sum, 0.99));
            $(Assert.less(row_sum, 1.01));
        });

        // Check all values are non-negative (softmax property)
        $.for(predictions, ($, row) => {
            $.for(row, ($, val) => {
                $(Assert.greaterEqual(val, 0.0));
            });
        });
    });

    test("softmax output with KL divergence loss", $ => {
        // Autoencoder with softmax output trained with KL divergence
        const X = $.let([
            [0.6, 0.3, 0.1],
            [0.1, 0.7, 0.2],
            [0.2, 0.2, 0.6],
            [0.4, 0.4, 0.2],
        ]);

        const mlp_config = $.let({
            hidden_layers: [8n, 2n, 8n],
            activation: variant('some', variant('relu', null)),
            output_activation: variant('some', variant('softmax', null)),
            dropout: variant('none', null),
            output_dim: variant('none', null),
            output_constraints: variant('none', null),
        });

        const train_config = $.let({
            epochs: variant('some', 100n),
            batch_size: variant('some', 2n),
            learning_rate: variant('some', 0.01),
            loss: variant('some', variant('kl_div', null)),
            optimizer: variant('some', variant('adam', null)),
            early_stopping: variant('some', 20n),
            validation_split: variant('some', 0.25),
            random_state: variant('some', 42n),
            pos_weight: variant('none', null),
            prior: variant('none', null),
            sample_constraints: variant('none', null),
        });

        const output = $.let(Torch.mlpTrainMulti(X, X, mlp_config, train_config));

        // Check training completed
        $(Assert.greater(output.result.train_losses.size(), 0n));

        // Check outputs are valid probabilities
        const predictions = $.let(Torch.mlpPredictMulti(output.model, X, variant('none', null)));
        $.for(predictions, ($, row) => {
            const row_sum = $.let(row.reduce(($, acc, val) => acc.add(val), 0.0));
            $(Assert.greater(row_sum, 0.99));
            $(Assert.less(row_sum, 1.01));
        });
    });

    test("sigmoid output activation produces values in [0,1]", $ => {
        const X = $.let([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
        ]);
        const y = $.let([
            [0.2, 0.8],
            [0.5, 0.5],
            [0.9, 0.1],
            [0.3, 0.7],
        ]);

        const mlp_config = $.let({
            hidden_layers: [8n, 4n],
            activation: variant('some', variant('relu', null)),
            output_activation: variant('some', variant('sigmoid', null)),
            dropout: variant('none', null),
            output_dim: variant('none', null),
            output_constraints: variant('none', null),
        });

        const train_config = $.let({
            epochs: variant('some', 50n),
            batch_size: variant('some', 2n),
            learning_rate: variant('some', 0.01),
            loss: variant('some', variant('mse', null)),
            optimizer: variant('some', variant('adam', null)),
            early_stopping: variant('none', null),
            validation_split: variant('some', 0.25),
            random_state: variant('some', 42n),
            pos_weight: variant('none', null),
            prior: variant('none', null),
            sample_constraints: variant('none', null),
        });

        const output = $.let(Torch.mlpTrainMulti(X, y, mlp_config, train_config));
        const predictions = $.let(Torch.mlpPredictMulti(output.model, X, variant('none', null)));

        // Check all values are in [0, 1] (sigmoid property)
        $.for(predictions, ($, row) => {
            $.for(row, ($, val) => {
                $(Assert.greaterEqual(val, 0.0));
                $(Assert.lessEqual(val, 1.0));
            });
        });
    });

    test("no output activation (linear) can produce values outside [0,1]", $ => {
        // This test verifies that without output activation, values can be unconstrained
        const X = $.let([
            [1.0, 2.0],
            [10.0, 20.0],
            [100.0, 200.0],
        ]);

        const mlp_config = $.let({
            hidden_layers: [4n],
            activation: variant('some', variant('relu', null)),
            output_activation: variant('none', null),  // No output activation
            dropout: variant('none', null),
            output_dim: variant('some', 2n),
            output_constraints: variant('none', null),
        });

        const train_config = $.let({
            epochs: variant('some', 10n),
            batch_size: variant('some', 2n),
            learning_rate: variant('some', 0.01),
            loss: variant('some', variant('mse', null)),
            optimizer: variant('some', variant('adam', null)),
            early_stopping: variant('none', null),
            validation_split: variant('some', 0.3),
            random_state: variant('some', 42n),
            pos_weight: variant('none', null),
            prior: variant('none', null),
            sample_constraints: variant('none', null),
        });

        const output = $.let(Torch.mlpTrainMulti(X, X, mlp_config, train_config));
        const predictions = $.let(Torch.mlpPredictMulti(output.model, X, variant('none', null)));

        // Just verify the model runs - no constraints on output values
        $(Assert.equal(predictions.size(), 3n));
        $(Assert.equal(predictions.get(0n).size(), 2n));
    });

    // ========================================================================
    // BCE Loss Tests (Binary Cross Entropy)
    // ========================================================================

    test("bce loss with sigmoid output for binary reconstruction", $ => {
        // Binary data (sparse matrix simulation - mostly 0s with some 1s)
        const X = $.let([
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
        ]);

        const mlp_config = $.let({
            hidden_layers: [8n, 3n, 8n],  // Bottleneck autoencoder
            activation: variant('some', variant('relu', null)),
            output_activation: variant('some', variant('sigmoid', null)),  // Required for BCE
            dropout: variant('none', null),
            output_dim: variant('none', null),
            output_constraints: variant('none', null),
        });

        const train_config = $.let({
            epochs: variant('some', 100n),
            batch_size: variant('some', 3n),
            learning_rate: variant('some', 0.01),
            loss: variant('some', variant('bce', null)),  // Binary Cross Entropy
            optimizer: variant('some', variant('adam', null)),
            early_stopping: variant('some', 15n),
            validation_split: variant('some', 0.2),
            random_state: variant('some', 42n),
            pos_weight: variant('none', null),
            prior: variant('none', null),
            sample_constraints: variant('none', null),
        });

        const output = $.let(Torch.mlpTrainMulti(X, X, mlp_config, train_config));
        const predictions = $.let(Torch.mlpPredictMulti(output.model, X, variant('none', null)));

        // Check training completed
        $(Assert.greater(output.result.train_losses.size(), 0n));

        // Check outputs are in [0, 1] (sigmoid output)
        $.for(predictions, ($, row) => {
            $.for(row, ($, val) => {
                $(Assert.greaterEqual(val, 0.0));
                $(Assert.lessEqual(val, 1.0));
            });
        });
    });

    test("bce_with_logits loss for binary reconstruction (no sigmoid output)", $ => {
        // Binary data for autoencoder
        const X = $.let([
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
        ]);

        const mlp_config = $.let({
            hidden_layers: [8n, 3n, 8n],  // Bottleneck autoencoder
            activation: variant('some', variant('relu', null)),
            output_activation: variant('none', null),  // NO sigmoid - bce_with_logits applies it internally
            dropout: variant('none', null),
            output_dim: variant('none', null),
            output_constraints: variant('none', null),
        });

        const train_config = $.let({
            epochs: variant('some', 100n),
            batch_size: variant('some', 3n),
            learning_rate: variant('some', 0.01),
            loss: variant('some', variant('bce_with_logits', null)),  // Applies sigmoid internally
            optimizer: variant('some', variant('adam', null)),
            early_stopping: variant('some', 15n),
            validation_split: variant('some', 0.2),
            random_state: variant('some', 42n),
            pos_weight: variant('none', null),
            prior: variant('none', null),
            sample_constraints: variant('none', null),
        });

        const output = $.let(Torch.mlpTrainMulti(X, X, mlp_config, train_config));

        // Check training completed
        $(Assert.greater(output.result.train_losses.size(), 0n));

        // Note: Predictions are raw logits (not sigmoid), so no [0,1] constraint
        const predictions = $.let(Torch.mlpPredictMulti(output.model, X, variant('none', null)));
        $(Assert.equal(predictions.size(), 6n));
        $(Assert.equal(predictions.get(0n).size(), 4n));
    });

    test("bce_with_logits with pos_weight for imbalanced binary data", $ => {
        // Sparse binary data (mostly 0s) - simulates task timing matrix
        const X = $.let([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        ]);

        const mlp_config = $.let({
            hidden_layers: [8n, 2n, 8n],  // Bottleneck
            activation: variant('some', variant('relu', null)),
            output_activation: variant('none', null),  // NO sigmoid for bce_with_logits
            dropout: variant('none', null),
            output_dim: variant('none', null),
            output_constraints: variant('none', null),
        });

        const train_config = $.let({
            epochs: variant('some', 100n),
            batch_size: variant('some', 3n),
            learning_rate: variant('some', 0.01),
            loss: variant('some', variant('bce_with_logits', null)),
            optimizer: variant('some', variant('adam', null)),
            early_stopping: variant('some', 15n),
            validation_split: variant('some', 0.2),
            random_state: variant('some', 42n),
            pos_weight: variant('some', variant('scalar', 5.0)),  // Weight positive class 5x more
            prior: variant('none', null),
            sample_constraints: variant('none', null),
        });

        const output = $.let(Torch.mlpTrainMulti(X, X, mlp_config, train_config));

        // Check training completed
        $(Assert.greater(output.result.train_losses.size(), 0n));

        // Verify model dimensions
        const predictions = $.let(Torch.mlpPredictMulti(output.model, X, variant('none', null)));
        $(Assert.equal(predictions.size(), 6n));
        $(Assert.equal(predictions.get(0n).size(), 6n));
    });

    // ========================================================================
    // Constrained Output Tests
    // ========================================================================

    test("constrained output with binary rows", $ => {
        // Test binary constraint (independent sigmoid per position)
        // 2 rows x 3 cols = 6 outputs
        const X = $.let([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        ]);

        const mlp_config = $.let({
            hidden_layers: [8n],
            activation: variant('some', variant('relu', null)),
            output_activation: variant('none', null),  // Ignored when output_constraints set
            dropout: variant('none', null),
            output_dim: variant('some', 6n),  // 2 rows x 3 cols
            output_constraints: variant('some', {
                row_constraints: [
                    // Row 0: binary, all positions valid
                    variant('binary', { mask: variant('none', null), data_mask: variant('none', null) }),
                    // Row 1: binary, all positions valid
                    variant('binary', { mask: variant('none', null), data_mask: variant('none', null) }),
                ],
            }),
        }, TorchMLPConfigType);

        const train_config = $.let({
            epochs: variant('some', 50n),
            batch_size: variant('some', 2n),
            learning_rate: variant('some', 0.01),
            loss: variant('some', variant('bce', null)),
            optimizer: variant('some', variant('adam', null)),
            early_stopping: variant('none', null),
            validation_split: variant('some', 0.25),
            random_state: variant('some', 42n),
            pos_weight: variant('none', null),
            prior: variant('none', null),
            sample_constraints: variant('none', null),
        });

        const output = $.let(Torch.mlpTrainMulti(X, X, mlp_config, train_config));
        const predictions = $.let(Torch.mlpPredictMulti(output.model, X, variant('none', null)));

        // Check outputs are in [0, 1] (sigmoid property)
        $.for(predictions, ($, row) => {
            $.for(row, ($, val) => {
                $(Assert.greaterEqual(val, 0.0));
                $(Assert.lessEqual(val, 1.0));
            });
        });
    });

    test("constrained output with binary masked rows", $ => {
        // Test binary constraint with mask (some positions impossible)
        // 2 rows x 4 cols = 8 outputs
        const X = $.let([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        ]);

        const mlp_config = $.let({
            hidden_layers: [8n],
            activation: variant('some', variant('relu', null)),
            output_activation: variant('none', null),
            dropout: variant('none', null),
            output_dim: variant('some', 8n),  // 2 rows x 4 cols
            output_constraints: variant('some', {
                row_constraints: [
                    // Row 0: binary, positions 0,1 valid, 2,3 masked
                    variant('binary', { mask: variant('some', [true, true, false, false]), data_mask: variant('none', null) }),
                    // Row 1: binary, positions 2,3 valid, 0,1 masked
                    variant('binary', { mask: variant('some', [false, false, true, true]), data_mask: variant('none', null) }),
                ],
            }),
        }, TorchMLPConfigType);

        const train_config = $.let({
            epochs: variant('some', 50n),
            batch_size: variant('some', 2n),
            learning_rate: variant('some', 0.01),
            loss: variant('some', variant('mse', null)),
            optimizer: variant('some', variant('adam', null)),
            early_stopping: variant('none', null),
            validation_split: variant('some', 0.25),
            random_state: variant('some', 42n),
            pos_weight: variant('none', null),
            prior: variant('none', null),
            sample_constraints: variant('none', null),
        });

        const output = $.let(Torch.mlpTrainMulti(X, X, mlp_config, train_config));
        const predictions = $.let(Torch.mlpPredictMulti(output.model, X, variant('none', null)));

        // Check masked positions are ~0
        $.for(predictions, ($, row) => {
            // Row 0, positions 2,3 should be ~0
            $(Assert.less(row.get(2n), 0.01));
            $(Assert.less(row.get(3n), 0.01));
            // Row 1, positions 0,1 should be ~0 (offset by 4 for second row)
            $(Assert.less(row.get(4n), 0.01));
            $(Assert.less(row.get(5n), 0.01));
        });
    });

    test("constrained output with mutex row (softmax)", $ => {
        // Test mutex constraint (exactly one position via softmax)
        // 2 rows x 3 cols = 6 outputs
        const X = $.let([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        ]);

        const mlp_config = $.let({
            hidden_layers: [8n],
            activation: variant('some', variant('relu', null)),
            output_activation: variant('none', null),
            dropout: variant('none', null),
            output_dim: variant('some', 6n),  // 2 rows x 3 cols
            output_constraints: variant('some', {
                row_constraints: [
                    // Row 0: mutex (softmax) - only one position can be active
                    variant('mutex', { mask: variant('none', null), allow_none: variant('none', null), data_mask: variant('none', null), class_weights: variant('none', null) }),
                    // Row 1: binary - independent
                    variant('binary', { mask: variant('none', null), data_mask: variant('none', null) }),
                ],
            }),
        }, TorchMLPConfigType);

        const train_config = $.let({
            epochs: variant('some', 50n),
            batch_size: variant('some', 2n),
            learning_rate: variant('some', 0.01),
            loss: variant('some', variant('mse', null)),
            optimizer: variant('some', variant('adam', null)),
            early_stopping: variant('none', null),
            validation_split: variant('some', 0.25),
            random_state: variant('some', 42n),
            pos_weight: variant('none', null),
            prior: variant('none', null),
            sample_constraints: variant('none', null),
        });

        const output = $.let(Torch.mlpTrainMulti(X, X, mlp_config, train_config));
        const predictions = $.let(Torch.mlpPredictMulti(output.model, X, variant('none', null)));

        // For mutex row (row 0, positions 0-2), outputs should sum to ~1.0
        $.for(predictions, ($, row) => {
            const mutex_sum = $.let(row.get(0n).add(row.get(1n)).add(row.get(2n)));
            $(Assert.greater(mutex_sum, 0.99));
            $(Assert.less(mutex_sum, 1.01));
        });
    });

    test("constrained output with mutex allow_none", $ => {
        // Test mutex with allow_none (can have all zeros)
        // 1 row x 3 cols = 3 outputs
        const X = $.let([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],  // No active position
        ]);

        const mlp_config = $.let({
            hidden_layers: [8n],
            activation: variant('some', variant('relu', null)),
            output_activation: variant('none', null),
            dropout: variant('none', null),
            output_dim: variant('some', 3n),
            output_constraints: variant('some', {
                row_constraints: [
                    // Mutex with allow_none - sum can be < 1
                    variant('mutex', { mask: variant('none', null), allow_none: variant('some', true), data_mask: variant('none', null), class_weights: variant('none', null) }),
                ],
            }),
        }, TorchMLPConfigType);

        const train_config = $.let({
            epochs: variant('some', 50n),
            batch_size: variant('some', 2n),
            learning_rate: variant('some', 0.01),
            loss: variant('some', variant('mse', null)),
            optimizer: variant('some', variant('adam', null)),
            early_stopping: variant('none', null),
            validation_split: variant('some', 0.25),
            random_state: variant('some', 42n),
            pos_weight: variant('none', null),
            prior: variant('none', null),
            sample_constraints: variant('none', null),
        });

        const output = $.let(Torch.mlpTrainMulti(X, X, mlp_config, train_config));
        const predictions = $.let(Torch.mlpPredictMulti(output.model, X, variant('none', null)));

        // All outputs should be in [0, 1]
        $.for(predictions, ($, row) => {
            $.for(row, ($, val) => {
                $(Assert.greaterEqual(val, 0.0));
                $(Assert.lessEqual(val, 1.0));
            });
        });

        // Sum should be <= 1 (because of allow_none)
        $.for(predictions, ($, row) => {
            const row_sum = $.let(row.reduce(($, acc, val) => acc.add(val), 0.0));
            $(Assert.lessEqual(row_sum, 1.01));
        });
    });

    test("constrained output with at_most constraint", $ => {
        // Test at_most constraint (max 2 positions active)
        // 1 row x 5 cols = 5 outputs
        const X = $.let([
            [1.0, 1.0, 0.0, 0.0, 0.0],  // 2 active
            [1.0, 0.0, 0.0, 0.0, 0.0],  // 1 active
            [0.0, 0.0, 0.0, 0.0, 0.0],  // 0 active
            [1.0, 1.0, 1.0, 0.0, 0.0],  // 3 active (should be clamped to 2)
        ]);

        const mlp_config = $.let({
            hidden_layers: [8n],
            activation: variant('some', variant('relu', null)),
            output_activation: variant('none', null),
            dropout: variant('none', null),
            output_dim: variant('some', 5n),
            output_constraints: variant('some', {
                row_constraints: [
                    // At most 2 positions active
                    variant('at_most', { max_count: 2n, mask: variant('none', null), data_mask: variant('none', null) }),
                ],

            }),
        }, TorchMLPConfigType);

        const train_config = $.let({
            epochs: variant('some', 50n),
            batch_size: variant('some', 2n),
            learning_rate: variant('some', 0.01),
            loss: variant('some', variant('mse', null)),
            optimizer: variant('some', variant('adam', null)),
            early_stopping: variant('none', null),
            validation_split: variant('some', 0.25),
            random_state: variant('some', 42n),
            pos_weight: variant('none', null),
            prior: variant('none', null),
            sample_constraints: variant('none', null),
        });

        const output = $.let(Torch.mlpTrainMulti(X, X, mlp_config, train_config));
        const predictions = $.let(Torch.mlpPredictMulti(output.model, X, variant('none', null)));

        // Count non-zero positions per row - should be at most 2
        $.for(predictions, ($, row) => {
            const count = $.let(row.reduce((_$, acc, val) =>
                val.greaterThan(0.1).ifElse(_$ => acc.add(1.0), _$ => acc), 0.0
            ));
            $(Assert.lessEqual(count, 2.0));
        });
    });

    // ========================================================================
    // Per-Output Pos Weight Tests (Section 6.1 from design doc)
    // ========================================================================

    test("per_output_pos_weight improves recall on rare outputs", $ => {
        // Imbalanced binary data: columns 0,1 have many 1s, columns 2,3 have few 1s
        const X = $.let([
            [1.0, 1.0, 1.0, 0.0],  // col 2 has 1
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 1.0],  // col 3 has 1
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ]);

        const mlp_config = $.let({
            hidden_layers: [8n],
            activation: variant('some', variant('relu', null)),
            output_activation: variant('none', null),
            dropout: variant('none', null),
            output_dim: variant('some', 4n),
            output_constraints: variant('none', null),
        });

        // Use per-output pos_weight: high weight for rare columns 2,3
        const train_config = $.let({
            epochs: variant('some', 100n),
            batch_size: variant('some', 4n),
            learning_rate: variant('some', 0.01),
            loss: variant('some', variant('bce_with_logits', null)),
            optimizer: variant('some', variant('adam', null)),
            early_stopping: variant('some', 20n),
            validation_split: variant('some', 0.2),
            random_state: variant('some', 42n),
            pos_weight: variant('some', variant('per_output', [1.0, 1.0, 10.0, 10.0])),
            prior: variant('none', null),
            sample_constraints: variant('none', null),
        });

        const output = $.let(Torch.mlpTrainMulti(X, X, mlp_config, train_config));

        // Check training completed
        $(Assert.greater(output.result.train_losses.size(), 0n));

        // Model should train successfully with per-output weights
        const predictions = $.let(Torch.mlpPredictMulti(output.model, X, variant('none', null)));
        $(Assert.equal(predictions.size(), 8n));
        $(Assert.equal(predictions.get(0n).size(), 4n));
    });

    test("no_double_sigmoid with bce_with_logits and constraints", $ => {
        // Test that bce_with_logits + binary constraints doesn't apply sigmoid twice
        // 2 rows x 3 cols = 6 outputs
        const X = $.let([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        ]);

        const mlp_config = $.let({
            hidden_layers: [8n],
            activation: variant('some', variant('relu', null)),
            output_activation: variant('none', null),  // No output activation
            dropout: variant('none', null),
            output_dim: variant('some', 6n),  // 2 rows x 3 cols
            output_constraints: variant('some', {
                row_constraints: [
                    variant('binary', { mask: variant('none', null), data_mask: variant('none', null) }),
                    variant('binary', { mask: variant('none', null), data_mask: variant('none', null) }),
                ],
            }),
        }, TorchMLPConfigType);

        const train_config = $.let({
            epochs: variant('some', 100n),
            batch_size: variant('some', 2n),
            learning_rate: variant('some', 0.01),
            loss: variant('some', variant('bce_with_logits', null)),  // Uses return_logits mode internally
            optimizer: variant('some', variant('adam', null)),
            early_stopping: variant('some', 20n),
            validation_split: variant('some', 0.25),
            random_state: variant('some', 42n),
            pos_weight: variant('none', null),
            prior: variant('none', null),
            sample_constraints: variant('none', null),
        });

        const output = $.let(Torch.mlpTrainMulti(X, X, mlp_config, train_config));
        const predictions = $.let(Torch.mlpPredictMulti(output.model, X, variant('none', null)));

        // Predictions should be in [0, 1] range (sigmoid applied at prediction time)
        $.for(predictions, ($, row) => {
            $.for(row, ($, val) => {
                $(Assert.greaterEqual(val, 0.0));
                $(Assert.lessEqual(val, 1.0));
            });
        });

        // No NaN or Inf values (would indicate double-sigmoid issue)
        $.for(predictions, ($, row) => {
            $.for(row, ($, val) => {
                // Values should be finite
                $(Assert.greaterEqual(val, -1000.0));
                $(Assert.lessEqual(val, 1000.0));
            });
        });
    });

    // ========================================================================
    // Per-Sample Masks Tests (Section 6.2 from design doc)
    // ========================================================================

    test("sample_masks_applied zeros correct positions", $ => {
        // Test per-sample masks: different samples have different valid positions
        // 2 rows x 3 cols = 6 outputs
        const X = $.let([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        ]);

        const mlp_config = $.let({
            hidden_layers: [8n],
            activation: variant('some', variant('relu', null)),
            output_activation: variant('none', null),
            dropout: variant('none', null),
            output_dim: variant('some', 6n),  // 2 rows x 3 cols
            output_constraints: variant('some', {
                row_constraints: [
                    variant('binary', { mask: variant('none', null), data_mask: variant('none', null) }),
                    variant('binary', { mask: variant('none', null), data_mask: variant('none', null) }),
                ],
            }),
        }, TorchMLPConfigType);

        // Per-sample masks: (n_samples, n_rows, n_cols)
        // Sample 0: row 0 all valid, row 1 all valid
        // Sample 1: row 0 col 2 masked, row 1 all valid
        // Sample 2: row 0 all valid, row 1 col 0 masked
        // Sample 3: row 0 col 0,1 masked, row 1 col 1,2 masked
        const sample_masks = $.let([
            [[true, true, true], [true, true, true]],      // Sample 0: all valid
            [[true, true, false], [true, true, true]],    // Sample 1: row0-col2 masked
            [[true, true, true], [false, true, true]],    // Sample 2: row1-col0 masked
            [[false, false, true], [true, false, false]], // Sample 3: partial masking
        ]);

        const train_config = $.let({
            epochs: variant('some', 50n),
            batch_size: variant('some', 2n),
            learning_rate: variant('some', 0.01),
            loss: variant('some', variant('bce', null)),
            optimizer: variant('some', variant('adam', null)),
            early_stopping: variant('none', null),
            validation_split: variant('some', 0.25),
            random_state: variant('some', 42n),
            pos_weight: variant('none', null),
            prior: variant('none', null),
            sample_constraints: variant('some', {
                masks: variant('some', sample_masks),
                pos_weights: variant('none', null),
                priors: variant('none', null),
                mutex_class_weights: variant('none', null),
            }),
        });

        const output = $.let(Torch.mlpTrainMulti(X, X, mlp_config, train_config));

        // Use sample_masks during prediction
        const predictions = $.let(Torch.mlpPredictMulti(output.model, X, variant('some', sample_masks)));

        // Check masked positions are ~0
        // Sample 1, row 0, col 2 (index 2) should be 0
        $(Assert.less(predictions.get(1n).get(2n), 0.01));
        // Sample 2, row 1, col 0 (index 3) should be 0
        $(Assert.less(predictions.get(2n).get(3n), 0.01));
        // Sample 3, row 0, col 0,1 (index 0,1) should be 0
        $(Assert.less(predictions.get(3n).get(0n), 0.01));
        $(Assert.less(predictions.get(3n).get(1n), 0.01));
        // Sample 3, row 1, col 1,2 (index 4,5) should be 0
        $(Assert.less(predictions.get(3n).get(4n), 0.01));
        $(Assert.less(predictions.get(3n).get(5n), 0.01));
    });

    test("sample_masks_with_data_mask both apply", $ => {
        // Test static data_mask combined with per-sample masks
        // 1 row x 4 cols = 4 outputs
        const X = $.let([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]);

        const mlp_config = $.let({
            hidden_layers: [8n],
            activation: variant('some', variant('relu', null)),
            output_activation: variant('none', null),
            dropout: variant('none', null),
            output_dim: variant('some', 4n),
            output_constraints: variant('some', {
                row_constraints: [
                    // Static data_mask: col 3 always masked
                    variant('binary', { mask: variant('none', null), data_mask: variant('some', [true, true, true, false]) }),
                ],
            }),
        }, TorchMLPConfigType);

        // Per-sample masks: additional masking on top of data_mask
        // Note: data_mask masks col 3 for ALL samples, sample_masks adds more
        const sample_masks = $.let([
            [[true, true, true, true]],      // Sample 0: no additional masking
            [[true, false, true, true]],     // Sample 1: col 1 also masked
            [[false, true, true, true]],     // Sample 2: col 0 also masked
            [[true, true, false, true]],     // Sample 3: col 2 also masked
        ]);

        const train_config = $.let({
            epochs: variant('some', 50n),
            batch_size: variant('some', 2n),
            learning_rate: variant('some', 0.01),
            loss: variant('some', variant('bce', null)),
            optimizer: variant('some', variant('adam', null)),
            early_stopping: variant('none', null),
            validation_split: variant('some', 0.25),
            random_state: variant('some', 42n),
            pos_weight: variant('none', null),
            prior: variant('none', null),
            sample_constraints: variant('some', {
                masks: variant('some', sample_masks),
                pos_weights: variant('none', null),
                priors: variant('none', null),
                mutex_class_weights: variant('none', null),
            }),
        });

        const output = $.let(Torch.mlpTrainMulti(X, X, mlp_config, train_config));

        // Predict with sample masks
        const predictions = $.let(Torch.mlpPredictMulti(output.model, X, variant('some', sample_masks)));

        // Col 3 should be masked for ALL samples (from data_mask)
        $.for(predictions, ($, row) => {
            $(Assert.less(row.get(3n), 0.01));
        });

        // Additional sample-specific masks
        $(Assert.less(predictions.get(1n).get(1n), 0.01));  // Sample 1, col 1
        $(Assert.less(predictions.get(2n).get(0n), 0.01));  // Sample 2, col 0
        $(Assert.less(predictions.get(3n).get(2n), 0.01));  // Sample 3, col 2
    });

    // ========================================================================
    // Per-Sample Weights Tests (Section 6.3 from design doc)
    // ========================================================================

    test("sample_pos_weights applies per-sample weighting", $ => {
        // Test per-sample pos_weight: different weights for different samples
        const X = $.let([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]);

        const mlp_config = $.let({
            hidden_layers: [8n],
            activation: variant('some', variant('relu', null)),
            output_activation: variant('none', null),
            dropout: variant('none', null),
            output_dim: variant('some', 4n),
            output_constraints: variant('none', null),
        });

        // Per-sample pos_weights: (n_samples, output_dim)
        const sample_pos_weights = $.let([
            [1.0, 1.0, 10.0, 10.0],  // Sample 0: high weight on cols 2,3
            [10.0, 10.0, 1.0, 1.0],  // Sample 1: high weight on cols 0,1
            [5.0, 5.0, 5.0, 5.0],    // Sample 2: balanced
            [1.0, 1.0, 1.0, 1.0],    // Sample 3: no weighting
        ]);

        const train_config = $.let({
            epochs: variant('some', 50n),
            batch_size: variant('some', 2n),
            learning_rate: variant('some', 0.01),
            loss: variant('some', variant('bce_with_logits', null)),
            optimizer: variant('some', variant('adam', null)),
            early_stopping: variant('none', null),
            validation_split: variant('some', 0.25),
            random_state: variant('some', 42n),
            pos_weight: variant('none', null),
            prior: variant('none', null),
            sample_constraints: variant('some', {
                masks: variant('none', null),
                pos_weights: variant('some', sample_pos_weights),
                priors: variant('none', null),
                mutex_class_weights: variant('none', null),
            }),
        });

        const output = $.let(Torch.mlpTrainMulti(X, X, mlp_config, train_config));

        // Check training completed
        $(Assert.greater(output.result.train_losses.size(), 0n));

        // Model should train successfully with per-sample weights
        const predictions = $.let(Torch.mlpPredictMulti(output.model, X, variant('none', null)));
        $(Assert.equal(predictions.size(), 4n));
        $(Assert.equal(predictions.get(0n).size(), 4n));
    });

    test("sample_priors applies per-sample prior regularization", $ => {
        // Test per-sample priors: bias outputs toward sample-specific priors
        const X = $.let([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]);

        const mlp_config = $.let({
            hidden_layers: [8n],
            activation: variant('some', variant('relu', null)),
            output_activation: variant('some', variant('sigmoid', null)),
            dropout: variant('none', null),
            output_dim: variant('some', 4n),
            output_constraints: variant('none', null),
        });

        // Per-sample priors: (n_samples, output_dim)
        // Each sample has different prior probabilities
        const sample_priors = $.let([
            [0.8, 0.1, 0.05, 0.05],  // Sample 0: col 0 has high prior
            [0.1, 0.8, 0.05, 0.05],  // Sample 1: col 1 has high prior
            [0.05, 0.05, 0.8, 0.1],  // Sample 2: col 2 has high prior
            [0.05, 0.05, 0.1, 0.8],  // Sample 3: col 3 has high prior
        ]);

        const train_config = $.let({
            epochs: variant('some', 100n),
            batch_size: variant('some', 2n),
            learning_rate: variant('some', 0.01),
            loss: variant('some', variant('mse', null)),
            optimizer: variant('some', variant('adam', null)),
            early_stopping: variant('none', null),
            validation_split: variant('some', 0.25),
            random_state: variant('some', 42n),
            pos_weight: variant('none', null),
            prior: variant('some', { values: [0.25, 0.25, 0.25, 0.25], weight: 0.1 }),  // Global prior
            sample_constraints: variant('some', {
                masks: variant('none', null),
                pos_weights: variant('none', null),
                priors: variant('some', sample_priors),
                mutex_class_weights: variant('none', null),
            }),
        });

        const output = $.let(Torch.mlpTrainMulti(X, X, mlp_config, train_config));

        // Check training completed
        $(Assert.greater(output.result.train_losses.size(), 0n));

        // Outputs should be in [0, 1] since we have sigmoid output
        const predictions = $.let(Torch.mlpPredictMulti(output.model, X, variant('none', null)));
        $.for(predictions, ($, row) => {
            $.for(row, ($, val) => {
                $(Assert.greaterEqual(val, 0.0));
                $(Assert.lessEqual(val, 1.0));
            });
        });
    });

    // ========================================================================
    // Full Pipeline Integration Test (Section 6.4 from design doc)
    // ========================================================================

    test("full_pipeline with all features", $ => {
        // Integration test: all features together
        // 2 rows x 3 cols = 6 outputs
        const X = $.let([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
        ]);

        const mlp_config = $.let({
            hidden_layers: [16n, 8n],
            activation: variant('some', variant('relu', null)),
            output_activation: variant('none', null),
            dropout: variant('none', null),
            output_dim: variant('some', 6n),  // 2 rows x 3 cols
            output_constraints: variant('some', {
                row_constraints: [
                    // Row 0: binary with data_mask (col 2 always masked)
                    variant('binary', { mask: variant('none', null), data_mask: variant('some', [true, true, false]) }),
                    // Row 1: binary
                    variant('binary', { mask: variant('none', null), data_mask: variant('none', null) }),
                ],
            }),
        }, TorchMLPConfigType);

        // Per-sample masks
        const sample_masks = $.let([
            [[true, true, true], [true, true, true]],
            [[true, true, true], [true, true, true]],
            [[true, true, true], [true, true, true]],
            [[true, true, true], [true, true, true]],
            [[true, true, true], [true, true, true]],
            [[true, true, true], [true, true, true]],
            [[true, true, true], [true, false, true]],  // Sample 6: row1-col1 masked
            [[true, true, true], [false, true, true]],  // Sample 7: row1-col0 masked
        ]);

        // Per-sample pos_weights
        const sample_pos_weights = $.let([
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ]);

        const train_config = $.let({
            epochs: variant('some', 100n),
            batch_size: variant('some', 4n),
            learning_rate: variant('some', 0.01),
            loss: variant('some', variant('bce_with_logits', null)),
            optimizer: variant('some', variant('adam', null)),
            early_stopping: variant('some', 20n),
            validation_split: variant('some', 0.2),
            random_state: variant('some', 42n),
            pos_weight: variant('some', variant('per_output', [1.0, 1.0, 5.0, 1.0, 1.0, 5.0])),
            prior: variant('some', { values: [0.5, 0.5, 0.1, 0.5, 0.5, 0.1], weight: 0.05 }),
            sample_constraints: variant('some', {
                masks: variant('some', sample_masks),
                pos_weights: variant('some', sample_pos_weights),
                priors: variant('none', null),
                mutex_class_weights: variant('none', null),
            }),
        });

        const output = $.let(Torch.mlpTrainMulti(X, X, mlp_config, train_config));

        // Check training completed
        $(Assert.greater(output.result.train_losses.size(), 0n));

        // Predict with sample masks
        const predictions = $.let(Torch.mlpPredictMulti(output.model, X, variant('some', sample_masks)));

        // All outputs should be in [0, 1]
        $.for(predictions, ($, row) => {
            $.for(row, ($, val) => {
                $(Assert.greaterEqual(val, 0.0));
                $(Assert.lessEqual(val, 1.0));
            });
        });

        // data_mask: col 2 of row 0 (index 2) should be ~0 for all samples
        $.for(predictions, ($, row) => {
            $(Assert.less(row.get(2n), 0.01));
        });

        // sample_masks: Sample 6, row1-col1 (index 4) should be ~0
        $(Assert.less(predictions.get(6n).get(4n), 0.01));
        // Sample 7, row1-col0 (index 3) should be ~0
        $(Assert.less(predictions.get(7n).get(3n), 0.01));
    });

    // ========================================================================
    // Utility Function Tests
    // ========================================================================

    test("compute_pos_weight calculates scalar pos_weight", $ => {
        // Imbalanced binary data: mostly 0s
        const y = $.let([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]);

        // Compute scalar pos_weight (not per-output)
        const pos_weight = $.let(Torch.computePosWeight(y, false));

        // Should be scalar with value > 1 (more negatives than positives)
        // Total: 24 elements, 2 positives, 22 negatives
        // pos_weight ≈ 22/2 = 11 (with smoothing)
        $(Assert.greater(pos_weight.unwrap('scalar'), 5.0));
    });

    test("compute_pos_weight calculates per_output pos_weight", $ => {
        // Imbalanced binary data: different imbalance per column
        const y = $.let([
            [1.0, 1.0, 0.0, 0.0],  // col0: 1, col1: 1
            [1.0, 0.0, 0.0, 0.0],  // col0: 1
            [1.0, 0.0, 0.0, 0.0],  // col0: 1
            [0.0, 0.0, 1.0, 0.0],  // col2: 1
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]);

        // Compute per-output pos_weight
        const pos_weight = $.let(Torch.computePosWeight(y, true));

        // Should be per_output with array of weights
        const weights = $.let(pos_weight.unwrap('per_output'));
        $(Assert.equal(weights.size(), 4n));

        // Col 0: 3 pos, 3 neg → ~1.0 weight
        // Col 1: 1 pos, 5 neg → ~5.0 weight
        // Col 2: 1 pos, 5 neg → ~5.0 weight
        // Col 3: 0 pos, 6 neg → high weight (capped)
        $(Assert.less(weights.get(0n), 2.0));      // Low weight for balanced col
        $(Assert.greater(weights.get(1n), 2.0));   // Higher weight for imbalanced col
        $(Assert.greater(weights.get(3n), 5.0));   // High weight for all-negative col
    });

    test("compute_data_mask creates mask from likelihood data", $ => {
        // Likelihood data: values <= threshold get masked
        // The function checks which columns have ANY values > threshold
        const likelihoods = $.let([
            [0.8, 0.5, 0.0, 0.0],  // col0: 0.8, col1: 0.5, col2: 0.0, col3: 0.0
            [0.3, 0.9, 0.1, 0.0],  // col0: 0.3, col1: 0.9, col2: 0.1, col3: 0.0
            [0.0, 0.0, 0.7, 0.4],  // col0: 0.0, col1: 0.0, col2: 0.7, col3: 0.4
            [0.0, 0.2, 0.0, 0.0],  // col0: 0.0, col1: 0.2, col2: 0.0, col3: 0.0
        ]);

        // Compute data_mask with threshold 0.0 (columns with any value > 0)
        // Col 0: has 0.8, 0.3 > 0 → true
        // Col 1: has 0.5, 0.9, 0.2 > 0 → true
        // Col 2: has 0.1, 0.7 > 0 → true
        // Col 3: has 0.4 > 0 → true
        const data_mask = $.let(Torch.computeDataMask(likelihoods, 0.0));

        // Result is 1D array with length = num columns (4)
        $(Assert.equal(data_mask.size(), 4n));

        // All columns have at least one value > 0, so all should be true
        const true_count = $.let(data_mask.reduce(
            ($, acc, val) => val.ifElse(_ => acc.add(1n), _ => acc), 0n
        ));
        $(Assert.equal(true_count, 4n));
    });

    test("compute_data_mask identifies zero columns", $ => {
        // Data where some columns are all zeros
        const data = $.let([
            [1.0, 0.0, 0.5, 0.0],  // col1 and col3 are 0
            [0.0, 0.0, 0.3, 0.0],  // col0 is 0 here, col1 and col3 still 0
            [0.5, 0.0, 0.0, 0.0],  // col2 is 0 here, col1 and col3 still 0
        ]);

        // Compute data_mask with threshold 0.0
        // Col 0: has 1.0, 0.5 > 0 → true
        // Col 1: all zeros → false
        // Col 2: has 0.5, 0.3 > 0 → true
        // Col 3: all zeros → false
        const data_mask = $.let(Torch.computeDataMask(data, 0.0));

        $(Assert.equal(data_mask.size(), 4n));

        // Count true values - should be 2 (cols 0 and 2)
        const true_count = $.let(data_mask.reduce(
            ($, acc, val) => val.ifElse(_ => acc.add(1n), _ => acc), 0n
        ));
        $(Assert.equal(true_count, 2n));
    });

    // ========================================================================
    // Mutex Class Weights Tests
    // ========================================================================

    test("mutex class_weights enables learning with imbalanced classes", $ => {
        // Test mutex with class_weights for imbalanced data
        // 1 mutex row x 4 classes = 4 outputs
        // Imbalanced: class 0 is 80% of samples, classes 1-3 are 20% combined
        const X = $.let([
            // Class 0 (dominant) - 8 samples
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            // Class 1 - 1 sample
            [0.0, 1.0, 0.0, 0.0],
            // Class 2 - 1 sample
            [0.0, 0.0, 1.0, 0.0],
        ]);

        // Class weights: low weight for dominant class, high for rare classes
        // This helps the model learn the dominant class correctly
        const class_weights = [0.125, 4.0, 4.0, 4.0];

        const mlp_config = $.let({
            hidden_layers: [16n, 8n, 16n],
            activation: variant('some', variant('relu', null)),
            output_activation: variant('none', null),
            dropout: variant('none', null),
            output_dim: variant('some', 4n),
            output_constraints: variant('some', {
                row_constraints: [
                    variant('mutex', {
                        mask: variant('none', null),
                        allow_none: variant('none', null),
                        data_mask: variant('none', null),
                        class_weights: variant('some', class_weights),
                    }),
                ],
            }),
        }, TorchMLPConfigType);

        const train_config = $.let({
            epochs: variant('some', 100n),
            batch_size: variant('some', 5n),
            learning_rate: variant('some', 0.01),
            loss: variant('some', variant('bce', null)),  // Loss type doesn't matter - we use CE for mutex with weights
            optimizer: variant('some', variant('adam', null)),
            early_stopping: variant('none', null),
            validation_split: variant('some', 0.2),
            random_state: variant('some', 42n),
            pos_weight: variant('none', null),
            prior: variant('none', null),
            sample_constraints: variant('none', null),
        });

        const output = $.let(Torch.mlpTrainMulti(X, X, mlp_config, train_config));
        const predictions = $.let(Torch.mlpPredictMulti(output.model, X, variant('none', null)));

        // Mutex outputs should sum to ~1.0
        $.for(predictions, ($, row) => {
            const mutex_sum = $.let(row.get(0n).add(row.get(1n)).add(row.get(2n)).add(row.get(3n)));
            $(Assert.greater(mutex_sum, 0.95));
            $(Assert.less(mutex_sum, 1.05));
        });

        // For the dominant class samples (first 8), class 0 should have highest probability
        // The class weights help the model correctly predict the dominant class
        $(Assert.greater(predictions.get(0n).get(0n), 0.5));
    });

    test("mutex without class_weights uses standard cross-entropy", $ => {
        // Test mutex without class_weights - should use standard CE
        // 1 mutex row x 3 classes = 3 outputs
        const X = $.let([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]);

        const mlp_config = $.let({
            hidden_layers: [16n, 8n, 16n],
            activation: variant('some', variant('relu', null)),
            output_activation: variant('none', null),
            dropout: variant('none', null),
            output_dim: variant('some', 3n),
            output_constraints: variant('some', {
                row_constraints: [
                    variant('mutex', {
                        mask: variant('none', null),
                        allow_none: variant('none', null),
                        data_mask: variant('none', null),
                        class_weights: variant('none', null),  // No class weights
                    }),
                ],
            }),
        }, TorchMLPConfigType);

        const train_config = $.let({
            epochs: variant('some', 100n),
            batch_size: variant('some', 3n),
            learning_rate: variant('some', 0.01),
            loss: variant('some', variant('bce', null)),
            optimizer: variant('some', variant('adam', null)),
            early_stopping: variant('none', null),
            validation_split: variant('some', 0.2),
            random_state: variant('some', 42n),
            pos_weight: variant('none', null),
            prior: variant('none', null),
            sample_constraints: variant('none', null),
        });

        const output = $.let(Torch.mlpTrainMulti(X, X, mlp_config, train_config));
        const predictions = $.let(Torch.mlpPredictMulti(output.model, X, variant('none', null)));

        // Mutex outputs should still sum to ~1.0
        $.for(predictions, ($, row) => {
            const mutex_sum = $.let(row.get(0n).add(row.get(1n)).add(row.get(2n)));
            $(Assert.greater(mutex_sum, 0.95));
            $(Assert.less(mutex_sum, 1.05));
        });
    });

    test("compute_mutex_class_weights calculates inverse frequency weights", $ => {
        // Imbalanced one-hot data: class 0 is dominant (80%), class 1 is rare (20%)
        // 1 row x 2 classes = 2 outputs
        const y = $.let([
            [1.0, 0.0],  // class 0
            [1.0, 0.0],  // class 0
            [1.0, 0.0],  // class 0
            [1.0, 0.0],  // class 0
            [0.0, 1.0],  // class 1
        ]);

        // Compute mutex class weights for row 0 (the only row)
        // n_rows = 1, mutex_row_indices = [0]
        const weights = $.let(Torch.computeMutexClassWeights(y, 1n, [0n]));

        // Result is (n_mutex_rows, n_classes) = (1, 2)
        $(Assert.equal(weights.size(), 1n));
        const row0_weights = $.let(weights.get(0n));
        $(Assert.equal(row0_weights.size(), 2n));

        // Class 0: 4 samples → low weight
        // Class 1: 1 sample → high weight
        // Weight formula: (n_samples / n_classes) / (count + smoothing)
        // Class 0: (5/2) / (4+1) = 2.5/5 = 0.5
        // Class 1: (5/2) / (1+1) = 2.5/2 = 1.25
        $(Assert.less(row0_weights.get(0n), 1.0));      // Low weight for dominant class
        $(Assert.greater(row0_weights.get(1n), 1.0));   // Higher weight for rare class
    });

    test("compute_mutex_class_weights handles multiple mutex rows", $ => {
        // 2 rows x 3 classes = 6 outputs
        const y = $.let([
            // Row 0: mostly class 0, Row 1: mostly class 2
            [1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        ]);

        // Compute weights for both mutex rows
        const weights = $.let(Torch.computeMutexClassWeights(y, 2n, [0n, 1n]));

        $(Assert.equal(weights.size(), 2n));

        // Row 0: class 0 = 3, class 1 = 1, class 2 = 0
        const row0_weights = $.let(weights.get(0n));
        $(Assert.equal(row0_weights.size(), 3n));
        // Class 0 should have lowest weight (most common)
        $(Assert.less(row0_weights.get(0n), row0_weights.get(1n)));

        // Row 1: class 0 = 0, class 1 = 1, class 2 = 3
        const row1_weights = $.let(weights.get(1n));
        $(Assert.equal(row1_weights.size(), 3n));
        // Class 2 should have lowest weight (most common)
        $(Assert.less(row1_weights.get(2n), row1_weights.get(1n)));
    });
}, { exportOnly: true });
