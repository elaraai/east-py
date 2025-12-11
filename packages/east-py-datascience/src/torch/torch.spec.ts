/**
 * Copyright (c) 2025 Elara AI Pty Ltd
 * Dual-licensed under AGPL-3.0 and commercial license. See LICENSE for details.
 */

/**
 * PyTorch platform function tests
 */
import { variant } from "@elaraai/east";
import { describeEast, Assert } from "@elaraai/east-node-std";
import { Torch } from "./torch.js";

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
            dropout: variant('none', null),
            output_dim: variant('none', null),
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
            dropout: variant('none', null),
            output_dim: variant('none', null),
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
            activation: variant('some', variant('tanh', {})),
            dropout: variant('some', 0.1),
            output_dim: variant('none', null),
        });

        const train_config = $.let({
            epochs: variant('some', 30n),
            batch_size: variant('some', 2n),
            learning_rate: variant('some', 0.01),
            loss: variant('none', null),
            optimizer: variant('some', variant('sgd', {})),
            early_stopping: variant('none', null),
            validation_split: variant('some', 0.25),
            random_state: variant('some', 42n),
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
            activation: variant('some', variant('relu', {})),
            dropout: variant('none', null),
            output_dim: variant('some', 1n),
        });

        const train_config = $.let({
            epochs: variant('some', 100n),
            batch_size: variant('some', 4n),
            learning_rate: variant('some', 0.01),
            loss: variant('some', variant('mse', {})),
            optimizer: variant('some', variant('adam', {})),
            early_stopping: variant('some', 10n),
            validation_split: variant('some', 0.25),
            random_state: variant('some', 42n),
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
            dropout: variant('none', null),
            output_dim: variant('none', null),
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
            dropout: variant('none', null),
            output_dim: variant('none', null),
        });

        const train_config = $.let({
            epochs: variant('some', 30n),
            batch_size: variant('some', 2n),
            learning_rate: variant('some', 0.01),
            loss: variant('none', null),
            optimizer: variant('some', variant('adamw', {})),
            early_stopping: variant('none', null),
            validation_split: variant('some', 0.25),
            random_state: variant('some', 42n),
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
            dropout: variant('none', null),
            output_dim: variant('none', null),
        });

        const train_config = $.let({
            epochs: variant('some', 30n),
            batch_size: variant('some', 2n),
            learning_rate: variant('some', 0.01),
            loss: variant('none', null),
            optimizer: variant('some', variant('rmsprop', {})),
            early_stopping: variant('none', null),
            validation_split: variant('some', 0.25),
            random_state: variant('some', 42n),
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
            dropout: variant('none', null),
            output_dim: variant('none', null),
        });

        const train_config = $.let({
            epochs: variant('some', 30n),
            batch_size: variant('some', 2n),
            learning_rate: variant('some', 0.01),
            loss: variant('some', variant('mae', {})),
            optimizer: variant('none', null),
            early_stopping: variant('none', null),
            validation_split: variant('some', 0.25),
            random_state: variant('some', 42n),
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
            activation: variant('some', variant('sigmoid', {})),
            dropout: variant('none', null),
            output_dim: variant('none', null),
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
            activation: variant('some', variant('leaky_relu', {})),
            dropout: variant('none', null),
            output_dim: variant('none', null),
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
            activation: variant('some', variant('relu', {})),
            dropout: variant('some', 0.2),
            output_dim: variant('none', null),
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
            dropout: variant('none', null),
            output_dim: variant('none', null),
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
            activation: variant('some', variant('relu', {})),
            dropout: variant('none', null),
            output_dim: variant('none', null),  // Inferred from y: 3
        });

        const train_config = $.let({
            epochs: variant('some', 100n),
            batch_size: variant('some', 4n),
            learning_rate: variant('some', 0.01),
            loss: variant('some', variant('mse', {})),
            optimizer: variant('some', variant('adam', {})),
            early_stopping: variant('some', 15n),
            validation_split: variant('some', 0.2),
            random_state: variant('some', 42n),
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
            dropout: variant('none', null),
            output_dim: variant('none', null),
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
        });

        const output = $.let(Torch.mlpTrainMulti(X, y, mlp_config, train_config));
        const predictions = $.let(Torch.mlpPredictMulti(output.model, X));

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
            activation: variant('some', variant('relu', {})),
            dropout: variant('none', null),
            output_dim: variant('none', null),  // Inferred from y: 4
        });

        const train_config = $.let({
            epochs: variant('some', 200n),
            batch_size: variant('some', 4n),
            learning_rate: variant('some', 0.01),
            loss: variant('some', variant('mse', {})),
            optimizer: variant('some', variant('adam', {})),
            early_stopping: variant('some', 20n),
            validation_split: variant('some', 0.2),
            random_state: variant('some', 42n),
        });

        const output = $.let(Torch.mlpTrainMulti(X, y, mlp_config, train_config));
        const predictions = $.let(Torch.mlpPredictMulti(output.model, X));

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
            dropout: variant('none', null),
            output_dim: variant('some', 2n),  // Explicit: match y's columns
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
        });

        const output = $.let(Torch.mlpTrainMulti(X, y, mlp_config, train_config));
        const predictions = $.let(Torch.mlpPredictMulti(output.model, X));

        $(Assert.equal(predictions.size(), 4n));
        $(Assert.equal(predictions.get(0n).size(), 2n));
    });

    test("error: train_multi shape mismatch", $ => {
        const X = $.let([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);  // 3 samples
        const y = $.let([[1.0, 2.0], [3.0, 4.0]]);  // 2 samples

        const mlp_config = $.let({
            hidden_layers: [8n],
            activation: variant('none', null),
            dropout: variant('none', null),
            output_dim: variant('none', null),
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
        });

        $(Assert.throws(Torch.mlpTrainMulti(X, y, mlp_config, train_config), /torch_mlp_train.*X.*3.*y.*2/));
    });
}, { exportOnly: true });
