/**
 * Copyright (c) 2025 Elara AI Pty Ltd
 * Dual-licensed under AGPL-3.0 and commercial license. See LICENSE for details.
 */

/**
 * Lightning platform function tests
 *
 * Tests for PyTorch Lightning neural network training.
 */
import { East, variant, IntegerType, FloatType, NullType } from "@elaraai/east";
import { describeEast, Assert } from "@elaraai/east-node-std";
import { Lightning } from "./lightning.js";

describeEast("Lightning platform functions", (test) => {
    test("regression: train and predict works", $ => {
        // Simple linear data: y = x1 + x2
        const X = $.let([
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
            [5.0, 5.0],
            [6.0, 6.0],
            [7.0, 7.0],
            [8.0, 8.0],
        ]);
        // Target as matrix (n_samples, 1)
        const y = $.let([
            [2.0],
            [4.0],
            [6.0],
            [8.0],
            [10.0],
            [12.0],
            [14.0],
            [16.0],
        ]);

        const config = $.let({
            architecture: variant('mlp', {
                hidden_layers: [16n, 8n],
            }),
            output: variant('regression', null),
            learning_rate: variant('some', 0.01),
            max_epochs: variant('some', 50n),
            patience: variant('some', 10n),
            batch_size: variant('some', 4n),
            dropout: variant('some', 0.0),
            gradient_clip: variant('some', 1.0),
            weight_decay: variant('none', null),
            random_state: variant('some', 42n),
            epoch_callback: variant('none', null),
        });

        // Train model
        const result = $.let(Lightning.train(X, y, config, variant('none', null), variant('none', null), variant('none', null)));

        // Check result structure
        $(Assert.greaterEqual(result.best_epoch, 0n));

        // Predict on training data
        const y_pred = $.let(Lightning.predict(result.model, X, variant('none', null)));

        // Check dimensions
        $(Assert.equal(y_pred.size(), 8n));
        $(Assert.equal(y_pred.get(0n).size(), 1n));

        // Verify model quality - predictions should follow y = x1 + x2 pattern
        // Predictions should increase monotonically (larger inputs = larger outputs)
        $(Assert.less(y_pred.get(0n).get(0n), y_pred.get(4n).get(0n)));
        $(Assert.less(y_pred.get(4n).get(0n), y_pred.get(7n).get(0n)));
    });

    test("binary: train and predict works", $ => {
        // Binary classification data
        const X = $.let([
            [0.0, 0.0],
            [0.5, 0.5],
            [1.0, 1.0],
            [1.5, 1.5],
            [10.0, 10.0],
            [10.5, 10.5],
            [11.0, 11.0],
            [11.5, 11.5],
        ]);
        // Binary targets as matrix (n_samples, 1)
        const y = $.let([
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [1.0],
            [1.0],
            [1.0],
            [1.0],
        ]);

        const config = $.let({
            architecture: variant('mlp', {
                hidden_layers: [16n],
            }),
            output: variant('binary', {
                pos_weight: variant('none', null),
            }),
            learning_rate: variant('some', 0.01),
            max_epochs: variant('some', 100n),
            patience: variant('some', 20n),
            batch_size: variant('some', 4n),
            dropout: variant('some', 0.0),
            gradient_clip: variant('some', 1.0),
            weight_decay: variant('none', null),
            random_state: variant('some', 42n),
            epoch_callback: variant('none', null),
        });

        // Train model
        const result = $.let(Lightning.train(X, y, config, variant('none', null), variant('none', null), variant('none', null)));

        // Predict probabilities
        const y_pred = $.let(Lightning.predict(result.model, X, variant('none', null)));

        // Check dimensions
        $(Assert.equal(y_pred.size(), 8n));

        // First samples should have low probability (class 0)
        $(Assert.less(y_pred.get(0n).get(0n), East.value(0.5)));
        // Last samples should have high probability (class 1)
        $(Assert.greater(y_pred.get(7n).get(0n), East.value(0.5)));
    });

    test("multiclass: train and predict works", $ => {
        // 3-class classification data
        const X = $.let([
            [0.0, 0.0],
            [0.5, 0.5],
            [5.0, 5.0],
            [5.5, 5.5],
            [10.0, 10.0],
            [10.5, 10.5],
        ]);
        // One-hot encoded targets (n_samples, n_classes)
        const y = $.let([
            [1.0, 0.0, 0.0],  // class 0
            [1.0, 0.0, 0.0],  // class 0
            [0.0, 1.0, 0.0],  // class 1
            [0.0, 1.0, 0.0],  // class 1
            [0.0, 0.0, 1.0],  // class 2
            [0.0, 0.0, 1.0],  // class 2
        ]);

        const config = $.let({
            architecture: variant('mlp', {
                hidden_layers: [16n],
            }),
            output: variant('multiclass', {
                n_classes: 3n,
                class_weights: variant('none', null),
            }),
            learning_rate: variant('some', 0.01),
            max_epochs: variant('some', 100n),
            patience: variant('some', 20n),
            batch_size: variant('some', 4n),
            dropout: variant('some', 0.0),
            gradient_clip: variant('some', 1.0),
            weight_decay: variant('none', null),
            random_state: variant('some', 42n),
            epoch_callback: variant('none', null),
        });

        // Train model
        const result = $.let(Lightning.train(X, y, config, variant('none', null), variant('none', null), variant('none', null)));

        // Predict probabilities
        const y_pred = $.let(Lightning.predict(result.model, X, variant('none', null)));

        // Check dimensions: 6 samples x 3 classes
        $(Assert.equal(y_pred.size(), 6n));
        $(Assert.equal(y_pred.get(0n).size(), 3n));

        // Verify model outputs valid probabilities
        // Probabilities should sum to ~1 (softmax output)
        const sum0 = $.let(y_pred.get(0n).get(0n).add(y_pred.get(0n).get(1n)).add(y_pred.get(0n).get(2n)));
        $(Assert.greater(sum0, East.value(0.99)));
        $(Assert.less(sum0, East.value(1.01)));

        // Each probability should be between 0 and 1
        $(Assert.greaterEqual(y_pred.get(0n).get(0n), East.value(0.0)));
        $(Assert.lessEqual(y_pred.get(0n).get(0n), East.value(1.0)));
    });

    test("multi_head: train and predict works", $ => {
        // Multi-head classification: 2 heads x 3 classes each
        const X = $.let([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.0, 0.0],
        ]);
        // Targets: (n_samples, n_heads * n_classes) = (4, 6)
        // Each row has 2 one-hot encoded heads
        const y = $.let([
            [1.0, 0.0, 0.0,  0.0, 1.0, 0.0],  // head0=class0, head1=class1
            [0.0, 1.0, 0.0,  0.0, 0.0, 1.0],  // head0=class1, head1=class2
            [0.0, 0.0, 1.0,  1.0, 0.0, 0.0],  // head0=class2, head1=class0
            [1.0, 0.0, 0.0,  0.0, 1.0, 0.0],  // head0=class0, head1=class1
        ]);

        const config = $.let({
            architecture: variant('mlp', {
                hidden_layers: [16n],
            }),
            output: variant('multi_head', {
                n_heads: 2n,
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

        // Train model
        const result = $.let(Lightning.train(X, y, config, variant('none', null), variant('none', null), variant('none', null)));

        // Predict
        const y_pred = $.let(Lightning.predict(result.model, X, variant('none', null)));

        // Check dimensions: 4 samples x 6 outputs (2 heads x 3 classes)
        $(Assert.equal(y_pred.size(), 4n));
        $(Assert.equal(y_pred.get(0n).size(), 6n));

        // Verify each head's probs sum to ~1 (softmax per head)
        const head0_sum = $.let(y_pred.get(0n).get(0n).add(y_pred.get(0n).get(1n)).add(y_pred.get(0n).get(2n)));
        const head1_sum = $.let(y_pred.get(0n).get(3n).add(y_pred.get(0n).get(4n)).add(y_pred.get(0n).get(5n)));
        $(Assert.greater(head0_sum, East.value(0.99)));
        $(Assert.less(head0_sum, East.value(1.01)));
        $(Assert.greater(head1_sum, East.value(0.99)));
        $(Assert.less(head1_sum, East.value(1.01)));

        // Each probability should be between 0 and 1
        $(Assert.greaterEqual(y_pred.get(0n).get(0n), East.value(0.0)));
        $(Assert.lessEqual(y_pred.get(0n).get(0n), East.value(1.0)));
    });

    test("autoencoder: train, encode, decode works", $ => {
        // Data for autoencoder
        const X = $.let([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
        ]);

        const config = $.let({
            architecture: variant('autoencoder', {
                encoder_layers: [8n],
                latent_dim: 2n,
                decoder_layers: [8n],
            }),
            output: variant('regression', null),
            learning_rate: variant('some', 0.01),
            max_epochs: variant('some', 100n),
            patience: variant('some', 20n),
            batch_size: variant('some', 4n),
            dropout: variant('some', 0.0),
            gradient_clip: variant('some', 1.0),
            weight_decay: variant('none', null),
            random_state: variant('some', 42n),
            epoch_callback: variant('none', null),
        });

        // Train autoencoder (X -> X)
        const result = $.let(Lightning.train(X, X, config, variant('none', null), variant('none', null), variant('none', null)));

        // Encode to latent space
        const z = $.let(Lightning.encode(result.model, X));

        // Check latent dimensions: 6 samples x 2 latent
        $(Assert.equal(z.size(), 6n));
        $(Assert.equal(z.get(0n).size(), 2n));

        // Decode back
        const X_reconstructed = $.let(Lightning.decode(result.model, z));

        // Check reconstruction dimensions
        $(Assert.equal(X_reconstructed.size(), 6n));
        $(Assert.equal(X_reconstructed.get(0n).size(), 4n));

        // Verify autoencoder quality - reconstructions should be close to inputs
        // For one-hot input [1,0,0,0], reconstruction should have highest value at position 0
        $(Assert.greater(X_reconstructed.get(0n).get(0n), X_reconstructed.get(0n).get(1n)));
        $(Assert.greater(X_reconstructed.get(0n).get(0n), X_reconstructed.get(0n).get(2n)));
        $(Assert.greater(X_reconstructed.get(0n).get(0n), X_reconstructed.get(0n).get(3n)));

        // For [0,1,0,0], position 1 should be highest
        $(Assert.greater(X_reconstructed.get(1n).get(1n), X_reconstructed.get(1n).get(0n)));
        $(Assert.greater(X_reconstructed.get(1n).get(1n), X_reconstructed.get(1n).get(2n)));
        $(Assert.greater(X_reconstructed.get(1n).get(1n), X_reconstructed.get(1n).get(3n)));

        // Embeddings should be different for different inputs
        $(Assert.notEqual(z.get(0n).get(0n), z.get(1n).get(0n)));
    });

    test("respects random_state for reproducibility", $ => {
        const X = $.let([
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
        ]);
        const y = $.let([
            [3.0],
            [5.0],
            [7.0],
            [9.0],
        ]);

        const config = $.let({
            architecture: variant('mlp', {
                hidden_layers: [8n],
            }),
            output: variant('regression', null),
            learning_rate: variant('some', 0.01),
            max_epochs: variant('some', 20n),
            patience: variant('some', 5n),
            batch_size: variant('some', 2n),
            dropout: variant('some', 0.0),
            gradient_clip: variant('some', 1.0),
            weight_decay: variant('none', null),
            random_state: variant('some', 123n),
            epoch_callback: variant('none', null),
        });

        // Train two models with same seed
        const result1 = $.let(Lightning.train(X, y, config, variant('none', null), variant('none', null), variant('none', null)));
        const result2 = $.let(Lightning.train(X, y, config, variant('none', null), variant('none', null), variant('none', null)));

        // Predictions should be identical
        const pred1 = $.let(Lightning.predict(result1.model, X, variant('none', null)));
        const pred2 = $.let(Lightning.predict(result2.model, X, variant('none', null)));

        $(Assert.equal(pred1.get(0n).get(0n), pred2.get(0n).get(0n)));
        $(Assert.equal(pred1.get(1n).get(0n), pred2.get(1n).get(0n)));
    });

    test("binary with vector pos_weight works", $ => {
        // Imbalanced binary data: output_dim = 2, first output is rare, second is common
        const X = $.let([
            [0.0, 0.0],
            [0.5, 0.5],
            [1.0, 1.0],
            [1.5, 1.5],
            [10.0, 10.0],
            [10.5, 10.5],
            [11.0, 11.0],
            [11.5, 11.5],
        ]);
        // Two binary outputs: first rarely 1, second commonly 1
        const y = $.let([
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0],  // rare: first output = 1
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]);

        const config = $.let({
            architecture: variant('mlp', {
                hidden_layers: [16n],
            }),
            output: variant('binary', {
                // Per-position pos_weight: upweight first output (rare), downweight second
                pos_weight: variant('some', [3.0, 0.5]),
            }),
            learning_rate: variant('some', 0.01),
            max_epochs: variant('some', 100n),
            patience: variant('some', 20n),
            batch_size: variant('some', 4n),
            dropout: variant('some', 0.0),
            gradient_clip: variant('some', 1.0),
            weight_decay: variant('none', null),
            random_state: variant('some', 42n),
            epoch_callback: variant('none', null),
        });

        // Train model
        const result = $.let(Lightning.train(X, y, config, variant('none', null), variant('none', null), variant('none', null)));

        // Should train successfully
        $(Assert.greaterEqual(result.best_epoch, 0n));

        // Predict and verify output dimensions
        const y_pred = $.let(Lightning.predict(result.model, X, variant('none', null)));
        $(Assert.equal(y_pred.size(), 8n));
        $(Assert.equal(y_pred.get(0n).size(), 2n));

        // Predictions should be between 0 and 1 (sigmoid)
        $(Assert.greaterEqual(y_pred.get(0n).get(0n), East.value(0.0)));
        $(Assert.lessEqual(y_pred.get(0n).get(0n), East.value(1.0)));
    });

    test("multiclass with class_weights works", $ => {
        // 3-class classification with weights
        const X = $.let([
            [0.0, 0.0],
            [0.5, 0.5],
            [5.0, 5.0],
            [5.5, 5.5],
            [10.0, 10.0],
            [10.5, 10.5],
        ]);
        const y = $.let([
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ]);

        const config = $.let({
            architecture: variant('mlp', {
                hidden_layers: [16n],
            }),
            output: variant('multiclass', {
                n_classes: 3n,
                class_weights: variant('some', [1.0, 2.0, 1.0]),  // Upweight class 1
            }),
            learning_rate: variant('some', 0.01),
            max_epochs: variant('some', 50n),
            patience: variant('some', 10n),
            batch_size: variant('some', 4n),
            dropout: variant('some', 0.0),
            gradient_clip: variant('some', 1.0),
            weight_decay: variant('none', null),
            random_state: variant('some', 42n),
            epoch_callback: variant('none', null),
        });

        // Train model
        const result = $.let(Lightning.train(X, y, config, variant('none', null), variant('none', null), variant('none', null)));

        // Should train successfully
        $(Assert.greaterEqual(result.best_epoch, 0n));

        // Verify model outputs valid probabilities
        const y_pred = $.let(Lightning.predict(result.model, X, variant('none', null)));

        // Probabilities should sum to ~1
        const sum0 = $.let(y_pred.get(0n).get(0n).add(y_pred.get(0n).get(1n)).add(y_pred.get(0n).get(2n)));
        $(Assert.greater(sum0, East.value(0.99)));
        $(Assert.less(sum0, East.value(1.01)));
    });

    test("multi_head with class_weights works", $ => {
        // Multi-head with imbalanced classes - upweight rare classes
        // 2 heads x 3 classes, where class 0 dominates in both heads
        const X = $.let([
            [1.0, 0.0],
            [0.9, 0.1],
            [0.8, 0.2],
            [0.7, 0.3],
            [0.0, 1.0],  // rare: head0=class1
            [0.5, 0.5],  // rare: head1=class2
        ]);
        const y = $.let([
            [1.0, 0.0, 0.0,  1.0, 0.0, 0.0],  // head0=class0, head1=class0
            [1.0, 0.0, 0.0,  1.0, 0.0, 0.0],  // head0=class0, head1=class0
            [1.0, 0.0, 0.0,  1.0, 0.0, 0.0],  // head0=class0, head1=class0
            [1.0, 0.0, 0.0,  1.0, 0.0, 0.0],  // head0=class0, head1=class0
            [0.0, 1.0, 0.0,  1.0, 0.0, 0.0],  // head0=class1 (rare), head1=class0
            [1.0, 0.0, 0.0,  0.0, 0.0, 1.0],  // head0=class0, head1=class2 (rare)
        ]);

        // Class weights: upweight rare classes (1 and 2)
        const class_weights = $.let([
            [1.0, 4.0, 4.0],  // head 0 weights
            [1.0, 4.0, 4.0],  // head 1 weights
        ]);

        const config = $.let({
            architecture: variant('mlp', {
                hidden_layers: [32n, 16n],
            }),
            output: variant('multi_head', {
                n_heads: 2n,
                n_classes_per_head: 3n,
                class_weights: variant('some', class_weights),
            }),
            learning_rate: variant('some', 0.01),
            max_epochs: variant('some', 200n),
            patience: variant('some', 30n),
            batch_size: variant('some', 3n),
            dropout: variant('some', 0.0),
            gradient_clip: variant('some', 1.0),
            weight_decay: variant('none', null),
            random_state: variant('some', 42n),
            epoch_callback: variant('none', null),
        });

        const result = $.let(Lightning.train(X, y, config, variant('none', null), variant('none', null), variant('none', null)));

        // Should train successfully
        $(Assert.greaterEqual(result.best_epoch, 0n));

        // Verify predictions
        const y_pred = $.let(Lightning.predict(result.model, X, variant('none', null)));

        // Check dimensions: 6 samples x 6 outputs (2 heads x 3 classes)
        $(Assert.equal(y_pred.size(), 6n));
        $(Assert.equal(y_pred.get(0n).size(), 6n));

        // Verify each head's probs sum to ~1
        const head0_sum = $.let(y_pred.get(0n).get(0n).add(y_pred.get(0n).get(1n)).add(y_pred.get(0n).get(2n)));
        const head1_sum = $.let(y_pred.get(0n).get(3n).add(y_pred.get(0n).get(4n)).add(y_pred.get(0n).get(5n)));
        $(Assert.greater(head0_sum, East.value(0.99)));
        $(Assert.less(head0_sum, East.value(1.01)));
        $(Assert.greater(head1_sum, East.value(0.99)));
        $(Assert.less(head1_sum, East.value(1.01)));
    });

    test("multi_head with masks works", $ => {
        // Multi-head where certain classes are masked (invalid) per sample
        const X = $.let([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, 0.5],
        ]);
        // 2 heads x 3 classes
        const y = $.let([
            [1.0, 0.0, 0.0,  0.0, 1.0, 0.0],  // head0=class0, head1=class1
            [0.0, 1.0, 0.0,  0.0, 0.0, 1.0],  // head0=class1, head1=class2
            [0.0, 0.0, 1.0,  1.0, 0.0, 0.0],  // head0=class2, head1=class0
            [1.0, 0.0, 0.0,  0.0, 1.0, 0.0],  // head0=class0, head1=class1
        ]);

        // Masks: (n_samples, n_heads, n_classes) - True = valid
        // Sample 0: all valid
        // Sample 1: head0 class2 masked, head1 class0 masked
        // Sample 2: head0 class0 masked, head1 class2 masked
        // Sample 3: all valid
        const masks = $.let([
            [[true, true, true], [true, true, true]],      // Sample 0: all valid
            [[true, true, false], [false, true, true]],    // Sample 1: some masked
            [[false, true, true], [true, true, false]],    // Sample 2: some masked
            [[true, true, true], [true, true, true]],      // Sample 3: all valid
        ]);

        const config = $.let({
            architecture: variant('mlp', {
                hidden_layers: [16n],
            }),
            output: variant('multi_head', {
                n_heads: 2n,
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

        // Train with masks
        const result = $.let(Lightning.train(X, y, config, variant('some', masks), variant('none', null), variant('none', null)));

        // Should train successfully
        $(Assert.greaterEqual(result.best_epoch, 0n));

        // Predict with masks
        const y_pred = $.let(Lightning.predict(result.model, X, variant('some', masks)));

        // Check dimensions
        $(Assert.equal(y_pred.size(), 4n));
        $(Assert.equal(y_pred.get(0n).size(), 6n));

        // For sample 1, head0 class2 is masked - its probability should be very low
        // (masked positions get -inf logits, so softmax gives ~0)
        $(Assert.less(y_pred.get(1n).get(2n), East.value(0.001)));

        // For sample 2, head1 class2 is masked
        $(Assert.less(y_pred.get(2n).get(5n), East.value(0.001)));

        // Probabilities should still sum to ~1 per head (softmax renormalizes)
        const s1_h0_sum = $.let(y_pred.get(1n).get(0n).add(y_pred.get(1n).get(1n)).add(y_pred.get(1n).get(2n)));
        $(Assert.greater(s1_h0_sum, East.value(0.99)));
        $(Assert.less(s1_h0_sum, East.value(1.01)));
    });

    test("binary with masks works", $ => {
        // Binary classification with some positions masked
        const X = $.let([
            [0.0, 0.0],
            [1.0, 1.0],
            [5.0, 5.0],
            [6.0, 6.0],
            [10.0, 10.0],
            [11.0, 11.0],
        ]);
        // 4 binary outputs per sample
        const y = $.let([
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
        ]);

        // Masks: (n_samples, 1, output_dim) - True = valid
        // Some positions are masked for training
        const masks = $.let([
            [[true, true, true, true]],     // all valid
            [[true, true, false, false]],   // outputs 2,3 masked
            [[false, true, true, false]],   // outputs 0,3 masked
            [[true, true, true, true]],     // all valid
            [[false, false, true, true]],   // outputs 0,1 masked
            [[true, true, true, true]],     // all valid
        ]);

        const config = $.let({
            architecture: variant('mlp', {
                hidden_layers: [16n],
            }),
            output: variant('binary', {
                pos_weight: variant('none', null),
            }),
            learning_rate: variant('some', 0.01),
            max_epochs: variant('some', 100n),
            patience: variant('some', 20n),
            batch_size: variant('some', 3n),
            dropout: variant('some', 0.0),
            gradient_clip: variant('some', 1.0),
            weight_decay: variant('none', null),
            random_state: variant('some', 42n),
            epoch_callback: variant('none', null),
        });

        // Train with masks
        const result = $.let(Lightning.train(X, y, config, variant('some', masks), variant('none', null), variant('none', null)));

        // Should train successfully
        $(Assert.greaterEqual(result.best_epoch, 0n));

        // Predict with masks - masked positions should be 0
        const y_pred = $.let(Lightning.predict(result.model, X, variant('some', masks)));
        $(Assert.equal(y_pred.size(), 6n));
        $(Assert.equal(y_pred.get(0n).size(), 4n));

        // Sample 1: outputs 2,3 are masked - should be 0
        $(Assert.equal(y_pred.get(1n).get(2n), East.value(0.0)));
        $(Assert.equal(y_pred.get(1n).get(3n), East.value(0.0)));

        // Sample 2: outputs 0,3 are masked - should be 0
        $(Assert.equal(y_pred.get(2n).get(0n), East.value(0.0)));
        $(Assert.equal(y_pred.get(2n).get(3n), East.value(0.0)));

        // Unmasked positions should have valid probabilities (0-1)
        $(Assert.greaterEqual(y_pred.get(0n).get(0n), East.value(0.0)));
        $(Assert.lessEqual(y_pred.get(0n).get(0n), East.value(1.0)));
    });

    test("epoch_callback is called with metrics", $ => {
        const X = $.let([
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
            [5.0, 5.0],
            [6.0, 6.0],
        ]);
        const y = $.let([
            [2.0],
            [4.0],
            [6.0],
            [8.0],
            [10.0],
            [12.0],
        ]);

        // Track that callback was called by counting epochs
        const epochCount = $.let(0n);
        const lastTrainLoss = $.let(0.0);

        const callback = East.function(
            [IntegerType, FloatType, FloatType],
            NullType,
            ($, epoch, train_loss) => {
                // Increment counter each time callback is called
                $.assign(epochCount, epochCount.add(1n));
                // Store train loss to verify it's reasonable
                $.assign(lastTrainLoss, train_loss);
                return $.return(null);
            }
        );

        const config = $.let({
            architecture: variant('mlp', {
                hidden_layers: [8n],
            }),
            output: variant('regression', null),
            learning_rate: variant('some', 0.01),
            max_epochs: variant('some', 20n),
            patience: variant('some', 5n),
            batch_size: variant('some', 2n),
            dropout: variant('some', 0.0),
            gradient_clip: variant('some', 1.0),
            weight_decay: variant('none', null),
            random_state: variant('some', 42n),
            epoch_callback: variant('some', callback),
        });

        // Train with callback
        const result = $.let(Lightning.train(X, y, config, variant('none', null), variant('none', null), variant('none', null)));

        // Callback should have been called at least once (epochCount > 0)
        $(Assert.greater(epochCount, 0n));

        // Last train loss should be non-negative
        $(Assert.greaterEqual(lastTrainLoss, East.value(0.0)));

        // Should train successfully
        $(Assert.greaterEqual(result.best_epoch, 0n));

        // Model should have learned the linear pattern
        const y_pred = $.let(Lightning.predict(result.model, X, variant('none', null)));

        // Predictions should be reasonably close to targets
        $(Assert.greater(y_pred.get(0n).get(0n), East.value(0.0)));
        $(Assert.less(y_pred.get(0n).get(0n), East.value(6.0)));
    });

    // =========================================================================
    // Model-specific tests matching design-lightning-module.md configurations
    // =========================================================================

    test("autoencoder + multiclass with encode/decode", $ => {
        // Autoencoder for categorical embeddings
        // One-hot encoded origins (simulating n_origins = 4)
        const X = $.let([
            [1.0, 0.0, 0.0, 0.0],  // origin 0
            [0.0, 1.0, 0.0, 0.0],  // origin 1
            [0.0, 0.0, 1.0, 0.0],  // origin 2
            [0.0, 0.0, 0.0, 1.0],  // origin 3
            [1.0, 0.0, 0.0, 0.0],  // origin 0
            [0.0, 1.0, 0.0, 0.0],  // origin 1
        ]);

        const config = $.let({
            architecture: variant('autoencoder', {
                encoder_layers: [8n],
                latent_dim: 2n,
                decoder_layers: [8n],
            }),
            output: variant('multiclass', {
                n_classes: 4n,
                class_weights: variant('none', null),
            }),
            learning_rate: variant('some', 0.01),
            max_epochs: variant('some', 100n),
            patience: variant('some', 20n),
            batch_size: variant('some', 3n),
            dropout: variant('some', 0.0),
            gradient_clip: variant('some', 1.0),
            weight_decay: variant('none', null),
            random_state: variant('some', 42n),
            epoch_callback: variant('none', null),
        });

        // Train autoencoder (X -> X reconstruction with multiclass output)
        const result = $.let(Lightning.train(X, X, config, variant('none', null), variant('none', null), variant('none', null)));

        // Should train successfully
        $(Assert.greaterEqual(result.best_epoch, 0n));

        // Encode to latent space
        const embeddings = $.let(Lightning.encode(result.model, X));

        // Check latent dimensions: 6 samples x 2 latent
        $(Assert.equal(embeddings.size(), 6n));
        $(Assert.equal(embeddings.get(0n).size(), 2n));

        // Similar origins should have similar embeddings
        // origin 0 appears at samples 0 and 4
        const emb0 = $.let(embeddings.get(0n));
        const emb4 = $.let(embeddings.get(4n));
        const dist_same = $.let(
            emb0.get(0n).subtract(emb4.get(0n)).abs()
                .add(emb0.get(1n).subtract(emb4.get(1n)).abs())
        );

        // Different origins should have different embeddings
        const emb1 = $.let(embeddings.get(1n));
        const dist_diff = $.let(
            emb0.get(0n).subtract(emb1.get(0n)).abs()
                .add(emb0.get(1n).subtract(emb1.get(1n)).abs())
        );

        // Same origin distance should be less than different origin distance
        $(Assert.less(dist_same, dist_diff));

        // Decode should produce valid probabilities
        const decoded = $.let(Lightning.decode(result.model, embeddings));
        $(Assert.equal(decoded.size(), 6n));
        $(Assert.equal(decoded.get(0n).size(), 4n));

        // Probabilities should sum to ~1 (softmax output)
        const prob_sum = $.let(
            decoded.get(0n).get(0n)
                .add(decoded.get(0n).get(1n))
                .add(decoded.get(0n).get(2n))
                .add(decoded.get(0n).get(3n))
        );
        $(Assert.greater(prob_sum, East.value(0.99)));
        $(Assert.less(prob_sum, East.value(1.01)));
    });

    test("autoencoder + binary + vector pos_weight + masks", $ => {
        // Binary autoencoder for sparse feature embeddings
        // Binary task vectors (n_tasks = 4)
        const X = $.let([
            [1.0, 0.0, 1.0, 0.0],  // tasks 0,2 active
            [0.0, 1.0, 0.0, 1.0],  // tasks 1,3 active
            [1.0, 1.0, 0.0, 0.0],  // tasks 0,1 active
            [0.0, 0.0, 1.0, 1.0],  // tasks 2,3 active
            [1.0, 0.0, 0.0, 0.0],  // only task 0 active
            [0.0, 0.0, 0.0, 1.0],  // only task 3 active
        ]);

        // Masks: some positions are never valid for certain samples
        // (n_samples, 1, n_outputs) - 3D with middle dim = 1 for binary
        const masks = $.let([
            [[true, true, true, true]],      // all valid
            [[true, true, true, true]],      // all valid
            [[true, true, false, false]],    // tasks 2,3 masked
            [[false, false, true, true]],    // tasks 0,1 masked
            [[true, false, false, false]],   // only task 0 valid
            [[false, false, false, true]],   // only task 3 valid
        ]);

        const config = $.let({
            architecture: variant('autoencoder', {
                encoder_layers: [8n],
                latent_dim: 2n,
                decoder_layers: [8n],
            }),
            output: variant('binary', {
                // Per-position pos_weight: upweight all positive classes
                pos_weight: variant('some', [3.0, 3.0, 3.0, 3.0]),
            }),
            learning_rate: variant('some', 0.01),
            max_epochs: variant('some', 100n),
            patience: variant('some', 20n),
            batch_size: variant('some', 3n),
            dropout: variant('some', 0.0),
            gradient_clip: variant('some', 1.0),
            weight_decay: variant('none', null),
            random_state: variant('some', 42n),
            epoch_callback: variant('none', null),
        });

        // Train with masks
        const result = $.let(Lightning.train(X, X, config, variant('some', masks), variant('none', null), variant('none', null)));

        // Should train successfully
        $(Assert.greaterEqual(result.best_epoch, 0n));

        // Encode to latent space
        const embeddings = $.let(Lightning.encode(result.model, X));
        $(Assert.equal(embeddings.size(), 6n));
        $(Assert.equal(embeddings.get(0n).size(), 2n));

        // Predict with masks - masked positions should be 0
        const y_pred = $.let(Lightning.predict(result.model, X, variant('some', masks)));
        $(Assert.equal(y_pred.size(), 6n));
        $(Assert.equal(y_pred.get(0n).size(), 4n));

        // Sample 2: tasks 2,3 are masked - should be 0
        $(Assert.equal(y_pred.get(2n).get(2n), East.value(0.0)));
        $(Assert.equal(y_pred.get(2n).get(3n), East.value(0.0)));

        // Sample 4: only task 0 valid - tasks 1,2,3 should be 0
        $(Assert.equal(y_pred.get(4n).get(1n), East.value(0.0)));
        $(Assert.equal(y_pred.get(4n).get(2n), East.value(0.0)));
        $(Assert.equal(y_pred.get(4n).get(3n), East.value(0.0)));

        // Unmasked positions should be between 0 and 1 (sigmoid output)
        $(Assert.greaterEqual(y_pred.get(0n).get(0n), East.value(0.0)));
        $(Assert.lessEqual(y_pred.get(0n).get(0n), East.value(1.0)));
    });

    test("autoencoder + multi_head + class_weights + masks", $ => {
        // Multi-head autoencoder for structured plans
        // Simulating 3 heads x 3 classes (simplified from 84 x 4)
        // Each head is mutex (one-hot per head)
        const X = $.let([
            [1.0, 0.0, 0.0,  0.0, 1.0, 0.0,  0.0, 0.0, 1.0],  // h0=c0, h1=c1, h2=c2
            [0.0, 1.0, 0.0,  1.0, 0.0, 0.0,  0.0, 1.0, 0.0],  // h0=c1, h1=c0, h2=c1
            [0.0, 0.0, 1.0,  0.0, 0.0, 1.0,  1.0, 0.0, 0.0],  // h0=c2, h1=c2, h2=c0
            [1.0, 0.0, 0.0,  0.0, 1.0, 0.0,  0.0, 0.0, 1.0],  // same as sample 0
            [0.0, 1.0, 0.0,  0.0, 1.0, 0.0,  0.0, 1.0, 0.0],  // h0=c1, h1=c1, h2=c1
            [1.0, 0.0, 0.0,  1.0, 0.0, 0.0,  1.0, 0.0, 0.0],  // h0=c0, h1=c0, h2=c0
        ]);

        // Class weights: upweight rare classes (3 heads x 3 classes)
        const class_weights = $.let([
            [1.0, 2.0, 2.0],  // head 0: class 0 common, 1,2 rare
            [2.0, 1.0, 2.0],  // head 1: class 1 common, 0,2 rare
            [2.0, 2.0, 1.0],  // head 2: class 2 common, 0,1 rare
        ]);

        // Masks: (n_samples, n_heads, n_classes)
        const masks = $.let([
            [[true, true, true], [true, true, true], [true, true, true]],      // all valid
            [[true, true, false], [true, true, true], [true, true, true]],     // h0 c2 masked
            [[true, true, true], [true, true, true], [true, true, true]],      // all valid
            [[true, true, true], [true, true, true], [true, true, true]],      // all valid
            [[true, true, true], [true, true, false], [true, true, true]],     // h1 c2 masked
            [[true, true, true], [true, true, true], [false, true, true]],     // h2 c0 masked
        ]);

        const config = $.let({
            architecture: variant('autoencoder', {
                encoder_layers: [16n],
                latent_dim: 4n,
                decoder_layers: [16n],
            }),
            output: variant('multi_head', {
                n_heads: 3n,
                n_classes_per_head: 3n,
                class_weights: variant('some', class_weights),
            }),
            learning_rate: variant('some', 0.01),
            max_epochs: variant('some', 150n),
            patience: variant('some', 30n),
            batch_size: variant('some', 3n),
            dropout: variant('some', 0.0),
            gradient_clip: variant('some', 1.0),
            weight_decay: variant('none', null),
            random_state: variant('some', 42n),
            epoch_callback: variant('none', null),
        });

        // Train with masks
        const result = $.let(Lightning.train(X, X, config, variant('some', masks), variant('none', null), variant('none', null)));

        // Should train successfully
        $(Assert.greaterEqual(result.best_epoch, 0n));

        // Encode to latent space
        const embeddings = $.let(Lightning.encode(result.model, X));
        $(Assert.equal(embeddings.size(), 6n));
        $(Assert.equal(embeddings.get(0n).size(), 4n));

        // Similar plans should have similar embeddings (samples 0 and 3 are identical)
        const emb0 = $.let(embeddings.get(0n));
        const emb3 = $.let(embeddings.get(3n));
        const dist_same = $.let(
            emb0.get(0n).subtract(emb3.get(0n)).abs()
                .add(emb0.get(1n).subtract(emb3.get(1n)).abs())
                .add(emb0.get(2n).subtract(emb3.get(2n)).abs())
                .add(emb0.get(3n).subtract(emb3.get(3n)).abs())
        );

        // Different plans should have different embeddings
        const emb1 = $.let(embeddings.get(1n));
        const dist_diff = $.let(
            emb0.get(0n).subtract(emb1.get(0n)).abs()
                .add(emb0.get(1n).subtract(emb1.get(1n)).abs())
                .add(emb0.get(2n).subtract(emb1.get(2n)).abs())
                .add(emb0.get(3n).subtract(emb1.get(3n)).abs())
        );

        // Same plan distance should be less than different plan distance
        $(Assert.less(dist_same, dist_diff));

        // Predict with masks
        const y_pred = $.let(Lightning.predict(result.model, X, variant('some', masks)));
        $(Assert.equal(y_pred.size(), 6n));
        $(Assert.equal(y_pred.get(0n).size(), 9n));

        // Each head's probs should sum to ~1
        const h0_sum = $.let(y_pred.get(0n).get(0n).add(y_pred.get(0n).get(1n)).add(y_pred.get(0n).get(2n)));
        const h1_sum = $.let(y_pred.get(0n).get(3n).add(y_pred.get(0n).get(4n)).add(y_pred.get(0n).get(5n)));
        const h2_sum = $.let(y_pred.get(0n).get(6n).add(y_pred.get(0n).get(7n)).add(y_pred.get(0n).get(8n)));
        $(Assert.greater(h0_sum, East.value(0.99)));
        $(Assert.less(h0_sum, East.value(1.01)));
        $(Assert.greater(h1_sum, East.value(0.99)));
        $(Assert.less(h1_sum, East.value(1.01)));
        $(Assert.greater(h2_sum, East.value(0.99)));
        $(Assert.less(h2_sum, East.value(1.01)));

        // Masked positions should have ~0 probability
        // Sample 1, head 0, class 2 is masked
        $(Assert.less(y_pred.get(1n).get(2n), East.value(0.001)));
        // Sample 5, head 2, class 0 is masked
        $(Assert.less(y_pred.get(5n).get(6n), East.value(0.001)));
    });

    // =========================================================================
    // Group weights tests
    // =========================================================================

    test("multi_head with group weights", $ => {
        // 2 groups with different class distributions
        // Group 0: mostly class 0, Group 1: mostly class 1
        const X = $.let([
            [1.0, 0.0], [1.1, 0.1], [0.9, 0.1],  // group 0
            [0.0, 1.0], [0.1, 1.1], [0.1, 0.9],  // group 1
        ]);
        // 2 heads x 3 classes = 6 outputs
        const y = $.let([
            [1.0, 0.0, 0.0,  1.0, 0.0, 0.0],  // group 0: class 0
            [1.0, 0.0, 0.0,  1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0,  1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0,  0.0, 1.0, 0.0],  // group 1: class 1
            [0.0, 1.0, 0.0,  0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0,  0.0, 1.0, 0.0],
        ]);

        // Group weights: [n_groups][n_heads][n_classes]
        const group_weights = $.let({
            weights: variant('multi_head', [
                [[1.0, 2.0, 2.0], [1.0, 2.0, 2.0]],  // group 0: upweight rare classes
                [[2.0, 1.0, 2.0], [2.0, 1.0, 2.0]],  // group 1: upweight rare classes
            ]),
            sample_groups: [0n, 0n, 0n, 1n, 1n, 1n],
        });

        const config = $.let({
            architecture: variant('mlp', { hidden_layers: [16n] }),
            output: variant('multi_head', {
                n_heads: 2n,
                n_classes_per_head: 3n,
                class_weights: variant('none', null),
            }),
            learning_rate: variant('some', 0.01),
            max_epochs: variant('some', 100n),
            patience: variant('some', 20n),
            batch_size: variant('some', 3n),
            dropout: variant('some', 0.0),
            gradient_clip: variant('some', 1.0),
            weight_decay: variant('none', null),
            random_state: variant('some', 42n),
            epoch_callback: variant('none', null),
        });

        const result = $.let(Lightning.train(X, y, config, variant('none', null), variant('some', group_weights), variant('none', null)));

        $(Assert.greaterEqual(result.best_epoch, 0n));

        const y_pred = $.let(Lightning.predict(result.model, X, variant('none', null)));
        $(Assert.equal(y_pred.size(), 6n));

        // Each head's probs should sum to ~1
        const h0_sum = $.let(y_pred.get(0n).get(0n).add(y_pred.get(0n).get(1n)).add(y_pred.get(0n).get(2n)));
        $(Assert.greater(h0_sum, East.value(0.99)));
        $(Assert.less(h0_sum, East.value(1.01)));
    });

    test("binary with group weights", $ => {
        // 2 groups with different sparsity
        const X = $.let([
            [1.0, 0.0], [1.1, 0.1], [0.9, 0.1],  // group 0
            [0.0, 1.0], [0.1, 1.1], [0.1, 0.9],  // group 1
        ]);
        // 4 binary outputs
        const y = $.let([
            [1.0, 0.0, 0.0, 0.0],  // group 0: sparse
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.0],  // group 1: denser
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
        ]);

        // Group weights (pos_weight per group): [n_groups][output_dim]
        const group_weights = $.let({
            weights: variant('binary', [
                [5.0, 5.0, 5.0, 5.0],  // group 0: high pos_weight (sparse)
                [1.0, 1.0, 1.0, 1.0],  // group 1: low pos_weight (denser)
            ]),
            sample_groups: [0n, 0n, 0n, 1n, 1n, 1n],
        });

        const config = $.let({
            architecture: variant('mlp', { hidden_layers: [16n] }),
            output: variant('binary', { pos_weight: variant('none', null) }),
            learning_rate: variant('some', 0.01),
            max_epochs: variant('some', 100n),
            patience: variant('some', 20n),
            batch_size: variant('some', 3n),
            dropout: variant('some', 0.0),
            gradient_clip: variant('some', 1.0),
            weight_decay: variant('none', null),
            random_state: variant('some', 42n),
            epoch_callback: variant('none', null),
        });

        const result = $.let(Lightning.train(X, y, config, variant('none', null), variant('some', group_weights), variant('none', null)));

        $(Assert.greaterEqual(result.best_epoch, 0n));

        const y_pred = $.let(Lightning.predict(result.model, X, variant('none', null)));
        $(Assert.equal(y_pred.size(), 6n));
        $(Assert.equal(y_pred.get(0n).size(), 4n));

        // Predictions should be between 0 and 1
        $(Assert.greaterEqual(y_pred.get(0n).get(0n), East.value(0.0)));
        $(Assert.lessEqual(y_pred.get(0n).get(0n), East.value(1.0)));
    });

    test("error: group_weights with regression output", $ => {
        const X = $.let([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]);
        const y = $.let([[1.0], [2.0], [3.0], [4.0]]);

        const group_weights = $.let({
            weights: variant('multi_head', [[[1.0]]]),
            sample_groups: [0n, 0n, 0n, 0n],
        });

        const config = $.let({
            architecture: variant('mlp', { hidden_layers: [8n] }),
            output: variant('regression', null),
            learning_rate: variant('none', null),
            max_epochs: variant('none', null),
            patience: variant('none', null),
            batch_size: variant('none', null),
            dropout: variant('none', null),
            gradient_clip: variant('none', null),
            weight_decay: variant('none', null),
            random_state: variant('none', null),
            epoch_callback: variant('none', null),
        });

        $(Assert.throws(
            Lightning.train(X, y, config, variant('none', null), variant('some', group_weights), variant('none', null)),
            /group_weights only supported for multi_head and binary output/
        ));
    });

    test("error: weights variant does not match output type", $ => {
        const X = $.let([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]);
        const y = $.let([[1.0], [0.0], [1.0], [0.0]]);

        // Using multi_head variant with binary output
        const group_weights = $.let({
            weights: variant('multi_head', [[[1.0, 1.0]]]),
            sample_groups: [0n, 0n, 0n, 0n],
        });

        const config = $.let({
            architecture: variant('mlp', { hidden_layers: [8n] }),
            output: variant('binary', { pos_weight: variant('none', null) }),
            learning_rate: variant('none', null),
            max_epochs: variant('none', null),
            patience: variant('none', null),
            batch_size: variant('none', null),
            dropout: variant('none', null),
            gradient_clip: variant('none', null),
            weight_decay: variant('none', null),
            random_state: variant('none', null),
            epoch_callback: variant('none', null),
        });

        $(Assert.throws(
            Lightning.train(X, y, config, variant('none', null), variant('some', group_weights), variant('none', null)),
            /group_weights variant 'multi_head' does not match output type 'binary'/
        ));
    });

    test("error: sample_groups index out of bounds", $ => {
        const X = $.let([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]);
        const y = $.let([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ]);

        // Only 1 group but sample_groups references index 1
        const group_weights = $.let({
            weights: variant('multi_head', [[[1.0, 1.0, 1.0]]]),  // 1 group
            sample_groups: [0n, 0n, 1n, 1n],  // ERROR: index 1 out of bounds
        });

        const config = $.let({
            architecture: variant('mlp', { hidden_layers: [8n] }),
            output: variant('multi_head', {
                n_heads: 1n,
                n_classes_per_head: 3n,
                class_weights: variant('none', null),
            }),
            learning_rate: variant('none', null),
            max_epochs: variant('none', null),
            patience: variant('none', null),
            batch_size: variant('none', null),
            dropout: variant('none', null),
            gradient_clip: variant('none', null),
            weight_decay: variant('none', null),
            random_state: variant('none', null),
            epoch_callback: variant('none', null),
        });

        $(Assert.throws(
            Lightning.train(X, y, config, variant('none', null), variant('some', group_weights), variant('none', null)),
            /sample_groups contains index 1 but only 1 groups provided/
        ));
    });

    test("error: sample_groups length mismatch", $ => {
        const X = $.let([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]);  // 4 samples
        const y = $.let([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ]);

        const group_weights = $.let({
            weights: variant('multi_head', [[[1.0, 1.0, 1.0]]]),
            sample_groups: [0n, 0n],  // ERROR: only 2 indices for 4 samples
        });

        const config = $.let({
            architecture: variant('mlp', { hidden_layers: [8n] }),
            output: variant('multi_head', {
                n_heads: 1n,
                n_classes_per_head: 3n,
                class_weights: variant('none', null),
            }),
            learning_rate: variant('none', null),
            max_epochs: variant('none', null),
            patience: variant('none', null),
            batch_size: variant('none', null),
            dropout: variant('none', null),
            gradient_clip: variant('none', null),
            weight_decay: variant('none', null),
            random_state: variant('none', null),
            epoch_callback: variant('none', null),
        });

        $(Assert.throws(
            Lightning.train(X, y, config, variant('none', null), variant('some', group_weights), variant('none', null)),
            /sample_groups length 2 does not match X rows 4/
        ));
    });

    // =========================================================================
    // Error handling tests
    // =========================================================================

    test("error: X and y shape mismatch", $ => {
        const X = $.let([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);  // 3 samples
        const y = $.let([[1.0], [2.0]]);  // 2 samples

        const config = $.let({
            architecture: variant('mlp', {
                hidden_layers: [8n],
            }),
            output: variant('regression', null),
            learning_rate: variant('none', null),
            max_epochs: variant('none', null),
            patience: variant('none', null),
            batch_size: variant('none', null),
            dropout: variant('none', null),
            gradient_clip: variant('none', null),
            weight_decay: variant('none', null),
            random_state: variant('none', null),
            epoch_callback: variant('none', null),
        });

        $(Assert.throws(
            Lightning.train(X, y, config, variant('none', null), variant('none', null), variant('none', null)),
            /Size mismatch between tensors/
        ));
    });

    test("error: encode on non-autoencoder model", $ => {
        const X = $.let([
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
        ]);
        const y = $.let([
            [3.0],
            [5.0],
            [7.0],
            [9.0],
        ]);

        const config = $.let({
            architecture: variant('mlp', {
                hidden_layers: [8n],
            }),
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

        const result = $.let(Lightning.train(X, y, config, variant('none', null), variant('none', null), variant('none', null)));

        $(Assert.throws(
            Lightning.encode(result.model, X),
            /encode\(\) not available for mlp architecture/
        ));
    });

    test("error: decode on non-autoencoder model", $ => {
        const X = $.let([
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
        ]);
        const y = $.let([
            [3.0],
            [5.0],
            [7.0],
            [9.0],
        ]);

        const config = $.let({
            architecture: variant('mlp', {
                hidden_layers: [8n],
            }),
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

        const result = $.let(Lightning.train(X, y, config, variant('none', null), variant('none', null), variant('none', null)));
        const z = $.let([[0.5, 0.5], [0.3, 0.7]]);

        $(Assert.throws(
            Lightning.decode(result.model, z),
            /decode\(\) not available for mlp architecture/
        ));
    });

    // =========================================================================
    // Temporal Architecture Tests
    // =========================================================================

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
                condition_dim: variant('none', null),
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

        const result = $.let(Lightning.train(X, X, config, variant('none', null), variant('none', null), variant('none', null)));
        $(Assert.greaterEqual(result.best_epoch, 0n));

        // Encode to latent
        const z = $.let(Lightning.encode(result.model, X));
        $(Assert.equal(z.size(), 4n));
        $(Assert.equal(z.get(0n).size(), 4n));

        // Decode should produce valid output
        const X_decoded = $.let(Lightning.decode(result.model, z));
        $(Assert.equal(X_decoded.size(), 4n));
        $(Assert.equal(X_decoded.get(0n).size(), 24n));
    });

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
                condition_dim: variant('none', null),
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

        const result = $.let(Lightning.train(X, X, config, variant('none', null), variant('none', null), variant('none', null)));
        $(Assert.greaterEqual(result.best_epoch, 0n));

        const z = $.let(Lightning.encode(result.model, X));
        $(Assert.equal(z.size(), 4n));
        $(Assert.equal(z.get(0n).size(), 4n));

        const X_decoded = $.let(Lightning.decode(result.model, z));
        $(Assert.equal(X_decoded.size(), 4n));
        $(Assert.equal(X_decoded.get(0n).size(), 24n));
    });

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
                condition_dim: variant('none', null),
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

        const result = $.let(Lightning.train(X, X, config, variant('none', null), variant('none', null), variant('none', null)));
        $(Assert.greaterEqual(result.best_epoch, 0n));

        const z = $.let(Lightning.encode(result.model, X));
        $(Assert.equal(z.size(), 2n));
        $(Assert.equal(z.get(0n).size(), 4n));
    });

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
                condition_dim: variant('none', null),
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

        const result = $.let(Lightning.train(X, X, config, variant('none', null), variant('none', null), variant('none', null)));
        $(Assert.greaterEqual(result.best_epoch, 0n));

        const z = $.let(Lightning.encode(result.model, X));
        $(Assert.equal(z.size(), 4n));
        $(Assert.equal(z.get(0n).size(), 4n));

        const X_decoded = $.let(Lightning.decode(result.model, z));
        $(Assert.equal(X_decoded.size(), 4n));
        $(Assert.equal(X_decoded.get(0n).size(), 24n));
    });

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
    });

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
                condition_dim: variant('none', null),
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
            Lightning.train(X, y, config, variant('none', null), variant('none', null), variant('none', null)),
            /Temporal architecture 'conv1d' requires multi_head output/
        ));
    });

    test("error: n_heads must equal n_channels * sequence_length", $ => {
        const X = $.let([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0]]);

        const config = $.let({
            architecture: variant('conv1d', {
                n_channels: 2n,
                sequence_length: 3n,  // 2 * 3 = 6 expected heads
                conv_channels: [8n],
                kernel_size: 3n,
                latent_dim: 4n,
                condition_dim: variant('none', null),
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
            Lightning.train(X, X, config, variant('none', null), variant('none', null), variant('none', null)),
            /n_heads \(4\) must equal n_channels \* sequence_length/
        ));
    });

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
                condition_dim: variant('none', null),
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

        const result = $.let(Lightning.train(X, X, config, variant('some', masks), variant('some', group_weights), variant('none', null)));
        $(Assert.greaterEqual(result.best_epoch, 0n));

        // Predict with masks
        const y_pred = $.let(Lightning.predict(result.model, X, variant('some', masks)));
        $(Assert.equal(y_pred.size(), 4n));

        // Encode/decode should work
        const z = $.let(Lightning.encode(result.model, X));
        $(Assert.equal(z.size(), 4n));
        $(Assert.equal(z.get(0n).size(), 4n));
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
                n_attention_heads: 2n,
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

        // Train without conditions
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

    test("error: condition_dim set but no conditions provided", $ => {
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
                condition_dim: variant('some', 3n),  // condition_dim is set
            }),
            output: variant('multi_head', {
                n_heads: 6n,
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

        // ERROR: condition_dim is set but no conditions provided (6th param is none)
        $(Assert.throws(
            Lightning.train(X, X, config, variant('none', null), variant('none', null), variant('none', null)),
            /architecture has condition_dim set but no conditions provided/
        ));
    });
}, { exportOnly: true });
