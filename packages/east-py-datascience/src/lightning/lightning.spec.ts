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
        const result = $.let(Lightning.train(X, y, config, variant('none', null)));

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
        const result = $.let(Lightning.train(X, y, config, variant('none', null)));

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
        const result = $.let(Lightning.train(X, y, config, variant('none', null)));

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
        const result = $.let(Lightning.train(X, y, config, variant('none', null)));

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
        const result = $.let(Lightning.train(X, X, config, variant('none', null)));

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
        const result1 = $.let(Lightning.train(X, y, config, variant('none', null)));
        const result2 = $.let(Lightning.train(X, y, config, variant('none', null)));

        // Predictions should be identical
        const pred1 = $.let(Lightning.predict(result1.model, X, variant('none', null)));
        const pred2 = $.let(Lightning.predict(result2.model, X, variant('none', null)));

        $(Assert.equal(pred1.get(0n).get(0n), pred2.get(0n).get(0n)));
        $(Assert.equal(pred1.get(1n).get(0n), pred2.get(1n).get(0n)));
    });

    test("binary with pos_weight works", $ => {
        // Imbalanced binary data (more 0s than 1s)
        const X = $.let([
            [0.0, 0.0],
            [0.5, 0.5],
            [1.0, 1.0],
            [1.5, 1.5],
            [2.0, 2.0],
            [2.5, 2.5],
            [10.0, 10.0],
            [10.5, 10.5],
        ]);
        const y = $.let([
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [1.0],
            [1.0],
        ]);

        const config = $.let({
            architecture: variant('mlp', {
                hidden_layers: [16n],
            }),
            output: variant('binary', {
                pos_weight: variant('some', 3.0),  // Upweight positive class
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
        const result = $.let(Lightning.train(X, y, config, variant('none', null)));

        // Should train successfully
        $(Assert.greaterEqual(result.best_epoch, 0n));
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
        const result = $.let(Lightning.train(X, y, config, variant('none', null)));

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

        const result = $.let(Lightning.train(X, y, config, variant('none', null)));

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
        const result = $.let(Lightning.train(X, y, config, variant('some', masks)));

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
        const result = $.let(Lightning.train(X, y, config, variant('none', null)));

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
        const result = $.let(Lightning.train(X, X, config, variant('none', null)));

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

    test("autoencoder + binary + pos_weight + masks", $ => {
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
        // (n_samples, n_outputs) - 2D for binary
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
                pos_weight: variant('some', 3.0),  // upweight positive class
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
        const result = $.let(Lightning.train(X, X, config, variant('some', masks)));

        // Should train successfully
        $(Assert.greaterEqual(result.best_epoch, 0n));

        // Encode to latent space
        const embeddings = $.let(Lightning.encode(result.model, X));
        $(Assert.equal(embeddings.size(), 6n));
        $(Assert.equal(embeddings.get(0n).size(), 2n));

        // Predict with masks
        const y_pred = $.let(Lightning.predict(result.model, X, variant('some', masks)));
        $(Assert.equal(y_pred.size(), 6n));
        $(Assert.equal(y_pred.get(0n).size(), 4n));

        // Predictions should be between 0 and 1 (sigmoid output)
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
        const result = $.let(Lightning.train(X, X, config, variant('some', masks)));

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
            Lightning.train(X, y, config, variant('none', null)),
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

        const result = $.let(Lightning.train(X, y, config, variant('none', null)));

        $(Assert.throws(
            Lightning.encode(result.model, X),
            /encode\(\) only available for autoencoder architecture/
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

        const result = $.let(Lightning.train(X, y, config, variant('none', null)));
        const z = $.let([[0.5, 0.5], [0.3, 0.7]]);

        $(Assert.throws(
            Lightning.decode(result.model, z),
            /decode\(\) only available for autoencoder architecture/
        ));
    });
}, { exportOnly: true });
