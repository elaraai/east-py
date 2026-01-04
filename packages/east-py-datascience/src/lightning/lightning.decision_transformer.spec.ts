/**
 * Copyright (c) 2025 Elara AI Pty Ltd
 * Dual-licensed under AGPL-3.0 and commercial license. See LICENSE for details.
 */

/**
 * Lightning Decision Transformer architecture tests
 */
import { variant, East, FloatType, ArrayType } from "@elaraai/east";
import { describeEast, Assert } from "@elaraai/east-node-std";
import { HeadConfigType, Lightning, LightningConfigType, TrajectoryGenerateConfigType } from "./lightning.js";

describeEast("Lightning Decision Transformer", (test) => {
    // =========================================================================
    // Basic Training Tests
    // =========================================================================

    test("decision_transformer: basic multi_head_mixed training", $ => {
        // 4 samples, seq_len=3, state_dim=2, action_dim=5 (2 binary + 3-class multiclass)
        const returns = $.let([1.0, 0.8, 0.5, 0.2]);

        const states = $.let([
            // Sample 0: 3 timesteps, 2 state features
            [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]],
            // Sample 1
            [[0.8, 0.2], [0.4, 0.6], [0.2, 0.8]],
            // Sample 2
            [[0.6, 0.4], [0.3, 0.7], [0.1, 0.9]],
            // Sample 3
            [[0.4, 0.6], [0.2, 0.8], [0.0, 1.0]],
        ]);

        // Action: 2 binary + 1 multiclass(3) = 5 logits
        // [bin0, bin1, class0, class1, class2]
        const actions = $.let([
            // Sample 0
            [[1.0, 0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0, 1.0]],
            // Sample 1
            [[0.0, 1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0]],
            // Sample 2
            [[1.0, 1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 1.0, 0.0]],
            // Sample 3
            [[0.0, 0.0, 0.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0, 0.0]],
        ]);

        // All timesteps valid
        const masks = $.let([
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ]);

        const config = $.let({
            architecture: variant('decision_transformer', {
                sequence_length: 3n,
                state_dim: 2n,
                action_dim: 5n,
                d_model: 16n,
                n_attention_heads: 2n,
                n_layers: 1n,
                d_ff: variant('none', null),
                dropout: variant('some', 0.0),
                return_embedding: variant('global', null),
            }),
            output: variant('multi_head_mixed', {
                heads: [
                    { head_type: variant('binary', null), class_weights: variant('none', null), conditional_on: variant('none', null) },
                    { head_type: variant('binary', null), class_weights: variant('none', null), conditional_on: variant('none', null) },
                    { head_type: variant('multiclass', { n_classes: 3n }), class_weights: variant('none', null), conditional_on: variant('none', null) },
                ],
            }),
            learning_rate: variant('some', 0.01),
            max_epochs: variant('some', 50n),
            patience: variant('some', 20n),
            batch_size: variant('some', 2n),
            dropout: variant('some', 0.0),
            gradient_clip: variant('some', 1.0),
            weight_decay: variant('none', null),
            random_state: variant('some', 42n),
            epoch_callback: variant('none', null),
        }, LightningConfigType);

        const result = $.let(Lightning.trainTrajectory(returns, states, actions, masks, config));
        $(Assert.greaterEqual(result.best_epoch, 0n));
        $(Assert.greaterEqual(result.train_loss, 0.0));
    });

    test("decision_transformer: generation with deterministic sampling", $ => {
        // Same setup as basic training
        const returns = $.let([1.0, 0.8, 0.5, 0.2]);

        const states = $.let([
            [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]],
            [[0.8, 0.2], [0.4, 0.6], [0.2, 0.8]],
            [[0.6, 0.4], [0.3, 0.7], [0.1, 0.9]],
            [[0.4, 0.6], [0.2, 0.8], [0.0, 1.0]],
        ]);

        const actions = $.let([
            [[1.0, 0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0, 1.0]],
            [[0.0, 1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0]],
            [[1.0, 1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 1.0, 0.0]],
            [[0.0, 0.0, 0.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0, 0.0]],
        ]);

        const masks = $.let([
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ]);

        const config = $.let({
            architecture: variant('decision_transformer', {
                sequence_length: 3n,
                state_dim: 2n,
                action_dim: 5n,
                d_model: 16n,
                n_attention_heads: 2n,
                n_layers: 1n,
                d_ff: variant('none', null),
                dropout: variant('some', 0.0),
                return_embedding: variant('global', null),
            }),
            output: variant('multi_head_mixed', {
                heads: [
                    { head_type: variant('binary', null), class_weights: variant('none', null), conditional_on: variant('none', null) },
                    { head_type: variant('binary', null), class_weights: variant('none', null), conditional_on: variant('none', null) },
                    { head_type: variant('multiclass', { n_classes: 3n }), class_weights: variant('none', null), conditional_on: variant('none', null) },
                ],
            }),
            learning_rate: variant('some', 0.01),
            max_epochs: variant('some', 50n),
            patience: variant('some', 20n),
            batch_size: variant('some', 2n),
            dropout: variant('some', 0.0),
            gradient_clip: variant('some', 1.0),
            weight_decay: variant('none', null),
            random_state: variant('some', 42n),
            epoch_callback: variant('none', null),
        }, LightningConfigType);

        const result = $.let(Lightning.trainTrajectory(returns, states, actions, masks, config));

        // Generate with deterministic sampling (temperature=0)
        const target_returns = $.let([1.0, 0.5]);
        const gen_states = $.let([
            [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]],
            [[0.6, 0.4], [0.3, 0.7], [0.1, 0.9]],
        ]);

        const gen_config = $.let({
            temperature: 0.0,
            return_probs: false,
            action_constraints: variant('none', null),
            temporal_mask: variant('none', null),
            head_configs: variant('some', [
                { head_type: variant('binary', null), class_weights: variant('none', null), conditional_on: variant('none', null) },
                { head_type: variant('binary', null), class_weights: variant('none', null), conditional_on: variant('none', null) },
                { head_type: variant('multiclass', { n_classes: 3n }), class_weights: variant('none', null), conditional_on: variant('none', null) },
            ]),
            action_prefix: variant('none', null),
            start_timestep: variant('none', null),
        }, TrajectoryGenerateConfigType);

        const generated = $.let(Lightning.generateTrajectory(result.model, gen_states, target_returns, gen_config));

        // Should return 2 samples x 3 timesteps x 5 action dims
        $(Assert.equal(generated.size(), 2n));
        $(Assert.equal(generated.get(0n).size(), 3n));
        $(Assert.equal(generated.get(0n).get(0n).size(), 5n));
    });

    // =========================================================================
    // Temporal Masking Tests
    // =========================================================================

    test("decision_transformer: temporal masking for variable length sequences", $ => {
        // 2 samples with different valid lengths
        const returns = $.let([1.0, 0.5]);

        const states = $.let([
            // Sample 0: 3 timesteps valid
            [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]],
            // Sample 1: only 2 timesteps valid (padded with zeros)
            [[0.8, 0.2], [0.4, 0.6], [0.0, 0.0]],
        ]);

        const actions = $.let([
            [[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 1.0]],
            [[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0]],  // Last timestep masked
        ]);

        // Mask: 1 = valid, 0 = invalid
        const masks = $.let([
            [1.0, 1.0, 1.0],  // All valid
            [1.0, 1.0, 0.0],  // Last timestep invalid
        ]);

        const config = $.let({
            architecture: variant('decision_transformer', {
                sequence_length: 3n,
                state_dim: 2n,
                action_dim: 4n,
                d_model: 16n,
                n_attention_heads: 2n,
                n_layers: 1n,
                d_ff: variant('none', null),
                dropout: variant('some', 0.0),
                return_embedding: variant('global', null),
            }),
            output: variant('multi_head_mixed', {
                heads: [
                    { head_type: variant('binary', null), class_weights: variant('none', null), conditional_on: variant('none', null) },
                    { head_type: variant('binary', null), class_weights: variant('none', null), conditional_on: variant('none', null) },
                    { head_type: variant('multiclass', { n_classes: 2n }), class_weights: variant('none', null), conditional_on: variant('none', null) },
                ],
            }),
            learning_rate: variant('some', 0.01),
            max_epochs: variant('some', 30n),
            patience: variant('some', 15n),
            batch_size: variant('some', 2n),
            dropout: variant('some', 0.0),
            gradient_clip: variant('some', 1.0),
            weight_decay: variant('none', null),
            random_state: variant('some', 42n),
            epoch_callback: variant('none', null),
        }, LightningConfigType);

        const result = $.let(Lightning.trainTrajectory(returns, states, actions, masks, config));
        $(Assert.greaterEqual(result.best_epoch, 0n));
    });

    // =========================================================================
    // Conditional Head Tests
    // =========================================================================

    test("decision_transformer: conditional heads", $ => {
        // 3 heads: occurrence (binary), occurrence_type (multiclass, conditional on occurrence)
        // action_dim = 1 (occurrence) + 3 (occurrence_type) = 4
        const returns = $.let([1.0, 0.8, 0.5, 0.2]);

        const states = $.let([
            [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]],
            [[0.8, 0.2], [0.4, 0.6], [0.2, 0.8]],
            [[0.6, 0.4], [0.3, 0.7], [0.1, 0.9]],
            [[0.4, 0.6], [0.2, 0.8], [0.0, 1.0]],
        ]);

        // Actions: [occurrence, type_0, type_1, type_2]
        // When occurrence=0, type doesn't matter (but we set one-hot anyway for valid training)
        const actions = $.let([
            [[1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0]],  // Last has no occurrence
            [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 1.0]],
            [[1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0]],
        ]);

        const masks = $.let([
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ]);

        const config = $.let({
            architecture: variant('decision_transformer', {
                sequence_length: 3n,
                state_dim: 2n,
                action_dim: 4n,
                d_model: 16n,
                n_attention_heads: 2n,
                n_layers: 1n,
                d_ff: variant('none', null),
                dropout: variant('some', 0.0),
                return_embedding: variant('global', null),
            }),
            output: variant('multi_head_mixed', {
                heads: [
                    // Head 0: occurrence (binary)
                    { head_type: variant('binary', null), class_weights: variant('none', null), conditional_on: variant('none', null) },
                    // Head 1: occurrence_type (multiclass) - conditional on head 0
                    { head_type: variant('multiclass', { n_classes: 3n }), class_weights: variant('none', null), conditional_on: variant('some', 0n) },
                ],
            }),
            learning_rate: variant('some', 0.01),
            max_epochs: variant('some', 50n),
            patience: variant('some', 20n),
            batch_size: variant('some', 2n),
            dropout: variant('some', 0.0),
            gradient_clip: variant('some', 1.0),
            weight_decay: variant('none', null),
            random_state: variant('some', 42n),
            epoch_callback: variant('none', null),
        }, LightningConfigType);

        const result = $.let(Lightning.trainTrajectory(returns, states, actions, masks, config));
        $(Assert.greaterEqual(result.best_epoch, 0n));
    });

    // =========================================================================
    // Binary Output Tests
    // =========================================================================

    test("decision_transformer: binary output mode", $ => {
        // All outputs treated as independent binary (simpler use case)
        const returns = $.let([1.0, 0.8, 0.5, 0.2]);

        const states = $.let([
            [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]],
            [[0.8, 0.2], [0.4, 0.6], [0.2, 0.8]],
            [[0.6, 0.4], [0.3, 0.7], [0.1, 0.9]],
            [[0.4, 0.6], [0.2, 0.8], [0.0, 1.0]],
        ]);

        // 3 independent binary outputs
        const actions = $.let([
            [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0]],
            [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
            [[1.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]],
        ]);

        const masks = $.let([
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ]);

        const config = $.let({
            architecture: variant('decision_transformer', {
                sequence_length: 3n,
                state_dim: 2n,
                action_dim: 3n,
                d_model: 16n,
                n_attention_heads: 2n,
                n_layers: 1n,
                d_ff: variant('none', null),
                dropout: variant('some', 0.0),
                return_embedding: variant('global', null),
            }),
            output: variant('binary', {
                pos_weight: variant('none', null),
            }),
            learning_rate: variant('some', 0.01),
            max_epochs: variant('some', 50n),
            patience: variant('some', 20n),
            batch_size: variant('some', 2n),
            dropout: variant('some', 0.0),
            gradient_clip: variant('some', 1.0),
            weight_decay: variant('none', null),
            random_state: variant('some', 42n),
            epoch_callback: variant('none', null),
        }, LightningConfigType);

        const result = $.let(Lightning.trainTrajectory(returns, states, actions, masks, config));
        $(Assert.greaterEqual(result.best_epoch, 0n));
    });

    // =========================================================================
    // Generation with Stochastic Sampling Tests
    // =========================================================================

    test("decision_transformer: stochastic generation with temperature", $ => {
        const returns = $.let([1.0, 0.8, 0.5, 0.2]);

        const states = $.let([
            [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]],
            [[0.8, 0.2], [0.4, 0.6], [0.2, 0.8]],
            [[0.6, 0.4], [0.3, 0.7], [0.1, 0.9]],
            [[0.4, 0.6], [0.2, 0.8], [0.0, 1.0]],
        ]);

        const actions = $.let([
            [[1.0, 0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0, 1.0]],
            [[0.0, 1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0]],
            [[1.0, 1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 1.0, 0.0]],
            [[0.0, 0.0, 0.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0, 0.0]],
        ]);

        const masks = $.let([
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ]);

        const config = $.let({
            architecture: variant('decision_transformer', {
                sequence_length: 3n,
                state_dim: 2n,
                action_dim: 5n,
                d_model: 16n,
                n_attention_heads: 2n,
                n_layers: 1n,
                d_ff: variant('none', null),
                dropout: variant('some', 0.0),
                return_embedding: variant('global', null),
            }),
            output: variant('multi_head_mixed', {
                heads: [
                    { head_type: variant('binary', null), class_weights: variant('none', null), conditional_on: variant('none', null) },
                    { head_type: variant('binary', null), class_weights: variant('none', null), conditional_on: variant('none', null) },
                    { head_type: variant('multiclass', { n_classes: 3n }), class_weights: variant('none', null), conditional_on: variant('none', null) },
                ],
            }),
            learning_rate: variant('some', 0.01),
            max_epochs: variant('some', 50n),
            patience: variant('some', 20n),
            batch_size: variant('some', 2n),
            dropout: variant('some', 0.0),
            gradient_clip: variant('some', 1.0),
            weight_decay: variant('none', null),
            random_state: variant('some', 42n),
            epoch_callback: variant('none', null),
        }, LightningConfigType);

        const result = $.let(Lightning.trainTrajectory(returns, states, actions, masks, config));

        // Generate with stochastic sampling (temperature=1.0)
        const target_returns = $.let([1.0]);
        const gen_states = $.let([
            [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]],
        ]);

        const gen_config = $.let({
            temperature: 1.0,  // Stochastic sampling
            return_probs: false,
            action_constraints: variant('none', null),
            temporal_mask: variant('none', null),
            head_configs: variant('some', [
                { head_type: variant('binary', null), class_weights: variant('none', null), conditional_on: variant('none', null) },
                { head_type: variant('binary', null), class_weights: variant('none', null), conditional_on: variant('none', null) },
                { head_type: variant('multiclass', { n_classes: 3n }), class_weights: variant('none', null), conditional_on: variant('none', null) },
            ]),
            action_prefix: variant('none', null),
            start_timestep: variant('none', null),
        }, TrajectoryGenerateConfigType);

        const generated = $.let(Lightning.generateTrajectory(result.model, gen_states, target_returns, gen_config));

        // Should return valid actions
        $(Assert.equal(generated.size(), 1n));
        $(Assert.equal(generated.get(0n).size(), 3n));
        $(Assert.equal(generated.get(0n).get(0n).size(), 5n));
    });

    // =========================================================================
    // Performance Tests
    // =========================================================================

    test("decision_transformer: learns return-conditioned policy", $ => {
        // Create a dataset where:
        // - High returns (0.8-1.0) -> action pattern [1, 0, 1, 0, 0] (one-hot class 0)
        // - Low returns (0.0-0.2)  -> action pattern [0, 1, 0, 0, 1] (one-hot class 2)
        // CRITICAL: Use IDENTICAL states for both classes so return is the ONLY differentiator
        // This tests if the model learns to predict different actions based on target return

        // Generate 50 high-return samples + 50 low-return samples = 100 total
        const nSamplesPerClass = $.let(50n);

        // Shared state pattern - both high and low return samples use the SAME states
        // This forces the model to learn return→action mapping, not state→action mapping
        const sharedStates = $.let(East.Array.generate(nSamplesPerClass, ArrayType(ArrayType(FloatType)), ($, i) => {
            const noise = i.toFloat().multiply(0.002);
            return [
                [noise.add(0.5), noise.negate().add(0.5)],
                [noise.add(0.5), noise.negate().add(0.5)],
                [noise.add(0.5), noise.negate().add(0.5)],
            ];
        }));

        // High-return data: returns in [0.85, 0.95], action = [1,0,1,0,0]
        const highReturns = $.let(East.Array.generate(nSamplesPerClass, FloatType, ($, i) => i.toFloat().multiply(0.002).add(0.85)));
        const highActions = $.let(East.Array.generate(nSamplesPerClass, ArrayType(ArrayType(FloatType)), _ => [
            [1.0, 0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0, 0.0],
        ]));

        // Low-return data: returns in [0.05, 0.15], action = [0,1,0,0,1]
        const lowReturns = $.let(East.Array.generate(nSamplesPerClass, FloatType, ($, i) => i.toFloat().multiply(0.002).add(0.05)));
        const lowActions = $.let(East.Array.generate(nSamplesPerClass, ArrayType(ArrayType(FloatType)), _ => [
            [0.0, 1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0, 1.0],
        ]));

        // Combine datasets - use sharedStates for BOTH high and low
        const returns = $.let(highReturns.concat(lowReturns));
        const states = $.let(sharedStates.concat(sharedStates));
        const actions = $.let(highActions.concat(lowActions));
        const masks = $.let(East.Array.generate(nSamplesPerClass.multiply(2n), ArrayType(FloatType), _ => [1.0, 1.0, 1.0]));

        const config = $.let({
            architecture: variant('decision_transformer', {
                sequence_length: 3n,
                state_dim: 2n,
                action_dim: 5n,
                d_model: 32n,
                n_attention_heads: 4n,
                n_layers: 2n,
                d_ff: variant('some', 64n),
                dropout: variant('some', 0.0),
                return_embedding: variant('global', null),
            }),
            output: variant('multi_head_mixed', {
                heads: [
                    { head_type: variant('binary', null), class_weights: variant('none', null), conditional_on: variant('none', null) },
                    { head_type: variant('binary', null), class_weights: variant('none', null), conditional_on: variant('none', null) },
                    { head_type: variant('multiclass', { n_classes: 3n }), class_weights: variant('none', null), conditional_on: variant('none', null) },
                ],
            }),
            learning_rate: variant('some', 0.001),
            max_epochs: variant('some', 200n),
            patience: variant('some', 50n),
            batch_size: variant('some', 8n),
            dropout: variant('some', 0.0),
            gradient_clip: variant('some', 1.0),
            weight_decay: variant('some', 0.0001),
            random_state: variant('some', 42n),
            epoch_callback: variant('none', null),
        }, LightningConfigType);

        const result = $.let(Lightning.trainTrajectory(returns, states, actions, masks, config));

        // Model should converge (loss should decrease significantly)
        $(Assert.less(result.val_loss, 1.0));

        // Now test generation: give high target return, expect high-return actions
        // Use the same state pattern as training (centered at 0.5)
        const testStates = $.let([
            [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]],
        ]);
        const highTargetReturn = $.let([1.0]);

        const genConfig = $.let({
            temperature: 0.0,  // Deterministic
            return_probs: false,
            action_constraints: variant('none', null),
            temporal_mask: variant('none', null),
            head_configs: variant('some', [
                { head_type: variant('binary', null), class_weights: variant('none', null), conditional_on: variant('none', null) },
                { head_type: variant('binary', null), class_weights: variant('none', null), conditional_on: variant('none', null) },
                { head_type: variant('multiclass', { n_classes: 3n }), class_weights: variant('none', null), conditional_on: variant('none', null) },
            ]),
            action_prefix: variant('none', null),
            start_timestep: variant('none', null),
        }, TrajectoryGenerateConfigType);

        const generatedHigh = $.let(Lightning.generateTrajectory(result.model, testStates, highTargetReturn, genConfig));

        // With high target return, first binary should be 1 (high-return pattern)
        // Check first timestep, first action dimension (binary1)
        $(Assert.equal(generatedHigh.get(0n).get(0n).get(0n), 1.0));

        // And multiclass should be class 0 (indices 2,3,4 = [1,0,0])
        $(Assert.equal(generatedHigh.get(0n).get(0n).get(2n), 1.0));

        // Now test with low target return
        const lowTargetReturn = $.let([0.1]);
        const generatedLow = $.let(Lightning.generateTrajectory(result.model, testStates, lowTargetReturn, genConfig));

        // With low target return, first binary should be 0 (low-return pattern)
        $(Assert.equal(generatedLow.get(0n).get(0n).get(0n), 0.0));

        // And multiclass should be class 2 (indices 2,3,4 = [0,0,1])
        $(Assert.equal(generatedLow.get(0n).get(0n).get(4n), 1.0));
    });

    test("decision_transformer: learns autoregressive action history", $ => {
        // Test that model uses previous actions when predicting next action
        // Setup: Constant states, constant return. Actions follow alternating pattern.
        // This pattern can ONLY be learned if model attends to a_{t-1} when predicting a_t.

        const nSamples = $.let(100n);

        // All samples have same return and same states
        const returns = $.let(East.Array.generate(nSamples, FloatType, _ => 0.5));
        const states = $.let(East.Array.generate(nSamples, ArrayType(ArrayType(FloatType)), _ => [
            [0.5, 0.5],
            [0.5, 0.5],
            [0.5, 0.5],
            [0.5, 0.5],
        ]));

        // Alternating pattern: [1,0], [0,1], [1,0], [0,1]
        // Model must learn to look at previous action to predict next
        const actions = $.let(East.Array.generate(nSamples, ArrayType(ArrayType(FloatType)), _ => [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]));

        const masks = $.let(East.Array.generate(nSamples, ArrayType(FloatType), _ => [1.0, 1.0, 1.0, 1.0]));

        const config = $.let({
            architecture: variant('decision_transformer', {
                sequence_length: 4n,
                state_dim: 2n,
                action_dim: 2n,
                d_model: 32n,
                n_attention_heads: 4n,
                n_layers: 2n,
                d_ff: variant('some', 64n),
                dropout: variant('some', 0.0),
                return_embedding: variant('global', null),
            }),
            output: variant('binary', { pos_weight: variant('none', null) }),
            learning_rate: variant('some', 0.001),
            max_epochs: variant('some', 300n),
            patience: variant('some', 100n),
            batch_size: variant('some', 16n),
            dropout: variant('some', 0.0),
            gradient_clip: variant('some', 1.0),
            weight_decay: variant('some', 0.0001),
            random_state: variant('some', 42n),
            epoch_callback: variant('none', null),
        }, LightningConfigType);

        const result = $.let(Lightning.trainTrajectory(returns, states, actions, masks, config));
        $(Assert.less(result.val_loss, 0.5));

        // Generate and check alternating pattern
        const testStates = $.let([[[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]]);
        const testReturn = $.let([0.5]);

        const genConfig = $.let({
            temperature: 0.0,
            return_probs: false,
            action_constraints: variant('none', null),
            temporal_mask: variant('none', null),
            head_configs: variant('none', null),
            action_prefix: variant('none', null),
            start_timestep: variant('none', null),
        }, TrajectoryGenerateConfigType);

        const generated = $.let(Lightning.generateTrajectory(result.model, testStates, testReturn, genConfig));

        // Check alternating pattern: t=0 [1,0], t=1 [0,1], t=2 [1,0], t=3 [0,1]
        $(Assert.equal(generated.get(0n).get(0n).get(0n), 1.0));  // t=0, first dim
        $(Assert.equal(generated.get(0n).get(1n).get(1n), 1.0));  // t=1, second dim
        $(Assert.equal(generated.get(0n).get(2n).get(0n), 1.0));  // t=2, first dim
        $(Assert.equal(generated.get(0n).get(3n).get(1n), 1.0));  // t=3, second dim
    });

    test("decision_transformer: learns state-dependent actions", $ => {
        // Test that model uses state information (not just return)
        // Setup: Constant return. If state[0] > 0.5 then action [1,0], else [0,1]

        const nSamples = $.let(100n);

        // All samples have same return
        const returns = $.let(East.Array.generate(nSamples, FloatType, _ => 0.5));

        // Half samples have high state[0], half have low state[0]
        const nHalf = $.let(50n);
        const highStates = $.let(East.Array.generate(nHalf, ArrayType(ArrayType(FloatType)), ($, i) => {
            const v = i.toFloat().multiply(0.01).add(0.6);
            return [[v, v.negate().add(1.0)], [v, v.negate().add(1.0)], [v, v.negate().add(1.0)]];
        }));
        const lowStates = $.let(East.Array.generate(nHalf, ArrayType(ArrayType(FloatType)), ($, i) => {
            const v = i.toFloat().multiply(0.01).add(0.1);
            return [[v, v.negate().add(1.0)], [v, v.negate().add(1.0)], [v, v.negate().add(1.0)]];
        }));

        const highActions = $.let(East.Array.generate(nHalf, ArrayType(ArrayType(FloatType)), _ => [
            [1.0, 0.0], [1.0, 0.0], [1.0, 0.0],
        ]));
        const lowActions = $.let(East.Array.generate(nHalf, ArrayType(ArrayType(FloatType)), _ => [
            [0.0, 1.0], [0.0, 1.0], [0.0, 1.0],
        ]));

        const states = $.let(highStates.concat(lowStates));
        const actions = $.let(highActions.concat(lowActions));
        const masks = $.let(East.Array.generate(nSamples, ArrayType(FloatType), _ => [1.0, 1.0, 1.0]));

        const config = $.let({
            architecture: variant('decision_transformer', {
                sequence_length: 3n,
                state_dim: 2n,
                action_dim: 2n,
                d_model: 32n,
                n_attention_heads: 4n,
                n_layers: 2n,
                d_ff: variant('some', 64n),
                dropout: variant('some', 0.0),
                return_embedding: variant('global', null),
            }),
            output: variant('binary', { pos_weight: variant('none', null) }),
            learning_rate: variant('some', 0.001),
            max_epochs: variant('some', 200n),
            patience: variant('some', 50n),
            batch_size: variant('some', 16n),
            dropout: variant('some', 0.0),
            gradient_clip: variant('some', 1.0),
            weight_decay: variant('some', 0.0001),
            random_state: variant('some', 42n),
            epoch_callback: variant('none', null),
        }, LightningConfigType);

        const result = $.let(Lightning.trainTrajectory(returns, states, actions, masks, config));
        $(Assert.less(result.val_loss, 0.5));

        const genConfig = $.let({
            temperature: 0.0,
            return_probs: false,
            action_constraints: variant('none', null),
            temporal_mask: variant('none', null),
            head_configs: variant('none', null),
            action_prefix: variant('none', null),
            start_timestep: variant('none', null),
        }, TrajectoryGenerateConfigType);

        // Test with high state
        const highTestStates = $.let([[[0.8, 0.2], [0.8, 0.2], [0.8, 0.2]]]);
        const genHigh = $.let(Lightning.generateTrajectory(result.model, highTestStates, $.let([0.5]), genConfig));
        $(Assert.equal(genHigh.get(0n).get(0n).get(0n), 1.0));  // Should be [1,0]

        // Test with low state
        const lowTestStates = $.let([[[0.2, 0.8], [0.2, 0.8], [0.2, 0.8]]]);
        const genLow = $.let(Lightning.generateTrajectory(result.model, lowTestStates, $.let([0.5]), genConfig));
        $(Assert.equal(genLow.get(0n).get(0n).get(1n), 1.0));  // Should be [0,1]
    });

    test("decision_transformer: learns temporal position awareness", $ => {
        // Test that model knows which timestep it's at via positional encoding
        // Setup: Constant states, constant return. Actions depend on timestep position.
        // t=0 → [1,0,0], t=1 → [0,1,0], t=2 → [0,0,1]

        const nSamples = $.let(100n);

        const returns = $.let(East.Array.generate(nSamples, FloatType, _ => 0.5));
        const states = $.let(East.Array.generate(nSamples, ArrayType(ArrayType(FloatType)), _ => [
            [0.5, 0.5], [0.5, 0.5], [0.5, 0.5],
        ]));

        // Actions encode the timestep position
        const actions = $.let(East.Array.generate(nSamples, ArrayType(ArrayType(FloatType)), _ => [
            [1.0, 0.0, 0.0],  // t=0
            [0.0, 1.0, 0.0],  // t=1
            [0.0, 0.0, 1.0],  // t=2
        ]));

        const masks = $.let(East.Array.generate(nSamples, ArrayType(FloatType), _ => [1.0, 1.0, 1.0]));

        const config = $.let({
            architecture: variant('decision_transformer', {
                sequence_length: 3n,
                state_dim: 2n,
                action_dim: 3n,
                d_model: 32n,
                n_attention_heads: 4n,
                n_layers: 2n,
                d_ff: variant('some', 64n),
                dropout: variant('some', 0.0),
                return_embedding: variant('global', null),
            }),
            output: variant('multi_head', {
                n_heads: 1n,
                n_classes_per_head: 3n,
                class_weights: variant('none', null),
            }),
            learning_rate: variant('some', 0.001),
            max_epochs: variant('some', 200n),
            patience: variant('some', 50n),
            batch_size: variant('some', 16n),
            dropout: variant('some', 0.0),
            gradient_clip: variant('some', 1.0),
            weight_decay: variant('some', 0.0001),
            random_state: variant('some', 42n),
            epoch_callback: variant('none', null),
        }, LightningConfigType);

        const result = $.let(Lightning.trainTrajectory(returns, states, actions, masks, config));
        $(Assert.less(result.val_loss, 0.5));

        const genConfig = $.let({
            temperature: 0.0,
            return_probs: false,
            action_constraints: variant('none', null),
            temporal_mask: variant('none', null),
            head_configs: variant('some', [
                { head_type: variant('multiclass', { n_classes: 3n }), class_weights: variant('none', null), conditional_on: variant('none', null) },
            ]),
            action_prefix: variant('none', null),
            start_timestep: variant('none', null),
        }, TrajectoryGenerateConfigType);

        const testStates = $.let([[[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]]);
        const generated = $.let(Lightning.generateTrajectory(result.model, testStates, $.let([0.5]), genConfig));

        // Check each timestep predicts the correct class
        $(Assert.equal(generated.get(0n).get(0n).get(0n), 1.0));  // t=0 → class 0
        $(Assert.equal(generated.get(0n).get(1n).get(1n), 1.0));  // t=1 → class 1
        $(Assert.equal(generated.get(0n).get(2n).get(2n), 1.0));  // t=2 → class 2
    });

    test("decision_transformer: respects action constraints", $ => {
        // Test that generation respects constraint masks
        // Train a model, then generate with constraints that disable certain actions

        const nSamples = $.let(100n);

        const returns = $.let(East.Array.generate(nSamples, FloatType, _ => 0.5));
        const states = $.let(East.Array.generate(nSamples, ArrayType(ArrayType(FloatType)), _ => [
            [0.5, 0.5], [0.5, 0.5], [0.5, 0.5],
        ]));

        // Train with all actions enabled at all timesteps
        const actions = $.let(East.Array.generate(nSamples, ArrayType(ArrayType(FloatType)), _ => [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]));

        const masks = $.let(East.Array.generate(nSamples, ArrayType(FloatType), _ => [1.0, 1.0, 1.0]));

        const config = $.let({
            architecture: variant('decision_transformer', {
                sequence_length: 3n,
                state_dim: 2n,
                action_dim: 3n,
                d_model: 32n,
                n_attention_heads: 4n,
                n_layers: 2n,
                d_ff: variant('some', 64n),
                dropout: variant('some', 0.0),
                return_embedding: variant('global', null),
            }),
            output: variant('multi_head', {
                n_heads: 1n,
                n_classes_per_head: 3n,
                class_weights: variant('none', null),
            }),
            learning_rate: variant('some', 0.001),
            max_epochs: variant('some', 100n),
            patience: variant('some', 30n),
            batch_size: variant('some', 16n),
            dropout: variant('some', 0.0),
            gradient_clip: variant('some', 1.0),
            weight_decay: variant('some', 0.0001),
            random_state: variant('some', 42n),
            epoch_callback: variant('none', null),
        }, LightningConfigType);

        const result = $.let(Lightning.trainTrajectory(returns, states, actions, masks, config));

        // Generate with constraints: at t=1, disable class 0 (the one the model prefers)
        // Constraint matrix: (seq_len, action_dim) - 1.0 means allowed, 0.0 means disabled
        const constraints = $.let([
            [1.0, 1.0, 1.0],   // t=0: all allowed
            [0.0, 1.0, 1.0],  // t=1: class 0 disabled
            [1.0, 1.0, 1.0],   // t=2: all allowed
        ]);

        const genConfig = $.let({
            temperature: 0.0,
            return_probs: false,
            action_constraints: variant('some', constraints),
            temporal_mask: variant('none', null),
            head_configs: variant('some', [
                { head_type: variant('multiclass', { n_classes: 3n }), class_weights: variant('none', null), conditional_on: variant('none', null) },
            ]),
            action_prefix: variant('none', null),
            start_timestep: variant('none', null),
        }, TrajectoryGenerateConfigType);

        const testStates = $.let([[[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]]);
        const generated = $.let(Lightning.generateTrajectory(result.model, testStates, $.let([0.5]), genConfig));

        // t=0 should be class 0 (unconstrained, model preference)
        $(Assert.equal(generated.get(0n).get(0n).get(0n), 1.0));

        // t=1 should NOT be class 0 (constrained out)
        $(Assert.equal(generated.get(0n).get(1n).get(0n), 0.0));

        // t=2 should be class 0 (unconstrained)
        $(Assert.equal(generated.get(0n).get(2n).get(0n), 1.0));
    });

    test("decision_transformer: interpolates unseen return values", $ => {
        // Test that model generalizes to unseen return values
        // Train on low [0.1-0.2] and high [0.8-0.9] returns, test with mid return 0.5

        const nPerClass = $.let(50n);

        // Low returns → action [0,1]
        const lowReturns = $.let(East.Array.generate(nPerClass, FloatType, ($, i) => i.toFloat().multiply(0.002).add(0.1)));
        const lowActions = $.let(East.Array.generate(nPerClass, ArrayType(ArrayType(FloatType)), _ => [
            [0.0, 1.0], [0.0, 1.0], [0.0, 1.0],
        ]));

        // High returns → action [1,0]
        const highReturns = $.let(East.Array.generate(nPerClass, FloatType, ($, i) => i.toFloat().multiply(0.002).add(0.8)));
        const highActions = $.let(East.Array.generate(nPerClass, ArrayType(ArrayType(FloatType)), _ => [
            [1.0, 0.0], [1.0, 0.0], [1.0, 0.0],
        ]));

        // Shared states
        const sharedStates = $.let(East.Array.generate(nPerClass, ArrayType(ArrayType(FloatType)), _ => [
            [0.5, 0.5], [0.5, 0.5], [0.5, 0.5],
        ]));

        const returns = $.let(lowReturns.concat(highReturns));
        const states = $.let(sharedStates.concat(sharedStates));
        const actions = $.let(lowActions.concat(highActions));
        const masks = $.let(East.Array.generate(nPerClass.multiply(2n), ArrayType(FloatType), _ => [1.0, 1.0, 1.0]));

        const config = $.let({
            architecture: variant('decision_transformer', {
                sequence_length: 3n,
                state_dim: 2n,
                action_dim: 2n,
                d_model: 32n,
                n_attention_heads: 4n,
                n_layers: 2n,
                d_ff: variant('some', 64n),
                dropout: variant('some', 0.0),
                return_embedding: variant('global', null),
            }),
            output: variant('binary', { pos_weight: variant('none', null) }),
            learning_rate: variant('some', 0.001),
            max_epochs: variant('some', 200n),
            patience: variant('some', 50n),
            batch_size: variant('some', 16n),
            dropout: variant('some', 0.0),
            gradient_clip: variant('some', 1.0),
            weight_decay: variant('some', 0.0001),
            random_state: variant('some', 42n),
            epoch_callback: variant('none', null),
        }, LightningConfigType);

        const result = $.let(Lightning.trainTrajectory(returns, states, actions, masks, config));
        $(Assert.less(result.val_loss, 0.5));

        const genConfig = $.let({
            temperature: 0.0,
            return_probs: false,
            action_constraints: variant('none', null),
            temporal_mask: variant('none', null),
            head_configs: variant('none', null),
            action_prefix: variant('none', null),
            start_timestep: variant('none', null),
        }, TrajectoryGenerateConfigType);

        const testStates = $.let([[[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]]);

        // Verify extremes still work
        const genLow = $.let(Lightning.generateTrajectory(result.model, testStates, $.let([0.15]), genConfig));
        $(Assert.equal(genLow.get(0n).get(0n).get(1n), 1.0));  // Low return → [0,1]

        const genHigh = $.let(Lightning.generateTrajectory(result.model, testStates, $.let([0.85]), genConfig));
        $(Assert.equal(genHigh.get(0n).get(0n).get(0n), 1.0));  // High return → [1,0]

        // Mid return (0.5) - model should pick one consistently (not crash or produce garbage)
        // We just verify it produces a valid binary output
        const genMid = $.let(Lightning.generateTrajectory(result.model, testStates, $.let([0.5]), genConfig));
        const midSum = $.let(genMid.get(0n).get(0n).get(0n).add(genMid.get(0n).get(0n).get(1n)));
        $(Assert.equal(midSum, 1.0));  // Should be exactly one of [1,0] or [0,1]
    });

    test("decision_transformer: learns multi-step consistency", $ => {
        // Test that generated sequences are internally coherent
        // Setup: If action at t=0 is [1,0], then actions at t=1,2 should also be [1,0]
        // If action at t=0 is [0,1], then actions at t=1,2 should also be [0,1]
        // This tests autoregressive coherence across multiple steps

        const nPerPattern = $.let(50n);

        // Pattern A: all [1,0]
        const patternA = $.let(East.Array.generate(nPerPattern, ArrayType(ArrayType(FloatType)), _ => [
            [1.0, 0.0], [1.0, 0.0], [1.0, 0.0],
        ]));

        // Pattern B: all [0,1]
        const patternB = $.let(East.Array.generate(nPerPattern, ArrayType(ArrayType(FloatType)), _ => [
            [0.0, 1.0], [0.0, 1.0], [0.0, 1.0],
        ]));

        // Use return to differentiate patterns
        const returnsA = $.let(East.Array.generate(nPerPattern, FloatType, ($, i) => i.toFloat().multiply(0.002).add(0.8)));
        const returnsB = $.let(East.Array.generate(nPerPattern, FloatType, ($, i) => i.toFloat().multiply(0.002).add(0.1)));

        const sharedStates = $.let(East.Array.generate(nPerPattern, ArrayType(ArrayType(FloatType)), _ => [
            [0.5, 0.5], [0.5, 0.5], [0.5, 0.5],
        ]));

        const returns = $.let(returnsA.concat(returnsB));
        const states = $.let(sharedStates.concat(sharedStates));
        const actions = $.let(patternA.concat(patternB));
        const masks = $.let(East.Array.generate(nPerPattern.multiply(2n), ArrayType(FloatType), _ => [1.0, 1.0, 1.0]));

        const config = $.let({
            architecture: variant('decision_transformer', {
                sequence_length: 3n,
                state_dim: 2n,
                action_dim: 2n,
                d_model: 32n,
                n_attention_heads: 4n,
                n_layers: 2n,
                d_ff: variant('some', 64n),
                dropout: variant('some', 0.0),
                return_embedding: variant('global', null),
            }),
            output: variant('binary', { pos_weight: variant('none', null) }),
            learning_rate: variant('some', 0.001),
            max_epochs: variant('some', 200n),
            patience: variant('some', 50n),
            batch_size: variant('some', 16n),
            dropout: variant('some', 0.0),
            gradient_clip: variant('some', 1.0),
            weight_decay: variant('some', 0.0001),
            random_state: variant('some', 42n),
            epoch_callback: variant('none', null),
        }, LightningConfigType);

        const result = $.let(Lightning.trainTrajectory(returns, states, actions, masks, config));
        $(Assert.less(result.val_loss, 0.5));

        const genConfig = $.let({
            temperature: 0.0,
            return_probs: false,
            action_constraints: variant('none', null),
            temporal_mask: variant('none', null),
            head_configs: variant('none', null),
            action_prefix: variant('none', null),
            start_timestep: variant('none', null),
        }, TrajectoryGenerateConfigType);

        const testStates = $.let([[[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]]);

        // High return should produce consistent [1,0] pattern
        const genHigh = $.let(Lightning.generateTrajectory(result.model, testStates, $.let([0.85]), genConfig));
        $(Assert.equal(genHigh.get(0n).get(0n).get(0n), 1.0));
        $(Assert.equal(genHigh.get(0n).get(1n).get(0n), 1.0));
        $(Assert.equal(genHigh.get(0n).get(2n).get(0n), 1.0));

        // Low return should produce consistent [0,1] pattern
        const genLow = $.let(Lightning.generateTrajectory(result.model, testStates, $.let([0.15]), genConfig));
        $(Assert.equal(genLow.get(0n).get(0n).get(1n), 1.0));
        $(Assert.equal(genLow.get(0n).get(1n).get(1n), 1.0));
        $(Assert.equal(genLow.get(0n).get(2n).get(1n), 1.0));
    });

    test("decision_transformer: handles single timestep", $ => {
        // Edge case: sequence_length=1, no action history to attend to

        const nSamples = $.let(100n);

        // Use return to differentiate actions
        const highReturns = $.let(East.Array.generate($.let(50n), FloatType, ($, i) => i.toFloat().multiply(0.002).add(0.8)));
        const lowReturns = $.let(East.Array.generate($.let(50n), FloatType, ($, i) => i.toFloat().multiply(0.002).add(0.1)));

        const sharedStates = $.let(East.Array.generate($.let(50n), ArrayType(ArrayType(FloatType)), _ => [[0.5, 0.5]]));

        const highActions = $.let(East.Array.generate($.let(50n), ArrayType(ArrayType(FloatType)), _ => [[1.0, 0.0]]));
        const lowActions = $.let(East.Array.generate($.let(50n), ArrayType(ArrayType(FloatType)), _ => [[0.0, 1.0]]));

        const returns = $.let(highReturns.concat(lowReturns));
        const states = $.let(sharedStates.concat(sharedStates));
        const actions = $.let(highActions.concat(lowActions));
        const masks = $.let(East.Array.generate(nSamples, ArrayType(FloatType), _ => [1.0]));

        const config = $.let({
            architecture: variant('decision_transformer', {
                sequence_length: 1n,
                state_dim: 2n,
                action_dim: 2n,
                d_model: 32n,
                n_attention_heads: 4n,
                n_layers: 2n,
                d_ff: variant('some', 64n),
                dropout: variant('some', 0.0),
                return_embedding: variant('global', null),
            }),
            output: variant('binary', { pos_weight: variant('none', null) }),
            learning_rate: variant('some', 0.001),
            max_epochs: variant('some', 200n),
            patience: variant('some', 50n),
            batch_size: variant('some', 16n),
            dropout: variant('some', 0.0),
            gradient_clip: variant('some', 1.0),
            weight_decay: variant('some', 0.0001),
            random_state: variant('some', 42n),
            epoch_callback: variant('none', null),
        }, LightningConfigType);

        const result = $.let(Lightning.trainTrajectory(returns, states, actions, masks, config));
        $(Assert.less(result.val_loss, 0.5));

        const genConfig = $.let({
            temperature: 0.0,
            return_probs: false,
            action_constraints: variant('none', null),
            temporal_mask: variant('none', null),
            head_configs: variant('none', null),
            action_prefix: variant('none', null),
            start_timestep: variant('none', null),
        }, TrajectoryGenerateConfigType);

        const testStates = $.let([[[0.5, 0.5]]]);

        const genHigh = $.let(Lightning.generateTrajectory(result.model, testStates, $.let([0.85]), genConfig));
        $(Assert.equal(genHigh.get(0n).get(0n).get(0n), 1.0));

        const genLow = $.let(Lightning.generateTrajectory(result.model, testStates, $.let([0.15]), genConfig));
        $(Assert.equal(genLow.get(0n).get(0n).get(1n), 1.0));
    });

    test("decision_transformer: handles constant actions", $ => {
        // Edge case: All training samples have identical actions regardless of return/state
        // Model should learn the constant pattern without collapsing

        const nSamples = $.let(100n);

        // Varying returns and states, but constant actions
        const returns = $.let(East.Array.generate(nSamples, FloatType, ($, i) => i.toFloat().multiply(0.01)));
        const states = $.let(East.Array.generate(nSamples, ArrayType(ArrayType(FloatType)), ($, i) => {
            const v = i.toFloat().multiply(0.01);
            return [[v, v.negate().add(1.0)], [v, v.negate().add(1.0)], [v, v.negate().add(1.0)]];
        }));

        // All actions are the same
        const actions = $.let(East.Array.generate(nSamples, ArrayType(ArrayType(FloatType)), _ => [
            [1.0, 0.0], [1.0, 0.0], [1.0, 0.0],
        ]));

        const masks = $.let(East.Array.generate(nSamples, ArrayType(FloatType), _ => [1.0, 1.0, 1.0]));

        const config = $.let({
            architecture: variant('decision_transformer', {
                sequence_length: 3n,
                state_dim: 2n,
                action_dim: 2n,
                d_model: 32n,
                n_attention_heads: 4n,
                n_layers: 2n,
                d_ff: variant('some', 64n),
                dropout: variant('some', 0.0),
                return_embedding: variant('global', null),
            }),
            output: variant('binary', { pos_weight: variant('none', null) }),
            learning_rate: variant('some', 0.001),
            max_epochs: variant('some', 100n),
            patience: variant('some', 30n),
            batch_size: variant('some', 16n),
            dropout: variant('some', 0.0),
            gradient_clip: variant('some', 1.0),
            weight_decay: variant('some', 0.0001),
            random_state: variant('some', 42n),
            epoch_callback: variant('none', null),
        }, LightningConfigType);

        const result = $.let(Lightning.trainTrajectory(returns, states, actions, masks, config));
        // Loss should be very low since there's only one pattern to learn
        $(Assert.less(result.val_loss, 0.1));

        const genConfig = $.let({
            temperature: 0.0,
            return_probs: false,
            action_constraints: variant('none', null),
            temporal_mask: variant('none', null),
            head_configs: variant('none', null),
            action_prefix: variant('none', null),
            start_timestep: variant('none', null),
        }, TrajectoryGenerateConfigType);

        // Any input should produce the constant pattern
        const testStates = $.let([[[0.3, 0.7], [0.3, 0.7], [0.3, 0.7]]]);
        const generated = $.let(Lightning.generateTrajectory(result.model, testStates, $.let([0.5]), genConfig));

        $(Assert.equal(generated.get(0n).get(0n).get(0n), 1.0));
        $(Assert.equal(generated.get(0n).get(1n).get(0n), 1.0));
        $(Assert.equal(generated.get(0n).get(2n).get(0n), 1.0));
    });

    test("decision_transformer: class_weights for imbalanced binary heads", $ => {
        // Test that class_weights (pos_weight) affects training for imbalanced data
        // Setup: 95% negative, 5% positive - without pos_weight, model will predict all zeros
        // With pos_weight, model should learn to predict positives

        const nSamples = $.let(100n);

        // All samples have same return and states
        const returns = $.let(East.Array.generate(nSamples, FloatType, _ => 0.5));
        const states = $.let(East.Array.generate(nSamples, ArrayType(ArrayType(FloatType)), _ => [
            [0.5, 0.5], [0.5, 0.5], [0.5, 0.5],
        ]));

        // Imbalanced actions: 5% positive (first 5 samples), 95% negative
        const actions = $.let(East.Array.generate(nSamples, ArrayType(ArrayType(FloatType)), ($, i) => {
            // First 5 samples are positive, rest are negative
            const isPositive = i.lessThan(5n);
            return isPositive.ifElse(
                _ => [[1.0], [1.0], [1.0]],
                _ => [[0.0], [0.0], [0.0]]
            );
        }));

        const masks = $.let(East.Array.generate(nSamples, ArrayType(FloatType), _ => [1.0, 1.0, 1.0]));

        // Train WITHOUT class weights - model will likely predict all zeros
        const configNoWeight = $.let({
            architecture: variant('decision_transformer', {
                sequence_length: 3n,
                state_dim: 2n,
                action_dim: 1n,
                d_model: 16n,
                n_attention_heads: 2n,
                n_layers: 1n,
                d_ff: variant('none', null),
                dropout: variant('some', 0.0),
                return_embedding: variant('global', null),
            }),
            output: variant('multi_head_mixed', {
                heads: [
                    { head_type: variant('binary', null), class_weights: variant('none', null), conditional_on: variant('none', null) },
                ],
            }),
            learning_rate: variant('some', 0.01),
            max_epochs: variant('some', 100n),
            patience: variant('some', 30n),
            batch_size: variant('some', 16n),
            dropout: variant('some', 0.0),
            gradient_clip: variant('some', 1.0),
            weight_decay: variant('none', null),
            random_state: variant('some', 42n),
            epoch_callback: variant('none', null),
        }, LightningConfigType);

        const resultNoWeight = $.let(Lightning.trainTrajectory(returns, states, actions, masks, configNoWeight));

        // Train WITH class weights - pos_weight = 19.0 (95/5 ratio)
        const configWithWeight = $.let({
            architecture: variant('decision_transformer', {
                sequence_length: 3n,
                state_dim: 2n,
                action_dim: 1n,
                d_model: 16n,
                n_attention_heads: 2n,
                n_layers: 1n,
                d_ff: variant('none', null),
                dropout: variant('some', 0.0),
                return_embedding: variant('global', null),
            }),
            output: variant('multi_head_mixed', {
                heads: [
                    { head_type: variant('binary', null), class_weights: variant('some', [19.0]), conditional_on: variant('none', null) },
                ],
            }),
            learning_rate: variant('some', 0.01),
            max_epochs: variant('some', 100n),
            patience: variant('some', 30n),
            batch_size: variant('some', 16n),
            dropout: variant('some', 0.0),
            gradient_clip: variant('some', 1.0),
            weight_decay: variant('none', null),
            random_state: variant('some', 42n),
            epoch_callback: variant('none', null),
        }, LightningConfigType);

        const resultWithWeight = $.let(Lightning.trainTrajectory(returns, states, actions, masks, configWithWeight));

        // The weighted model should have higher loss (because positive errors are weighted more)
        // OR the losses should at least be different (showing weights are applied)
        $(Assert.notEqual(resultNoWeight.train_loss, resultWithWeight.train_loss));
    });

    test("decision_transformer: generation with action prefix", $ => {
        // Test continuing generation from a partial action history
        // Train a model that learns return-conditioned policy
        // Then test that providing a prefix maintains consistency

        const nSamplesPerClass = $.let(50n);

        // Shared state pattern for all samples
        const sharedStates = $.let(East.Array.generate(nSamplesPerClass, ArrayType(ArrayType(FloatType)), ($, i) => {
            const noise = i.toFloat().multiply(0.002);
            return [
                [noise.add(0.5), noise.negate().add(0.5)],
                [noise.add(0.5), noise.negate().add(0.5)],
                [noise.add(0.5), noise.negate().add(0.5)],
                [noise.add(0.5), noise.negate().add(0.5)],
            ];
        }));

        // High-return data: returns in [0.85, 0.95], action = [1,0,1,0,0]
        const highReturns = $.let(East.Array.generate(nSamplesPerClass, FloatType, ($, i) => i.toFloat().multiply(0.002).add(0.85)));
        const highActions = $.let(East.Array.generate(nSamplesPerClass, ArrayType(ArrayType(FloatType)), _ => [
            [1.0, 0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0, 0.0],
        ]));

        // Low-return data: returns in [0.05, 0.15], action = [0,1,0,0,1]
        const lowReturns = $.let(East.Array.generate(nSamplesPerClass, FloatType, ($, i) => i.toFloat().multiply(0.002).add(0.05)));
        const lowActions = $.let(East.Array.generate(nSamplesPerClass, ArrayType(ArrayType(FloatType)), _ => [
            [0.0, 1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0, 1.0],
        ]));

        // Combine datasets
        const returns = $.let(highReturns.concat(lowReturns));
        const states = $.let(sharedStates.concat(sharedStates));
        const actions = $.let(highActions.concat(lowActions));
        const masks = $.let(East.Array.generate(nSamplesPerClass.multiply(2n), ArrayType(FloatType), _ => [1.0, 1.0, 1.0, 1.0]));

        const config = $.let({
            architecture: variant('decision_transformer', {
                sequence_length: 4n,
                state_dim: 2n,
                action_dim: 5n,
                d_model: 32n,
                n_attention_heads: 4n,
                n_layers: 2n,
                d_ff: variant('some', 64n),
                dropout: variant('some', 0.0),
                return_embedding: variant('global', null),
            }),
            output: variant('multi_head_mixed', {
                heads: [
                    { head_type: variant('binary', null), class_weights: variant('none', null), conditional_on: variant('none', null) },
                    { head_type: variant('binary', null), class_weights: variant('none', null), conditional_on: variant('none', null) },
                    { head_type: variant('multiclass', { n_classes: 3n }), class_weights: variant('none', null), conditional_on: variant('none', null) },
                ],
            }),
            learning_rate: variant('some', 0.001),
            max_epochs: variant('some', 200n),
            patience: variant('some', 50n),
            batch_size: variant('some', 8n),
            dropout: variant('some', 0.0),
            gradient_clip: variant('some', 1.0),
            weight_decay: variant('some', 0.0001),
            random_state: variant('some', 42n),
            epoch_callback: variant('none', null),
        }, LightningConfigType);

        const result = $.let(Lightning.trainTrajectory(returns, states, actions, masks, config));
        $(Assert.less(result.val_loss, 1.0));

        const headConfigs = $.let([
            { head_type: variant('binary', null), class_weights: variant('none', null), conditional_on: variant('none', null) },
            { head_type: variant('binary', null), class_weights: variant('none', null), conditional_on: variant('none', null) },
            { head_type: variant('multiclass', { n_classes: 3n }), class_weights: variant('none', null), conditional_on: variant('none', null) },
        ], ArrayType(HeadConfigType));

        // Test 1: Generate from scratch with high return (baseline)
        const testStates = $.let([[[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]]);
        const highTargetReturn = $.let([1.0]);

        const genConfigNoPrefix = $.let({
            temperature: 0.0,
            return_probs: false,
            action_constraints: variant('none', null),
            temporal_mask: variant('none', null),
            head_configs: variant('some', headConfigs),
            action_prefix: variant('none', null),
            start_timestep: variant('none', null),
        }, TrajectoryGenerateConfigType);

        const generatedFull = $.let(Lightning.generateTrajectory(result.model, testStates, highTargetReturn, genConfigNoPrefix));

        // With high return, should get high-return pattern
        $(Assert.equal(generatedFull.get(0n).get(0n).get(0n), 1.0));

        // Test 2: Generate with prefix (first 2 timesteps from high-return pattern)
        // Use high-return actions as prefix
        const prefix = $.let([
            [1.0, 0.0, 1.0, 0.0, 0.0],  // t=0: high-return action
            [1.0, 0.0, 1.0, 0.0, 0.0],  // t=1: high-return action
            [0.0, 0.0, 0.0, 0.0, 0.0],  // t=2: placeholder (will be generated)
            [0.0, 0.0, 0.0, 0.0, 0.0],  // t=3: placeholder (will be generated)
        ]);

        const genConfigWithPrefix = $.let({
            temperature: 0.0,
            return_probs: false,
            action_constraints: variant('none', null),
            temporal_mask: variant('none', null),
            head_configs: variant('some', headConfigs),
            action_prefix: variant('some', prefix),
            start_timestep: variant('some', 2n),  // Generate from t=2 onwards
        }, TrajectoryGenerateConfigType);

        const generatedWithPrefix = $.let(Lightning.generateTrajectory(result.model, testStates, highTargetReturn, genConfigWithPrefix));

        // Verify prefix was copied exactly
        $(Assert.equal(generatedWithPrefix.get(0n).get(0n).get(0n), 1.0));  // t=0 from prefix
        $(Assert.equal(generatedWithPrefix.get(0n).get(0n).get(2n), 1.0));  // t=0 multiclass from prefix
        $(Assert.equal(generatedWithPrefix.get(0n).get(1n).get(0n), 1.0));  // t=1 from prefix
        $(Assert.equal(generatedWithPrefix.get(0n).get(1n).get(2n), 1.0));  // t=1 multiclass from prefix

        // Verify t=2 and t=3 were generated (should continue high-return pattern)
        $(Assert.equal(generatedWithPrefix.get(0n).get(2n).get(0n), 1.0));  // t=2 generated - binary 1
        $(Assert.equal(generatedWithPrefix.get(0n).get(2n).get(2n), 1.0));  // t=2 generated - multiclass class 0
        $(Assert.equal(generatedWithPrefix.get(0n).get(3n).get(0n), 1.0));  // t=3 generated - binary 1
        $(Assert.equal(generatedWithPrefix.get(0n).get(3n).get(2n), 1.0));  // t=3 generated - multiclass class 0
    });

}, { exportOnly: true });
