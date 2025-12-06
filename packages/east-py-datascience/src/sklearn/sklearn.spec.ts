/**
 * Copyright (c) 2025 Elara AI Pty Ltd
 * Dual-licensed under AGPL-3.0 and commercial license. See LICENSE for details.
 */

/**
 * Sklearn platform function tests
 *
 * These tests use describeEast following east-node conventions.
 * Tests compile East functions and run them to validate platform function behavior.
 *
 * Note: These tests require scikit-learn to be installed in the Python environment.
 */
import { East, variant } from "@elaraai/east";
import { describeEast, Assert } from "@elaraai/east-node-std";
import { Sklearn } from "./sklearn.js";

describeEast("Sklearn platform functions", (test) => {
    test("train_test_split splits data correctly", $ => {
        // Create sample data
        const X = $.let([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
            [9.0, 10.0],
        ]);
        const y = $.let([1.0, 2.0, 3.0, 4.0, 5.0]);

        const config = $.let({
            test_size: variant('some', 0.4),
            random_state: variant('some', 42n),
            shuffle: variant('some', true),
        });

        const result = $.let(Sklearn.trainTestSplit(X, y, config));

        // With 5 samples and 0.4 test_size, expect 3 train and 2 test
        $(Assert.equal(result.X_train.size(), 3n));
        $(Assert.equal(result.X_test.size(), 2n));
        $(Assert.equal(result.y_train.size(), 3n));
        $(Assert.equal(result.y_test.size(), 2n));
    });

    test("standard_scaler_fit and transform works", $ => {
        // Create sample data with different scales
        const X = $.let([
            [0.0, 0.0],
            [1.0, 100.0],
            [2.0, 200.0],
        ]);

        // Fit scaler
        const scaler = $.let(Sklearn.standardScalerFit(X));

        // Transform data
        const X_scaled = $.let(Sklearn.standardScalerTransform(scaler, X));

        // Scaled data should have roughly zero mean
        // Check that dimensions are preserved
        $(Assert.equal(X_scaled.size(), 3n));
    });

    test("min_max_scaler_fit and transform works", $ => {
        const X = $.let([
            [0.0, 0.0],
            [5.0, 50.0],
            [10.0, 100.0],
        ]);

        // Fit scaler
        const scaler = $.let(Sklearn.minMaxScalerFit(X));

        // Transform data
        const X_scaled = $.let(Sklearn.minMaxScalerTransform(scaler, X));

        // Check dimensions preserved
        $(Assert.equal(X_scaled.size(), 3n));
    });

    test("metrics_regression computes correct metrics", $ => {
        const y_true = $.let([1.0, 2.0, 3.0, 4.0, 5.0]);
        const y_pred = $.let([1.1, 2.1, 2.9, 4.2, 4.8]);

        const metrics = $.let(Sklearn.metricsRegression(y_true, y_pred));

        // MSE should be small since predictions are close
        $(Assert.less(metrics.mse, East.value(0.1)));
        // R2 should be high
        $(Assert.greater(metrics.r2, East.value(0.9)));
    });

    test("metrics_classification computes correct metrics", $ => {
        const y_true = $.let([0n, 0n, 1n, 1n, 2n, 2n]);
        const y_pred = $.let([0n, 0n, 1n, 1n, 2n, 2n]);

        const metrics = $.let(Sklearn.metricsClassification(y_true, y_pred));

        // Perfect predictions should have accuracy 1.0
        $(Assert.equal(metrics.accuracy, East.value(1.0)));
        $(Assert.equal(metrics.f1, East.value(1.0)));
    });

    test("regressor_chain with xgboost base estimator", $ => {
        // Multi-target regression: predict y1 = x1 + x2, y2 = x1 * 2
        const X = $.let([
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0],
        ]);
        const Y = $.let([
            [3.0, 2.0],   // y1 = 1+2, y2 = 1*2
            [5.0, 4.0],   // y1 = 2+3, y2 = 2*2
            [7.0, 6.0],   // y1 = 3+4, y2 = 3*2
            [9.0, 8.0],   // y1 = 4+5, y2 = 4*2
            [11.0, 10.0], // y1 = 5+6, y2 = 5*2
        ]);

        const config = $.let({
            base_estimator: variant('xgboost', {
                n_estimators: variant('some', 50n),
                max_depth: variant('some', 3n),
                learning_rate: variant('some', 0.1),
                min_child_weight: variant('none', null),
                subsample: variant('none', null),
                colsample_bytree: variant('none', null),
                reg_alpha: variant('none', null),
                reg_lambda: variant('none', null),
                random_state: variant('some', 42n),
                n_jobs: variant('none', null),
            }),
            order: variant('none', null),
            random_state: variant('some', 42n),
        });

        const model = $.let(Sklearn.regressorChainTrain(X, Y, config));
        const predictions = $.let(Sklearn.regressorChainPredict(model, X));

        // Should return predictions for all samples
        $(Assert.equal(predictions.size(), 5n));
        // Each prediction should have 2 targets
        $(Assert.equal(predictions.get(0n).size(), 2n));
    });

    test("regressor_chain with lightgbm base estimator", $ => {
        const X = $.let([
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0],
        ]);
        const Y = $.let([
            [3.0, 2.0],
            [5.0, 4.0],
            [7.0, 6.0],
            [9.0, 8.0],
            [11.0, 10.0],
        ]);

        const config = $.let({
            base_estimator: variant('lightgbm', {
                n_estimators: variant('some', 50n),
                max_depth: variant('some', 3n),
                learning_rate: variant('some', 0.1),
                num_leaves: variant('none', null),
                min_child_samples: variant('none', null),
                subsample: variant('none', null),
                colsample_bytree: variant('none', null),
                reg_alpha: variant('none', null),
                reg_lambda: variant('none', null),
                random_state: variant('some', 42n),
                n_jobs: variant('none', null),
            }),
            order: variant('none', null),
            random_state: variant('some', 42n),
        });

        const model = $.let(Sklearn.regressorChainTrain(X, Y, config));
        const predictions = $.let(Sklearn.regressorChainPredict(model, X));

        $(Assert.equal(predictions.size(), 5n));
        $(Assert.equal(predictions.get(0n).size(), 2n));
    });

    test("regressor_chain with ngboost base estimator", $ => {
        const X = $.let([
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0],
        ]);
        const Y = $.let([
            [3.0, 2.0],
            [5.0, 4.0],
            [7.0, 6.0],
            [9.0, 8.0],
            [11.0, 10.0],
        ]);

        const config = $.let({
            base_estimator: variant('ngboost', {
                n_estimators: variant('some', 50n),
                learning_rate: variant('some', 0.1),
                minibatch_frac: variant('none', null),
                col_sample: variant('none', null),
                random_state: variant('some', 42n),
                distribution: variant('none', null),
            }),
            order: variant('none', null),
            random_state: variant('some', 42n),
        });

        const model = $.let(Sklearn.regressorChainTrain(X, Y, config));
        const predictions = $.let(Sklearn.regressorChainPredict(model, X));

        $(Assert.equal(predictions.size(), 5n));
        $(Assert.equal(predictions.get(0n).size(), 2n));
    });

    test("regressor_chain with gp base estimator", $ => {
        const X = $.let([
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0],
        ]);
        const Y = $.let([
            [3.0, 2.0],
            [5.0, 4.0],
            [7.0, 6.0],
            [9.0, 8.0],
            [11.0, 10.0],
        ]);

        const config = $.let({
            base_estimator: variant('gp', {
                kernel: variant('some', variant('rbf', {})),
                alpha: variant('some', 1e-10),
                n_restarts_optimizer: variant('some', 0n),
                normalize_y: variant('some', true),
                random_state: variant('some', 42n),
            }),
            order: variant('none', null),
            random_state: variant('some', 42n),
        });

        const model = $.let(Sklearn.regressorChainTrain(X, Y, config));
        const predictions = $.let(Sklearn.regressorChainPredict(model, X));

        $(Assert.equal(predictions.size(), 5n));
        $(Assert.equal(predictions.get(0n).size(), 2n));
        // GP should interpolate training data well
        $(Assert.less(predictions.get(0n).get(0n).subtract(East.value(3.0)).abs(), East.value(0.5)));
    });

    test("regressor_chain with custom order", $ => {
        const X = $.let([
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0],
        ]);
        // 3 targets
        const Y = $.let([
            [3.0, 2.0, 5.0],
            [5.0, 4.0, 9.0],
            [7.0, 6.0, 13.0],
            [9.0, 8.0, 17.0],
            [11.0, 10.0, 21.0],
        ]);

        // Predict in order: target 2, then 0, then 1
        const config = $.let({
            base_estimator: variant('xgboost', {
                n_estimators: variant('some', 50n),
                max_depth: variant('some', 3n),
                learning_rate: variant('some', 0.1),
                min_child_weight: variant('none', null),
                subsample: variant('none', null),
                colsample_bytree: variant('none', null),
                reg_alpha: variant('none', null),
                reg_lambda: variant('none', null),
                random_state: variant('some', 42n),
                n_jobs: variant('none', null),
            }),
            order: variant('some', [2n, 0n, 1n]),
            random_state: variant('some', 42n),
        });

        const model = $.let(Sklearn.regressorChainTrain(X, Y, config));
        const predictions = $.let(Sklearn.regressorChainPredict(model, X));

        $(Assert.equal(predictions.size(), 5n));
        $(Assert.equal(predictions.get(0n).size(), 3n));
    });

    test("error: train_test_split shape mismatch", $ => {
        const X = $.let([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);  // 3 samples
        const y = $.let([1.0, 2.0]);  // 2 samples

        const config = $.let({
            test_size: variant('some', 0.2),
            random_state: variant('none', null),
            shuffle: variant('none', null),
        });

        $(Assert.throws(Sklearn.trainTestSplit(X, y, config), /sklearn_train_test_split.*X has 3 samples.*y has 2 samples/));
    });
}, { exportOnly: true });
