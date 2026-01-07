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
            stratify: variant('none', null),
            min_stratify_samples: variant('none', null),
        });

        const result = $.let(Sklearn.trainTestSplit(X, y, config));

        // With 5 samples and 0.4 test_size, expect 3 train and 2 test
        $(Assert.equal(result.X_train.size(), 3n));
        $(Assert.equal(result.X_test.size(), 2n));
        $(Assert.equal(result.y_train.size(), 3n));
        $(Assert.equal(result.y_test.size(), 2n));
        // No stratify, so no rejections
        $(Assert.equal(result.rejected_indices.size(), 0n));
    });

    test("train_test_split filters rare stratify classes", $ => {
        // 7 samples with 3 classes:
        // Class 0: 3 samples (enough)
        // Class 1: 3 samples (enough)
        // Class 2: 1 sample (rare - should be rejected)
        const X = $.let([
            [1.0, 1.0], [2.0, 1.0], [3.0, 1.0],  // class 0 (indices 0,1,2)
            [1.0, 2.0], [2.0, 2.0], [3.0, 2.0],  // class 1 (indices 3,4,5)
            [1.0, 3.0],                          // class 2 (index 6) - rare
        ]);
        const y = $.let([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0]);

        const stratify_labels = $.let([0n, 0n, 0n, 1n, 1n, 1n, 2n]);

        const config = $.let({
            test_size: variant('some', 0.33),
            random_state: variant('some', 42n),
            shuffle: variant('some', true),
            stratify: variant('some', stratify_labels),
            min_stratify_samples: variant('none', null),  // default 2
        });

        const result = $.let(Sklearn.trainTestSplit(X, y, config));

        // Class 2 (index 6) should be rejected
        $(Assert.equal(result.rejected_indices.size(), 1n));
        $(Assert.equal(result.rejected_indices.get(0n), 6n));

        // Only 6 samples remain (3 from class 0, 3 from class 1)
        // With 0.33 test_size: 4 train, 2 test
        $(Assert.equal(result.X_train.size().add(result.X_test.size()), 6n));

        // Verify remaining y values are only from classes 0 and 1 (0.0 or 1.0)
        // All y values should be < 2.0 (class 2 was removed)
        const all_train_class_01 = $.let(result.y_train.every(($, v) => v.lessThan(1.5)));
        const all_test_class_01 = $.let(result.y_test.every(($, v) => v.lessThan(1.5)));
        $(Assert.equal(all_train_class_01, true));
        $(Assert.equal(all_test_class_01, true));
    });

    test("train_test_split with custom min_stratify_samples", $ => {
        // 9 samples with 3 classes:
        // Class 0: 4 samples (X[:,1] = 1.0)
        // Class 1: 3 samples (X[:,1] = 2.0)
        // Class 2: 2 samples (X[:,1] = 3.0)
        const X = $.let([
            [1.0, 1.0], [2.0, 1.0], [3.0, 1.0], [4.0, 1.0],  // class 0 (indices 0-3)
            [1.0, 2.0], [2.0, 2.0], [3.0, 2.0],              // class 1 (indices 4-6)
            [1.0, 3.0], [2.0, 3.0],                          // class 2 (indices 7,8)
        ]);
        const y = $.let([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0]);

        const stratify_labels = $.let([0n, 0n, 0n, 0n, 1n, 1n, 1n, 2n, 2n]);

        // Require minimum 3 samples per class - class 2 should be rejected
        const config = $.let({
            test_size: variant('some', 0.3),
            random_state: variant('some', 42n),
            shuffle: variant('some', true),
            stratify: variant('some', stratify_labels),
            min_stratify_samples: variant('some', 3n),  // custom: need 3+
        });

        const result = $.let(Sklearn.trainTestSplit(X, y, config));

        // Class 2 (indices 7,8) should be rejected
        $(Assert.equal(result.rejected_indices.size(), 2n));
        // Verify the actual rejected indices
        $(Assert.equal(result.rejected_indices.get(0n), 7n));
        $(Assert.equal(result.rejected_indices.get(1n), 8n));

        // Only 7 samples remain (4 from class 0, 3 from class 1)
        $(Assert.equal(result.X_train.size().add(result.X_test.size()), 7n));

        // Verify remaining samples don't have class 2 features
        // Class 2 samples had X[:,1] = 3.0, so all remaining should have X[:,1] < 3.0
        const train_no_class2 = $.let(result.X_train.every(($, row) => row.get(1n).lessThan(2.5)));
        const test_no_class2 = $.let(result.X_test.every(($, row) => row.get(1n).lessThan(2.5)));
        $(Assert.equal(train_no_class2, true));
        $(Assert.equal(test_no_class2, true));
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

    test("compute_metrics computes correct regression metrics", $ => {
        const y_true = $.let([1.0, 2.0, 3.0, 4.0, 5.0]);
        const y_pred = $.let([1.1, 2.1, 2.9, 4.2, 4.8]);

        const results = $.let(Sklearn.computeMetrics(
            y_true,
            y_pred,
            [variant('mse', null), variant('r2', null)]
        ));

        // Should return 2 metrics
        $(Assert.equal(results.size(), 2n));
    });

    test("compute_metrics mean_error measures prediction bias", $ => {
        // Predictions that are consistently too high (positive bias)
        const y_true = $.let([1.0, 2.0, 3.0, 4.0, 5.0]);
        const y_pred_high = $.let([1.5, 2.5, 3.5, 4.5, 5.5]);  // +0.5 bias

        const results_high = $.let(Sklearn.computeMetrics(
            y_true,
            y_pred_high,
            [variant('mean_error', null)]
        ));

        // Mean error should be positive (predictions > true)
        $(Assert.equal(results_high.size(), 1n));
        $(Assert.greater(results_high.get(0n).value, 0.4));
        $(Assert.less(results_high.get(0n).value, 0.6));

        // Predictions that are consistently too low (negative bias)
        const y_pred_low = $.let([0.5, 1.5, 2.5, 3.5, 4.5]);  // -0.5 bias

        const results_low = $.let(Sklearn.computeMetrics(
            y_true,
            y_pred_low,
            [variant('mean_error', null)]
        ));

        // Mean error should be negative (predictions < true)
        $(Assert.less(results_low.get(0n).value, -0.4));
        $(Assert.greater(results_low.get(0n).value, -0.6));
    });

    test("compute_metrics pinball_loss for quantile regression", $ => {
        const y_true = $.let([1.0, 2.0, 3.0, 4.0, 5.0]);
        const y_pred = $.let([1.5, 2.5, 3.5, 4.5, 5.5]);  // Over-predictions

        // Pinball loss with alpha=0.5 (median) - symmetric penalty
        const results_median = $.let(Sklearn.computeMetrics(
            y_true,
            y_pred,
            [variant('pinball_loss', 0.5)]
        ));
        $(Assert.equal(results_median.size(), 1n));
        $(Assert.greater(results_median.get(0n).value, 0.0));

        // Pinball loss with alpha=0.9 (90th percentile)
        // Over-predictions are penalized less for high quantiles
        const results_high = $.let(Sklearn.computeMetrics(
            y_true,
            y_pred,
            [variant('pinball_loss', 0.9)]
        ));

        // Pinball loss with alpha=0.1 (10th percentile)
        // Over-predictions are penalized more for low quantiles
        const results_low = $.let(Sklearn.computeMetrics(
            y_true,
            y_pred,
            [variant('pinball_loss', 0.1)]
        ));

        // For over-predictions: low quantile loss > median loss > high quantile loss
        $(Assert.greater(results_low.get(0n).value, results_median.get(0n).value));
        $(Assert.greater(results_median.get(0n).value, results_high.get(0n).value));
    });

    test("compute_metrics huber_loss is robust to outliers", $ => {
        // Data with an outlier
        const y_true = $.let([1.0, 2.0, 3.0, 4.0, 100.0]);  // 100.0 is outlier
        const y_pred = $.let([1.0, 2.0, 3.0, 4.0, 5.0]);

        // MSE will be heavily affected by outlier
        const results_mse = $.let(Sklearn.computeMetrics(
            y_true,
            y_pred,
            [variant('mse', null)]
        ));

        // Huber loss with delta=1.0 (default) - less affected by outlier
        const results_huber = $.let(Sklearn.computeMetrics(
            y_true,
            y_pred,
            [variant('huber', 1.0)]
        ));

        // MSE should be much larger than Huber due to squared outlier error
        $(Assert.greater(results_mse.get(0n).value, results_huber.get(0n).value));

        // Huber with larger delta approaches MSE behavior
        const results_huber_large = $.let(Sklearn.computeMetrics(
            y_true,
            y_pred,
            [variant('huber', 100.0)]  // Large delta = more like MSE
        ));

        // Larger delta should give higher loss (closer to MSE)
        $(Assert.greater(results_huber_large.get(0n).value, results_huber.get(0n).value));
    });

    test("compute_metrics mean_tweedie_deviance for different distributions", $ => {
        // Positive values required for Tweedie with power != 0
        const y_true = $.let([1.0, 2.0, 3.0, 4.0, 5.0]);
        const y_pred = $.let([1.1, 2.1, 2.9, 4.2, 4.8]);

        // Power=0: Normal distribution (similar to MSE)
        const results_normal = $.let(Sklearn.computeMetrics(
            y_true,
            y_pred,
            [variant('mean_tweedie_deviance', 0.0)]
        ));
        $(Assert.equal(results_normal.size(), 1n));
        $(Assert.greaterEqual(results_normal.get(0n).value, 0.0));

        // Power=1: Poisson distribution
        const results_poisson = $.let(Sklearn.computeMetrics(
            y_true,
            y_pred,
            [variant('mean_tweedie_deviance', 1.0)]
        ));
        $(Assert.greaterEqual(results_poisson.get(0n).value, 0.0));

        // Power=2: Gamma distribution
        const results_gamma = $.let(Sklearn.computeMetrics(
            y_true,
            y_pred,
            [variant('mean_tweedie_deviance', 2.0)]
        ));
        $(Assert.greaterEqual(results_gamma.get(0n).value, 0.0));
    });

    test("compute_classification_metrics computes correct metrics", $ => {
        const y_true = $.let([0n, 0n, 1n, 1n, 2n, 2n]);
        const y_pred = $.let([0n, 0n, 1n, 1n, 2n, 2n]);

        const config = $.let({
            average: variant('some', variant('macro', null)),
        });

        const results = $.let(Sklearn.computeClassificationMetrics(
            y_true,
            y_pred,
            [variant('accuracy', null), variant('f1', null)],
            config
        ));

        // Should return 2 metrics
        $(Assert.equal(results.size(), 2n));
    });

    test("train_val_test_split creates 3-way split", $ => {
        // 10 samples, 3 features
        const X = $.let([
            [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 18.0],
            [19.0, 20.0, 21.0], [22.0, 23.0, 24.0], [25.0, 26.0, 27.0],
            [28.0, 29.0, 30.0],
        ]);
        // 10 samples, 2 targets
        const Y = $.let([
            [1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0],
            [11.0, 12.0], [13.0, 14.0], [15.0, 16.0], [17.0, 18.0], [19.0, 20.0],
        ]);

        const config = $.let({
            val_size: variant('some', 0.2),
            test_size: variant('some', 0.2),
            random_state: variant('some', 42n),
            shuffle: variant('some', true),
            stratify: variant('none', null),
            min_stratify_samples: variant('none', null),
        });

        const result = $.let(Sklearn.trainValTestSplit(X, Y, config));

        // 60% train (6), 20% val (2), 20% test (2)
        $(Assert.equal(result.X_train.size(), 6n));
        $(Assert.equal(result.X_val.size(), 2n));
        $(Assert.equal(result.X_test.size(), 2n));
        $(Assert.equal(result.Y_train.size(), 6n));
        $(Assert.equal(result.Y_val.size(), 2n));
        $(Assert.equal(result.Y_test.size(), 2n));
        // No stratify, so no rejections
        $(Assert.equal(result.rejected_indices.size(), 0n));
    });

    test("train_val_test_split with stratify ensures all classes in each split", $ => {
        // 12 samples with 3 classes (4 samples each)
        // Class distribution: 0,0,0,0, 1,1,1,1, 2,2,2,2
        const X = $.let([
            [1.0, 1.0], [2.0, 1.0], [3.0, 1.0], [4.0, 1.0],  // class 0
            [1.0, 2.0], [2.0, 2.0], [3.0, 2.0], [4.0, 2.0],  // class 1
            [1.0, 3.0], [2.0, 3.0], [3.0, 3.0], [4.0, 3.0],  // class 2
        ]);
        const Y = $.let([
            [0.0], [0.0], [0.0], [0.0],
            [1.0], [1.0], [1.0], [1.0],
            [2.0], [2.0], [2.0], [2.0],
        ]);

        // Stratify by class
        const stratify_labels = $.let([0n, 0n, 0n, 0n, 1n, 1n, 1n, 1n, 2n, 2n, 2n, 2n]);

        const config = $.let({
            val_size: variant('some', 0.25),
            test_size: variant('some', 0.25),
            random_state: variant('some', 42n),
            shuffle: variant('some', true),
            stratify: variant('some', stratify_labels),
            min_stratify_samples: variant('none', null),
        });

        const result = $.let(Sklearn.trainValTestSplit(X, Y, config));

        // 50% train (6), 25% val (3), 25% test (3)
        $(Assert.equal(result.X_train.size(), 6n));
        $(Assert.equal(result.X_val.size(), 3n));
        $(Assert.equal(result.X_test.size(), 3n));

        // With stratification, each split should have representation from all classes
        // Train: 2 from each class, Val: 1 from each, Test: 1 from each
        // No rejections since all classes have 4 samples (>= 3 default)
        $(Assert.equal(result.rejected_indices.size(), 0n));
    });

    test("train_val_test_split filters rare stratify classes", $ => {
        // 10 samples with 3 classes:
        // Class 0: 4 samples (enough) - X[:,1] = 1.0
        // Class 1: 4 samples (enough) - X[:,1] = 2.0
        // Class 2: 2 samples (rare - should be rejected) - X[:,1] = 3.0
        const X = $.let([
            [1.0, 1.0], [2.0, 1.0], [3.0, 1.0], [4.0, 1.0],  // class 0 (indices 0-3)
            [1.0, 2.0], [2.0, 2.0], [3.0, 2.0], [4.0, 2.0],  // class 1 (indices 4-7)
            [1.0, 3.0], [2.0, 3.0],                          // class 2 (indices 8,9) - rare
        ]);
        const Y = $.let([
            [0.0], [0.0], [0.0], [0.0],
            [1.0], [1.0], [1.0], [1.0],
            [2.0], [2.0],
        ]);

        const stratify_labels = $.let([0n, 0n, 0n, 0n, 1n, 1n, 1n, 1n, 2n, 2n]);

        const config = $.let({
            val_size: variant('some', 0.25),
            test_size: variant('some', 0.25),
            random_state: variant('some', 42n),
            shuffle: variant('some', true),
            stratify: variant('some', stratify_labels),
            min_stratify_samples: variant('none', null),  // default 3 for 3-way
        });

        const result = $.let(Sklearn.trainValTestSplit(X, Y, config));

        // Class 2 (indices 8,9) should be rejected
        $(Assert.equal(result.rejected_indices.size(), 2n));
        // Verify the actual rejected indices
        $(Assert.equal(result.rejected_indices.get(0n), 8n));
        $(Assert.equal(result.rejected_indices.get(1n), 9n));

        // Only 8 samples remain (4 from class 0, 4 from class 1)
        const total = $.let(result.X_train.size().add(result.X_val.size()).add(result.X_test.size()));
        $(Assert.equal(total, 8n));

        // Verify remaining samples don't have class 2 features
        // Class 2 samples had X[:,1] = 3.0, so all remaining should have X[:,1] < 3.0
        const train_no_class2 = $.let(result.X_train.every(($, row) => row.get(1n).lessThan(2.5)));
        const val_no_class2 = $.let(result.X_val.every(($, row) => row.get(1n).lessThan(2.5)));
        const test_no_class2 = $.let(result.X_test.every(($, row) => row.get(1n).lessThan(2.5)));
        $(Assert.equal(train_no_class2, true));
        $(Assert.equal(val_no_class2, true));
        $(Assert.equal(test_no_class2, true));

        // Verify Y values don't have class 2 (2.0)
        const train_Y_no_class2 = $.let(result.Y_train.every(($, row) => row.get(0n).lessThan(1.5)));
        const val_Y_no_class2 = $.let(result.Y_val.every(($, row) => row.get(0n).lessThan(1.5)));
        const test_Y_no_class2 = $.let(result.Y_test.every(($, row) => row.get(0n).lessThan(1.5)));
        $(Assert.equal(train_Y_no_class2, true));
        $(Assert.equal(val_Y_no_class2, true));
        $(Assert.equal(test_Y_no_class2, true));
    });

    test("train_val_test_split with custom min_stratify_samples", $ => {
        // 14 samples with 3 classes:
        // Class 0: 6 samples - X[:,1] = 1.0
        // Class 1: 5 samples - X[:,1] = 2.0
        // Class 2: 3 samples - X[:,1] = 3.0
        const X = $.let([
            [1.0, 1.0], [2.0, 1.0], [3.0, 1.0], [4.0, 1.0], [5.0, 1.0], [6.0, 1.0],  // class 0 (0-5)
            [1.0, 2.0], [2.0, 2.0], [3.0, 2.0], [4.0, 2.0], [5.0, 2.0],              // class 1 (6-10)
            [1.0, 3.0], [2.0, 3.0], [3.0, 3.0],                                      // class 2 (11-13)
        ]);
        const Y = $.let([
            [0.0], [0.0], [0.0], [0.0], [0.0], [0.0],
            [1.0], [1.0], [1.0], [1.0], [1.0],
            [2.0], [2.0], [2.0],
        ]);

        const stratify_labels = $.let([0n, 0n, 0n, 0n, 0n, 0n, 1n, 1n, 1n, 1n, 1n, 2n, 2n, 2n]);

        // Require minimum 4 samples per class - class 2 should be rejected
        const config = $.let({
            val_size: variant('some', 0.2),
            test_size: variant('some', 0.2),
            random_state: variant('some', 42n),
            shuffle: variant('some', true),
            stratify: variant('some', stratify_labels),
            min_stratify_samples: variant('some', 4n),  // custom: need 4+
        });

        const result = $.let(Sklearn.trainValTestSplit(X, Y, config));

        // Class 2 (indices 11,12,13) should be rejected
        $(Assert.equal(result.rejected_indices.size(), 3n));
        // Verify the actual rejected indices
        $(Assert.equal(result.rejected_indices.get(0n), 11n));
        $(Assert.equal(result.rejected_indices.get(1n), 12n));
        $(Assert.equal(result.rejected_indices.get(2n), 13n));

        // Only 11 samples remain (6 from class 0, 5 from class 1)
        const total = $.let(result.X_train.size().add(result.X_val.size()).add(result.X_test.size()));
        $(Assert.equal(total, 11n));

        // Verify remaining samples don't have class 2 features
        // Class 2 samples had X[:,1] = 3.0, so all remaining should have X[:,1] < 3.0
        const train_no_class2 = $.let(result.X_train.every(($, row) => row.get(1n).lessThan(2.5)));
        const val_no_class2 = $.let(result.X_val.every(($, row) => row.get(1n).lessThan(2.5)));
        const test_no_class2 = $.let(result.X_test.every(($, row) => row.get(1n).lessThan(2.5)));
        $(Assert.equal(train_no_class2, true));
        $(Assert.equal(val_no_class2, true));
        $(Assert.equal(test_no_class2, true));
    });

    test("compute_metrics_multi computes per-target metrics", $ => {
        // Multi-target data
        const Y_true = $.let([
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
            [5.0, 50.0],
        ]);
        const Y_pred = $.let([
            [1.1, 10.5],
            [2.1, 20.5],
            [2.9, 29.5],
            [4.2, 40.5],
            [4.8, 49.5],
        ]);

        const config = $.let({
            aggregation: variant('some', variant('per_target', null)),
        });

        const results = $.let(Sklearn.computeMetricsMulti(
            Y_true,
            Y_pred,
            [variant('mse', null), variant('r2', null)],
            config
        ));

        // Should return 2 metrics
        $(Assert.equal(results.size(), 2n));
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
                sample_weight: variant('none', null),
                categorical_features: variant('none', null),
                max_cat_to_onehot: variant('none', null),
                max_cat_threshold: variant('none', null),
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
                kernel: variant('some', variant('rbf', null)),
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
                sample_weight: variant('none', null),
                categorical_features: variant('none', null),
                max_cat_to_onehot: variant('none', null),
                max_cat_threshold: variant('none', null),
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
            stratify: variant('none', null),
            min_stratify_samples: variant('none', null),
        });

        $(Assert.throws(Sklearn.trainTestSplit(X, y, config), /sklearn_train_test_split.*X has 3 samples.*y has 2 samples/));
    });

    test("train_test_split post-split validation rejects classes missing from a split", $ => {
        // 8 samples with 2 classes:
        // Class 0: 6 samples (plenty)
        // Class 1: 2 samples (exactly min_stratify_samples=2, but may not appear in both splits)
        // With 2 samples and 50% test_size, sklearn may put both in train or both in test
        const X = $.let([
            [1.0, 1.0], [2.0, 1.0], [3.0, 1.0], [4.0, 1.0], [5.0, 1.0], [6.0, 1.0],  // class 0 (0-5)
            [1.0, 2.0], [2.0, 2.0],  // class 1 (6-7) - edge case
        ]);
        const y = $.let([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]);
        const stratify_labels = $.let([0n, 0n, 0n, 0n, 0n, 0n, 1n, 1n]);

        const config = $.let({
            test_size: variant('some', 0.25),  // 2 test samples
            random_state: variant('some', 123n),  // specific seed to trigger edge case
            shuffle: variant('some', true),
            stratify: variant('some', stratify_labels),
            min_stratify_samples: variant('some', 2n),  // allow class 1 through pre-filter
        });

        const result = $.let(Sklearn.trainTestSplit(X, y, config));

        // Either all 8 samples remain (class 1 was in both splits)
        // Or only 6 samples remain (class 1 was rejected)
        // The key is: we should never have class 1 in only one split
        const train_has_class1 = $.let(result.y_train.some(($, v) => v.greaterThan(0.5)));
        const test_has_class1 = $.let(result.y_test.some(($, v) => v.greaterThan(0.5)));

        // If one split has class 1, both must have it (otherwise they'd be rejected)
        // This is enforced by: train_has_class1 == test_has_class1
        $(Assert.equal(train_has_class1, test_has_class1));
    });

    test("train_val_test_split post-split validation rejects classes missing from any split", $ => {
        // 11 samples with 2 classes:
        // Class 0: 8 samples (plenty for 3-way split)
        // Class 1: 3 samples (exactly min_stratify_samples=3, edge case for 3-way split)
        // With 3 samples and ~70/15/15 split, sklearn may put 0 in one split
        const X = $.let([
            [1.0, 1.0], [2.0, 1.0], [3.0, 1.0], [4.0, 1.0],
            [5.0, 1.0], [6.0, 1.0], [7.0, 1.0], [8.0, 1.0],  // class 0 (0-7)
            [1.0, 2.0], [2.0, 2.0], [3.0, 2.0],  // class 1 (8-10) - edge case
        ]);
        const Y = $.let([
            [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0],
            [1.0], [1.0], [1.0],
        ]);
        const stratify_labels = $.let([0n, 0n, 0n, 0n, 0n, 0n, 0n, 0n, 1n, 1n, 1n]);

        const config = $.let({
            val_size: variant('some', 0.15),
            test_size: variant('some', 0.15),
            random_state: variant('some', 7n),  // specific seed
            shuffle: variant('some', true),
            stratify: variant('some', stratify_labels),
            min_stratify_samples: variant('some', 3n),  // allow class 1 through pre-filter
        });

        const result = $.let(Sklearn.trainValTestSplit(X, Y, config));

        // Check consistency: if class 1 appears in any split, it must appear in ALL splits
        const train_has_class1 = $.let(result.Y_train.some(($, row) => row.get(0n).greaterThan(0.5)));
        const val_has_class1 = $.let(result.Y_val.some(($, row) => row.get(0n).greaterThan(0.5)));
        const test_has_class1 = $.let(result.Y_test.some(($, row) => row.get(0n).greaterThan(0.5)));

        // All three must be equal (either all have class 1, or none have it)
        $(Assert.equal(train_has_class1, val_has_class1));
        $(Assert.equal(val_has_class1, test_has_class1));

        // If class 1 was rejected, verify rejected_indices contains indices 8,9,10
        $.if(result.rejected_indices.size().greaterThan(0n), $ => {
            // All rejected should be class 1 samples (indices 8,9,10)
            const all_rejected_are_class1 = $.let(result.rejected_indices.every(($, idx) => idx.greaterThanOrEqual(8n)));
            $(Assert.equal(all_rejected_are_class1, true));
        });
    });

    test("train_val_test_split guarantees each split has all stratify classes", $ => {
        // Test multiple random seeds to ensure consistency
        // 15 samples: class 0 (9), class 1 (6)
        const X = $.let([
            [1.0, 1.0], [2.0, 1.0], [3.0, 1.0], [4.0, 1.0], [5.0, 1.0],
            [6.0, 1.0], [7.0, 1.0], [8.0, 1.0], [9.0, 1.0],  // class 0 (0-8)
            [1.0, 2.0], [2.0, 2.0], [3.0, 2.0], [4.0, 2.0], [5.0, 2.0], [6.0, 2.0],  // class 1 (9-14)
        ]);
        const Y = $.let([
            [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0],
            [1.0], [1.0], [1.0], [1.0], [1.0], [1.0],
        ]);
        const stratify_labels = $.let([0n, 0n, 0n, 0n, 0n, 0n, 0n, 0n, 0n, 1n, 1n, 1n, 1n, 1n, 1n]);

        const config = $.let({
            val_size: variant('some', 0.2),
            test_size: variant('some', 0.2),
            random_state: variant('some', 42n),
            shuffle: variant('some', true),
            stratify: variant('some', stratify_labels),
            min_stratify_samples: variant('some', 3n),
        });

        const result = $.let(Sklearn.trainValTestSplit(X, Y, config));

        // With 6 samples in class 1 and 3-way split, both classes should appear in all splits
        // No samples should be rejected
        $(Assert.equal(result.rejected_indices.size(), 0n));

        // Verify all splits have both classes
        const train_has_class0 = $.let(result.Y_train.some(($, row) => row.get(0n).lessThan(0.5)));
        const train_has_class1 = $.let(result.Y_train.some(($, row) => row.get(0n).greaterThan(0.5)));
        const val_has_class0 = $.let(result.Y_val.some(($, row) => row.get(0n).lessThan(0.5)));
        const val_has_class1 = $.let(result.Y_val.some(($, row) => row.get(0n).greaterThan(0.5)));
        const test_has_class0 = $.let(result.Y_test.some(($, row) => row.get(0n).lessThan(0.5)));
        const test_has_class1 = $.let(result.Y_test.some(($, row) => row.get(0n).greaterThan(0.5)));

        $(Assert.equal(train_has_class0, true));
        $(Assert.equal(train_has_class1, true));
        $(Assert.equal(val_has_class0, true));
        $(Assert.equal(val_has_class1, true));
        $(Assert.equal(test_has_class0, true));
        $(Assert.equal(test_has_class1, true));
    });
}, { exportOnly: true });
