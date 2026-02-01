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
    test("split creates 2-way split correctly", $ => {
        // Create sample data
        const X = $.let([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
            [9.0, 10.0],
        ]);
        const Y = $.let([[1.0], [2.0], [3.0], [4.0], [5.0]]);

        const config = $.let({
            split_sizes: [0.6, 0.4],  // 60% train, 40% test
            random_state: variant('some', 42n),
            shuffle: variant('some', true),
            stratify: variant('none', null),
            overlap: variant('none', null),
            min_overlap: variant('none', null),
        });

        const result = $.let(Sklearn.split(X, Y, config));

        // With 5 samples and [0.6, 0.4], expect 3 train and 2 test
        $(Assert.equal(result.X_splits.get(0n).size(), 3n));  // train
        $(Assert.equal(result.X_splits.get(1n).size(), 2n));  // test
        $(Assert.equal(result.Y_splits.get(0n).size(), 3n));
        $(Assert.equal(result.Y_splits.get(1n).size(), 2n));
        // No stratify, so no rejections
        $(Assert.equal(result.rejected_indices.size(), 0n));
    });

    test("split filters rare overlap classes", $ => {
        // 7 samples with 3 classes:
        // Class 0: 3 samples (enough)
        // Class 1: 3 samples (enough)
        // Class 2: 1 sample (rare - should be rejected by min_overlap)
        const X = $.let([
            [1.0, 1.0], [2.0, 1.0], [3.0, 1.0],  // class 0 (indices 0,1,2)
            [1.0, 2.0], [2.0, 2.0], [3.0, 2.0],  // class 1 (indices 3,4,5)
            [1.0, 3.0],                          // class 2 (index 6) - rare
        ]);
        const Y = $.let([[0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [2.0]]);

        const overlap_labels = $.let([[0n, 0n, 0n, 1n, 1n, 1n, 2n]]);  // Array of arrays

        const config = $.let({
            split_sizes: [0.67, 0.33],
            random_state: variant('some', 42n),
            shuffle: variant('some', true),
            stratify: variant('none', null),
            overlap: variant('some', overlap_labels),
            min_overlap: variant('none', null),  // default 2
        });

        const result = $.let(Sklearn.split(X, Y, config));

        // Class 2 (index 6) should be rejected (only 1 sample, needs 2)
        $(Assert.equal(result.rejected_indices.size(), 1n));
        $(Assert.equal(result.rejected_indices.get(0n), 6n));

        // Only 6 samples remain (3 from class 0, 3 from class 1)
        $(Assert.equal(result.X_splits.get(0n).size().add(result.X_splits.get(1n).size()), 6n));
    });

    test("split with custom min_overlap", $ => {
        // 9 samples with 3 classes:
        // Class 0: 4 samples (X[:,1] = 1.0)
        // Class 1: 3 samples (X[:,1] = 2.0)
        // Class 2: 2 samples (X[:,1] = 3.0)
        const X = $.let([
            [1.0, 1.0], [2.0, 1.0], [3.0, 1.0], [4.0, 1.0],  // class 0 (indices 0-3)
            [1.0, 2.0], [2.0, 2.0], [3.0, 2.0],              // class 1 (indices 4-6)
            [1.0, 3.0], [2.0, 3.0],                          // class 2 (indices 7,8)
        ]);
        const Y = $.let([[0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [2.0], [2.0]]);

        const overlap_labels = $.let([[0n, 0n, 0n, 0n, 1n, 1n, 1n, 2n, 2n]]);

        // Require minimum 3 samples per overlap value - class 2 should be rejected
        const config = $.let({
            split_sizes: [0.7, 0.3],
            random_state: variant('some', 42n),
            shuffle: variant('some', true),
            stratify: variant('none', null),
            overlap: variant('some', overlap_labels),
            min_overlap: variant('some', 3n),  // custom: need 3+
        });

        const result = $.let(Sklearn.split(X, Y, config));

        // Class 2 (indices 7,8) should be rejected
        $(Assert.equal(result.rejected_indices.size(), 2n));
        // Verify the actual rejected indices
        $(Assert.equal(result.rejected_indices.get(0n), 7n));
        $(Assert.equal(result.rejected_indices.get(1n), 8n));

        // Only 7 samples remain (4 from class 0, 3 from class 1)
        $(Assert.equal(result.X_splits.get(0n).size().add(result.X_splits.get(1n).size()), 7n));

        // Verify remaining samples don't have class 2 features
        // Class 2 samples had X[:,1] = 3.0, so all remaining should have X[:,1] < 3.0
        const train_no_class2 = $.let(result.X_splits.get(0n).every(($, row) => row.get(1n).lessThan(2.5)));
        const test_no_class2 = $.let(result.X_splits.get(1n).every(($, row) => row.get(1n).lessThan(2.5)));
        $(Assert.equal(train_no_class2, true));
        $(Assert.equal(test_no_class2, true));
    });

    test("split with multi-column stratification", $ => {
        // 12 samples with 2 stratify columns:
        // Column A (origin): 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1
        // Column B (type):   0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1
        // Compound strata: A*2 + B = 0,0,1,1, 2,2,3,3, 0,1,2,3
        // Each compound stratum has: 0->3, 1->3, 2->3, 3->3 samples (enough for 3-way)
        const X = $.let([
            [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 1.0], [4.0, 0.0, 1.0],
            [5.0, 1.0, 0.0], [6.0, 1.0, 0.0], [7.0, 1.0, 1.0], [8.0, 1.0, 1.0],
            [9.0, 0.0, 0.0], [10.0, 0.0, 1.0], [11.0, 1.0, 0.0], [12.0, 1.0, 1.0],
        ]);
        const Y = $.let([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0], [10.0], [11.0], [12.0]]);

        // Two stratify columns with similar value ranges (both 0 and 1)
        const col_A = $.let([0n, 0n, 0n, 0n, 1n, 1n, 1n, 1n, 0n, 0n, 1n, 1n]);
        const col_B = $.let([0n, 0n, 1n, 1n, 0n, 0n, 1n, 1n, 0n, 1n, 0n, 1n]);
        const stratify_cols = $.let([col_A, col_B]);

        const config = $.let({
            split_sizes: [0.5, 0.25, 0.25],  // 6 train, 3 val, 3 test
            random_state: variant('some', 42n),
            shuffle: variant('some', true),
            stratify: variant('some', stratify_cols),
            min_overlap: variant('some', 3n),
            overlap: variant('none', null),
        });

        const result = $.let(Sklearn.split(X, Y, config));

        // 3-way split
        $(Assert.equal(result.X_splits.size(), 3n));

        // Total samples should be 12 (no rejections since each compound stratum has 3+ samples)
        const total = $.let(
            result.X_splits.get(0n).size()
                .add(result.X_splits.get(1n).size())
                .add(result.X_splits.get(2n).size())
        );
        $(Assert.equal(total, 12n));
        $(Assert.equal(result.rejected_indices.size(), 0n));
    });

    test("split with multi-column stratification rejects rare compound strata", $ => {
        // 9 samples with 2 stratify columns:
        // Column A (origin): 0, 0, 0, 0, 0, 1, 1, 1, 1
        // Column B (type):   0, 0, 0, 1, 1, 0, 0, 1, 1
        // Compound strata: A*2 + B = 0,0,0,1,1, 2,2,3,3
        // Stratum 0: 3 samples, Stratum 1: 2 samples, Stratum 2: 2 samples, Stratum 3: 2 samples
        const X = $.let([
            [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0],  // stratum 0 (A=0, B=0) - 3 samples
            [4.0, 0.0, 1.0], [5.0, 0.0, 1.0],                    // stratum 1 (A=0, B=1) - 2 samples
            [6.0, 1.0, 0.0], [7.0, 1.0, 0.0],                    // stratum 2 (A=1, B=0) - 2 samples
            [8.0, 1.0, 1.0], [9.0, 1.0, 1.0],                    // stratum 3 (A=1, B=1) - 2 samples
        ]);
        const Y = $.let([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0]]);

        const col_A = $.let([0n, 0n, 0n, 0n, 0n, 1n, 1n, 1n, 1n]);
        const col_B = $.let([0n, 0n, 0n, 1n, 1n, 0n, 0n, 1n, 1n]);
        const stratify_cols = $.let([col_A, col_B]);

        const config = $.let({
            split_sizes: [0.7, 0.3],
            random_state: variant('some', 42n),
            shuffle: variant('some', true),
            stratify: variant('some', stratify_cols),
            min_overlap: variant('some', 3n),  // Require 3+ per compound stratum
            overlap: variant('none', null),
        });

        const result = $.let(Sklearn.split(X, Y, config));

        // Strata 1, 2, 3 have only 2 samples each -> rejected (indices 3,4,5,6,7,8)
        // Only stratum 0 (indices 0,1,2) remains
        $(Assert.equal(result.rejected_indices.size(), 6n));

        const total = $.let(result.X_splits.get(0n).size().add(result.X_splits.get(1n).size()));
        $(Assert.equal(total, 3n));
    });

    test("split with overlap column ensures values in all splits", $ => {
        // 12 samples with a class column that should overlap
        // Class values: 10, 10, 10, 10, 20, 20, 20, 20, 30, 30, 30, 30
        const X = $.let([
            [1.0], [2.0], [3.0], [4.0],   // class 10 (indices 0-3)
            [5.0], [6.0], [7.0], [8.0],   // class 20 (indices 4-7)
            [9.0], [10.0], [11.0], [12.0], // class 30 (indices 8-11)
        ]);
        const Y = $.let([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0], [10.0], [11.0], [12.0]]);

        // Overlap column - all 3 classes should appear in all splits
        const overlap_col = $.let([[10n, 10n, 10n, 10n, 20n, 20n, 20n, 20n, 30n, 30n, 30n, 30n]]);

        const config = $.let({
            split_sizes: [0.5, 0.25, 0.25],
            random_state: variant('some', 42n),
            shuffle: variant('some', true),
            stratify: variant('none', null),
            min_overlap: variant('none', null),
            overlap: variant('some', overlap_col),
        });

        const result = $.let(Sklearn.split(X, Y, config));

        // All 12 samples should remain if all classes appear in all splits
        // Or some may be rejected if a class doesn't appear in all splits
        $(Assert.equal(result.X_splits.size(), 3n));
    });

    test("split multi-column stratification correctly distinguishes (A=0,B=1) from (A=1,B=0)", $ => {
        // EDGE CASE: Both columns have identical value ranges [0, 1]
        // We need to ensure compound strata correctly distinguish:
        //   (A=0, B=1) vs (A=1, B=0) - these should be DIFFERENT strata
        //
        // 8 samples with deliberately ambiguous values:
        // Sample 0-1: A=0, B=0 -> compound = 0*2 + 0 = 0
        // Sample 2-3: A=0, B=1 -> compound = 0*2 + 1 = 1
        // Sample 4-5: A=1, B=0 -> compound = 1*2 + 0 = 2
        // Sample 6-7: A=1, B=1 -> compound = 1*2 + 1 = 3
        const X = $.let([
            [1.0], [2.0],   // A=0, B=0 (indices 0,1) - mark with x=1,2
            [3.0], [4.0],   // A=0, B=1 (indices 2,3) - mark with x=3,4
            [5.0], [6.0],   // A=1, B=0 (indices 4,5) - mark with x=5,6
            [7.0], [8.0],   // A=1, B=1 (indices 6,7) - mark with x=7,8
        ]);
        const Y = $.let([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]]);

        // CRITICAL: Both columns use values 0 and 1
        const col_A = $.let([0n, 0n, 0n, 0n, 1n, 1n, 1n, 1n]);
        const col_B = $.let([0n, 0n, 1n, 1n, 0n, 0n, 1n, 1n]);
        const stratify_cols = $.let([col_A, col_B]);

        const config = $.let({
            split_sizes: [0.5, 0.5],
            random_state: variant('some', 42n),
            shuffle: variant('some', true),
            stratify: variant('some', stratify_cols),
            min_overlap: variant('some', 2n),  // Each compound stratum has exactly 2
            overlap: variant('none', null),
        });

        const result = $.let(Sklearn.split(X, Y, config));

        // All 8 samples should be included (each compound stratum has 2 samples)
        const total = $.let(result.X_splits.get(0n).size().add(result.X_splits.get(1n).size()));
        $(Assert.equal(total, 8n));
        $(Assert.equal(result.rejected_indices.size(), 0n));

        // Verify stratification worked: each split should have 4 samples
        // (1 from each of the 4 compound strata)
        $(Assert.equal(result.X_splits.get(0n).size(), 4n));
        $(Assert.equal(result.X_splits.get(1n).size(), 4n));

        // Verify that A=0,B=1 samples (x=3,4) are separate from A=1,B=0 samples (x=5,6)
        // Both splits should have exactly one sample with x in [3,4] and one with x in [5,6]
        const split0_has_AB01 = $.let(result.X_splits.get(0n).some(($, row) =>
            row.get(0n).greaterThanOrEqual(3.0).bitAnd(row.get(0n).lessThan(5.0))
        ));
        const split0_has_AB10 = $.let(result.X_splits.get(0n).some(($, row) =>
            row.get(0n).greaterThanOrEqual(5.0).bitAnd(row.get(0n).lessThan(7.0))
        ));
        const split1_has_AB01 = $.let(result.X_splits.get(1n).some(($, row) =>
            row.get(0n).greaterThanOrEqual(3.0).bitAnd(row.get(0n).lessThan(5.0))
        ));
        const split1_has_AB10 = $.let(result.X_splits.get(1n).some(($, row) =>
            row.get(0n).greaterThanOrEqual(5.0).bitAnd(row.get(0n).lessThan(7.0))
        ));

        // Each split should have representation from each compound stratum
        $(Assert.equal(split0_has_AB01, true));
        $(Assert.equal(split0_has_AB10, true));
        $(Assert.equal(split1_has_AB01, true));
        $(Assert.equal(split1_has_AB10, true));
    });

    test("split multi-column handles large value ranges", $ => {
        // EDGE CASE: One column has much larger values than another
        // Column A: [0, 1, 2]
        // Column B: [0, 1000000]
        // The multiplier for column A must be > 1000001 to avoid collision
        const X = $.let([
            [1.0], [2.0], [3.0],   // A=0, B=0 (indices 0-2)
            [4.0], [5.0], [6.0],   // A=0, B=1000000 (indices 3-5)
            [7.0], [8.0], [9.0],   // A=1, B=0 (indices 6-8)
            [10.0], [11.0], [12.0], // A=1, B=1000000 (indices 9-11)
        ]);
        const Y = $.let([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0], [10.0], [11.0], [12.0]]);

        const col_A = $.let([0n, 0n, 0n, 0n, 0n, 0n, 1n, 1n, 1n, 1n, 1n, 1n]);
        const col_B = $.let([0n, 0n, 0n, 1000000n, 1000000n, 1000000n, 0n, 0n, 0n, 1000000n, 1000000n, 1000000n]);
        const stratify_cols = $.let([col_A, col_B]);

        const config = $.let({
            split_sizes: [0.5, 0.5],
            random_state: variant('some', 42n),
            shuffle: variant('some', true),
            stratify: variant('some', stratify_cols),
            min_overlap: variant('some', 3n),
            overlap: variant('none', null),
        });

        const result = $.let(Sklearn.split(X, Y, config));

        // All 12 samples should be included (each compound stratum has 3 samples)
        const total = $.let(result.X_splits.get(0n).size().add(result.X_splits.get(1n).size()));
        $(Assert.equal(total, 12n));
        $(Assert.equal(result.rejected_indices.size(), 0n));
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

    test("robust_scaler_fit and transform works", $ => {
        // Data with outliers - RobustScaler should handle these better
        const X = $.let([
            [0.0, 0.0],
            [1.0, 100.0],
            [2.0, 200.0],
            [100.0, 1000.0],  // outlier
        ]);

        // Fit scaler
        const scaler = $.let(Sklearn.robustScalerFit(X));

        // Transform data
        const X_scaled = $.let(Sklearn.robustScalerTransform(scaler, X));

        // Check dimensions preserved
        $(Assert.equal(X_scaled.size(), 4n));
        $(Assert.equal(X_scaled.get(0n).size(), 2n));
    });

    test("compute_class_weight returns balanced weights", $ => {
        // Imbalanced classes: class 0 has 5 samples, class 1 has 2 samples
        const y = $.let([0n, 0n, 0n, 0n, 0n, 1n, 1n]);

        const weights = $.let(Sklearn.computeClassWeight(variant('balanced', null), y));

        // Should return 2 weights (one per class)
        $(Assert.equal(weights.size(), 2n));

        // Class 0 (majority) should have lower weight than class 1 (minority)
        $(Assert.less(weights.get(0n), weights.get(1n)));

        // Weights should be positive
        $(Assert.greater(weights.get(0n), 0.0));
        $(Assert.greater(weights.get(1n), 0.0));
    });

    test("confusion_matrix computes correct matrix", $ => {
        // Perfect predictions for 3 classes
        const y_true = $.let([0n, 0n, 1n, 1n, 2n, 2n]);
        const y_pred = $.let([0n, 0n, 1n, 1n, 2n, 2n]);

        const result = $.let(Sklearn.confusionMatrix(y_true, y_pred));

        // Should have 3x3 matrix (3 classes)
        $(Assert.equal(result.matrix.size(), 3n));
        $(Assert.equal(result.matrix.get(0n).size(), 3n));

        // Diagonal should be 2 (2 correct predictions per class)
        $(Assert.equal(result.matrix.get(0n).get(0n), 2.0));
        $(Assert.equal(result.matrix.get(1n).get(1n), 2.0));
        $(Assert.equal(result.matrix.get(2n).get(2n), 2.0));

        // Off-diagonal should be 0 (no misclassifications)
        $(Assert.equal(result.matrix.get(0n).get(1n), 0.0));
        $(Assert.equal(result.matrix.get(1n).get(0n), 0.0));

        // Classes should be [0, 1, 2]
        $(Assert.equal(result.classes.size(), 3n));
    });

    test("confusion_matrix with misclassifications", $ => {
        // Some misclassifications
        const y_true = $.let([0n, 0n, 1n, 1n]);
        const y_pred = $.let([0n, 1n, 1n, 0n]);  // 1 error in class 0, 1 error in class 1

        const result = $.let(Sklearn.confusionMatrix(y_true, y_pred));

        // Class 0: 1 correct, 1 predicted as class 1
        $(Assert.equal(result.matrix.get(0n).get(0n), 1.0));  // True 0, Pred 0
        $(Assert.equal(result.matrix.get(0n).get(1n), 1.0));  // True 0, Pred 1

        // Class 1: 1 correct, 1 predicted as class 0
        $(Assert.equal(result.matrix.get(1n).get(0n), 1.0));  // True 1, Pred 0
        $(Assert.equal(result.matrix.get(1n).get(1n), 1.0));  // True 1, Pred 1
    });

    test("roc_auc_score computes binary classification score", $ => {
        // Binary classification with good predictions
        const y_true = $.let([0n, 0n, 1n, 1n]);
        // Probabilities: [P(class=0), P(class=1)]
        const y_proba = $.let([
            [0.9, 0.1],  // High confidence for class 0
            [0.8, 0.2],  // High confidence for class 0
            [0.2, 0.8],  // High confidence for class 1
            [0.1, 0.9],  // High confidence for class 1
        ]);

        const config = $.let({
            multi_class: variant('none', null),
            average: variant('none', null),
        });

        const score = $.let(Sklearn.rocAucScore(y_true, y_proba, config));

        // Perfect predictions should give AUC close to 1.0
        $(Assert.greater(score, 0.9));
    });

    test("log_loss computes cross-entropy loss", $ => {
        // Binary classification
        const y_true = $.let([0n, 0n, 1n, 1n]);
        const y_proba = $.let([
            [0.9, 0.1],
            [0.8, 0.2],
            [0.2, 0.8],
            [0.1, 0.9],
        ]);

        const loss = $.let(Sklearn.logLoss(y_true, y_proba));

        // Good predictions should have low log loss
        $(Assert.less(loss, 0.5));
        $(Assert.greater(loss, 0.0));
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

    test("compute_classification_metrics with cohen_kappa weights", $ => {
        // Ordinal classification where distance between classes matters
        const y_true = $.let([0n, 1n, 2n, 3n, 4n, 0n, 1n, 2n, 3n, 4n]);
        const y_pred = $.let([0n, 2n, 2n, 3n, 3n, 0n, 0n, 2n, 4n, 4n]);

        const config = $.let({
            average: variant('none', null),
        });

        // Test with no weighting (default)
        const results_none = $.let(Sklearn.computeClassificationMetrics(
            y_true,
            y_pred,
            [variant('cohen_kappa', variant('none', null))],
            config
        ));
        $(Assert.equal(results_none.size(), 1n));

        // Test with linear weighting
        const results_linear = $.let(Sklearn.computeClassificationMetrics(
            y_true,
            y_pred,
            [variant('cohen_kappa', variant('linear', null))],
            config
        ));
        $(Assert.equal(results_linear.size(), 1n));

        // Test with quadratic weighting
        const results_quadratic = $.let(Sklearn.computeClassificationMetrics(
            y_true,
            y_pred,
            [variant('cohen_kappa', variant('quadratic', null))],
            config
        ));
        $(Assert.equal(results_quadratic.size(), 1n));

        // Weighted kappa should generally give higher values than unweighted
        // for ordinal scales (since close errors are penalized less)
        $(Assert.greaterEqual(results_quadratic.get(0n).value, results_none.get(0n).value));
    });

    test("split creates 3-way split", $ => {
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
            split_sizes: [0.6, 0.2, 0.2],  // 60% train, 20% val, 20% test
            random_state: variant('some', 42n),
            shuffle: variant('some', true),
            stratify: variant('none', null),
            min_overlap: variant('none', null),
            overlap: variant('none', null),
        });

        const result = $.let(Sklearn.split(X, Y, config));

        // 60% train (6), 20% val (2), 20% test (2)
        $(Assert.equal(result.X_splits.get(0n).size(), 6n));
        $(Assert.equal(result.X_splits.get(1n).size(), 2n));
        $(Assert.equal(result.X_splits.get(2n).size(), 2n));
        $(Assert.equal(result.Y_splits.get(0n).size(), 6n));
        $(Assert.equal(result.Y_splits.get(1n).size(), 2n));
        $(Assert.equal(result.Y_splits.get(2n).size(), 2n));
        // No stratify, so no rejections
        $(Assert.equal(result.rejected_indices.size(), 0n));
    });

    test("split with stratify ensures all classes in each split", $ => {
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

        // Stratify by class (now as array of arrays)
        const stratify_labels = $.let([[0n, 0n, 0n, 0n, 1n, 1n, 1n, 1n, 2n, 2n, 2n, 2n]]);

        const config = $.let({
            split_sizes: [0.5, 0.25, 0.25],
            random_state: variant('some', 42n),
            shuffle: variant('some', true),
            stratify: variant('some', stratify_labels),
            min_overlap: variant('none', null),
            overlap: variant('none', null),
        });

        const result = $.let(Sklearn.split(X, Y, config));

        // 50% train (6), 25% val (3), 25% test (3)
        $(Assert.equal(result.X_splits.get(0n).size(), 6n));
        $(Assert.equal(result.X_splits.get(1n).size(), 3n));
        $(Assert.equal(result.X_splits.get(2n).size(), 3n));

        // With stratification, each split should have representation from all classes
        // No rejections since all classes have 4 samples (>= 3 default)
        $(Assert.equal(result.rejected_indices.size(), 0n));
    });

    test("split 3-way filters rare stratify classes", $ => {
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

        const stratify_labels = $.let([[0n, 0n, 0n, 0n, 1n, 1n, 1n, 1n, 2n, 2n]]);

        const config = $.let({
            split_sizes: [0.5, 0.25, 0.25],
            random_state: variant('some', 42n),
            shuffle: variant('some', true),
            stratify: variant('some', stratify_labels),
            min_overlap: variant('none', null),  // default 3 for 3-way
            overlap: variant('none', null),
        });

        const result = $.let(Sklearn.split(X, Y, config));

        // Class 2 (indices 8,9) should be rejected
        $(Assert.equal(result.rejected_indices.size(), 2n));

        // Only 8 samples remain (4 from class 0, 4 from class 1)
        const total = $.let(result.X_splits.get(0n).size()
            .add(result.X_splits.get(1n).size())
            .add(result.X_splits.get(2n).size()));
        $(Assert.equal(total, 8n));

        // Verify remaining samples don't have class 2 features
        // Class 2 samples had X[:,1] = 3.0, so all remaining should have X[:,1] < 3.0
        const train_no_class2 = $.let(result.X_splits.get(0n).every(($, row) => row.get(1n).lessThan(2.5)));
        const val_no_class2 = $.let(result.X_splits.get(1n).every(($, row) => row.get(1n).lessThan(2.5)));
        const test_no_class2 = $.let(result.X_splits.get(2n).every(($, row) => row.get(1n).lessThan(2.5)));
        $(Assert.equal(train_no_class2, true));
        $(Assert.equal(val_no_class2, true));
        $(Assert.equal(test_no_class2, true));
    });

    test("split 3-way with custom min_overlap", $ => {
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

        const stratify_labels = $.let([[0n, 0n, 0n, 0n, 0n, 0n, 1n, 1n, 1n, 1n, 1n, 2n, 2n, 2n]]);

        // Require minimum 4 samples per class - class 2 should be rejected
        const config = $.let({
            split_sizes: [0.6, 0.2, 0.2],
            random_state: variant('some', 42n),
            shuffle: variant('some', true),
            stratify: variant('some', stratify_labels),
            min_overlap: variant('some', 4n),  // custom: need 4+
            overlap: variant('none', null),
        });

        const result = $.let(Sklearn.split(X, Y, config));

        // Class 2 (indices 11,12,13) should be rejected
        $(Assert.equal(result.rejected_indices.size(), 3n));

        // Only 11 samples remain (6 from class 0, 5 from class 1)
        const total = $.let(result.X_splits.get(0n).size()
            .add(result.X_splits.get(1n).size())
            .add(result.X_splits.get(2n).size()));
        $(Assert.equal(total, 11n));

        // Verify remaining samples don't have class 2 features
        // Class 2 samples had X[:,1] = 3.0, so all remaining should have X[:,1] < 3.0
        const train_no_class2 = $.let(result.X_splits.get(0n).every(($, row) => row.get(1n).lessThan(2.5)));
        const val_no_class2 = $.let(result.X_splits.get(1n).every(($, row) => row.get(1n).lessThan(2.5)));
        const test_no_class2 = $.let(result.X_splits.get(2n).every(($, row) => row.get(1n).lessThan(2.5)));
        $(Assert.equal(train_no_class2, true));
        $(Assert.equal(val_no_class2, true));
        $(Assert.equal(test_no_class2, true));
    });

    test("split creates 4-way split", $ => {
        // 20 samples for train/val/calib/test
        const X = $.let([
            [1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0], [10.0],
            [11.0], [12.0], [13.0], [14.0], [15.0], [16.0], [17.0], [18.0], [19.0], [20.0],
        ]);
        const Y = $.let([
            [1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0], [10.0],
            [11.0], [12.0], [13.0], [14.0], [15.0], [16.0], [17.0], [18.0], [19.0], [20.0],
        ]);

        const config = $.let({
            split_sizes: [0.5, 0.2, 0.15, 0.15],  // train/val/calib/test
            random_state: variant('some', 42n),
            shuffle: variant('some', true),
            stratify: variant('none', null),
            min_overlap: variant('none', null),
            overlap: variant('none', null),
        });

        const result = $.let(Sklearn.split(X, Y, config));

        // Should have 4 splits
        $(Assert.equal(result.X_splits.size(), 4n));
        $(Assert.equal(result.Y_splits.size(), 4n));

        // Verify approximate sizes: 10, 4, 3, 3
        $(Assert.equal(result.X_splits.get(0n).size(), 10n));
        $(Assert.equal(result.X_splits.get(1n).size(), 4n));
        $(Assert.equal(result.X_splits.get(2n).size(), 3n));
        $(Assert.equal(result.X_splits.get(3n).size(), 3n));
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
                gamma: variant('none', null),
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
                gamma: variant('none', null),
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

    test("error: split shape mismatch", $ => {
        const X = $.let([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);  // 3 samples
        const Y = $.let([[1.0], [2.0]]);  // 2 samples

        const config = $.let({
            split_sizes: [0.8, 0.2],
            random_state: variant('none', null),
            shuffle: variant('none', null),
            stratify: variant('none', null),
            min_overlap: variant('none', null),
            overlap: variant('none', null),
        });

        $(Assert.throws(Sklearn.split(X, Y, config), /sklearn_split.*X has 3 samples.*Y has 2 samples/));
    });

    test("split post-split validation rejects classes missing from a split", $ => {
        // 8 samples with 2 classes:
        // Class 0: 6 samples (plenty)
        // Class 1: 2 samples (exactly min_overlap=2, but may not appear in both splits)
        const X = $.let([
            [1.0, 1.0], [2.0, 1.0], [3.0, 1.0], [4.0, 1.0], [5.0, 1.0], [6.0, 1.0],  // class 0 (0-5)
            [1.0, 2.0], [2.0, 2.0],  // class 1 (6-7) - edge case
        ]);
        const Y = $.let([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0]]);
        const stratify_labels = $.let([[0n, 0n, 0n, 0n, 0n, 0n, 1n, 1n]]);

        const config = $.let({
            split_sizes: [0.75, 0.25],
            random_state: variant('some', 123n),
            shuffle: variant('some', true),
            stratify: variant('some', stratify_labels),
            min_overlap: variant('some', 2n),
            overlap: variant('none', null),
        });

        const result = $.let(Sklearn.split(X, Y, config));

        // Either all 8 samples remain (class 1 was in both splits)
        // Or only 6 samples remain (class 1 was rejected)
        // The key is: we should never have class 1 in only one split
        const train_has_class1 = $.let(result.Y_splits.get(0n).some(($, row) => row.get(0n).greaterThan(0.5)));
        const test_has_class1 = $.let(result.Y_splits.get(1n).some(($, row) => row.get(0n).greaterThan(0.5)));

        // If one split has class 1, both must have it (otherwise they'd be rejected)
        $(Assert.equal(train_has_class1, test_has_class1));
    });

    test("split 3-way post-split validation rejects classes missing from any split", $ => {
        // 11 samples with 2 classes:
        // Class 0: 8 samples (plenty for 3-way split)
        // Class 1: 3 samples (exactly min_overlap=3, edge case)
        const X = $.let([
            [1.0, 1.0], [2.0, 1.0], [3.0, 1.0], [4.0, 1.0],
            [5.0, 1.0], [6.0, 1.0], [7.0, 1.0], [8.0, 1.0],  // class 0 (0-7)
            [1.0, 2.0], [2.0, 2.0], [3.0, 2.0],  // class 1 (8-10) - edge case
        ]);
        const Y = $.let([
            [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0],
            [1.0], [1.0], [1.0],
        ]);
        const stratify_labels = $.let([[0n, 0n, 0n, 0n, 0n, 0n, 0n, 0n, 1n, 1n, 1n]]);

        const config = $.let({
            split_sizes: [0.7, 0.15, 0.15],
            random_state: variant('some', 7n),
            shuffle: variant('some', true),
            stratify: variant('some', stratify_labels),
            min_overlap: variant('some', 3n),
            overlap: variant('none', null),
        });

        const result = $.let(Sklearn.split(X, Y, config));

        // Check consistency: if class 1 appears in any split, it must appear in ALL splits
        const train_has_class1 = $.let(result.Y_splits.get(0n).some(($, row) => row.get(0n).greaterThan(0.5)));
        const val_has_class1 = $.let(result.Y_splits.get(1n).some(($, row) => row.get(0n).greaterThan(0.5)));
        const test_has_class1 = $.let(result.Y_splits.get(2n).some(($, row) => row.get(0n).greaterThan(0.5)));

        // All three must be equal (either all have class 1, or none have it)
        $(Assert.equal(train_has_class1, val_has_class1));
        $(Assert.equal(val_has_class1, test_has_class1));
    });

    test("split guarantees each split has all stratify classes", $ => {
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
        const stratify_labels = $.let([[0n, 0n, 0n, 0n, 0n, 0n, 0n, 0n, 0n, 1n, 1n, 1n, 1n, 1n, 1n]]);

        const config = $.let({
            split_sizes: [0.6, 0.2, 0.2],
            random_state: variant('some', 42n),
            shuffle: variant('some', true),
            stratify: variant('some', stratify_labels),
            min_overlap: variant('some', 3n),
            overlap: variant('none', null),
        });

        const result = $.let(Sklearn.split(X, Y, config));

        // With 6 samples in class 1 and 3-way split, both classes should appear in all splits
        $(Assert.equal(result.rejected_indices.size(), 0n));

        // Verify all splits have both classes
        const train_has_class0 = $.let(result.Y_splits.get(0n).some(($, row) => row.get(0n).lessThan(0.5)));
        const train_has_class1 = $.let(result.Y_splits.get(0n).some(($, row) => row.get(0n).greaterThan(0.5)));
        const val_has_class0 = $.let(result.Y_splits.get(1n).some(($, row) => row.get(0n).lessThan(0.5)));
        const val_has_class1 = $.let(result.Y_splits.get(1n).some(($, row) => row.get(0n).greaterThan(0.5)));
        const test_has_class0 = $.let(result.Y_splits.get(2n).some(($, row) => row.get(0n).lessThan(0.5)));
        const test_has_class1 = $.let(result.Y_splits.get(2n).some(($, row) => row.get(0n).greaterThan(0.5)));

        $(Assert.equal(train_has_class0, true));
        $(Assert.equal(train_has_class1, true));
        $(Assert.equal(val_has_class0, true));
        $(Assert.equal(val_has_class1, true));
        $(Assert.equal(test_has_class0, true));
        $(Assert.equal(test_has_class1, true));
    });

    test("label_encoder fit/transform/inverse_transform", $ => {
        // Labels with gaps: 2, 5, 2, 8, 5 -> encoded as 0, 1, 0, 2, 1
        const y = $.let([2n, 5n, 2n, 8n, 5n]);

        const model = $.let(Sklearn.labelEncoderFit(y));

        // Transform: 2->0, 5->1, 8->2
        const transformed = $.let(Sklearn.labelEncoderTransform(model, y));
        $(Assert.equal(transformed.get(0n), 0n));  // 2 -> 0
        $(Assert.equal(transformed.get(1n), 1n));  // 5 -> 1
        $(Assert.equal(transformed.get(2n), 0n));  // 2 -> 0
        $(Assert.equal(transformed.get(3n), 2n));  // 8 -> 2
        $(Assert.equal(transformed.get(4n), 1n));  // 5 -> 1

        // Inverse transform back to original
        const inverse = $.let(Sklearn.labelEncoderInverseTransform(model, transformed));
        $(Assert.equal(inverse.get(0n), 2n));
        $(Assert.equal(inverse.get(1n), 5n));
        $(Assert.equal(inverse.get(2n), 2n));
        $(Assert.equal(inverse.get(3n), 8n));
        $(Assert.equal(inverse.get(4n), 5n));
    });

    test("ordinal_encoder fit/transform", $ => {
        // Each column represents a categorical feature encoded as floats
        // Column 0: categories 1.0, 2.0, 3.0 (encoded 0, 1, 2)
        // Column 1: categories 10.0, 20.0 (encoded 0, 1)
        const X = $.let([
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 10.0],
            [1.0, 20.0],
        ]);

        const model = $.let(Sklearn.ordinalEncoderFit(X));

        const transformed = $.let(Sklearn.ordinalEncoderTransform(model, X));

        // Check shape is preserved
        $(Assert.equal(transformed.size(), 4n));

        // Check column 0: 1.0->0.0, 2.0->1.0, 3.0->2.0
        $(Assert.equal(transformed.get(0n).get(0n), 0.0));
        $(Assert.equal(transformed.get(1n).get(0n), 1.0));
        $(Assert.equal(transformed.get(2n).get(0n), 2.0));
        $(Assert.equal(transformed.get(3n).get(0n), 0.0));

        // Check column 1: 10.0->0.0, 20.0->1.0
        $(Assert.equal(transformed.get(0n).get(1n), 0.0));
        $(Assert.equal(transformed.get(1n).get(1n), 1.0));
        $(Assert.equal(transformed.get(2n).get(1n), 0.0));
        $(Assert.equal(transformed.get(3n).get(1n), 1.0));
    });
}, { exportOnly: true });
