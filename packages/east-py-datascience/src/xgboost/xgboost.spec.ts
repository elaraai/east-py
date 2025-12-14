/**
 * Copyright (c) 2025 Elara AI Pty Ltd
 * Dual-licensed under AGPL-3.0 and commercial license. See LICENSE for details.
 */

/**
 * XGBoost platform function tests
 */
import { East, variant } from "@elaraai/east";
import { describeEast, Assert } from "@elaraai/east-node-std";
import { XGBoost } from "./xgboost.js";

describeEast("XGBoost platform functions", (test) => {
    test("train_regressor and predict works", $ => {
        // Simple linear data: y = x1 + x2
        const X = $.let([
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
            [5.0, 5.0],
        ]);
        const y = $.let([2.0, 4.0, 6.0, 8.0, 10.0]);

        const config = $.let({
            n_estimators: variant('some', 100n),
            max_depth: variant('some', 3n),
            learning_rate: variant('some', 0.3),
            min_child_weight: variant('none', null),
            subsample: variant('none', null),
            colsample_bytree: variant('none', null),
            reg_alpha: variant('none', null),
            reg_lambda: variant('none', null),
            random_state: variant('some', 42n),
            n_jobs: variant('none', null),
            sample_weight: variant('none', null),
        });

        // Train model
        const model = $.let(XGBoost.trainRegressor(X, y, config));

        // Predict on training data
        const y_pred = $.let(XGBoost.predict(model, X));

        // Check dimensions
        $(Assert.equal(y_pred.size(), 5n));

        // Check predictions are close to actual values (within 1.0)
        $(Assert.less(y_pred.get(0n).subtract(y.get(0n)).abs(), East.value(1.0)));
        $(Assert.less(y_pred.get(2n).subtract(y.get(2n)).abs(), East.value(1.0)));
        $(Assert.less(y_pred.get(4n).subtract(y.get(4n)).abs(), East.value(1.0)));
    });

    test("train_classifier and predict_class works", $ => {
        // Binary classification data - well-separated clusters (need enough data for XGBoost)
        const X = $.let([
            [0.0, 0.0],
            [0.5, 0.5],
            [1.0, 1.0],
            [1.5, 1.5],
            [2.0, 2.0],
            [10.0, 10.0],
            [10.5, 10.5],
            [11.0, 11.0],
            [11.5, 11.5],
            [12.0, 12.0],
        ]);
        const y = $.let([0n, 0n, 0n, 0n, 0n, 1n, 1n, 1n, 1n, 1n]);

        const config = $.let({
            n_estimators: variant('some', 100n),
            max_depth: variant('some', 4n),
            learning_rate: variant('some', 0.3),
            min_child_weight: variant('none', null),
            subsample: variant('none', null),
            colsample_bytree: variant('none', null),
            reg_alpha: variant('none', null),
            reg_lambda: variant('none', null),
            random_state: variant('some', 42n),
            n_jobs: variant('none', null),
            sample_weight: variant('none', null),
        });

        // Train model
        const model = $.let(XGBoost.trainClassifier(X, y, config));

        // Predict on training data
        const y_pred = $.let(XGBoost.predictClass(model, X));

        // Check dimensions
        $(Assert.equal(y_pred.size(), 10n));

        // Check predictions are correct (data is well-separated)
        $(Assert.equal(y_pred.get(0n), 0n));  // First cluster should be class 0
        $(Assert.equal(y_pred.get(2n), 0n));
        $(Assert.equal(y_pred.get(5n), 1n));  // Second cluster should be class 1
        $(Assert.equal(y_pred.get(9n), 1n));
    });

    test("predict_proba returns probability matrix", $ => {
        // Binary classification data - well separated (need enough data for XGBoost)
        const X = $.let([
            [0.0, 0.0],
            [0.5, 0.5],
            [1.0, 1.0],
            [1.5, 1.5],
            [2.0, 2.0],
            [10.0, 10.0],
            [10.5, 10.5],
            [11.0, 11.0],
            [11.5, 11.5],
            [12.0, 12.0],
        ]);
        const y = $.let([0n, 0n, 0n, 0n, 0n, 1n, 1n, 1n, 1n, 1n]);

        const config = $.let({
            n_estimators: variant('some', 100n),
            max_depth: variant('some', 4n),
            learning_rate: variant('some', 0.3),
            min_child_weight: variant('none', null),
            subsample: variant('none', null),
            colsample_bytree: variant('none', null),
            reg_alpha: variant('none', null),
            reg_lambda: variant('none', null),
            random_state: variant('some', 42n),
            n_jobs: variant('none', null),
            sample_weight: variant('none', null),
        });

        // Train model
        const model = $.let(XGBoost.trainClassifier(X, y, config));

        // Get probabilities
        const proba = $.let(XGBoost.predictProba(model, X));

        // Check dimensions: 10 samples x 2 classes
        $(Assert.equal(proba.size(), 10n));
        $(Assert.equal(proba.get(0n).size(), 2n));

        // First sample (class 0) should have high prob for class 0
        $(Assert.greater(proba.get(0n).get(0n), East.value(0.7)));
        // Sixth sample (class 1) should have high prob for class 1
        $(Assert.greater(proba.get(5n).get(1n), East.value(0.7)));

        // Probabilities should sum to 1 (within tolerance)
        const sum0 = $.let(proba.get(0n).get(0n).add(proba.get(0n).get(1n)));
        $(Assert.greater(sum0, East.value(0.99)));
        $(Assert.less(sum0, East.value(1.01)));
    });

    test("respects random_state for reproducibility", $ => {
        const X = $.let([
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
        ]);
        const y = $.let([1.0, 2.0, 3.0, 4.0]);

        const config = $.let({
            n_estimators: variant('some', 10n),
            max_depth: variant('some', 2n),
            learning_rate: variant('some', 0.1),
            min_child_weight: variant('none', null),
            subsample: variant('none', null),
            colsample_bytree: variant('none', null),
            reg_alpha: variant('none', null),
            reg_lambda: variant('none', null),
            random_state: variant('some', 123n),
            n_jobs: variant('none', null),
            sample_weight: variant('none', null),
        });

        // Train two models with same seed
        const model1 = $.let(XGBoost.trainRegressor(X, y, config));
        const model2 = $.let(XGBoost.trainRegressor(X, y, config));

        // Predictions should be identical
        const pred1 = $.let(XGBoost.predict(model1, X));
        const pred2 = $.let(XGBoost.predict(model2, X));

        $(Assert.equal(pred1.get(0n), pred2.get(0n)));
        $(Assert.equal(pred1.get(1n), pred2.get(1n)));
    });

    test("error: train_regressor shape mismatch", $ => {
        const X = $.let([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);  // 3 samples
        const y = $.let([1.0, 2.0]);  // 2 samples

        const config = $.let({
            n_estimators: variant('none', null),
            max_depth: variant('none', null),
            learning_rate: variant('none', null),
            min_child_weight: variant('none', null),
            subsample: variant('none', null),
            colsample_bytree: variant('none', null),
            reg_alpha: variant('none', null),
            reg_lambda: variant('none', null),
            random_state: variant('none', null),
            n_jobs: variant('none', null),
            sample_weight: variant('none', null),
        });

        $(Assert.throws(XGBoost.trainRegressor(X, y, config), /xgboost_train_regressor.*X has 3 samples.*y has 2 samples/));
    });

    test("error: train_classifier shape mismatch", $ => {
        const X = $.let([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);  // 3 samples
        const y = $.let([0n, 1n]);  // 2 samples

        const config = $.let({
            n_estimators: variant('none', null),
            max_depth: variant('none', null),
            learning_rate: variant('none', null),
            min_child_weight: variant('none', null),
            subsample: variant('none', null),
            colsample_bytree: variant('none', null),
            reg_alpha: variant('none', null),
            reg_lambda: variant('none', null),
            random_state: variant('none', null),
            n_jobs: variant('none', null),
            sample_weight: variant('none', null),
        });

        $(Assert.throws(XGBoost.trainClassifier(X, y, config), /xgboost_train_classifier.*X has 3 samples.*y has 2 samples/));
    });

    test("error: predict with wrong model type", $ => {
        // Train a classifier but try to use it with regressor predict
        const X = $.let([[0.0, 0.0], [1.0, 1.0], [10.0, 10.0], [11.0, 11.0]]);
        const y = $.let([0n, 0n, 1n, 1n]);

        const config = $.let({
            n_estimators: variant('some', 10n),
            max_depth: variant('some', 2n),
            learning_rate: variant('some', 0.1),
            min_child_weight: variant('none', null),
            subsample: variant('none', null),
            colsample_bytree: variant('none', null),
            reg_alpha: variant('none', null),
            reg_lambda: variant('none', null),
            random_state: variant('some', 42n),
            n_jobs: variant('none', null),
            sample_weight: variant('none', null),
        });

        const classifierModel = $.let(XGBoost.trainClassifier(X, y, config));
        $(Assert.throws(XGBoost.predict(classifierModel, X), /xgboost_predict.*Expected xgboost_regressor.*got xgboost_classifier/));
    });

    test("error: predict_class with wrong model type", $ => {
        // Train a regressor but try to use it with classifier predict
        const X = $.let([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]);
        const y = $.let([2.0, 4.0, 6.0, 8.0]);

        const config = $.let({
            n_estimators: variant('some', 10n),
            max_depth: variant('some', 2n),
            learning_rate: variant('some', 0.1),
            min_child_weight: variant('none', null),
            subsample: variant('none', null),
            colsample_bytree: variant('none', null),
            reg_alpha: variant('none', null),
            reg_lambda: variant('none', null),
            random_state: variant('some', 42n),
            n_jobs: variant('none', null),
            sample_weight: variant('none', null),
        });

        const regressorModel = $.let(XGBoost.trainRegressor(X, y, config));
        $(Assert.throws(XGBoost.predictClass(regressorModel, X), /xgboost_predict_class.*Expected xgboost_classifier.*got xgboost_regressor/));
    });

    test("error: predict_proba with wrong model type", $ => {
        // Train a regressor but try to use it with predict_proba
        const X = $.let([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]);
        const y = $.let([2.0, 4.0, 6.0, 8.0]);

        const config = $.let({
            n_estimators: variant('some', 10n),
            max_depth: variant('some', 2n),
            learning_rate: variant('some', 0.1),
            min_child_weight: variant('none', null),
            subsample: variant('none', null),
            colsample_bytree: variant('none', null),
            reg_alpha: variant('none', null),
            reg_lambda: variant('none', null),
            random_state: variant('some', 42n),
            n_jobs: variant('none', null),
            sample_weight: variant('none', null),
        });

        const regressorModel = $.let(XGBoost.trainRegressor(X, y, config));
        $(Assert.throws(XGBoost.predictProba(regressorModel, X), /xgboost_predict_proba.*Expected xgboost_classifier.*got xgboost_regressor/));
    });

    test("train_quantile and predict_quantile works", $ => {
        // Linear data with noise for quantile regression
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
        const y = $.let([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]);

        const config = $.let({
            quantiles: [0.1, 0.5, 0.9],  // 80% prediction interval + median
            n_estimators: variant('some', 50n),
            max_depth: variant('some', 3n),
            learning_rate: variant('some', 0.3),
            min_child_weight: variant('none', null),
            subsample: variant('none', null),
            colsample_bytree: variant('none', null),
            reg_alpha: variant('none', null),
            reg_lambda: variant('none', null),
            random_state: variant('some', 42n),
            n_jobs: variant('none', null),
            sample_weight: variant('none', null),
        });

        // Train quantile model
        const model = $.let(XGBoost.trainQuantile(X, y, config));

        // Predict quantiles
        const result = $.let(XGBoost.predictQuantile(model, X));

        // Check result structure
        $(Assert.equal(result.quantiles.size(), 3n));  // 3 quantiles
        $(Assert.equal(result.predictions.size(), 8n));  // 8 samples
        $(Assert.equal(result.predictions.get(0n).size(), 3n));  // 3 quantiles per sample

        // Check quantile values are returned correctly
        $(Assert.less(result.quantiles.get(0n).subtract(East.value(0.1)).abs(), East.value(0.01)));
        $(Assert.less(result.quantiles.get(1n).subtract(East.value(0.5)).abs(), East.value(0.01)));
        $(Assert.less(result.quantiles.get(2n).subtract(East.value(0.9)).abs(), East.value(0.01)));
    });

    test("quantile predictions maintain ordering (lower <= median <= upper)", $ => {
        // Data for quantile regression
        const X = $.let([
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
            [5.0, 5.0],
            [6.0, 6.0],
        ]);
        const y = $.let([2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);

        const config = $.let({
            quantiles: [0.1, 0.5, 0.9],
            n_estimators: variant('some', 50n),
            max_depth: variant('some', 3n),
            learning_rate: variant('some', 0.3),
            min_child_weight: variant('none', null),
            subsample: variant('none', null),
            colsample_bytree: variant('none', null),
            reg_alpha: variant('none', null),
            reg_lambda: variant('none', null),
            random_state: variant('some', 42n),
            n_jobs: variant('none', null),
            sample_weight: variant('none', null),
        });

        const model = $.let(XGBoost.trainQuantile(X, y, config));
        const result = $.let(XGBoost.predictQuantile(model, X));

        // Check first sample: q0.1 <= q0.5 <= q0.9
        const q10_0 = $.let(result.predictions.get(0n).get(0n));
        const q50_0 = $.let(result.predictions.get(0n).get(1n));
        const q90_0 = $.let(result.predictions.get(0n).get(2n));

        $(Assert.lessEqual(q10_0, q50_0));
        $(Assert.lessEqual(q50_0, q90_0));

        // Check last sample
        const q10_5 = $.let(result.predictions.get(5n).get(0n));
        const q50_5 = $.let(result.predictions.get(5n).get(1n));
        const q90_5 = $.let(result.predictions.get(5n).get(2n));

        $(Assert.lessEqual(q10_5, q50_5));
        $(Assert.lessEqual(q50_5, q90_5));
    });

    test("error: predict_quantile with wrong model type", $ => {
        // Train a regressor but try to use it with quantile predict
        const X = $.let([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]);
        const y = $.let([2.0, 4.0, 6.0, 8.0]);

        const config = $.let({
            n_estimators: variant('some', 10n),
            max_depth: variant('some', 2n),
            learning_rate: variant('some', 0.1),
            min_child_weight: variant('none', null),
            subsample: variant('none', null),
            colsample_bytree: variant('none', null),
            reg_alpha: variant('none', null),
            reg_lambda: variant('none', null),
            random_state: variant('some', 42n),
            n_jobs: variant('none', null),
            sample_weight: variant('none', null),
        });

        const regressorModel = $.let(XGBoost.trainRegressor(X, y, config));
        $(Assert.throws(XGBoost.predictQuantile(regressorModel, X), /xgboost_predict_quantile.*Expected xgboost_quantile.*got xgboost_regressor/));
    });

    test("error: train_quantile with invalid quantiles", $ => {
        const X = $.let([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]);
        const y = $.let([2.0, 4.0, 6.0, 8.0]);

        // Quantile outside valid range (must be in (0, 1))
        const config = $.let({
            quantiles: [0.0, 0.5, 1.0],  // 0.0 and 1.0 are invalid
            n_estimators: variant('some', 10n),
            max_depth: variant('none', null),
            learning_rate: variant('none', null),
            min_child_weight: variant('none', null),
            subsample: variant('none', null),
            colsample_bytree: variant('none', null),
            reg_alpha: variant('none', null),
            reg_lambda: variant('none', null),
            random_state: variant('none', null),
            n_jobs: variant('none', null),
            sample_weight: variant('none', null),
        });

        $(Assert.throws(XGBoost.trainQuantile(X, y, config), /xgboost_train_quantile.*Quantiles must be in \(0, 1\)/));
    });

    test("error: train_quantile shape mismatch", $ => {
        const X = $.let([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);  // 3 samples
        const y = $.let([1.0, 2.0]);  // 2 samples

        const config = $.let({
            quantiles: [0.1, 0.5, 0.9],
            n_estimators: variant('none', null),
            max_depth: variant('none', null),
            learning_rate: variant('none', null),
            min_child_weight: variant('none', null),
            subsample: variant('none', null),
            colsample_bytree: variant('none', null),
            reg_alpha: variant('none', null),
            reg_lambda: variant('none', null),
            random_state: variant('none', null),
            n_jobs: variant('none', null),
            sample_weight: variant('none', null),
        });

        $(Assert.throws(XGBoost.trainQuantile(X, y, config), /xgboost_train_quantile.*X has 3 samples.*y has 2 samples/));
    });

    test("train_regressor with sample_weight works", $ => {
        // Data where first samples are more important
        const X = $.let([
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
            [5.0, 5.0],
        ]);
        const y = $.let([2.0, 4.0, 6.0, 8.0, 10.0]);
        // Give higher weight to first samples
        const weights = $.let([10.0, 10.0, 1.0, 1.0, 1.0]);

        const config = $.let({
            n_estimators: variant('some', 50n),
            max_depth: variant('some', 3n),
            learning_rate: variant('some', 0.3),
            min_child_weight: variant('none', null),
            subsample: variant('none', null),
            colsample_bytree: variant('none', null),
            reg_alpha: variant('none', null),
            reg_lambda: variant('none', null),
            random_state: variant('some', 42n),
            n_jobs: variant('none', null),
            sample_weight: variant('some', weights),
        });

        // Train model with sample weights
        const model = $.let(XGBoost.trainRegressor(X, y, config));

        // Predict on training data - should work
        const y_pred = $.let(XGBoost.predict(model, X));

        // Check dimensions
        $(Assert.equal(y_pred.size(), 5n));
    });

    test("error: sample_weight length mismatch", $ => {
        const X = $.let([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]);  // 4 samples
        const y = $.let([1.0, 2.0, 3.0, 4.0]);
        const weights = $.let([1.0, 1.0]);  // Only 2 weights

        const config = $.let({
            n_estimators: variant('some', 10n),
            max_depth: variant('none', null),
            learning_rate: variant('none', null),
            min_child_weight: variant('none', null),
            subsample: variant('none', null),
            colsample_bytree: variant('none', null),
            reg_alpha: variant('none', null),
            reg_lambda: variant('none', null),
            random_state: variant('none', null),
            n_jobs: variant('none', null),
            sample_weight: variant('some', weights),
        });

        $(Assert.throws(XGBoost.trainRegressor(X, y, config), /xgboost_train_regressor.*sample_weight has 2 elements.*X has 4 samples/));
    });
}, { exportOnly: true });
