/**
 * Copyright (c) 2025 Elara AI Pty Ltd
 * Dual-licensed under AGPL-3.0 and commercial license. See LICENSE for details.
 */

/**
 * SHAP platform function tests
 */
import { East, variant } from "@elaraai/east";
import { describeEast, Assert } from "@elaraai/east-node-std";
import { Shap } from "./shap.js";
import { LightGBM } from "../lightgbm/lightgbm.js";
import { XGBoost } from "../xgboost/xgboost.js";
import { NGBoost } from "../ngboost/ngboost.js";
import { GP } from "../gp/gp.js";
import { Torch } from "../torch/torch.js";
import { Sklearn } from "../sklearn/sklearn.js";

describeEast("SHAP platform functions", (test) => {
    test("tree_explainer works with LightGBM regressor", $ => {
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

        const config = $.let({
            n_estimators: variant('some', 50n),
            max_depth: variant('some', 3n),
            learning_rate: variant('some', 0.1),
            num_leaves: variant('some', 31n),
            min_child_samples: variant('some', 1n),
            subsample: variant('none', null),
            colsample_bytree: variant('none', null),
            reg_alpha: variant('none', null),
            reg_lambda: variant('none', null),
            random_state: variant('some', 42n),
            n_jobs: variant('none', null),
        });

        const model = $.let(LightGBM.trainRegressor(X, y, config));
        const explainer = $.let(Shap.treeExplainerCreate(model));
        const feature_names = $.let(["feature1", "feature2"]);
        const result = $.let(Shap.computeValues(explainer, X, feature_names));

        // Regression returns matrix_2d variant
        $.match(result.shap_values, {
            matrix_2d: ($, shap_matrix) => {
                $(Assert.equal(shap_matrix.size(), 8n));
                $(Assert.equal(shap_matrix.get(0n).size(), 2n));
            },
            tensor_3d: () => $(Assert.fail("Expected matrix_2d for regression")),
        });
        $(Assert.equal(result.feature_names.size(), 2n));
    });

    test("tree_explainer works with LightGBM classifier", $ => {
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
        const y = $.let([0n, 0n, 0n, 0n, 1n, 1n, 1n, 1n]);

        const config = $.let({
            n_estimators: variant('some', 50n),
            max_depth: variant('some', 3n),
            learning_rate: variant('some', 0.1),
            num_leaves: variant('some', 31n),
            min_child_samples: variant('some', 1n),
            subsample: variant('none', null),
            colsample_bytree: variant('none', null),
            reg_alpha: variant('none', null),
            reg_lambda: variant('none', null),
            random_state: variant('some', 42n),
            n_jobs: variant('none', null),
        });

        const model = $.let(LightGBM.trainClassifier(X, y, config));
        const explainer = $.let(Shap.treeExplainerCreate(model));
        const feature_names = $.let(["feature1", "feature2"]);
        const result = $.let(Shap.computeValues(explainer, X, feature_names));

        // Binary classification returns matrix_2d variant
        $.match(result.shap_values, {
            matrix_2d: ($, shap_matrix) => {
                $(Assert.equal(shap_matrix.size(), 8n));
                $(Assert.equal(shap_matrix.get(0n).size(), 2n));
            },
            tensor_3d: ($) => $(Assert.fail("Expected matrix_2d for binary classification")),
        });
    });

    test("tree_explainer works with XGBoost regressor", $ => {
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

        const config = $.let({
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
        });

        const model = $.let(XGBoost.trainRegressor(X, y, config));
        const explainer = $.let(Shap.treeExplainerCreate(model));
        const feature_names = $.let(["feature1", "feature2"]);
        const result = $.let(Shap.computeValues(explainer, X, feature_names));

        // Regression returns matrix_2d variant
        $.match(result.shap_values, {
            matrix_2d: ($, shap_matrix) => {
                $(Assert.equal(shap_matrix.size(), 8n));
                $(Assert.equal(shap_matrix.get(0n).size(), 2n));
            },
            tensor_3d: ($) => $(Assert.fail("Expected matrix_2d for regression")),
        });
    });

    test("tree_explainer works with XGBoost classifier", $ => {
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
        const y = $.let([0n, 0n, 0n, 0n, 1n, 1n, 1n, 1n]);

        const config = $.let({
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
        });

        const model = $.let(XGBoost.trainClassifier(X, y, config));
        const explainer = $.let(Shap.treeExplainerCreate(model));
        const feature_names = $.let(["feature1", "feature2"]);
        const result = $.let(Shap.computeValues(explainer, X, feature_names));

        // Binary classification returns matrix_2d variant
        $.match(result.shap_values, {
            matrix_2d: ($, shap_matrix) => {
                $(Assert.equal(shap_matrix.size(), 8n));
                $(Assert.equal(shap_matrix.get(0n).size(), 2n));
            },
            tensor_3d: ($) => $(Assert.fail("Expected matrix_2d for binary classification")),
        });
    });

    test("tree_explainer works with XGBoost quantile regressor", $ => {
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

        const config = $.let({
            quantiles: [0.1, 0.5, 0.9],
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
        });

        // Train quantile model
        const model = $.let(XGBoost.trainQuantile(X, y, config));
        // TreeExplainer uses the median (0.5) quantile model
        const explainer = $.let(Shap.treeExplainerCreate(model));
        const feature_names = $.let(["feature1", "feature2"]);
        const result = $.let(Shap.computeValues(explainer, X, feature_names));

        // Quantile regression returns matrix_2d variant
        $.match(result.shap_values, {
            matrix_2d: ($, shap_matrix) => {
                $(Assert.equal(shap_matrix.size(), 8n));
                $(Assert.equal(shap_matrix.get(0n).size(), 2n));
            },
            tensor_3d: ($) => $(Assert.fail("Expected matrix_2d for quantile regression")),
        });
    });

    test("feature_importance computes mean absolute SHAP", $ => {
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

        const config = $.let({
            n_estimators: variant('some', 50n),
            max_depth: variant('some', 3n),
            learning_rate: variant('some', 0.1),
            num_leaves: variant('some', 31n),
            min_child_samples: variant('some', 1n),
            subsample: variant('none', null),
            colsample_bytree: variant('none', null),
            reg_alpha: variant('none', null),
            reg_lambda: variant('none', null),
            random_state: variant('some', 42n),
            n_jobs: variant('none', null),
        });

        const model = $.let(LightGBM.trainRegressor(X, y, config));
        const explainer = $.let(Shap.treeExplainerCreate(model));
        const feature_names = $.let(["feature1", "feature2"]);
        const shap_result = $.let(Shap.computeValues(explainer, X, feature_names));
        const importance = $.let(Shap.featureImportance(shap_result.shap_values, feature_names));

        $(Assert.equal(importance.feature_names.size(), 2n));
        $(Assert.equal(importance.importances.size(), 2n));
        $(Assert.greaterEqual(importance.importances.get(0n), East.value(0.0)));
        $(Assert.greaterEqual(importance.importances.get(1n), East.value(0.0)));
    });

    test("kernel_explainer works with NGBoost regressor", $ => {
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

        const config = $.let({
            n_estimators: variant('some', 50n),
            learning_rate: variant('some', 0.1),
            minibatch_frac: variant('none', null),
            col_sample: variant('none', null),
            random_state: variant('some', 42n),
            distribution: variant('none', null),
        });

        const model = $.let(NGBoost.trainRegressor(X, y, config));
        // Use subset of data as background for KernelExplainer
        const X_background = $.let([
            [1.0, 2.0],
            [4.0, 5.0],
            [8.0, 9.0],
        ]);
        const explainer = $.let(Shap.kernelExplainerCreate(model, X_background));
        const feature_names = $.let(["feature1", "feature2"]);
        // Explain just 2 samples to keep test fast
        const X_explain = $.let([
            [2.0, 3.0],
            [5.0, 6.0],
        ]);
        const result = $.let(Shap.computeValues(explainer, X_explain, feature_names));

        // Regression returns matrix_2d variant
        $.match(result.shap_values, {
            matrix_2d: ($, shap_matrix) => {
                $(Assert.equal(shap_matrix.size(), 2n));
                $(Assert.equal(shap_matrix.get(0n).size(), 2n));
            },
            tensor_3d: ($) => $(Assert.fail("Expected matrix_2d for regression")),
        });
    });

    test("kernel_explainer works with GP regressor", $ => {
        const X = $.let([
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0],
        ]);
        const y = $.let([3.0, 5.0, 7.0, 9.0, 11.0]);

        const config = $.let({
            kernel: variant('some', variant('rbf', null)),
            alpha: variant('some', 1e-10),
            n_restarts_optimizer: variant('some', 0n),
            normalize_y: variant('some', true),
            random_state: variant('some', 42n),
        });

        const model = $.let(GP.train(X, y, config));
        const X_background = $.let([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]);
        const explainer = $.let(Shap.kernelExplainerCreate(model, X_background));
        const feature_names = $.let(["feature1", "feature2"]);
        const X_explain = $.let([
            [2.0, 3.0],
            [4.0, 5.0],
        ]);
        const result = $.let(Shap.computeValues(explainer, X_explain, feature_names));

        // Regression returns matrix_2d variant
        $.match(result.shap_values, {
            matrix_2d: ($, shap_matrix) => {
                $(Assert.equal(shap_matrix.size(), 2n));
                $(Assert.equal(shap_matrix.get(0n).size(), 2n));
            },
            tensor_3d: ($) => $(Assert.fail("Expected matrix_2d for regression")),
        });
    });

    test("kernel_explainer works with Torch MLP", $ => {
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
            activation: variant('some', variant('relu', null)),
            output_activation: variant('none', null),
            dropout: variant('none', null),
            output_dim: variant('some', 1n),
            output_constraints: variant('none', null),
        });
        const train_config = $.let({
            epochs: variant('some', 50n),
            batch_size: variant('some', 4n),
            learning_rate: variant('some', 0.01),
            loss: variant('none', null),
            optimizer: variant('none', null),
            early_stopping: variant('none', null),
            validation_split: variant('none', null),
            random_state: variant('some', 42n),
            pos_weight: variant('none', null),
        });

        const train_result = $.let(Torch.mlpTrain(X, y, mlp_config, train_config));
        const model = $.let(train_result.model);
        const X_background = $.let([
            [1.0, 2.0],
            [4.0, 5.0],
            [8.0, 9.0],
        ]);
        const explainer = $.let(Shap.kernelExplainerCreate(model, X_background));
        const feature_names = $.let(["feature1", "feature2"]);
        const X_explain = $.let([
            [2.0, 3.0],
            [5.0, 6.0],
        ]);
        const result = $.let(Shap.computeValues(explainer, X_explain, feature_names));

        // Regression returns matrix_2d variant
        $.match(result.shap_values, {
            matrix_2d: ($, shap_matrix) => {
                $(Assert.equal(shap_matrix.size(), 2n));
                $(Assert.equal(shap_matrix.get(0n).size(), 2n));
            },
            tensor_3d: ($) => $(Assert.fail("Expected matrix_2d for regression")),
        });
    });

    test("tree_explainer works with XGBoost multi-class classifier", $ => {
        // Multi-class classification with 3 classes
        const X = $.let([
            [0.0, 0.0],
            [0.5, 0.5],
            [1.0, 1.0],
            [5.0, 5.0],
            [5.5, 5.5],
            [6.0, 6.0],
            [10.0, 10.0],
            [10.5, 10.5],
            [11.0, 11.0],
        ]);
        const y = $.let([0n, 0n, 0n, 1n, 1n, 1n, 2n, 2n, 2n]);

        const config = $.let({
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
        });

        const model = $.let(XGBoost.trainClassifier(X, y, config));
        const explainer = $.let(Shap.treeExplainerCreate(model));
        const feature_names = $.let(["feature1", "feature2"]);
        const result = $.let(Shap.computeValues(explainer, X, feature_names));

        // Multi-class (>2 classes) returns tensor_3d variant
        $.match(result.shap_values, {
            matrix_2d: ($) => $(Assert.fail("Expected tensor_3d for multi-class classification")),
            tensor_3d: ($, shap_tensor) => {
                // tensor_3d is list of (n_features, n_classes) matrices, one per sample
                $(Assert.equal(shap_tensor.size(), 9n));  // 9 samples
                $(Assert.equal(shap_tensor.get(0n).size(), 2n));  // 2 features
                $(Assert.equal(shap_tensor.get(0n).get(0n).size(), 3n));  // 3 classes
            },
        });

        // base_value should be per_class
        $.match(result.base_value, {
            single: ($) => $(Assert.fail("Expected per_class for multi-class classification")),
            per_class: ($, base_values) => {
                $(Assert.equal(base_values.size(), 3n));  // 3 classes
            },
        });
    });

    test("feature_importance works with multi-class tensor_3d", $ => {
        // Multi-class classification with 3 classes
        const X = $.let([
            [0.0, 0.0],
            [0.5, 0.5],
            [1.0, 1.0],
            [5.0, 5.0],
            [5.5, 5.5],
            [6.0, 6.0],
            [10.0, 10.0],
            [10.5, 10.5],
            [11.0, 11.0],
        ]);
        const y = $.let([0n, 0n, 0n, 1n, 1n, 1n, 2n, 2n, 2n]);

        const config = $.let({
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
        });

        const model = $.let(XGBoost.trainClassifier(X, y, config));
        const explainer = $.let(Shap.treeExplainerCreate(model));
        const feature_names = $.let(["feature1", "feature2"]);
        const shap_result = $.let(Shap.computeValues(explainer, X, feature_names));
        const importance = $.let(Shap.featureImportance(shap_result.shap_values, feature_names));

        // Feature importance aggregates across samples and classes
        $(Assert.equal(importance.feature_names.size(), 2n));
        $(Assert.equal(importance.importances.size(), 2n));
        $(Assert.greaterEqual(importance.importances.get(0n), East.value(0.0)));
        $(Assert.greaterEqual(importance.importances.get(1n), East.value(0.0)));
    });

    test("kernel_explainer works with RegressorChain", $ => {
        // Multi-target regression data
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
        // Multi-target: Y has 2 targets (columns)
        const Y = $.let([
            [3.0, 6.0],
            [5.0, 10.0],
            [7.0, 14.0],
            [9.0, 18.0],
            [11.0, 22.0],
            [13.0, 26.0],
            [15.0, 30.0],
            [17.0, 34.0],
        ]);

        // Configure RegressorChain with XGBoost base estimator
        const xgboost_config = $.let({
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
        });

        const chain_config = $.let({
            base_estimator: variant('xgboost', xgboost_config),
            order: variant('none', null),
            random_state: variant('some', 42n),
        });

        const model = $.let(Sklearn.regressorChainTrain(X, Y, chain_config));

        // Use subset of data as background for KernelExplainer
        const X_background = $.let([
            [1.0, 2.0],
            [4.0, 5.0],
            [8.0, 9.0],
        ]);
        const explainer = $.let(Shap.kernelExplainerCreate(model, X_background));
        const feature_names = $.let(["feature1", "feature2"]);

        // Explain just 2 samples to keep test fast
        const X_explain = $.let([
            [2.0, 3.0],
            [5.0, 6.0],
        ]);
        const result = $.let(Shap.computeValues(explainer, X_explain, feature_names));

        // RegressorChain returns first target's predictions, so SHAP gives matrix_2d
        $.match(result.shap_values, {
            matrix_2d: ($, shap_matrix) => {
                $(Assert.equal(shap_matrix.size(), 2n));
                $(Assert.equal(shap_matrix.get(0n).size(), 2n));
            },
            tensor_3d: ($) => $(Assert.fail("Expected matrix_2d for RegressorChain")),
        });
    });
}, { exportOnly: true });
