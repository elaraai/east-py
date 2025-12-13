/**
 * Copyright (c) 2025 Elara AI Pty Ltd
 * Dual-licensed under AGPL-3.0 and commercial license. See LICENSE for details.
 */

/**
 * Integration tests for east-py-datascience.
 *
 * These tests demonstrate the full ML pipeline:
 * 1. Train/test split
 * 2. Hyperparameter tuning with Optuna
 * 3. Model training
 * 4. Feature importance with SHAP
 */
import { ArrayType, East, FloatType, variant } from "@elaraai/east";
import { describeEast, Assert } from "@elaraai/east-node-std";
import {
    Sklearn,
    XGBoost,
    LightGBM,
    NGBoost,
    GP,
    Torch,
    Optuna,
    Shap,
    NamedParamType,
    ParamSpaceType,
} from "./index.js";

describeEast("Integration tests", (test) => {
    test("XGBoost: split -> optuna tune -> train -> shap", $ => {
        // Dataset: y = 2*x1 + 3*x2
        const X = $.let([
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0],
            [6.0, 7.0],
            [7.0, 8.0],
            [8.0, 9.0],
            [9.0, 10.0],
            [10.0, 11.0],
        ]);
        const y = $.let([8.0, 13.0, 18.0, 23.0, 28.0, 33.0, 38.0, 43.0, 48.0, 53.0]);

        // 1. Train/test split
        const splitConfig = $.let({
            test_size: variant('some', 0.3),
            random_state: variant('some', 42n),
            shuffle: variant('some', true),
        });
        const split = $.let(Sklearn.trainTestSplit(X, y, splitConfig));

        // 2. Hyperparameter tuning with Optuna
        const search_space = $.let([
            {
                name: "n_estimators",
                kind: variant("int", null),
                low: variant("some", 10.0),
                high: variant("some", 100.0),
                choices: variant("none", null),
            },
            {
                name: "max_depth",
                kind: variant("int", null),
                low: variant("some", 2.0),
                high: variant("some", 6.0),
                choices: variant("none", null),
            },
        ], ArrayType(ParamSpaceType));

        const objective = East.function(
            [ArrayType(NamedParamType)],
            FloatType,
            ($inner, params) => {
                const nEst = $inner.let(params.get(0n).value.unwrap('int'));
                const depth = $inner.let(params.get(1n).value.unwrap('int'));

                const config = $inner.let({
                    n_estimators: variant('some', nEst),
                    max_depth: variant('some', depth),
                    learning_rate: variant('some', 0.1),
                    min_child_weight: variant('none', null),
                    subsample: variant('none', null),
                    colsample_bytree: variant('none', null),
                    reg_alpha: variant('none', null),
                    reg_lambda: variant('none', null),
                    random_state: variant('some', 42n),
                    n_jobs: variant('none', null),
                });

                const model = $inner.let(XGBoost.trainRegressor(split.X_train, split.y_train, config));
                const preds = $inner.let(XGBoost.predict(model, split.X_test));

                // Compute MSE
                const mse = $inner.let(
                    split.y_test.reduce(($r, acc, actual, i) => {
                        const pred = preds.get(i);
                        const diff = actual.subtract(pred);
                        return acc.add(diff.multiply(diff));
                    }, 0.0).divide(split.y_test.size().toFloat())
                );

                return $inner.return(mse);
            }
        );

        const studyConfig = $.let({
            direction: variant('some', variant('minimize', null)),
            n_trials: 5n,
            random_state: variant('some', 42n),
            pruner: variant('none', null),
        });

        const studyResult = $.let(Optuna.optimize(search_space, objective, studyConfig));

        // 3. Train final model with best params from tuning
        const bestNEst = $.let(studyResult.best_params.get(0n).value.unwrap('int'));
        const bestDepth = $.let(studyResult.best_params.get(1n).value.unwrap('int'));

        const finalConfig = $.let({
            n_estimators: variant('some', bestNEst),
            max_depth: variant('some', bestDepth),
            learning_rate: variant('some', 0.1),
            min_child_weight: variant('none', null),
            subsample: variant('none', null),
            colsample_bytree: variant('none', null),
            reg_alpha: variant('none', null),
            reg_lambda: variant('none', null),
            random_state: variant('some', 42n),
            n_jobs: variant('none', null),
        });

        const finalModel = $.let(XGBoost.trainRegressor(split.X_train, split.y_train, finalConfig));

        // 4. Feature importance with SHAP
        const explainer = $.let(Shap.treeExplainerCreate(finalModel));
        const featureNames = $.let(["x1", "x2"]);
        const shapResult = $.let(Shap.computeValues(explainer, split.X_test, featureNames));
        const importance = $.let(Shap.featureImportance(shapResult.shap_values, featureNames));

        // Assertions
        $(Assert.equal(studyResult.trials.size(), 5n));
        // Verify best params are within search space bounds
        $(Assert.greaterEqual(bestNEst, 10n));
        $(Assert.lessEqual(bestNEst, 100n));
        $(Assert.greaterEqual(bestDepth, 2n));
        $(Assert.lessEqual(bestDepth, 6n));
        // Verify best_score is finite (not NaN or Inf)
        $(Assert.greaterEqual(studyResult.best_score, East.value(0.0)));
        $(Assert.less(studyResult.best_score, East.value(10000.0)));
        // Verify SHAP values are valid
        $(Assert.equal(importance.feature_names.size(), 2n));
        $(Assert.greaterEqual(importance.importances.get(0n), East.value(0.0)));
        $(Assert.less(importance.importances.get(0n), East.value(1000.0)));
        $(Assert.greaterEqual(importance.importances.get(1n), East.value(0.0)));
        $(Assert.less(importance.importances.get(1n), East.value(1000.0)));
    });

    test("LightGBM: split -> optuna tune -> train -> shap", $ => {
        const X = $.let([
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0],
            [6.0, 7.0],
            [7.0, 8.0],
            [8.0, 9.0],
            [9.0, 10.0],
            [10.0, 11.0],
        ]);
        const y = $.let([8.0, 13.0, 18.0, 23.0, 28.0, 33.0, 38.0, 43.0, 48.0, 53.0]);

        // 1. Train/test split
        const splitConfig = $.let({
            test_size: variant('some', 0.3),
            random_state: variant('some', 42n),
            shuffle: variant('some', true),
        });
        const split = $.let(Sklearn.trainTestSplit(X, y, splitConfig));

        // 2. Hyperparameter tuning with Optuna
        const search_space = $.let([
            {
                name: "n_estimators",
                kind: variant("int", null),
                low: variant("some", 10.0),
                high: variant("some", 100.0),
                choices: variant("none", null),
            },
            {
                name: "num_leaves",
                kind: variant("int", null),
                low: variant("some", 10.0),
                high: variant("some", 50.0),
                choices: variant("none", null),
            },
        ], ArrayType(ParamSpaceType));

        const objective = East.function(
            [ArrayType(NamedParamType)],
            FloatType,
            ($inner, params) => {
                const nEst = $inner.let(params.get(0n).value.unwrap('int'));
                const leaves = $inner.let(params.get(1n).value.unwrap('int'));

                const config = $inner.let({
                    n_estimators: variant('some', nEst),
                    max_depth: variant('none', null),
                    learning_rate: variant('some', 0.1),
                    num_leaves: variant('some', leaves),
                    min_child_samples: variant('some', 1n),
                    subsample: variant('none', null),
                    colsample_bytree: variant('none', null),
                    reg_alpha: variant('none', null),
                    reg_lambda: variant('none', null),
                    random_state: variant('some', 42n),
                    n_jobs: variant('none', null),
                });

                const model = $inner.let(LightGBM.trainRegressor(split.X_train, split.y_train, config));
                const preds = $inner.let(LightGBM.predict(model, split.X_test));

                const mse = $inner.let(
                    split.y_test.reduce(($r, acc, actual, i) => {
                        const pred = preds.get(i);
                        const diff = actual.subtract(pred);
                        return acc.add(diff.multiply(diff));
                    }, 0.0).divide(split.y_test.size().toFloat())
                );

                return $inner.return(mse);
            }
        );

        const studyConfig = $.let({
            direction: variant('some', variant('minimize', null)),
            n_trials: 5n,
            random_state: variant('some', 42n),
            pruner: variant('none', null),
        });

        const studyResult = $.let(Optuna.optimize(search_space, objective, studyConfig));

        // 3. Train final model with best params from tuning
        const bestNEst = $.let(studyResult.best_params.get(0n).value.unwrap('int'));
        const bestLeaves = $.let(studyResult.best_params.get(1n).value.unwrap('int'));

        const finalConfig = $.let({
            n_estimators: variant('some', bestNEst),
            max_depth: variant('none', null),
            learning_rate: variant('some', 0.1),
            num_leaves: variant('some', bestLeaves),
            min_child_samples: variant('some', 1n),
            subsample: variant('none', null),
            colsample_bytree: variant('none', null),
            reg_alpha: variant('none', null),
            reg_lambda: variant('none', null),
            random_state: variant('some', 42n),
            n_jobs: variant('none', null),
        });

        const finalModel = $.let(LightGBM.trainRegressor(split.X_train, split.y_train, finalConfig));

        // 4. Feature importance with SHAP
        const explainer = $.let(Shap.treeExplainerCreate(finalModel));
        const featureNames = $.let(["x1", "x2"]);
        const shapResult = $.let(Shap.computeValues(explainer, split.X_test, featureNames));
        const importance = $.let(Shap.featureImportance(shapResult.shap_values, featureNames));

        // Assertions
        $(Assert.equal(studyResult.trials.size(), 5n));
        // Verify best params are within search space bounds
        $(Assert.greaterEqual(bestNEst, 10n));
        $(Assert.lessEqual(bestNEst, 100n));
        $(Assert.greaterEqual(bestLeaves, 10n));
        $(Assert.lessEqual(bestLeaves, 50n));
        // Verify best_score is finite
        $(Assert.greaterEqual(studyResult.best_score, East.value(0.0)));
        $(Assert.less(studyResult.best_score, East.value(10000.0)));
        // Verify SHAP values are valid
        $(Assert.equal(importance.feature_names.size(), 2n));
        $(Assert.greaterEqual(importance.importances.get(0n), East.value(0.0)));
        $(Assert.less(importance.importances.get(0n), East.value(1000.0)));
        $(Assert.greaterEqual(importance.importances.get(1n), East.value(0.0)));
        $(Assert.less(importance.importances.get(1n), East.value(1000.0)));
    });

    test("NGBoost: split -> optuna tune -> train -> shap", $ => {
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
        const y = $.let([8.0, 13.0, 18.0, 23.0, 28.0, 33.0, 38.0, 43.0]);

        // 1. Train/test split
        const splitConfig = $.let({
            test_size: variant('some', 0.3),
            random_state: variant('some', 42n),
            shuffle: variant('some', true),
        });
        const split = $.let(Sklearn.trainTestSplit(X, y, splitConfig));

        // 2. Hyperparameter tuning with Optuna
        const search_space = $.let([
            {
                name: "n_estimators",
                kind: variant("int", null),
                low: variant("some", 20.0),
                high: variant("some", 100.0),
                choices: variant("none", null),
            },
        ], ArrayType(ParamSpaceType));

        const objective = East.function(
            [ArrayType(NamedParamType)],
            FloatType,
            ($inner, params) => {
                const nEst = $inner.let(params.get(0n).value.unwrap('int'));

                const config = $inner.let({
                    n_estimators: variant('some', nEst),
                    learning_rate: variant('some', 0.1),
                    minibatch_frac: variant('none', null),
                    col_sample: variant('none', null),
                    random_state: variant('some', 42n),
                    distribution: variant('none', null),
                });

                const model = $inner.let(NGBoost.trainRegressor(split.X_train, split.y_train, config));
                const preds = $inner.let(NGBoost.predict(model, split.X_test));

                const mse = $inner.let(
                    split.y_test.reduce(($r, acc, actual, i) => {
                        const pred = preds.get(i);
                        const diff = actual.subtract(pred);
                        return acc.add(diff.multiply(diff));
                    }, 0.0).divide(split.y_test.size().toFloat())
                );

                return $inner.return(mse);
            }
        );

        const studyConfig = $.let({
            direction: variant('some', variant('minimize', null)),
            n_trials: 3n,
            random_state: variant('some', 42n),
            pruner: variant('none', null),
        });

        const studyResult = $.let(Optuna.optimize(search_space, objective, studyConfig));

        // 3. Train final model with best params from tuning
        const bestNEst = $.let(studyResult.best_params.get(0n).value.unwrap('int'));

        const finalConfig = $.let({
            n_estimators: variant('some', bestNEst),
            learning_rate: variant('some', 0.1),
            minibatch_frac: variant('none', null),
            col_sample: variant('none', null),
            random_state: variant('some', 42n),
            distribution: variant('none', null),
        });

        const finalModel = $.let(NGBoost.trainRegressor(split.X_train, split.y_train, finalConfig));

        // 4. Feature importance with SHAP (using KernelExplainer for NGBoost)
        const explainer = $.let(Shap.kernelExplainerCreate(finalModel, split.X_train));
        const featureNames = $.let(["x1", "x2"]);
        // Explain 2 samples to keep test fast
        const X_explain = $.let([
            [3.0, 4.0],
            [7.0, 8.0],
        ]);
        const shapResult = $.let(Shap.computeValues(explainer, X_explain, featureNames));
        const importance = $.let(Shap.featureImportance(shapResult.shap_values, featureNames));

        // Assertions
        $(Assert.equal(studyResult.trials.size(), 3n));
        // Verify best params are within search space bounds
        $(Assert.greaterEqual(bestNEst, 20n));
        $(Assert.lessEqual(bestNEst, 100n));
        // Verify best_score is finite
        $(Assert.greaterEqual(studyResult.best_score, East.value(0.0)));
        $(Assert.less(studyResult.best_score, East.value(10000.0)));
        // Verify SHAP values are valid
        $(Assert.equal(importance.feature_names.size(), 2n));
        $(Assert.greaterEqual(importance.importances.get(0n), East.value(0.0)));
        $(Assert.less(importance.importances.get(0n), East.value(1000.0)));
        $(Assert.greaterEqual(importance.importances.get(1n), East.value(0.0)));
        $(Assert.less(importance.importances.get(1n), East.value(1000.0)));
    });

    test("GP: split -> optuna tune -> train -> shap", $ => {
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
        const y = $.let([8.0, 13.0, 18.0, 23.0, 28.0, 33.0, 38.0, 43.0]);

        // 1. Train/test split
        const splitConfig = $.let({
            test_size: variant('some', 0.3),
            random_state: variant('some', 42n),
            shuffle: variant('some', true),
        });
        const split = $.let(Sklearn.trainTestSplit(X, y, splitConfig));

        // 2. Hyperparameter tuning with Optuna (tune n_restarts)
        const search_space = $.let([
            {
                name: "n_restarts",
                kind: variant("int", null),
                low: variant("some", 0.0),
                high: variant("some", 3.0),
                choices: variant("none", null),
            },
        ], ArrayType(ParamSpaceType));

        const objective = East.function(
            [ArrayType(NamedParamType)],
            FloatType,
            ($inner, params) => {
                const restarts = $inner.let(params.get(0n).value.unwrap('int'));

                const config = $inner.let({
                    kernel: variant('some', variant('rbf', null)),
                    alpha: variant('some', 1e-10),
                    n_restarts_optimizer: variant('some', restarts),
                    normalize_y: variant('some', true),
                    random_state: variant('some', 42n),
                });

                const model = $inner.let(GP.train(split.X_train, split.y_train, config));
                const preds = $inner.let(GP.predict(model, split.X_test));

                const mse = $inner.let(
                    split.y_test.reduce(($r, acc, actual, i) => {
                        const pred = preds.get(i);
                        const diff = actual.subtract(pred);
                        return acc.add(diff.multiply(diff));
                    }, 0.0).divide(split.y_test.size().toFloat())
                );

                return $inner.return(mse);
            }
        );

        const studyConfig = $.let({
            direction: variant('some', variant('minimize', null)),
            n_trials: 3n,
            random_state: variant('some', 42n),
            pruner: variant('none', null),
        });

        const studyResult = $.let(Optuna.optimize(search_space, objective, studyConfig));

        // 3. Train final model with best params from tuning
        const bestRestarts = $.let(studyResult.best_params.get(0n).value.unwrap('int'));

        const finalConfig = $.let({
            kernel: variant('some', variant('rbf', null)),
            alpha: variant('some', 1e-10),
            n_restarts_optimizer: variant('some', bestRestarts),
            normalize_y: variant('some', true),
            random_state: variant('some', 42n),
        });

        const finalModel = $.let(GP.train(split.X_train, split.y_train, finalConfig));

        // 4. Feature importance with SHAP (using KernelExplainer for GP)
        const explainer = $.let(Shap.kernelExplainerCreate(finalModel, split.X_train));
        const featureNames = $.let(["x1", "x2"]);
        const X_explain = $.let([
            [3.0, 4.0],
            [7.0, 8.0],
        ]);
        const shapResult = $.let(Shap.computeValues(explainer, X_explain, featureNames));
        const importance = $.let(Shap.featureImportance(shapResult.shap_values, featureNames));

        // Assertions
        $(Assert.equal(studyResult.trials.size(), 3n));
        // Verify best params are within search space bounds
        $(Assert.greaterEqual(bestRestarts, 0n));
        $(Assert.lessEqual(bestRestarts, 3n));
        // Verify best_score is finite
        $(Assert.greaterEqual(studyResult.best_score, East.value(0.0)));
        $(Assert.less(studyResult.best_score, East.value(10000.0)));
        // Verify SHAP values are valid
        $(Assert.equal(importance.feature_names.size(), 2n));
        $(Assert.greaterEqual(importance.importances.get(0n), East.value(0.0)));
        $(Assert.less(importance.importances.get(0n), East.value(1000.0)));
        $(Assert.greaterEqual(importance.importances.get(1n), East.value(0.0)));
        $(Assert.less(importance.importances.get(1n), East.value(1000.0)));
    });

    test("Torch MLP: split -> optuna tune -> train -> shap", $ => {
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
        const y = $.let([8.0, 13.0, 18.0, 23.0, 28.0, 33.0, 38.0, 43.0]);

        // 1. Train/test split
        const splitConfig = $.let({
            test_size: variant('some', 0.3),
            random_state: variant('some', 42n),
            shuffle: variant('some', true),
        });
        const split = $.let(Sklearn.trainTestSplit(X, y, splitConfig));

        // 2. Hyperparameter tuning with Optuna
        const search_space = $.let([
            {
                name: "hidden_size",
                kind: variant("int", null),
                low: variant("some", 8.0),
                high: variant("some", 32.0),
                choices: variant("none", null),
            },
        ], ArrayType(ParamSpaceType));

        const objective = East.function(
            [ArrayType(NamedParamType)],
            FloatType,
            ($inner, params) => {
                const hiddenSize = $inner.let(params.get(0n).value.unwrap('int'));

                const mlpConfig = $inner.let({
                    hidden_layers: [hiddenSize],
                    activation: variant('some', variant('relu', null)),
                    output_activation: variant('none', null),
                    dropout: variant('none', null),
                    output_dim: variant('some', 1n),
                });

                const trainConfig = $inner.let({
                    epochs: variant('some', 30n),
                    batch_size: variant('some', 4n),
                    learning_rate: variant('some', 0.01),
                    loss: variant('none', null),
                    optimizer: variant('none', null),
                    early_stopping: variant('none', null),
                    validation_split: variant('none', null),
                    random_state: variant('some', 42n),
                });

                const trainResult = $inner.let(Torch.mlpTrain(split.X_train, split.y_train, mlpConfig, trainConfig));
                const preds = $inner.let(Torch.mlpPredict(trainResult.model, split.X_test));

                const mse = $inner.let(
                    split.y_test.reduce(($r, acc, actual, i) => {
                        const pred = preds.get(i);
                        const diff = actual.subtract(pred);
                        return acc.add(diff.multiply(diff));
                    }, 0.0).divide(split.y_test.size().toFloat())
                );

                return $inner.return(mse);
            }
        );

        const studyConfig = $.let({
            direction: variant('some', variant('minimize', null)),
            n_trials: 3n,
            random_state: variant('some', 42n),
            pruner: variant('none', null),
        });

        const studyResult = $.let(Optuna.optimize(search_space, objective, studyConfig));

        // 3. Train final model with best params from tuning
        const bestHiddenSize = $.let(studyResult.best_params.get(0n).value.unwrap('int'));

        const finalMlpConfig = $.let({
            hidden_layers: [bestHiddenSize],
            activation: variant('some', variant('relu', null)),
            output_activation: variant('none', null),
            dropout: variant('none', null),
            output_dim: variant('some', 1n),
        });

        const finalTrainConfig = $.let({
            epochs: variant('some', 50n),
            batch_size: variant('some', 4n),
            learning_rate: variant('some', 0.01),
            loss: variant('none', null),
            optimizer: variant('none', null),
            early_stopping: variant('none', null),
            validation_split: variant('none', null),
            random_state: variant('some', 42n),
        });

        const finalResult = $.let(Torch.mlpTrain(split.X_train, split.y_train, finalMlpConfig, finalTrainConfig));

        // 4. Feature importance with SHAP (using KernelExplainer for Torch)
        const explainer = $.let(Shap.kernelExplainerCreate(finalResult.model, split.X_train));
        const featureNames = $.let(["x1", "x2"]);
        const X_explain = $.let([
            [3.0, 4.0],
            [7.0, 8.0],
        ]);
        const shapResult = $.let(Shap.computeValues(explainer, X_explain, featureNames));
        const importance = $.let(Shap.featureImportance(shapResult.shap_values, featureNames));

        // Assertions
        $(Assert.equal(studyResult.trials.size(), 3n));
        // Verify best params are within search space bounds
        $(Assert.greaterEqual(bestHiddenSize, 8n));
        $(Assert.lessEqual(bestHiddenSize, 32n));
        // Verify best_score is finite
        $(Assert.greaterEqual(studyResult.best_score, East.value(0.0)));
        $(Assert.less(studyResult.best_score, East.value(100000.0)));
        // Verify SHAP values are valid
        $(Assert.equal(importance.feature_names.size(), 2n));
        $(Assert.greaterEqual(importance.importances.get(0n), East.value(0.0)));
        $(Assert.less(importance.importances.get(0n), East.value(1000.0)));
        $(Assert.greaterEqual(importance.importances.get(1n), East.value(0.0)));
        $(Assert.less(importance.importances.get(1n), East.value(1000.0)));
    });
}, { exportOnly: true });
