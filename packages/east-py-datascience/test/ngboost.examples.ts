/**
 * Copyright (c) 2025 Elara AI Pty Ltd
 * Dual-licensed under AGPL-3.0 and commercial license. See LICENSE for details.
 */
import { East, IntegerType, BooleanType, variant, example } from "@elaraai/east";
import { NGBoost } from "@elaraai/east-py-datascience";

export const ngboostTrainPredict = example({
    keywords: ["ngboost", "trainRegressor", "predict", "regression", "probabilistic", "demand", "forecasting"],
    description: "Train probabilistic regressor on demand features and predict future demand",
    fn: East.function([], IntegerType, ($) => {
        // Features: day_of_week, promo_active, temperature
        // Target: units sold
        const X_train = $.let(East.Matrix.fromArray([
            [1.0, 0.0, 20.0],
            [2.0, 1.0, 22.0],
            [3.0, 0.0, 25.0],
            [4.0, 1.0, 18.0],
            [5.0, 0.0, 15.0],
            [6.0, 1.0, 28.0],
            [7.0, 0.0, 30.0],
            [1.0, 1.0, 21.0],
            [3.0, 1.0, 24.0],
            [5.0, 0.0, 19.0],
        ]));
        const y_train = $.let(new Float64Array([50.0, 80.0, 60.0, 75.0, 40.0, 90.0, 55.0, 70.0, 85.0, 45.0]));

        const config = $.let({
            n_estimators: variant('some', 100n),
            learning_rate: variant('some', 0.1),
            minibatch_frac: variant('none', null),
            col_sample: variant('none', null),
            random_state: variant('some', 42n),
            distribution: variant('none', null),
        });

        const model = $.let(NGBoost.trainRegressor(X_train, y_train, config));

        // Predict demand for new conditions
        const X_new = $.let(East.Matrix.fromArray([
            [2.0, 1.0, 23.0],
            [6.0, 0.0, 26.0],
        ]));
        const predictions = $.let(NGBoost.predict(model, X_new));

        // Should produce one prediction per input row
        return predictions.length();
    }),
    inputs: [],
    returns: 2n,
});

export const ngboostPredictDist = example({
    keywords: ["ngboost", "predictDist", "uncertainty", "confidence interval", "std", "probabilistic prediction"],
    description: "Get predictions with uncertainty estimates including confidence intervals",
    fn: East.function([], BooleanType, ($) => {
        // Simple linear data with noise
        const X = $.let(East.Matrix.fromArray([
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
            [5.0, 5.0],
            [6.0, 6.0],
            [7.0, 7.0],
            [8.0, 8.0],
            [9.0, 9.0],
            [10.0, 10.0],
        ]));
        const y = $.let(new Float64Array([2.1, 3.9, 6.2, 7.8, 10.1, 12.0, 13.9, 16.1, 18.0, 20.1]));

        const config = $.let({
            n_estimators: variant('some', 100n),
            learning_rate: variant('some', 0.1),
            minibatch_frac: variant('none', null),
            col_sample: variant('none', null),
            random_state: variant('some', 42n),
            distribution: variant('none', null),
        });

        const predictConfig = $.let({
            confidence_level: variant('some', 0.95),
        });

        const model = $.let(NGBoost.trainRegressor(X, y, config));
        const result = $.let(NGBoost.predictDist(model, X, predictConfig));

        // Predictions should have 10 elements
        return East.equal(result.predictions.length(), 10n);
    }),
    inputs: [],
    returns: true,
});
