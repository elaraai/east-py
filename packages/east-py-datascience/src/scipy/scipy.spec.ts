/**
 * Copyright (c) 2025 Elara AI Pty Ltd
 * Dual-licensed under AGPL-3.0 and commercial license. See LICENSE for details.
 */

/**
 * SciPy platform function tests
 */
import { East, FloatType, variant } from "@elaraai/east";
import { describeEast, Assert } from "@elaraai/east-node-std";
import { Scipy } from "./scipy.js";

describeEast("Scipy platform functions", (test) => {
    test("stats_describe computes correct statistics", $ => {
        const data = $.let([1.0, 2.0, 3.0, 4.0, 5.0]);

        const result = $.let(Scipy.statsDescribe(data));

        $(Assert.equal(result.count, 5n));
        $(Assert.equal(result.mean, East.value(3.0)));
        $(Assert.equal(result.min, East.value(1.0)));
        $(Assert.equal(result.max, East.value(5.0)));
    });

    test("stats_pearsonr computes correlation", $ => {
        // Perfect positive correlation
        const x = $.let([1.0, 2.0, 3.0, 4.0, 5.0]);
        const y = $.let([2.0, 4.0, 6.0, 8.0, 10.0]);

        const result = $.let(Scipy.statsPearsonr(x, y));

        // Should be ~1.0 for perfect correlation
        $(Assert.greater(result.correlation, East.value(0.99)));
    });

    test("stats_spearmanr computes rank correlation", $ => {
        // Perfect positive rank correlation
        const x = $.let([1.0, 2.0, 3.0, 4.0, 5.0]);
        const y = $.let([10.0, 20.0, 30.0, 40.0, 50.0]);

        const result = $.let(Scipy.statsSpearmanr(x, y));

        $(Assert.greater(result.correlation, East.value(0.99)));
    });

    test("curve_fit fits linear function", $ => {
        // Linear data: y = 2 + 3*x
        const x = $.let([0.0, 1.0, 2.0, 3.0, 4.0]);
        const y = $.let([2.0, 5.0, 8.0, 11.0, 14.0]);

        const config = $.let({
            max_iter: variant('some', 5000n),
            initial_guess: variant('none', null),
        });

        const result = $.let(Scipy.curveFit(
            variant('linear', null),
            x,
            y,
            config
        ));

        $(Assert.equal(result.success, true));
        $(Assert.greater(result.r_squared, East.value(0.99)));
        // params[0] should be ~2.0 (intercept), params[1] should be ~3.0 (slope)
    });

    test("curve_fit fits exponential decay", $ => {
        // Exponential decay: y = 10 * exp(-0.5 * x)
        const x = $.let([0.0, 1.0, 2.0, 3.0, 4.0]);
        const y = $.let([10.0, 6.065, 3.679, 2.231, 1.353]);

        const config = $.let({
            max_iter: variant('some', 5000n),
            initial_guess: variant('none', null),
        });

        const result = $.let(Scipy.curveFit(
            variant('exponential_decay', null),
            x,
            y,
            config
        ));

        $(Assert.equal(result.success, true));
        $(Assert.greater(result.r_squared, East.value(0.99)));
    });

    test("interpolate_1d_fit and predict works", $ => {
        // Known data points
        const x = $.let([0.0, 1.0, 2.0, 3.0, 4.0]);
        const y = $.let([0.0, 1.0, 4.0, 9.0, 16.0]);

        const config = $.let({
            kind: variant('some', variant('linear', null)),
        });

        // Fit interpolator
        const interp = $.let(Scipy.interpolate1dFit(x, y, config));

        // Predict at known and interpolated points
        const x_new = $.let([0.5, 1.5, 2.5]);
        const y_pred = $.let(Scipy.interpolate1dPredict(interp, x_new));

        // Check dimensions
        $(Assert.equal(y_pred.size(), 3n));
    });

    test("optimize_minimize finds minimum", $ => {
        // Minimize sum of squares (minimum at origin)
        const objective = East.function([Scipy.Types.VectorType], FloatType, ($, x) => {
            const x0 = $.let(x.get(0n));
            const x1 = $.let(x.get(1n));
            return $.return(x0.multiply(x0).add(x1.multiply(x1)));
        });

        const x0 = $.let([1.0, 1.0]);
        const config = $.let({
            method: variant('some', variant('l_bfgs_b', null)),
            max_iter: variant('some', 100n),
            tol: variant('some', 0.000001),
        });

        const result = $.let(Scipy.optimizeMinimize(objective, x0, config));

        $(Assert.equal(result.success, true));
        $(Assert.less(result.fun, East.value(0.01)));
    });

    test("optimize_minimize_quadratic finds minimum", $ => {
        // Minimize f(x) = 0.5 * x'Ax + b'x + c
        // A = [[2, 0], [0, 2]], b = [-2, -2], c = 0
        // Minimum at x = [1, 1], f(x) = -2

        const x0 = $.let([0.0, 0.0]);
        const quadratic = $.let({
            A: [[2.0, 0.0], [0.0, 2.0]],
            b: [-2.0, -2.0],
            c: 0.0,
        });
        const config = $.let({
            method: variant('some', variant('l_bfgs_b', null)),
            max_iter: variant('some', 100n),
            tol: variant('some', 0.000001),
        });

        const result = $.let(Scipy.optimizeMinimizeQuadratic(x0, quadratic, config));

        $(Assert.equal(result.success, true));
        $(Assert.less(result.fun, East.value(-1.9)));
    });

    test("curve_fit fits custom function", $ => {
        // Custom function: y = a * sin(b * x)
        // Use data: sin(x) at x = [0, π/2, π, 3π/2, 2π]
        // Expected params: a ~ 1.0, b ~ 1.0
        const x = $.let([0.0, 1.5708, 3.1416, 4.7124, 6.2832]);
        const y = $.let([0.0, 1.0, 0.0, -1.0, 0.0]);

        // Define custom curve function: a * sin(b * x)
        const customFn = East.function(
            [FloatType, Scipy.Types.VectorType],
            FloatType,
            ($, x_val, params) => {
                const a = $.let(params.get(0n));
                const b = $.let(params.get(1n));
                return $.return(a.multiply(b.multiply(x_val).sin()));
            }
        );

        const config = $.let({
            max_iter: variant('some', 5000n),
            initial_guess: variant('some', [1.0, 1.0]),
        });

        const result = $.let(Scipy.curveFit(
            variant('custom', {
                fn: customFn,
                n_params: 2n,
                param_bounds: variant('none', null),
            }),
            x,
            y,
            config
        ));

        $(Assert.equal(result.success, true));
        $(Assert.greater(result.r_squared, East.value(0.9)));
    });

    test("error: curve_fit unknown curve type", $ => {
        const x = $.let([1.0, 2.0, 3.0]);
        const y = $.let([1.0, 4.0, 9.0]);

        const config = $.let({
            max_iter: variant('none', null),
            initial_guess: variant('none', null),
        });

        // Use an unknown curve type variant
        $(Assert.throws(Scipy.curveFit(
            variant('unknown_curve' as any, null),
            x,
            y,
            config
        ), /scipy_curve_fit.*Unknown curve type/));
    });
}, { exportOnly: true });
