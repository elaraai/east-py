/**
 * Copyright (c) 2025 Elara AI Pty Ltd
 * Dual-licensed under AGPL-3.0 and commercial license. See LICENSE for details.
 */

/**
 * MADS platform function tests
 *
 * These tests use describeEast following east-node conventions.
 * Tests compile East functions and run them to validate platform function behavior.
 *
 * Note: These tests require PyNomadBBO to be installed in the Python environment.
 * The tests define East functions that call MADS optimization and verify results.
 */
import { ArrayType, East, FloatType, variant } from "@elaraai/east";
import { describeEast, Assert } from "@elaraai/east-node-std";
import { MADS, MADSConstraintType } from "./mads.js";

describeEast("MADS platform functions", (test) => {
    test("optimize minimizes sum of squares", $ => {
        // Define objective: minimize sum of squares (minimum at origin)
        const objective = East.function([MADS.Types.VectorType], FloatType, ($, x) => {
            // x[0]^2 + x[1]^2 + x[2]^2
            const x0 = $.let(x.get(0n));
            const x1 = $.let(x.get(1n));
            const x2 = $.let(x.get(2n));
            return $.return(
                x0.multiply(x0)
                    .add(x1.multiply(x1))
                    .add(x2.multiply(x2))
            );
        });

        // Starting point
        const x0 = $.let([0.71, 0.51, 0.51]);

        // Bounds
        const bounds = $.let({
            lower: [-1.0, -1.0, -1.0],
            upper: [1.0, 1.0, 1.0],
        });

        // Config
        const config = $.let({
            max_bb_eval: variant('some', 100n),
            display_degree: variant('some', 0n),
            direction_type: variant('none', null),
            initial_mesh_size: variant('none', null),
            min_mesh_size: variant('none', null),
            seed: variant('some', 42n),
        });

        // Run optimization
        const result = $.let(MADS.optimize(objective, x0, bounds, variant('none', null), config));

        // Verify success
        $(Assert.equal(East.less(result.f_best, 0.1), true));
        $(Assert.equal(result.success, true));
        $(Assert.greater(result.bb_eval, 0n));
    });

    test("optimize with constraints", $ => {
        // Minimize x[0] subject to x[0]^2 + x[1]^2 >= 1 (outside unit circle)
        const objective = East.function([MADS.Types.VectorType], FloatType, ($, x) => {
            return $.return(x.get(0n));
        });

        // Constraint: 1 - x[0]^2 - x[1]^2 <= 0 (must be outside unit circle)
        const constraint = East.function([MADS.Types.VectorType], FloatType, ($, x) => {
            const x0 = $.let(x.get(0n));
            const x1 = $.let(x.get(1n));
            return $.return(
                East.value(1.0)
                    .subtract(x0.multiply(x0))
                    .subtract(x1.multiply(x1))
            );
        });

        const x0 = $.let([2.0, 0.0]);
        const bounds = $.let({
            lower: [-5.0, -5.0],
            upper: [5.0, 5.0],
        });

        // Use extreme barrier constraint
        const constraints = $.let([variant('eb', constraint)], ArrayType(MADSConstraintType));

        const config = $.let({
            max_bb_eval: variant('some', 200n),
            display_degree: variant('some', 0n),
            direction_type: variant('none', null),
            initial_mesh_size: variant('none', null),
            min_mesh_size: variant('none', null),
            seed: variant('some', 42n),
        });

        const result = $.let(MADS.optimize(objective, x0, bounds, variant('some', constraints), config));

        // The minimum of x[0] on the unit circle boundary is -1
        $(Assert.equal(result.success, true));
        $(Assert.less(result.f_best, East.value(0.0))); // Should be negative (around -1)
    });

    test("optimize respects seed for reproducibility", $ => {
        const objective = East.function([MADS.Types.VectorType], FloatType, ($, x) => {
            const x0 = $.let(x.get(0n));
            return $.return(x0.multiply(x0));
        });

        const x0 = $.let([0.5]);
        const bounds = $.let({
            lower: [-1.0],
            upper: [1.0],
        });

        const config = $.let({
            max_bb_eval: variant('some', 50n),
            display_degree: variant('some', 0n),
            direction_type: variant('none', null),
            initial_mesh_size: variant('none', null),
            min_mesh_size: variant('none', null),
            seed: variant('some', 123n),
        });

        // Run twice with same seed
        const result1 = $.let(MADS.optimize(objective, x0, bounds, variant('none', null), config));
        const result2 = $.let(MADS.optimize(objective, x0, bounds, variant('none', null), config));

        // Results should be identical with same seed
        $(Assert.equal(result1.f_best, result2.f_best));
    });
}, { exportOnly: true });
