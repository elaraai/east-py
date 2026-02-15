/**
 * Copyright (c) 2025 Elara AI Pty Ltd
 * Dual-licensed under AGPL-3.0 and commercial license. See LICENSE for details.
 */

/**
 * Iterative optimization platform function tests.
 *
 * These tests use describeEast following east-node conventions.
 * Tests compile East functions and export IR for Python execution.
 */
import { East, FloatType, IntegerType, VectorType, variant } from "@elaraai/east";
import { describeEast, Assert } from "@elaraai/east-node-std";
import { Optimization } from "./optimization.js";

describeEast("Optimization platform functions", (test) => {
    test("iterative finds optimal task-worker assignment", $ => {
        // 5 tasks, 3 workers. skill[task][worker] = match score.
        const skill = $.let([
            [3.0, 1.0, 2.0],   // task 0: best with worker 0
            [1.0, 3.0, 2.0],   // task 1: best with worker 1
            [2.0, 2.0, 3.0],   // task 2: best with worker 2
            [3.0, 2.0, 1.0],   // task 3: best with worker 0
            [1.0, 2.0, 3.0],   // task 4: best with worker 2
        ]);

        // Objective: total skill score for given assignment
        const objective = East.function(
            [VectorType(IntegerType)], FloatType,
            ($, assignments) => {
                const total = $.let(0.0);
                $.for(East.Array.range(0n, East.value(5n)), ($, i) => {
                    const worker = $.let(assignments.get(i));
                    const score = $.let(skill.get(i).get(worker));
                    $.assign(total, total.add(score));
                });
                return $.return(total);
            }
        );

        // Each task can be assigned to worker 0, 1, or 2
        const spaces = $.let([
            new BigInt64Array([0n, 1n, 2n]),
            new BigInt64Array([0n, 1n, 2n]),
            new BigInt64Array([0n, 1n, 2n]),
            new BigInt64Array([0n, 1n, 2n]),
            new BigInt64Array([0n, 1n, 2n]),
        ]);

        const config = $.let({
            iterations: variant('some', 10n),
            samples: variant('some', 3n),
            initial: variant('some', variant('random', null)),
            order: variant('some', variant('sequential', null)),
            random_state: variant('some', 42n),
        });

        const result = $.let(Optimization.iterative(
            objective, spaces, config,
        ));

        $(Assert.equal(result.success, true));
        // Optimal: each task assigned to best worker -> 3+3+3+3+3 = 15.0
        $(Assert.equal(result.best_objective, East.value(15.0)));
    });

    test("iterative respects seed for reproducibility", $ => {
        // Maximize sum of values (trivial but verifies determinism)
        const objective = East.function(
            [VectorType(IntegerType)], FloatType,
            ($, params) => {
                const total = $.let(0.0);
                $.for(East.Array.range(0n, params.length()), ($, i) => {
                    $.assign(total, total.add(params.get(i).toFloat()));
                });
                return $.return(total);
            }
        );

        const spaces = $.let([
            new BigInt64Array([0n, 1n, 2n, 3n]),
            new BigInt64Array([0n, 1n, 2n, 3n]),
            new BigInt64Array([0n, 1n, 2n, 3n]),
        ]);

        const config = $.let({
            iterations: variant('some', 5n),
            samples: variant('some', 2n),
            initial: variant('some', variant('random', null)),
            order: variant('some', variant('random', null)),
            random_state: variant('some', 123n),
        });

        const result1 = $.let(Optimization.iterative(objective, spaces, config));
        const result2 = $.let(Optimization.iterative(objective, spaces, config));

        $(Assert.equal(result1.best_objective, result2.best_objective));
    });

    test("iterative works with default config", $ => {
        // Simple: maximize sum with all defaults
        const objective = East.function(
            [VectorType(IntegerType)], FloatType,
            ($, params) => {
                const total = $.let(0.0);
                $.for(East.Array.range(0n, params.length()), ($, i) => {
                    $.assign(total, total.add(params.get(i).toFloat()));
                });
                return $.return(total);
            }
        );

        const spaces = $.let([
            new BigInt64Array([0n, 1n, 2n]),
            new BigInt64Array([0n, 1n, 2n]),
        ]);

        // All-none config: use all defaults
        const config = $.let({
            iterations: variant('none', null),
            samples: variant('none', null),
            initial: variant('none', null),
            order: variant('none', null),
            random_state: variant('none', null),
        });

        const result = $.let(Optimization.iterative(objective, spaces, config));

        $(Assert.equal(result.success, true));
        // Should find [2, 2] -> 4.0
        $(Assert.equal(result.best_objective, East.value(4.0)));
    });

    test("iterative converges early when no improvement", $ => {
        // Objective: minimize distance from [1, 1, 1]
        // (negative squared distance = maximize to get closer)
        const objective = East.function(
            [VectorType(IntegerType)], FloatType,
            ($, params) => {
                const penalty = $.let(0.0);
                $.for(East.Array.range(0n, params.length()), ($, i) => {
                    const diff = $.let(params.get(i).subtract(1n).toFloat());
                    $.assign(penalty, penalty.subtract(diff.multiply(diff)));
                });
                return $.return(penalty);
            }
        );

        // Each param can be 0, 1, or 2. Best is [1, 1, 1] -> penalty 0.0
        const spaces = $.let([
            new BigInt64Array([0n, 1n, 2n]),
            new BigInt64Array([0n, 1n, 2n]),
            new BigInt64Array([0n, 1n, 2n]),
        ]);

        const config = $.let({
            iterations: variant('some', 100n),
            samples: variant('some', 1n),
            initial: variant('some', variant('first', null)),
            order: variant('some', variant('sequential', null)),
            random_state: variant('none', null),
        });

        const result = $.let(Optimization.iterative(objective, spaces, config));

        $(Assert.equal(result.success, true));
        // Should converge to [1,1,1] -> 0.0
        $(Assert.equal(result.best_objective, East.value(0.0)));
        // Should converge well before 100 iterations
        $(Assert.less(result.iterations, 100n));
    });

}, { exportOnly: true });
