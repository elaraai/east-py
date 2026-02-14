#
# Copyright (c) 2025 Elara AI Pty Ltd
# Licensed under the Business Source License 1.1. See LICENSE.md for details.
#
"""Iterative coordinate descent optimization for East.

Provides discrete combinatorial optimization by iteratively optimizing each
element of a parameter array over its candidate values. Supports multi-start
sampling for better exploration of the search space.

Ported from the Julia IterativeDecisionAlgorithm (ArrayParameterSpace branch).
"""

import copy
import random
from collections.abc import Callable
from typing import Any

from east.runtime.platform import GenericPlatformFunction
from east.types.types import (
    ArrayType,
    BooleanType,
    FloatType,
    IntegerType,
    NullType,
    OptionType,
    StructType,
    VariantType,
)
from east.types.values import EastArray, EastStruct, EastVariant

# ============================================================================
# Type Definitions (must match TypeScript exactly)
# ============================================================================

InitialStrategyType = VariantType(
    [
        ("first", NullType),
        ("random", NullType),
    ]
)

EvaluationOrderType = VariantType(
    [
        ("sequential", NullType),
        ("random", NullType),
    ]
)

IterativeConfigType = StructType(
    [
        ("iterations", OptionType(IntegerType)),
        ("samples", OptionType(IntegerType)),
        ("initial", OptionType(InitialStrategyType)),
        ("order", OptionType(EvaluationOrderType)),
        ("random_state", OptionType(IntegerType)),
    ]
)


def IterativeResultType(value_type: Any) -> StructType:
    """Create iterative optimization result type for a given value type."""
    return StructType(
        [
            ("best_parameters", ArrayType(value_type)),
            ("best_objective", FloatType),
            ("iterations", IntegerType),
            ("evaluations", IntegerType),
            ("success", BooleanType),
        ]
    )


# ============================================================================
# Helper Functions
# ============================================================================


def _get_option(opt: EastVariant | None, default: Any) -> Any:
    """Extract value from Option variant, returning default if None.

    Note: The runtime creates EastVariant instances, not EastOption instances,
    even for Option types. So we check the tag directly rather than using
    is_east_option().
    """
    if opt is None:
        return default
    if isinstance(opt, EastVariant) and opt.type == "some":
        return opt.value
    return default


# ============================================================================
# Platform Function Implementation
# ============================================================================


def _copy_params(params: EastArray) -> EastArray:
    """Deep copy an EastArray of parameters."""
    return EastArray(params.element_type, [copy.deepcopy(v) for v in params])


def optimization_iterative_impl(
    objective_fn: Callable[[EastArray], float],
    parameter_spaces: EastArray,
    config: EastStruct,
) -> EastStruct:
    """Run iterative coordinate descent optimization.

    Maximizes an objective function over an array of discrete parameters.
    Each parameter position has its own set of candidate values.
    The algorithm performs coordinate descent: for each element, try all
    candidate values while holding others fixed, keep the best.

    Multiple independent restarts (samples) improve exploration.

    Args:
        objective_fn: Objective function (Array<V> -> Float), higher is better
        parameter_spaces: Per-element candidate values (Array<Array<V>>)
        config: Optimization configuration

    Returns:
        EastStruct with best_parameters, best_objective, iterations, evaluations, success
    """
    # Extract config with defaults
    max_iterations = int(_get_option(config.get("iterations"), 100))
    num_samples = int(_get_option(config.get("samples"), 1))

    initial_opt = _get_option(config.get("initial"), None)
    use_random_init = (
        initial_opt is not None
        and isinstance(initial_opt, EastVariant)
        and initial_opt.type == "random"
    )

    order_opt = _get_option(config.get("order"), None)
    use_random_order = (
        order_opt is not None
        and isinstance(order_opt, EastVariant)
        and order_opt.type == "random"
    )

    seed = _get_option(config.get("random_state"), None)
    if seed is not None:
        seed = int(seed)

    rng = random.Random(seed)
    n = len(parameter_spaces)

    # Determine element type from the first space
    if n > 0 and len(parameter_spaces[0]) > 0:
        element_type = parameter_spaces[0].element_type
    else:
        element_type = IntegerType

    global_best_obj = float("-inf")
    global_best_params: EastArray | None = None
    total_iterations = 0
    total_evaluations = 0

    for _sample in range(num_samples):
        # Initialize parameters
        if use_random_init:
            init_values = [rng.choice(list(space)) for space in parameter_spaces]
        else:
            # first: use the first candidate value from each space
            init_values = [space[0] for space in parameter_spaces]

        params = EastArray(element_type, init_values)

        # Evaluate initial solution
        best_obj = float(objective_fn(params))
        best_params = _copy_params(params)
        total_evaluations += 1

        # Coordinate descent loop
        for iteration in range(1, max_iterations + 1):
            changed = False

            for i in range(n):
                candidates = list(parameter_spaces[i])
                if use_random_order:
                    rng.shuffle(candidates)

                current_best_val = params[i]

                for candidate in candidates:
                    params[i] = candidate
                    obj = float(objective_fn(params))
                    total_evaluations += 1

                    if obj > best_obj:
                        best_obj = obj
                        best_params = _copy_params(params)
                        current_best_val = candidate
                        changed = True

                # Restore best value for this element
                params[i] = current_best_val

            total_iterations += 1

            if not changed:
                break

        # Update global best across samples
        if best_obj > global_best_obj:
            global_best_obj = best_obj
            global_best_params = best_params

    return EastStruct(
        {
            "best_parameters": global_best_params
            if global_best_params is not None
            else EastArray(element_type, []),
            "best_objective": global_best_obj
            if global_best_params is not None
            else 0.0,
            "iterations": total_iterations,
            "evaluations": total_evaluations,
            "success": global_best_params is not None,
        }
    )


# ============================================================================
# Platform Function Registration
# ============================================================================

# Optimization is generic over value type V.
# The factory receives the type parameter and returns the implementation.
# Type safety is enforced at the TypeScript level.

optimization_impl = [
    GenericPlatformFunction(
        name="optimization_iterative",
        type_parameters=["V"],
        type="sync",
        fn=lambda V: optimization_iterative_impl,
    ),
]


__all__ = [
    # Platform implementation
    "optimization_impl",
    # Types
    "InitialStrategyType",
    "EvaluationOrderType",
    "IterativeConfigType",
    "IterativeResultType",
]
