#
# Copyright (c) 2025 Elara AI Pty Ltd
# Licensed under the Business Source License 1.1. See LICENSE.md for details.
#
"""Iterative coordinate descent optimization for East.

Provides discrete combinatorial optimization by iteratively optimizing each
element of a parameter vector over its candidate values. Supports multi-start
sampling for better exploration of the search space.

Ported from the Julia IterativeDecisionAlgorithm (ArrayParameterSpace branch).
"""

import random
from collections.abc import Callable
from typing import Any

import numpy as np
from east.runtime.platform import PlatformFunction
from east.types.types import (
    ArrayType,
    BooleanType,
    FloatType,
    FunctionType,
    IntegerType,
    NullType,
    OptionType,
    StructType,
    VariantType,
    VectorType,
)
from east.types.values import EastArray, EastStruct, EastVariant, EastVector, is_east_variant

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


IterativeResultType = StructType(
    [
        ("best_parameters", VectorType(IntegerType)),
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
    if is_east_variant(opt) and opt.type == "some":
        return opt.value
    return default


# ============================================================================
# Platform Function Implementation
# ============================================================================


def optimization_iterative_impl(
    objective_fn: Callable[[EastVector], float],
    parameter_spaces: EastArray,
    config: EastStruct,
) -> EastStruct:
    """Run iterative coordinate descent optimization.

    Maximizes an objective function over a vector of discrete parameters.
    Each parameter position has its own set of candidate values (vector).
    The algorithm performs coordinate descent: for each element, try all
    candidate values while holding others fixed, keep the best.

    Multiple independent restarts (samples) improve exploration.

    Args:
        objective_fn: Objective function (Vector<V> -> Float), higher is better
        parameter_spaces: Per-element candidate values (Array<Vector<V>>)
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
        and is_east_variant(initial_opt)
        and initial_opt.type == "random"
    )

    order_opt = _get_option(config.get("order"), None)
    use_random_order = (
        order_opt is not None
        and is_east_variant(order_opt)
        and order_opt.type == "random"
    )

    seed = _get_option(config.get("random_state"), None)
    if seed is not None:
        seed = int(seed)

    rng = random.Random(seed)
    n = len(parameter_spaces)

    # Determine element type and numpy dtype from the first space
    if n > 0:
        space0: EastVector = parameter_spaces[0]
        element_type = space0.element_type
        dtype = space0.data.dtype
    else:
        element_type = IntegerType
        dtype = np.dtype(np.int64)

    global_best_obj = float("-inf")
    global_best_params: EastVector | None = None
    total_iterations = 0
    total_evaluations = 0

    for _sample in range(num_samples):
        # Initialize parameters as numpy vector
        if use_random_init:
            init_values = np.array(
                [rng.choice(space.data.tolist()) for space in parameter_spaces],
                dtype=dtype,
            )
        else:
            # first: use the first candidate value from each space
            init_values = np.array(
                [space.data[0] for space in parameter_spaces],
                dtype=dtype,
            )

        params = EastVector(element_type, data=init_values)

        # Evaluate initial solution
        best_obj = float(objective_fn(params))
        best_params = EastVector(element_type, data=params.data.copy())
        total_evaluations += 1

        # Coordinate descent loop
        for _iteration in range(1, max_iterations + 1):
            changed = False

            for i in range(n):
                space: EastVector = parameter_spaces[i]
                candidates = space.data.tolist()
                if use_random_order:
                    rng.shuffle(candidates)

                current_best_val = params.data[i]

                for candidate in candidates:
                    params.data[i] = candidate
                    obj = float(objective_fn(params))
                    total_evaluations += 1

                    if obj > best_obj:
                        best_obj = obj
                        best_params = EastVector(element_type, data=params.data.copy())
                        current_best_val = candidate
                        changed = True

                # Restore best value for this element
                params.data[i] = current_best_val

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
            else EastVector(element_type, data=np.array([], dtype=dtype)),
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

optimization_impl = [
    PlatformFunction(
        name="optimization_iterative",
        inputs=[
            FunctionType([VectorType(IntegerType)], FloatType),
            ArrayType(VectorType(IntegerType)),
            IterativeConfigType,
        ],
        output=IterativeResultType,
        type="sync",
        fn=optimization_iterative_impl,
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
