"""Simulated Annealing platform functions for East.

Provides discrete optimization using the simanneal library.
Ideal for combinatorial problems like TSP, scheduling, and subset selection.
"""

from typing import Any, Callable
import random

from east.runtime.platform import PlatformFunction
from east.types.types import (
    ArrayType,
    BooleanType,
    FloatType,
    FunctionType,
    IntegerType,
    OptionType,
    StructType,
    VariantType,
)
from east.types.values import EastArray, EastStruct, EastVariant, is_east_variant

# ============================================================================
# Type Definitions
# ============================================================================

# Discrete state type (int_array or bool_array)
DiscreteStateType = VariantType(
    [
        ("int_array", ArrayType(IntegerType)),
        ("bool_array", ArrayType(BooleanType)),
    ]
)

# Energy function type: state -> score
EnergyFunctionType = FunctionType([DiscreteStateType], FloatType)

# Move function type: state -> neighbor
MoveFunctionType = FunctionType([DiscreteStateType], DiscreteStateType)

# Permutation energy function type
PermutationEnergyType = FunctionType([ArrayType(IntegerType)], FloatType)

# Subset energy function type
SubsetEnergyType = FunctionType([ArrayType(BooleanType)], FloatType)

# Annealing configuration
AnnealConfigType = StructType(
    [
        ("t_max", OptionType(FloatType)),
        ("t_min", OptionType(FloatType)),
        ("steps", OptionType(IntegerType)),
        ("updates", OptionType(IntegerType)),
        ("auto_schedule", OptionType(FloatType)),
        ("random_state", OptionType(IntegerType)),
    ]
)

# Annealing result
AnnealResultType = StructType(
    [
        ("best_state", DiscreteStateType),
        ("best_energy", FloatType),
        ("steps_taken", IntegerType),
        ("success", BooleanType),
    ]
)


# ============================================================================
# Helper Functions
# ============================================================================


def _get_option(opt: EastVariant | None, default: Any) -> Any:
    """Extract value from Option variant, returning default if None."""
    if opt is None:
        return default
    if is_east_variant(opt) and opt.type == "some":
        return opt.value
    return default


# ============================================================================
# Platform Function Implementations
# ============================================================================


def simanneal_optimize_impl(
    initial_state: EastVariant,
    energy_fn: Callable[[EastVariant], float],
    move_fn: Callable[[EastVariant], EastVariant],
    config: EastStruct,
) -> EastStruct:
    """Run simulated annealing with custom energy and move functions."""
    from simanneal import Annealer

    # Set random seed if provided
    random_state = _get_option(config.get("random_state"), None)
    if random_state is not None:
        random.seed(int(random_state))

    class EastAnnealer(Annealer):
        """Annealer that wraps East energy and move functions."""

        copy_strategy = "method"

        def __init__(self, state, energy_fn, move_fn):
            self.energy_fn = energy_fn
            self.move_fn = move_fn
            super().__init__(state)

        def copy_state(self, state):
            """Custom copy for EastVariant - states are immutable."""
            # EastVariant is immutable, so we can just return it
            return state

        def move(self):
            """Generate neighbor state using East move function."""
            self.state = self.move_fn(self.state)

        def energy(self):
            """Calculate energy using East energy function."""
            return self.energy_fn(self.state)

    # Create annealer instance
    annealer = EastAnnealer(initial_state, energy_fn, move_fn)

    # Configure schedule
    t_max = _get_option(config.get("t_max"), None)
    if t_max is not None:
        annealer.Tmax = float(t_max)

    t_min = _get_option(config.get("t_min"), None)
    if t_min is not None:
        annealer.Tmin = float(t_min)

    steps = _get_option(config.get("steps"), None)
    if steps is not None:
        annealer.steps = int(steps)

    updates = _get_option(config.get("updates"), None)
    if updates is not None:
        annealer.updates = int(updates)
    else:
        annealer.updates = 0  # Suppress output by default

    # Auto-calibrate if requested
    auto_minutes = _get_option(config.get("auto_schedule"), None)
    if auto_minutes is not None:
        schedule = annealer.auto(minutes=float(auto_minutes))
        annealer.set_schedule(schedule)

    # Run optimization
    best_state, best_energy = annealer.anneal()

    return EastStruct(
        {
            "best_state": best_state,
            "best_energy": float(best_energy),
            "steps_taken": int(annealer.steps),
            "success": True,
        }
    )


def simanneal_optimize_permutation_impl(
    initial_perm: EastArray,
    energy_fn: Callable[[EastArray], float],
    config: EastStruct,
) -> EastStruct:
    """Run simulated annealing on a permutation using swap moves."""
    from simanneal import Annealer

    # Set random seed if provided
    random_state = _get_option(config.get("random_state"), None)
    if random_state is not None:
        random.seed(int(random_state))

    # Convert to list for efficient swapping
    state_list = [int(x) for x in initial_perm]
    n = len(state_list)

    class PermutationAnnealer(Annealer):
        """Annealer for permutation problems with swap moves."""

        copy_strategy = "slice"  # Efficient list copying

        def __init__(self, state, energy_fn, n):
            self.energy_fn = energy_fn
            self.n = n
            super().__init__(state)

        def move(self):
            """Swap two random elements."""
            i = random.randint(0, self.n - 1)
            j = random.randint(0, self.n - 1)
            self.state[i], self.state[j] = self.state[j], self.state[i]

        def energy(self):
            """Calculate energy from permutation."""
            perm_array: EastArray = EastArray(IntegerType, self.state)
            return self.energy_fn(perm_array)

    annealer = PermutationAnnealer(state_list, energy_fn, n)

    # Configure schedule
    t_max = _get_option(config.get("t_max"), None)
    if t_max is not None:
        annealer.Tmax = float(t_max)

    t_min = _get_option(config.get("t_min"), None)
    if t_min is not None:
        annealer.Tmin = float(t_min)

    steps = _get_option(config.get("steps"), None)
    if steps is not None:
        annealer.steps = int(steps)

    updates = _get_option(config.get("updates"), None)
    if updates is not None:
        annealer.updates = int(updates)
    else:
        annealer.updates = 0

    auto_minutes = _get_option(config.get("auto_schedule"), None)
    if auto_minutes is not None:
        schedule = annealer.auto(minutes=float(auto_minutes))
        annealer.set_schedule(schedule)

    # Run optimization
    best_state_list, best_energy = annealer.anneal()

    return EastStruct(
        {
            "best_state": EastVariant(
                "int_array", EastArray(IntegerType, best_state_list)
            ),
            "best_energy": float(best_energy),
            "steps_taken": int(annealer.steps),
            "success": True,
        }
    )


def simanneal_optimize_subset_impl(
    initial_selection: EastArray,
    energy_fn: Callable[[EastArray], float],
    config: EastStruct,
) -> EastStruct:
    """Run simulated annealing on a subset selection using bit-flip moves."""
    from simanneal import Annealer

    # Set random seed if provided
    random_state = _get_option(config.get("random_state"), None)
    if random_state is not None:
        random.seed(int(random_state))

    state_list = [bool(x) for x in initial_selection]
    n = len(state_list)

    class SubsetAnnealer(Annealer):
        """Annealer for subset selection with bit-flip moves."""

        copy_strategy = "slice"

        def __init__(self, state, energy_fn, n):
            self.energy_fn = energy_fn
            self.n = n
            super().__init__(state)

        def move(self):
            """Flip a random bit."""
            i = random.randint(0, self.n - 1)
            self.state[i] = not self.state[i]

        def energy(self):
            """Calculate energy from selection."""
            selection_array: EastArray = EastArray(BooleanType, self.state)
            return self.energy_fn(selection_array)

    annealer = SubsetAnnealer(state_list, energy_fn, n)

    # Configure schedule
    t_max = _get_option(config.get("t_max"), None)
    if t_max is not None:
        annealer.Tmax = float(t_max)

    t_min = _get_option(config.get("t_min"), None)
    if t_min is not None:
        annealer.Tmin = float(t_min)

    steps = _get_option(config.get("steps"), None)
    if steps is not None:
        annealer.steps = int(steps)

    updates = _get_option(config.get("updates"), None)
    if updates is not None:
        annealer.updates = int(updates)
    else:
        annealer.updates = 0

    auto_minutes = _get_option(config.get("auto_schedule"), None)
    if auto_minutes is not None:
        schedule = annealer.auto(minutes=float(auto_minutes))
        annealer.set_schedule(schedule)

    best_state_list, best_energy = annealer.anneal()

    return EastStruct(
        {
            "best_state": EastVariant(
                "bool_array", EastArray(BooleanType, best_state_list)
            ),
            "best_energy": float(best_energy),
            "steps_taken": int(annealer.steps),
            "success": True,
        }
    )


# ============================================================================
# Platform Function Registration
# ============================================================================

simanneal_impl = [
    PlatformFunction(
        name="simanneal_optimize",
        inputs=[
            DiscreteStateType,
            EnergyFunctionType,
            MoveFunctionType,
            AnnealConfigType,
        ],
        output=AnnealResultType,
        type="sync",
        fn=simanneal_optimize_impl,
    ),
    PlatformFunction(
        name="simanneal_optimize_permutation",
        inputs=[
            ArrayType(IntegerType),
            PermutationEnergyType,
            AnnealConfigType,
        ],
        output=AnnealResultType,
        type="sync",
        fn=simanneal_optimize_permutation_impl,
    ),
    PlatformFunction(
        name="simanneal_optimize_subset",
        inputs=[
            ArrayType(BooleanType),
            SubsetEnergyType,
            AnnealConfigType,
        ],
        output=AnnealResultType,
        type="sync",
        fn=simanneal_optimize_subset_impl,
    ),
]


__all__ = [
    "simanneal_impl",
    "DiscreteStateType",
    "EnergyFunctionType",
    "MoveFunctionType",
    "PermutationEnergyType",
    "SubsetEnergyType",
    "AnnealConfigType",
    "AnnealResultType",
]
