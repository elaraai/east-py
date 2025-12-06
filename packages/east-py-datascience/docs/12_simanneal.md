# Module 12: Simulated Annealing (`simanneal_impl.py`)

## Purpose

Discrete optimization using Simulated Annealing via the `simanneal` package.

Simulated Annealing is a probabilistic optimization technique inspired by the metallurgic process
of annealing, where metals are cooled at a controlled rate to reach their lowest energy state.
It is particularly effective for:
- Combinatorial optimization problems (e.g., TSP, scheduling, assignment)
- Discrete search spaces where gradient-based methods don't apply
- Problems with many local minima where greedy methods get stuck
- State-based optimization where solutions are permutations, selections, or configurations

## Type Definitions

```python
from east.types.types import (
    ArrayType, FloatType, IntegerType, BooleanType, StringType,
    StructType, VariantType, FunctionType, OptionType, NullType,
)

# State type - variant to support different discrete state representations
DiscreteStateType = VariantType([
    ("int_array", ArrayType(IntegerType)),       # e.g., permutations, assignments
    ("bool_array", ArrayType(BooleanType)),      # e.g., subset selection
    ("string_array", ArrayType(StringType)),     # e.g., categorical sequences
])

# Energy function type: state -> score (lower is better)
EnergyFunctionType = FunctionType([DiscreteStateType], FloatType)

# Move function type: state -> new_state (random neighbor)
MoveFunctionType = FunctionType([DiscreteStateType], DiscreteStateType)

# Annealing schedule configuration
AnnealConfigType = StructType([
    ("t_max", OptionType(FloatType)),            # Starting temperature (default 25000.0)
    ("t_min", OptionType(FloatType)),            # Ending temperature (default 2.5)
    ("steps", OptionType(IntegerType)),          # Total iterations (default 50000)
    ("updates", OptionType(IntegerType)),        # Progress report frequency (default 100)
    ("auto_schedule", OptionType(FloatType)),    # Minutes for auto-calibration (default none)
    ("random_state", OptionType(IntegerType)),   # Random seed for reproducibility
])

# Annealing result
AnnealResultType = StructType([
    ("best_state", DiscreteStateType),           # Best state found
    ("best_energy", FloatType),                  # Energy of best state
    ("steps_taken", IntegerType),                # Actual iterations performed
    ("success", BooleanType),                    # Whether optimization completed
])
```

## Platform Functions

### `simanneal_optimize`

Run simulated annealing optimization on a discrete state space.

```python
PlatformFunction(
    name="simanneal_optimize",
    inputs=[
        DiscreteStateType,                        # initial_state
        EnergyFunctionType,                       # energy_fn: state -> score
        MoveFunctionType,                         # move_fn: state -> neighbor
        AnnealConfigType,                         # config
    ],
    output=AnnealResultType,
    type="sync",
    fn=simanneal_optimize_impl,
)

def simanneal_optimize_impl(
    initial_state: EastVariant,                   # DiscreteStateType
    energy_fn: Callable[[EastVariant], float],    # state -> score
    move_fn: Callable[[EastVariant], EastVariant],# state -> neighbor
    config: EastStruct                            # AnnealConfigType
) -> EastStruct:  # AnnealResultType
    """Run simulated annealing optimization using simanneal."""
    from simanneal import Annealer
    import random

    # Set random seed if provided
    random_state = _get_option(config.get("random_state"), None)
    if random_state is not None:
        random.seed(int(random_state))

    class EastAnnealer(Annealer):
        """Annealer that wraps East energy and move functions."""

        def __init__(self, state, energy_fn, move_fn):
            self.energy_fn = energy_fn
            self.move_fn = move_fn
            super().__init__(state)

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

    return EastStruct({
        "best_state": best_state,
        "best_energy": float(best_energy),
        "steps_taken": int(annealer.steps),
        "success": True,
    })
```

### `simanneal_optimize_permutation`

Convenience function for permutation-based optimization (e.g., TSP, scheduling).
Automatically provides a swap-based move function.

```python
PlatformFunction(
    name="simanneal_optimize_permutation",
    inputs=[
        ArrayType(IntegerType),                   # initial_permutation
        FunctionType([ArrayType(IntegerType)], FloatType),  # energy_fn
        AnnealConfigType,                         # config
    ],
    output=AnnealResultType,
    type="sync",
    fn=simanneal_optimize_permutation_impl,
)

def simanneal_optimize_permutation_impl(
    initial_perm: EastArray,                      # Array[Integer]
    energy_fn: Callable[[EastArray], float],      # perm -> score
    config: EastStruct                            # AnnealConfigType
) -> EastStruct:  # AnnealResultType
    """Run simulated annealing on a permutation using swap moves."""
    from simanneal import Annealer
    import random

    random_state = _get_option(config.get("random_state"), None)
    if random_state is not None:
        random.seed(int(random_state))

    # Convert to list for efficient swapping
    state_list = list(initial_perm)
    n = len(state_list)

    class PermutationAnnealer(Annealer):
        """Annealer for permutation problems with swap moves."""

        copy_strategy = 'slice'  # Efficient list copying

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
            perm_array = EastArray(IntegerType, self.state)
            return self.energy_fn(perm_array)

    annealer = PermutationAnnealer(state_list, energy_fn, n)

    # Configure schedule (same as simanneal_optimize)
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

    return EastStruct({
        "best_state": EastVariant("int_array", EastArray(IntegerType, best_state_list)),
        "best_energy": float(best_energy),
        "steps_taken": int(annealer.steps),
        "success": True,
    })
```

### `simanneal_optimize_subset`

Convenience function for subset selection optimization (e.g., feature selection, knapsack).
Uses bit-flip moves.

```python
PlatformFunction(
    name="simanneal_optimize_subset",
    inputs=[
        ArrayType(BooleanType),                   # initial_selection
        FunctionType([ArrayType(BooleanType)], FloatType),  # energy_fn
        AnnealConfigType,                         # config
    ],
    output=AnnealResultType,
    type="sync",
    fn=simanneal_optimize_subset_impl,
)

def simanneal_optimize_subset_impl(
    initial_selection: EastArray,                 # Array[Boolean]
    energy_fn: Callable[[EastArray], float],      # selection -> score
    config: EastStruct                            # AnnealConfigType
) -> EastStruct:  # AnnealResultType
    """Run simulated annealing on a subset selection using bit-flip moves."""
    from simanneal import Annealer
    import random

    random_state = _get_option(config.get("random_state"), None)
    if random_state is not None:
        random.seed(int(random_state))

    state_list = list(initial_selection)
    n = len(state_list)

    class SubsetAnnealer(Annealer):
        """Annealer for subset selection with bit-flip moves."""

        copy_strategy = 'slice'

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
            selection_array = EastArray(BooleanType, self.state)
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

    return EastStruct({
        "best_state": EastVariant("bool_array", EastArray(BooleanType, best_state_list)),
        "best_energy": float(best_energy),
        "steps_taken": int(annealer.steps),
        "success": True,
    })
```

## Usage Examples

### Travelling Salesman Problem (TSP)

```east
// Distance matrix (city i to city j)
let distances: Matrix = [
    [0.0,  29.0, 82.0, 46.0, 68.0],
    [29.0, 0.0,  55.0, 46.0, 42.0],
    [82.0, 55.0, 0.0,  68.0, 46.0],
    [46.0, 46.0, 68.0, 0.0,  82.0],
    [68.0, 42.0, 46.0, 82.0, 0.0],
];

// Energy: total route distance
let energy = fn(route: Array[Int]) -> Float {
    let total = 0.0;
    let n = Array.length(route);
    for i in 0..n {
        let from = route[i];
        let to = route[(i + 1) % n];
        total = total + distances[from][to];
    }
    total
};

// Initial route: 0, 1, 2, 3, 4
let initial = [0, 1, 2, 3, 4];

let config = {
    t_max: some(10000.0),
    t_min: some(1.0),
    steps: some(50000),
    updates: none,
    auto_schedule: none,
    random_state: some(42),
};

let result = simanneal_optimize_permutation(initial, energy, config);

if result.success {
    print("Best route: " ++ array_to_string(result.best_state));
    print("Total distance: " ++ float_to_string(result.best_energy));
}
```

### Feature Selection

```east
// Select best subset of features for a model
let n_features = 20;

// Energy: model error with selected features (lower is better)
// In practice, this would train/evaluate a model
let energy = fn(selected: Array[Bool]) -> Float {
    let n_selected = 0;
    for i in 0..Array.length(selected) {
        if selected[i] { n_selected = n_selected + 1; }
    }

    // Penalty for too many or too few features
    let ideal = 5;
    let penalty = abs(n_selected - ideal) * 0.1;

    // Simulated model error (replace with actual evaluation)
    let base_error = 0.5;
    base_error + penalty
};

// Start with all features selected
let initial = Array.repeat(true, n_features);

let config = {
    t_max: some(5000.0),
    t_min: some(0.5),
    steps: some(10000),
    updates: none,
    auto_schedule: none,
    random_state: some(123),
};

let result = simanneal_optimize_subset(initial, energy, config);

print("Selected features: ");
for i in 0..n_features {
    if result.best_state[i] {
        print("  Feature " ++ int_to_string(i));
    }
}
```

### Job Scheduling

```east
// Schedule n jobs on m machines to minimize makespan
let n_jobs = 10;
let m_machines = 3;
let job_times = [5, 8, 3, 7, 2, 9, 4, 6, 1, 8];

// State: assignment[i] = machine for job i
// Energy: makespan (max machine load)
let energy = fn(assignment: Array[Int]) -> Float {
    let loads = Array.repeat(0.0, m_machines);
    for i in 0..n_jobs {
        let machine = assignment[i];
        loads[machine] = loads[machine] + job_times[i];
    }
    Array.max(loads)  // Minimize maximum load
};

// Move: reassign random job to random machine
let move = fn(assignment: Array[Int]) -> Array[Int] {
    let new_assignment = Array.copy(assignment);
    let job = random_int(0, n_jobs - 1);
    let machine = random_int(0, m_machines - 1);
    new_assignment[job] = machine;
    new_assignment
};

// Initial: round-robin assignment
let initial = [];
for i in 0..n_jobs {
    initial = Array.push(initial, i % m_machines);
}

let config = {
    t_max: some(1000.0),
    t_min: some(0.1),
    steps: some(20000),
    updates: none,
    auto_schedule: none,
    random_state: some(42),
};

// Use generic optimizer with custom move function
let result = simanneal_optimize(
    #int_array(initial),
    fn(state) { match state { #int_array(a) => energy(a) } },
    fn(state) { match state { #int_array(a) => #int_array(move(a)) } },
    config
);
```

### Auto-Calibration

For problems where good temperature parameters are unknown:

```east
let config = {
    t_max: none,           // Will be auto-calibrated
    t_min: none,           // Will be auto-calibrated
    steps: none,           // Will be auto-calibrated
    updates: none,
    auto_schedule: some(2.0),  // Spend 2 minutes calibrating
    random_state: some(42),
};

let result = simanneal_optimize_permutation(initial, energy, config);
```

## Notes

- Simulated annealing is a metaheuristic - it does not guarantee finding the global optimum but often finds good solutions quickly
- The temperature schedule (Tmax, Tmin, steps) significantly affects performance and should be tuned for each problem
- Use `auto_schedule` for automatic parameter calibration when problem characteristics are unknown
- For large state spaces, ensure the move function generates "nearby" neighbors for efficient exploration
- The algorithm accepts worse solutions probabilistically based on temperature, allowing escape from local minima
- Lower temperatures reduce acceptance of worse solutions; the algorithm becomes more greedy as it cools
- For reproducible results, always set `random_state`
