# Module 12: ALNS (`alns_impl.py`)

## Purpose

Adaptive Large Neighborhood Search (ALNS) metaheuristic for combinatorial optimization problems like roster scheduling, vehicle routing, and resource allocation.

ALNS is designed for discrete optimization problems where:
- Solutions are combinatorial (assignments, schedules, routes)
- Domain-specific destroy/repair operators can be defined
- The objective function may be complex or black-box
- Local search alone gets stuck in local minima

## Key Concepts

### Destroy-Repair Paradigm

ALNS iteratively improves solutions by:
1. **Destroy**: Remove part of the current solution (e.g., unassign some shifts)
2. **Repair**: Reconstruct a complete solution (e.g., reassign shifts greedily)
3. **Accept/Reject**: Decide whether to keep the new solution
4. **Adapt**: Update operator weights based on performance

### Adaptive Operator Selection

ALNS maintains weights for each operator and selects them probabilistically. Operators that find improving solutions get higher weights over time.

## Platform Functions

### `alns_optimize`

Single-objective combinatorial optimization with injectable operators.

The function is generic over solution type `S` - users provide their own struct type.

```typescript
// TypeScript type definition
export const alns_optimize = <S extends EastType>(solutionType: S) =>
    East.platform(
        "alns_optimize",
        [
            solutionType,                                           // initial_solution: S
            FunctionType([solutionType], FloatType),                // objective: S -> Float
            ArrayType(FunctionType([solutionType], solutionType)),  // destroy_operators: S -> S
            ArrayType(FunctionType([solutionType], solutionType)),  // repair_operators: S -> S
            ALNSConfigType,
        ],
        ALNSResultType(solutionType)
    );
```

```python
# Python implementation
PlatformFunction(
    name="alns_optimize",
    inputs=[
        SolutionType,                              # initial_solution: S (user-defined)
        FunctionType([SolutionType], FloatType),   # objective: S -> Float
        ArrayType(FunctionType([SolutionType], SolutionType)),  # destroy_operators: S -> S
        ArrayType(FunctionType([SolutionType], SolutionType)),  # repair_operators: S -> S
        ALNSConfigType,
    ],
    output=ALNSResultType,
    type="sync",
    fn=alns_optimize_impl,
)

def alns_optimize_impl(
    initial_solution: Any,
    objective_fn: Callable[[Any], float],
    destroy_operators: EastArray,  # Array[S -> S]
    repair_operators: EastArray,   # Array[S -> S]
    config: EastStruct
) -> EastStruct:
    """Run ALNS optimization using the alns library.

    Note: East-compiled functions are callable from Python. When you define
    `East.function([SolutionType], SolutionType, ...)` and compile it, the
    result is a Python callable that accepts/returns East values.

    Helper functions used:
    - _get_option(value, default): Unwraps Option types, returns default if None
    - _get_enum_tag(variant): Returns the tag name of an EastVariant
    """
    import alns
    from alns.accept import SimulatedAnnealing, HillClimbing, RecordToRecordTravel
    from alns.select import RouletteWheel
    from alns.stop import MaxIterations, MaxRuntime, NoImprovement
    import numpy as np

    # Create random state
    seed = int(_get_option(config.get("seed"), 42))
    rng = np.random.default_rng(seed)

    # Wrap solution in State class (required by alns library)
    class SolutionState:
        def __init__(self, solution):
            self.solution = solution
            self._objective = None

        def objective(self) -> float:
            if self._objective is None:
                self._objective = objective_fn(self.solution)
            return self._objective

    # Wrap destroy operators (S -> S)
    # Note: alns library passes rng to operators, but East operators manage
    # their own randomness via platform functions. The seed in config ensures
    # reproducibility at the ALNS level (operator selection, acceptance).
    def make_destroy(destroy_fn):
        def destroy(state: SolutionState, rng) -> SolutionState:
            destroyed = destroy_fn(state.solution)
            return SolutionState(destroyed)
        return destroy

    # Wrap repair operators (S -> S)
    def make_repair(repair_fn):
        def repair(state: SolutionState, rng) -> SolutionState:
            repaired = repair_fn(state.solution)
            return SolutionState(repaired)
        return repair

    # Build ALNS instance
    alns_instance = alns.ALNS(rng)

    for i, destroy_fn in enumerate(destroy_operators):
        alns_instance.add_destroy_operator(make_destroy(destroy_fn), name=f"destroy_{i}")

    for i, repair_fn in enumerate(repair_operators):
        alns_instance.add_repair_operator(make_repair(repair_fn), name=f"repair_{i}")

    # Configure acceptance criterion
    accept_config = _get_option(config.get("acceptance"), None)
    if accept_config is None or _get_enum_tag(accept_config) == "simulated_annealing":
        sa_config = accept_config.value if accept_config else {}
        start_temp = float(_get_option(sa_config.get("start_temperature") if sa_config else None, 100.0))
        end_temp = float(_get_option(sa_config.get("end_temperature") if sa_config else None, 0.01))
        step = float(_get_option(sa_config.get("step") if sa_config else None, 0.99))
        accept = SimulatedAnnealing(start_temp, end_temp, step)
    elif _get_enum_tag(accept_config) == "hill_climbing":
        accept = HillClimbing()
    elif _get_enum_tag(accept_config) == "record_to_record":
        rtr_config = accept_config.value
        threshold = float(_get_option(rtr_config.get("threshold"), 0.05))
        accept = RecordToRecordTravel(threshold)

    # Configure operator selection
    select_config = _get_option(config.get("operator_selection"), None)
    rw_config = select_config.value if select_config else {}
    scores = _get_option(rw_config.get("scores") if rw_config else None, [33, 9, 3, 0])
    decay = float(_get_option(rw_config.get("decay") if rw_config else None, 0.8))
    num_destroy = len(destroy_operators)
    num_repair = len(repair_operators)
    select = RouletteWheel(scores, decay, num_destroy, num_repair)

    # Configure stopping criterion
    stop_config = _get_option(config.get("stop"), None)
    if stop_config is None or _get_enum_tag(stop_config) == "max_iterations":
        max_iter = int(stop_config.value) if stop_config else 1000
        stop = MaxIterations(max_iter)
    elif _get_enum_tag(stop_config) == "max_runtime":
        max_time = float(stop_config.value)
        stop = MaxRuntime(max_time)
    elif _get_enum_tag(stop_config) == "no_improvement":
        max_iter = int(stop_config.value)
        stop = NoImprovement(max_iter)

    # Run optimization with error handling
    initial_state = SolutionState(initial_solution)
    try:
        result = alns_instance.iterate(initial_state, select, accept, stop)
        best_state = result.best_state
        statistics = result.statistics

        return EastStruct({
            "best_solution": best_state.solution,
            "best_objective": float(best_state.objective()),
            "iterations": int(len(statistics.objectives)),
            "runtime": float(statistics.total_runtime),
            "success": True,
        })
    except Exception as e:
        # Return failure result with initial solution
        return EastStruct({
            "best_solution": initial_solution,
            "best_objective": float('inf'),
            "iterations": 0,
            "runtime": 0.0,
            "success": False,
        })
```

## Type Definitions

### Generic Solution Type

The solution type `S` is **entirely user-defined**. The library is generic over it - users pass in their own struct type representing their problem domain.

```typescript
// The library does NOT define a solution type.
// Users provide their own, e.g.:
//   - Roster: { assignments: Array<{employee, shift}>, unassigned: Array<String> }
//   - VRP: { routes: Array<{vehicle, stops}>, unvisited: Array<String> }
//   - Scheduling: { jobs: Array<{task, machine, start_time}> }

// Operator types are FunctionTypes parameterized by user's solution type S:

// Objective function type
FunctionType([S], FloatType)  // S -> Float (to minimize)

// Destroy operator type
FunctionType([S], S)  // S -> S (partially destroys the solution)

// Repair operator type
FunctionType([S], S)  // S -> S (repairs the destroyed solution)

// The platform function accepts arrays of these function types
ArrayType(FunctionType([S], S))  // destroy operators
ArrayType(FunctionType([S], S))  // repair operators
```

### Acceptance Criteria

```typescript
export const SimulatedAnnealingConfigType = StructType({
    start_temperature: OptionType(FloatType),  // default: 100.0
    end_temperature: OptionType(FloatType),    // default: 0.01
    step: OptionType(FloatType),               // default: 0.99 (multiplicative cooling)
});

export const RecordToRecordConfigType = StructType({
    threshold: OptionType(FloatType),  // default: 0.05 (5% deviation allowed)
});

export const AcceptanceCriterionType = VariantType({
    simulated_annealing: SimulatedAnnealingConfigType,
    hill_climbing: NullType,
    record_to_record: RecordToRecordConfigType,
});
```

### Operator Selection

```typescript
export const RouletteWheelConfigType = StructType({
    // Scores for: [new best, better, accepted, rejected]
    scores: OptionType(ArrayType(IntegerType)),  // default: [33, 9, 3, 0]
    decay: OptionType(FloatType),                // default: 0.8
});

export const SegmentedRouletteWheelConfigType = StructType({
    scores: OptionType(ArrayType(IntegerType)),
    decay: OptionType(FloatType),
    segment_length: OptionType(IntegerType),  // default: 100
});

export const OperatorSelectionType = VariantType({
    roulette_wheel: RouletteWheelConfigType,
    segmented_roulette_wheel: SegmentedRouletteWheelConfigType,
});
```

### Stopping Criteria

```typescript
export const StopCriterionType = VariantType({
    max_iterations: IntegerType,
    max_runtime: FloatType,       // seconds
    no_improvement: IntegerType,  // iterations without improvement
});
```

### Configuration

```typescript
export const ALNSConfigType = StructType({
    /** Stopping criterion */
    stop: OptionType(StopCriterionType),
    /** Acceptance criterion */
    acceptance: OptionType(AcceptanceCriterionType),
    /** Operator selection strategy */
    operator_selection: OptionType(OperatorSelectionType),
    /** Degree of destruction (0.0-1.0) */
    destroy_degree: OptionType(FloatType),
    /** Random seed for reproducibility */
    seed: OptionType(IntegerType),
});
```

### Result

```typescript
// Generic result type parameterized by solution type S
export const ALNSResultType = <S extends EastType>(solutionType: S) =>
    StructType({
        /** Best solution found */
        best_solution: solutionType,  // Same type S as input
        /** Best objective value */
        best_objective: FloatType,
        /** Number of iterations performed */
        iterations: IntegerType,
        /** Total runtime in seconds */
        runtime: FloatType,
        /** Whether optimization succeeded */
        success: BooleanType,
    });
```

## Usage Examples

### Roster Optimization

```typescript
import {
    East,
    StructType,
    ArrayType,
    SetType,
    DictType,
    StringType,
    DateTimeType,
    FloatType,
    IntegerType,
    BooleanType,
    OptionType,
    variant,
} from "@elaraai/east";
import { ALNS } from "@elaraai/east-py-datascience";

// =============================================================================
// Domain Types (user-defined, not part of library)
// =============================================================================

/** Employee with skills, availability, and cost */
const EmployeeType = StructType({
    id: StringType,
    name: StringType,
    skills: SetType(StringType),              // e.g., {"nursing", "medication"}
    hourly_rate: FloatType,
    max_hours_per_week: FloatType,
    availability: DictType(DateTimeType, ArrayType(IntegerType)),  // date -> available shift slots
});

/** Shift requirement */
const ShiftType = StructType({
    id: StringType,
    date: DateTimeType,
    start_hour: IntegerType,
    end_hour: IntegerType,
    required_skills: SetType(StringType),
    location: StringType,
    priority: IntegerType,                    // 1=critical, 2=important, 3=optional
});

/** Assignment of employee to shift */
const AssignmentType = StructType({
    employee_id: StringType,
    shift_id: StringType,
});

/** Complete roster solution */
const RosterSolutionType = StructType({
    assignments: ArrayType(AssignmentType),
    unassigned_shifts: ArrayType(StringType),
});

/** Problem data (employees, shifts, constraints) */
const RosterProblemType = StructType({
    employees: DictType(StringType, EmployeeType),
    shifts: DictType(StringType, ShiftType),
    min_rest_hours: IntegerType,              // minimum hours between shifts
});

// =============================================================================
// Helper Functions
// =============================================================================

/** Check if employee has required skills for shift */
const hasRequiredSkills = East.function(
    [EmployeeType, ShiftType],
    BooleanType,
    ($, employee, shift) => {
        $.return(shift.required_skills.isSubsetOf(employee.skills));
    }
);

/** Calculate hours worked by employee in current assignments */
const getEmployeeHours = East.function(
    [StringType, ArrayType(AssignmentType), DictType(StringType, ShiftType)],
    FloatType,
    ($, employeeId, assignments, shifts) => {
        const hours = $.let(assignments
            .filter(($, a) => a.employee_id.equals(employeeId))
            .reduce(($, acc, a) => {
                const shift = $.let(shifts.get(a.shift_id));
                const duration = $.let(shift.end_hour.subtract(shift.start_hour).toFloat());
                return acc.add(duration);
            }, East.value(0.0)));
        $.return(hours);
    }
);

// =============================================================================
// Main Optimization
// =============================================================================

/**
 * Main optimization function.
 *
 * IMPORTANT PATTERN: Operators are defined INLINE inside this function where
 * `problem` is in lexical scope. This allows operators to access problem data
 * without needing factory functions or closures.
 */
const optimizeRoster = East.function(
    [RosterProblemType, RosterSolutionType],
    ALNS.Types.ResultType(RosterSolutionType),
    ($, problem, initialSolution) => {
        // ---------------------------------------------------------------------
        // Objective function (defined inline - has access to `problem`)
        // ---------------------------------------------------------------------
        const objective = East.function([RosterSolutionType], FloatType, ($, solution) => {
            // Labor cost
            const laborCost = $.let(solution.assignments.reduce(($, acc, assignment) => {
                const employee = $.let(problem.employees.get(assignment.employee_id));
                const shift = $.let(problem.shifts.get(assignment.shift_id));
                const hours = $.let(shift.end_hour.subtract(shift.start_hour).toFloat());
                return acc.add(employee.hourly_rate.multiply(hours));
            }, East.value(0.0)));

            // Unassigned shift penalty
            const unassignedPenalty = $.let(solution.unassigned_shifts.reduce(($, acc, shiftId) => {
                const shift = $.let(problem.shifts.get(shiftId));
                const penalty = $.let(
                    shift.priority.equals(East.value(1n)).ifElse(
                        $ => East.value(10000.0),
                        $ => East.value(1000.0)
                    )
                );
                return acc.add(penalty);
            }, East.value(0.0)));

            $.return(laborCost.add(unassignedPenalty));
        });

        // ---------------------------------------------------------------------
        // Destroy operators (S -> S, defined inline - have access to `problem`)
        // ---------------------------------------------------------------------

        /** Random removal: remove ~20% of assignments randomly */
        const randomDestroy = East.function(
            [RosterSolutionType],
            RosterSolutionType,
            ($, solution) => {
                // Destroy degree is configured via ALNSConfig, not passed to operator
                const destroyFraction = 0.2;
                const nRemove = $.let(
                    East.max(
                        East.value(1n),
                        solution.assignments.size().toFloat().multiply(East.value(destroyFraction)).toInteger()
                    )
                );
                // ... random removal logic using platform Random functions
                // Returns solution with some assignments moved to unassigned_shifts
                $.return(solution);  // Simplified for illustration
            }
        );

        /** Day destruction: remove all assignments for a random day */
        const dayDestroy = East.function(
            [RosterSolutionType],
            RosterSolutionType,
            ($, solution) => {
                // Get unique days from assignments - uses `problem` from outer scope
                const days = $.let(solution.assignments.map(($, a) => {
                    return problem.shifts.get(a.shift_id).date;
                }).toSet());

                $.if(days.size().equals(East.value(0n)), $ => {
                    $.return(solution);
                });

                // Pick random day to clear, partition assignments
                // ... day selection and filtering logic
                $.return(solution);  // Simplified for illustration
            }
        );

        // ---------------------------------------------------------------------
        // Repair operators (defined inline - have access to `problem`)
        // ---------------------------------------------------------------------

        /** Greedy repair: assign each shift to cheapest available employee */
        const greedyRepair = East.function(
            [RosterSolutionType],
            RosterSolutionType,
            ($, solution) => {
                const assignments = $.let(solution.assignments.copy());
                const unassigned = $.let(solution.unassigned_shifts.copy());
                const stillUnassigned = $.let(East.Array.empty(StringType));

                $.while(unassigned.size().greaterThan(East.value(0n)), $ => {
                    const shiftId = $.let(unassigned.popFirst());
                    const shift = $.let(problem.shifts.get(shiftId));  // Uses `problem` from outer scope

                    // Find cheapest available employee
                    const bestEmployee = $.let(East.value(variant("none", null) as OptionType<string>));
                    const bestCost = $.let(East.value(Infinity));

                    $(problem.employees.forEach(($, employeeId, employee) => {
                        // Check skills
                        const hasSkills = $.let(shift.required_skills.isSubsetOf(employee.skills));

                        $.if(hasSkills, $ => {
                            const shiftHours = $.let(shift.end_hour.subtract(shift.start_hour).toFloat());
                            const cost = $.let(employee.hourly_rate.multiply(shiftHours));
                            $.if(cost.lessThan(bestCost), $ => {
                                $.assign(bestCost, cost);
                                $.assign(bestEmployee, variant("some", employeeId));
                            });
                        });
                    }));

                    $.if(bestEmployee.hasTag("some"), $ => {
                        $(assignments.pushLast({
                            employee_id: bestEmployee.unwrap("some"),
                            shift_id: shiftId,
                        }));
                    }).else($ => {
                        $(stillUnassigned.pushLast(shiftId));
                    });
                });

                $.return({ assignments: assignments, unassigned_shifts: stillUnassigned });
            }
        );

        // ---------------------------------------------------------------------
        // Configuration
        // ---------------------------------------------------------------------
        const config = $.let({
            stop: variant("some", variant("max_iterations", 10000n)),
            acceptance: variant("some", variant("simulated_annealing", {
                start_temperature: variant("some", 1000.0),
                end_temperature: variant("some", 0.1),
                step: variant("some", 0.9995),
            })),
            operator_selection: variant("some", variant("roulette_wheel", {
                scores: variant("some", [33n, 9n, 3n, 0n]),
                decay: variant("some", 0.8),
            })),
            destroy_degree: variant("some", 0.2),
            seed: variant("some", 42n),
        });

        // ---------------------------------------------------------------------
        // Run ALNS optimization
        // ---------------------------------------------------------------------
        // Operators are passed directly as arrays - they're defined above
        // with `problem` in scope, so no factory functions needed
        const result = $.let(ALNS.optimize(RosterSolutionType)(
            initialSolution,
            objective,                        // S -> Float
            [randomDestroy, dayDestroy],      // Array<S -> S>
            [greedyRepair],                   // Array<S -> S>
            config
        ));

        $(Console.log(East.str`=== ALNS Roster Optimization Complete ===`));
        $(Console.log(East.str`Best objective: $${result.best_objective}`));
        $(Console.log(East.str`Iterations: ${result.iterations}`));
        $(Console.log(East.str`Runtime: ${result.runtime}s`));
        $(Console.log(East.str`Assigned shifts: ${result.best_solution.assignments.size()}`));
        $(Console.log(East.str`Unassigned shifts: ${result.best_solution.unassigned_shifts.size()}`));

        $.return(result);
    }
);
```

### Vehicle Routing Problem

```typescript
import { East, StructType, ArrayType, DictType, StringType, FloatType, variant } from "@elaraai/east";
import { ALNS } from "@elaraai/east-py-datascience";

// User-defined types
const CustomerType = StructType({
    id: StringType,
    x: FloatType,
    y: FloatType,
    demand: FloatType,
});

const VRPProblemType = StructType({
    customers: DictType(StringType, CustomerType),
    vehicle_capacity: FloatType,
    depot: CustomerType,
});

const RouteType = StructType({
    vehicle_id: StringType,
    stops: ArrayType(StringType),
    load: FloatType,
});

const VRPSolutionType = StructType({
    routes: ArrayType(RouteType),
    unvisited: ArrayType(StringType),
});

// Main optimization - operators defined inline with `problem` in scope
const optimizeVRP = East.function(
    [VRPProblemType, VRPSolutionType],
    ALNS.Types.ResultType(VRPSolutionType),
    ($, problem, initialSolution) => {
        // Objective: minimize total distance + penalty for unvisited
        const objective = East.function([VRPSolutionType], FloatType, ($, solution) => {
            const totalDistance = $.let(solution.routes.reduce(($, acc, route) => {
                // Uses `problem` from outer scope for customer locations
                const routeDist = $.let(route.stops.reduce(($, d, stopId) => {
                    const customer = $.let(problem.customers.get(stopId));
                    // Simplified distance calculation
                    return d.add(customer.x.abs().add(customer.y.abs()));
                }, East.value(0.0)));
                return acc.add(routeDist);
            }, East.value(0.0)));
            const penalty = $.let(solution.unvisited.size().toFloat().multiply(East.value(10000.0)));
            $.return(totalDistance.add(penalty));
        });

        // Shaw removal: remove geographically similar customers
        const shawDestroy = East.function(
            [VRPSolutionType],
            VRPSolutionType,
            ($, solution) => {
                // Uses `problem` from outer scope for customer locations
                // ... implementation
                $.return(solution);
            }
        );

        // Greedy repair: insert customers at cheapest position
        const greedyRepair = East.function([VRPSolutionType], VRPSolutionType, ($, solution) => {
            // Uses `problem` from outer scope for distances and capacity
            // ... implementation
            $.return(solution);
        });

        const config = $.let({
            stop: variant("some", variant("max_iterations", 5000n)),
            destroy_degree: variant("some", 0.25),
            seed: variant("some", 42n),
            acceptance: variant("none", null),
            operator_selection: variant("none", null),
        });

        const result = $.let(ALNS.optimize(VRPSolutionType)(
            initialSolution,
            objective,
            [shawDestroy],
            [greedyRepair],
            config
        ));

        $.return(result);
    }
);
```

## Notes

**Pattern for defining operators**: Define operators inline inside the main `East.function()` where problem data is in lexical scope. This allows operators to access problem data (e.g., distance matrices, constraints) without passing it as a parameter.

**Destroy degree**: The `destroy_degree` in config is a hint for how much of the solution to destroy. Since operators have signature `S -> S`, they should either hardcode their destruction amount or access a shared degree value from their lexical scope. The ALNS library doesn't pass degree directly to operators.

- ALNS excels at problems where domain-specific destroy/repair operators can be designed
- The adaptive mechanism automatically learns which operators work best for the problem
- Simulated annealing acceptance allows escaping local minima early, then intensifies search
- Destroy degree controls exploration vs exploitation trade-off
- Multiple diverse operators typically outperform a single sophisticated operator
- The algorithm is stochastic; use seeds for reproducibility

## Dependencies

- `alns` - Python ALNS implementation by N. Wouda
- `numpy` - For random number generation

```bash
pip install alns numpy
```

## Comparison with MADS

| Aspect | ALNS | MADS |
|--------|------|------|
| Problem type | Combinatorial/discrete | Continuous |
| Solution space | User-defined structure | Real-valued vectors |
| Search mechanism | Destroy/repair operators | Mesh-based polling |
| Customization | Operators + objective | Objective only |
| Best for | Scheduling, routing, assignment | Parameter tuning, simulation |
