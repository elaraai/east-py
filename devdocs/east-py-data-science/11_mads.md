# Module 11: MADS (`mads_impl.py`)

## Purpose

Derivative-free blackbox optimization using NOMAD's MADS (Mesh Adaptive Direct Search) algorithm via PyNomadBBO.

MADS is designed for difficult blackbox optimization problems where:
- Functions have no exploitable derivatives
- Evaluations are computationally expensive
- Functions may be contaminated by noise
- Functions may fail for some feasible points

## Platform Functions

### `mads_optimize`

Single-objective optimization with optional constraints.

```python
PlatformFunction(
    name="mads_optimize",
    inputs=[
        ScalarObjectiveType,                    # objective: x -> f(x)
        VectorType,                             # x0 (starting point)
        MADSBoundsType,                         # bounds
        OptionType(ArrayType(MADSConstraintType)),  # constraints (optional)
        MADSConfigType,                         # config
    ],
    output=MADSResultType,
    type="sync",
    fn=mads_optimize_impl,
)

def mads_optimize_impl(
    objective_fn: Callable[[Vector], float],
    x0: Vector,
    bounds: EastStruct,
    constraints: EastArray | None,  # Array[MADSConstraint] or None
    config: EastStruct
) -> EastStruct:
    """Run MADS optimization using PyNomadBBO."""
    import PyNomad

    x0_list = list(east_vector_to_numpy(x0))
    lb_list = list(east_vector_to_numpy(bounds["lower"]))
    ub_list = list(east_vector_to_numpy(bounds["upper"]))
    dim = len(x0_list)

    # Extract constraints if provided
    constraint_list = []
    if is_east_option(constraints) and constraints.is_some():
        constraint_list = list(constraints.value)

    # Build blackbox function
    def bb(x):
        try:
            # Extract coordinates into East vector
            x_vec = EastArray(FloatType, [x.get_coord(i) for i in range(x.size())])

            # Evaluate objective
            f = objective_fn(x_vec)
            outputs = [str(f)]

            # Evaluate each constraint (variant value is the function)
            for constraint in constraint_list:
                c_fn = constraint.value  # The function is the variant value
                c_val = c_fn(x_vec)
                outputs.append(str(c_val))

            raw_bbo = " ".join(outputs)
            x.setBBO(raw_bbo.encode("UTF-8"))
            return 1
        except Exception:
            return 0

    # Build output type string: OBJ followed by constraint types
    output_types = ["OBJ"]
    for constraint in constraint_list:
        kind = constraint.type  # The variant tag indicates eb or pb
        output_types.append("EB" if kind == "eb" else "PB")

    # Build parameters
    params = [
        f"DIMENSION {dim}",
        f"BB_OUTPUT_TYPE {' '.join(output_types)}",
        f"MAX_BB_EVAL {_get_option(config.get('max_bb_eval'), 100)}",
        f"DISPLAY_DEGREE {_get_option(config.get('display_degree'), 0)}",
    ]

    direction = _get_option(config.get("direction_type"), None)
    if direction:
        direction_name = _get_enum_tag(direction)
        direction_map = {
            "ortho_2n": "ORTHO 2N",
            "ortho_n_plus_1": "ORTHO N+1",
            "lt_2n": "LT 2N",
            "single": "SINGLE",
        }
        params.append(f"DIRECTION_TYPE {direction_map.get(direction_name, 'ORTHO 2N')}")

    mesh_size = _get_option(config.get("initial_mesh_size"), None)
    if mesh_size:
        params.append(f"INITIAL_MESH_SIZE {mesh_size}")

    min_mesh = _get_option(config.get("min_mesh_size"), None)
    if min_mesh:
        params.append(f"MIN_MESH_SIZE {min_mesh}")

    seed = _get_option(config.get("seed"), None)
    if seed is not None:
        params.append(f"SEED {seed}")

    # Run optimization
    result = PyNomad.optimize(bb, x0_list, lb_list, ub_list, params)

    # Extract results
    x_best = result.get("x_best", x0_list)
    f_best = result.get("f_best", float("inf"))
    bb_eval = result.get("bb_eval", 0)

    return EastStruct({
        "x_best": EastArray(FloatType, [float(v) for v in x_best]),
        "f_best": float(f_best),
        "bb_eval": int(bb_eval),
        "success": f_best != float("inf"),
    })
```

### `mads_optimize_multi`

Multi-objective optimization returning a Pareto front.

```python
PlatformFunction(
    name="mads_optimize_multi",
    inputs=[
        ArrayType(ScalarObjectiveType),         # objectives (multiple)
        VectorType,                             # x0
        MADSBoundsType,                         # bounds
        MADSConfigType,                         # config
    ],
    output=MADSMultiResultType,
    type="sync",
    fn=mads_optimize_multi_impl,
)

def mads_optimize_multi_impl(
    objective_fns: EastArray,  # Array[ObjectiveFunction]
    x0: Vector,
    bounds: EastStruct,
    config: EastStruct
) -> EastStruct:
    """Run multi-objective MADS optimization using PyNomadBBO."""
    import PyNomad
    import numpy as np

    x0_list = list(east_vector_to_numpy(x0))
    lb_list = list(east_vector_to_numpy(bounds["lower"]))
    ub_list = list(east_vector_to_numpy(bounds["upper"]))
    dim = len(x0_list)
    n_obj = len(objective_fns)

    # Build blackbox function
    def bb(x):
        try:
            x_vec = EastArray(FloatType, [x.get_coord(i) for i in range(x.size())])

            # Evaluate all objectives
            outputs = []
            for obj_fn in objective_fns:
                f = obj_fn(x_vec)
                outputs.append(str(f))

            raw_bbo = " ".join(outputs)
            x.setBBO(raw_bbo.encode("UTF-8"))
            return 1
        except Exception:
            return 0

    # Build parameters for multi-objective
    params = [
        f"DIMENSION {dim}",
        f"BB_OUTPUT_TYPE {' '.join(['OBJ'] * n_obj)}",
        f"MAX_BB_EVAL {_get_option(config.get('max_bb_eval'), 100)}",
        f"DISPLAY_DEGREE {_get_option(config.get('display_degree'), 0)}",
        "DMULTIMADS_OPTIMIZATION yes",
    ]

    direction = _get_option(config.get("direction_type"), None)
    if direction:
        direction_name = _get_enum_tag(direction)
        direction_map = {
            "ortho_2n": "ORTHO 2N",
            "ortho_n_plus_1": "ORTHO N+1",
            "lt_2n": "LT 2N",
            "single": "SINGLE",
        }
        params.append(f"DIRECTION_TYPE {direction_map.get(direction_name, 'ORTHO 2N')}")

    seed = _get_option(config.get("seed"), None)
    if seed is not None:
        params.append(f"SEED {seed}")

    # Run optimization
    result = PyNomad.optimize(bb, x0_list, lb_list, ub_list, params)

    # Extract Pareto front
    pareto_x = result.get("pareto_front", [])
    pareto_f = result.get("pareto_values", [])
    bb_eval = result.get("bb_eval", 0)

    # Convert to matrices
    if pareto_x:
        pareto_front = numpy_to_east_matrix(np.array(pareto_x))
        pareto_values = numpy_to_east_matrix(np.array(pareto_f))
    else:
        pareto_front = EastArray(ArrayType(FloatType), [])
        pareto_values = EastArray(ArrayType(FloatType), [])

    return EastStruct({
        "pareto_front": pareto_front,
        "pareto_values": pareto_values,
        "bb_eval": int(bb_eval),
        "success": len(pareto_x) > 0,
    })
```

## Usage Examples

### Basic unconstrained optimization

```east
// Objective: minimize sum of squares
let objective = fn(x: Vector) -> Float {
    let sum = 0.0;
    for i in 0..Array.length(x) {
        sum = sum + x[i] * x[i];
    }
    sum
};

let x0 = [0.71, 0.51, 0.51];
let bounds = {
    lower: [-1.0, -1.0, -1.0],
    upper: [1.0, 1.0, 1.0],
};

let config = {
    max_bb_eval: some(100),
    display_degree: some(0),
    direction_type: none,
    initial_mesh_size: none,
    min_mesh_size: none,
    seed: some(42),
};

let result = mads_optimize(objective, x0, bounds, none, config);

if result.success {
    print("Best solution: " ++ vector_to_string(result.x_best));
    print("Best value: " ++ float_to_string(result.f_best));
    print("Evaluations: " ++ int_to_string(result.bb_eval));
}
```

### Constrained optimization

```east
// Objective: minimize x[4]
let objective = fn(x: Vector) -> Float {
    x[4]
};

// Constraint 1: sum((x[i]-1)^2) - 25 <= 0
let constraint1 = fn(x: Vector) -> Float {
    let sum = 0.0;
    for i in 0..Array.length(x) {
        sum = sum + (x[i] - 1.0) * (x[i] - 1.0);
    }
    sum - 25.0
};

// Constraint 2: 25 - sum((x[i]+1)^2) <= 0
let constraint2 = fn(x: Vector) -> Float {
    let sum = 0.0;
    for i in 0..Array.length(x) {
        sum = sum + (x[i] + 1.0) * (x[i] + 1.0);
    }
    25.0 - sum
};

let x0 = [0.0, 0.0, 0.0, 0.0, 0.0];
let bounds = {
    lower: [-6.0, -6.0, -6.0, -6.0, -6.0],
    upper: [6.0, 6.0, 6.0, 6.0, 6.0],
};

// Constraints: variant tag indicates the constraint kind
let constraints = some([
    #eb(constraint1),  // Extreme barrier
    #eb(constraint2),
]);

let config = {
    max_bb_eval: some(200),
    display_degree: some(0),
    direction_type: none,
    initial_mesh_size: none,
    min_mesh_size: none,
    seed: none,
};

let result = mads_optimize(objective, x0, bounds, constraints, config);
```

### Multi-objective optimization

```east
// Two objectives to minimize
let f1 = fn(x: Vector) -> Float {
    (x[0] - 1.0) * (x[0] - 1.0) + (x[0] - x[1]) * (x[0] - x[1])
};

let f2 = fn(x: Vector) -> Float {
    (x[0] - x[1]) * (x[0] - x[1]) + (x[1] - 3.0) * (x[1] - 3.0)
};

let x0 = [2.0, 2.0];
let bounds = {
    lower: [-1.0, -1.0],
    upper: [5.0, 5.0],
};

let config = {
    max_bb_eval: some(400),
    display_degree: some(0),
    direction_type: some(#ortho_2n),
    initial_mesh_size: none,
    min_mesh_size: none,
    seed: some(42),
};

let result = mads_optimize_multi([f1, f2], x0, bounds, config);

if result.success {
    print("Found " ++ int_to_string(Array.length(result.pareto_front)) ++ " Pareto-optimal solutions");
}
```

## Notes

- MADS is particularly effective for expensive blackbox functions where gradient information is unavailable
- Extreme barrier (EB) constraints reject infeasible points entirely
- Progressive barrier (PB) constraints allow temporary constraint violations during search
- Multi-objective optimization returns a set of Pareto-optimal solutions
- The algorithm is deterministic given the same seed
