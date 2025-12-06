# Module 6: Optuna (`optuna_impl.py`)

## Purpose

Hyperparameter optimization using Optuna's TPE sampler.

## Config Types

```python
OptunaStudyConfigType = StructType([
    ("direction", OptionType(OptimizationDirectionType)),  # default minimize
    ("n_trials", IntegerType),                             # number of trials
    ("random_state", OptionType(IntegerType)),             # default None
    ("pruner", OptionType(PrunerType)),                    # default none
])

# Objective function type: takes params, returns score
ObjectiveFunctionType = FunctionType(
    [ArrayType(NamedHyperparamType)],  # params input
    FloatType                           # score output
)
```

## Platform Functions

### `optuna_optimize`

Run hyperparameter optimization with an objective function.

The objective function is an East function that receives suggested parameters
and returns a score. The platform function calls this East function for each trial.

```python
PlatformFunction(
    name="optuna_optimize",
    inputs=[
        ArrayType(HyperparamSpaceType),  # search_space
        ObjectiveFunctionType,            # objective (East function)
        OptunaStudyConfigType,            # config
    ],
    output=StudyResultType,
    type="sync",
    fn=optuna_optimize_impl,
)

def optuna_optimize_impl(
    search_space: EastArray,  # Array[HyperparamSpace]
    objective_fn: Callable[[EastArray], float],  # Array[NamedHyperparam] -> Float
    config: EastStruct  # OptunaStudyConfigType
) -> EastStruct:  # StudyResultType
    """Run Optuna optimization with East objective function.

    The objective_fn is a compiled East function that the platform function
    can call directly - it receives an EastArray of NamedHyperparam and
    returns a float score.
    """
    import optuna

    direction_variant = _get_option(config.get("direction"), None)
    direction = _get_enum_tag(direction_variant) if direction_variant else "minimize"

    n_trials = config["n_trials"]
    random_state = _get_option(config.get("random_state"), None)

    pruner_variant = _get_option(config.get("pruner"), None)
    pruner_name = _get_enum_tag(pruner_variant) if pruner_variant else "none"

    if pruner_name == "median":
        pruner = optuna.pruners.MedianPruner()
    elif pruner_name == "hyperband":
        pruner = optuna.pruners.HyperbandPruner()
    else:
        pruner = optuna.pruners.NopPruner()

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction=direction, sampler=sampler, pruner=pruner)

    def wrapped_objective(trial: optuna.Trial) -> float:
        """Wrap Optuna trial to call East objective function."""
        params = _suggest_params_from_trial(trial, search_space)
        # Call the East function directly - it's a compiled Python callable
        return objective_fn(params)

    study.optimize(wrapped_objective, n_trials=n_trials)

    return _make_study_result(study)


def _suggest_params_from_trial(
    trial: "optuna.Trial",
    search_space: EastArray  # Array[HyperparamSpace]
) -> EastArray:  # Array[NamedHyperparam]
    """Suggest parameters from Optuna trial based on search space."""
    params = []

    for param_def in search_space:
        name = param_def["name"]
        kind_variant = param_def["kind"]
        kind = _get_enum_tag(kind_variant)

        if kind == "int":
            low = int(_get_option(param_def.get("low"), 1))
            high = int(_get_option(param_def.get("high"), 100))
            value = trial.suggest_int(name, low, high)
            params.append(EastStruct({
                "name": name,
                "value": EastVariant("int", value),
            }))

        elif kind == "float":
            low = float(_get_option(param_def.get("low"), 0.0))
            high = float(_get_option(param_def.get("high"), 1.0))
            value = trial.suggest_float(name, low, high)
            params.append(EastStruct({
                "name": name,
                "value": EastVariant("float", value),
            }))

        elif kind == "log_uniform":
            low = float(_get_option(param_def.get("low"), 1e-6))
            high = float(_get_option(param_def.get("high"), 1.0))
            value = trial.suggest_float(name, low, high, log=True)
            params.append(EastStruct({
                "name": name,
                "value": EastVariant("float", value),
            }))

        elif kind == "categorical":
            choices_arr = _get_option(param_def.get("choices"), None)
            if choices_arr is None:
                raise ValueError(f"categorical param {name} requires choices")

            py_choices = [choice.value for choice in choices_arr]
            value = trial.suggest_categorical(name, py_choices)

            # Convert back to HyperparamValueType variant
            if isinstance(value, bool):
                params.append(EastStruct({
                    "name": name,
                    "value": EastVariant("bool", value),
                }))
            elif isinstance(value, int):
                params.append(EastStruct({
                    "name": name,
                    "value": EastVariant("int", value),
                }))
            elif isinstance(value, float):
                params.append(EastStruct({
                    "name": name,
                    "value": EastVariant("float", value),
                }))
            elif isinstance(value, str):
                params.append(EastStruct({
                    "name": name,
                    "value": EastVariant("string", value),
                }))

    return EastArray(NamedHyperparamType, params)


def _make_study_result(study: "optuna.Study") -> EastStruct:
    """Convert Optuna study to East StudyResultType."""
    import optuna

    # Best params
    best_params = _params_to_east(study.best_params)

    # All completed trials
    trials = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            trials.append(EastStruct({
                "trial_id": trial.number,
                "params": _params_to_east(trial.params),
                "score": float(trial.value),
            }))

    return EastStruct({
        "best_params": EastArray(NamedHyperparamType, best_params),
        "best_score": float(study.best_value),
        "trials": EastArray(TrialResultType, trials),
    })


def _params_to_east(params: dict) -> list[EastStruct]:
    """Convert Python params dict to list of NamedHyperparam."""
    result = []
    for name, value in params.items():
        if isinstance(value, bool):
            result.append(EastStruct({
                "name": name,
                "value": EastVariant("bool", value),
            }))
        elif isinstance(value, int):
            result.append(EastStruct({
                "name": name,
                "value": EastVariant("int", value),
            }))
        elif isinstance(value, float):
            result.append(EastStruct({
                "name": name,
                "value": EastVariant("float", value),
            }))
        elif isinstance(value, str):
            result.append(EastStruct({
                "name": name,
                "value": EastVariant("string", value),
            }))
    return result
```

## Usage Example (East Code)

```east
// Define search space
let search_space = [
    { name: "n_estimators", kind: #int, low: some(50.0), high: some(500.0), choices: none },
    { name: "max_depth", kind: #int, low: some(3.0), high: some(10.0), choices: none },
    { name: "learning_rate", kind: #log_uniform, low: some(0.001), high: some(0.3), choices: none },
];

// Define objective function in East
let objective = fn(params: Array[NamedHyperparam]) -> Float {
    // Extract params
    let n_estimators = get_param_int(params, "n_estimators");
    let max_depth = get_param_int(params, "max_depth");
    let lr = get_param_float(params, "learning_rate");

    // Train model with these params
    let config = {
        n_estimators: some(n_estimators),
        max_depth: some(max_depth),
        learning_rate: some(lr),
        // ... other fields
    };
    let model = xgboost_train_regressor(X_train, y_train, config);

    // Evaluate and return score (lower is better for minimize)
    let preds = xgboost_predict(model, X_val);
    let metrics = sklearn_metrics_regression(y_val, preds);
    metrics.mse
};

// Run optimization
let config = {
    direction: some(#minimize),
    n_trials: 50,
    random_state: some(42),
    pruner: none,
};

let result = optuna_optimize(search_space, objective, config);

// Use best params
print("Best score: " ++ float_to_string(result.best_score));
print("Best params: " ++ params_to_string(result.best_params));
```
