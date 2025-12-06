# Module 6: Optuna (`optuna_impl.py`)

## Purpose

Bayesian optimization using Optuna's TPE (Tree-structured Parzen Estimator) sampler.
Supports general parameter optimization including hyperparameter tuning, design optimization,
and any blackbox optimization problem with categorical, integer, or continuous parameters.

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
    [ArrayType(NamedParamType)],  # params input
    FloatType                      # score output
)
```

## Platform Functions

### `optuna_optimize`

Run Bayesian optimization with an objective function.

The objective function is an East function that receives suggested parameters
and returns a score. The platform function calls this East function for each trial.

```python
PlatformFunction(
    name="optuna_optimize",
    inputs=[
        ArrayType(ParamSpaceType),    # search_space
        ObjectiveFunctionType,         # objective (East function)
        OptunaStudyConfigType,         # config
    ],
    output=StudyResultType,
    type="sync",
    fn=optuna_optimize_impl,
)

def optuna_optimize_impl(
    search_space: EastArray,  # Array[ParamSpace]
    objective_fn: Callable[[EastArray], float],  # Array[NamedParam] -> Float
    config: EastStruct  # OptunaStudyConfigType
) -> EastStruct:  # StudyResultType
    """Run Optuna optimization with East objective function.

    The objective_fn is a compiled East function that the platform function
    can call directly - it receives an EastArray of NamedParam and
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
    search_space: EastArray  # Array[ParamSpace]
) -> EastArray:  # Array[NamedParam]
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

            # Convert back to ParamValueType variant
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

    return EastArray(NamedParamType, params)


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
        "best_params": EastArray(NamedParamType, best_params),
        "best_score": float(study.best_value),
        "trials": EastArray(TrialResultType, trials),
    })


def _params_to_east(params: dict) -> list[EastStruct]:
    """Convert Python params dict to list of NamedParam."""
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
