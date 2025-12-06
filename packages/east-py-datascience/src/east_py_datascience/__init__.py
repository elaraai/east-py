"""East Data Science Platform Functions.

Python implementation of data science platform functions for the East programming language.
Provides ML and optimization capabilities for East programs running in Python.
"""

from east_py_datascience.mads import mads_impl
from east_py_datascience.optuna import optuna_impl
from east_py_datascience.simanneal import simanneal_impl
from east_py_datascience.sklearn import sklearn_impl
from east_py_datascience.scipy import scipy_impl
from east_py_datascience.xgboost import xgboost_impl
from east_py_datascience.lightgbm import lightgbm_impl
from east_py_datascience.ngboost import ngboost_impl
from east_py_datascience.shap import shap_impl
from east_py_datascience.torch import torch_impl
from east_py_datascience.gp import gp_impl
from east_py_datascience.types import (
    VectorType,
    MatrixType,
    IntVectorType,
    ScalarObjectiveType,
    VectorObjectiveType,
    LabelVectorType,
    SplitConfigType,
    SplitResultType,
    RegressionMetricsType,
    ClassificationMetricsType,
    ModelBlobType,
    # Scipy types
    OptimizeMethodType,
    InterpolationKindType,
    OptimizeConfigType,
    InterpolateConfigType,
    CurveFunctionType,
    CurveFitConfigType,
    StatsDescribeResultType,
    CorrelationResultType,
    CurveFitResultType,
    OptimizeResultType,
)

__version__ = "0.1.0"

# Complete data science platform implementation
# Pass this list to compile_async() to enable all platform functions
datascience_platform = [
    *mads_impl,
    *optuna_impl,
    *simanneal_impl,
    *sklearn_impl,
    *scipy_impl,
    *xgboost_impl,
    *lightgbm_impl,
    *ngboost_impl,
    *shap_impl,
    *torch_impl,
    *gp_impl,
]

__all__ = [
    "__version__",
    # Main platform exports
    "datascience_platform",
    # Module exports
    "mads_impl",
    "optuna_impl",
    "simanneal_impl",
    "sklearn_impl",
    "scipy_impl",
    "xgboost_impl",
    "lightgbm_impl",
    "ngboost_impl",
    "shap_impl",
    "torch_impl",
    "gp_impl",
    # Type exports
    "VectorType",
    "MatrixType",
    "IntVectorType",
    "ScalarObjectiveType",
    "VectorObjectiveType",
    "LabelVectorType",
    "SplitConfigType",
    "SplitResultType",
    "RegressionMetricsType",
    "ClassificationMetricsType",
    "ModelBlobType",
    # Scipy types
    "OptimizeMethodType",
    "InterpolationKindType",
    "OptimizeConfigType",
    "InterpolateConfigType",
    "CurveFunctionType",
    "CurveFitConfigType",
    "StatsDescribeResultType",
    "CorrelationResultType",
    "CurveFitResultType",
    "OptimizeResultType",
]
