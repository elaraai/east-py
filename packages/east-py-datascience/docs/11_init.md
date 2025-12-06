# Main Export (`__init__.py`)

```python
"""East Python Data Science Platform Functions.

Python implementation of data science platform functions for the East programming language.
Provides machine learning, optimization, and explainability operations.
"""

from east_py_data_science.scikit import sklearn_impl
from east_py_data_science.xgboost_impl import xgboost_impl
from east_py_data_science.lightgbm_impl import lightgbm_impl
from east_py_data_science.ngboost_impl import ngboost_impl
from east_py_data_science.optuna_impl import optuna_impl
from east_py_data_science.shap_impl import shap_impl
from east_py_data_science.scipy_impl import scipy_impl
from east_py_data_science.torch_impl import torch_impl
from east_py_data_science.gp_impl import gp_impl

__version__ = "0.1.0"

# Complete Python Data Science platform implementation
python_data_science_platform = [
    *sklearn_impl,
    *xgboost_impl,
    *lightgbm_impl,
    *ngboost_impl,
    *optuna_impl,
    *shap_impl,
    *scipy_impl,
    *torch_impl,
    *gp_impl,
]

__all__ = [
    "__version__",
    "python_data_science_platform",
    # Module exports
    "sklearn_impl",
    "xgboost_impl",
    "lightgbm_impl",
    "ngboost_impl",
    "optuna_impl",
    "shap_impl",
    "scipy_impl",
    "torch_impl",
    "gp_impl",
]
```
