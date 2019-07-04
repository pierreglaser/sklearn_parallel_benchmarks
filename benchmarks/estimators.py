import warnings
from sklearn.utils.testing import all_estimators
from sklearn.base import MetaEstimatorMixin
from .config import N_SAMPLES

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    ALL_REGRESSORS = {
        k: v
        for k, v in all_estimators(type_filter="regressor")
        if not k.startswith("_") and N_SAMPLES.get(k, -1) != -1
    }
    ALL_CLASSIFIERS = {
        k: v
        for k, v in all_estimators(type_filter="classifier")
        if not k.startswith("_") and N_SAMPLES.get(k, -1) != -1

    }
    ALL_TRANSFORMERS = {
        k: v
        for k, v in all_estimators(type_filter="transformer")
        if not k.startswith("_") and N_SAMPLES.get(k, -1) != -1
    }

ALL_REGRESSORS_WITH_INTERNAL_PARALLELISM = {}
ALL_TRANSFORMERS_WITH_INTERNAL_PARALLELISM = {}
ALL_CLASSIFIERS_WITH_INTERNAL_PARALLELISM = {}

blacklisted_regressors = []
for name, cls in ALL_REGRESSORS.items():
    if issubclass(cls, MetaEstimatorMixin):  # should be filtered by now
        blacklisted_regressors.append(name)
        continue
    if 'Dummy' in cls.__name__:              # same
        blacklisted_regressors.append(name)
        continue
    try:
        _estimator = cls()
    except Exception as e:
        pass
    else:
        if hasattr(_estimator, "n_jobs"):
            ALL_REGRESSORS_WITH_INTERNAL_PARALLELISM[name] = cls

for name in blacklisted_regressors:
    ALL_REGRESSORS.pop(name)

for name, cls in ALL_TRANSFORMERS.items():
    if name == "Imputer":  # deprecated
        continue
    try:
        _estimator = cls()
    except Exception as e:
        pass
    else:
        if hasattr(_estimator, "n_jobs"):
            ALL_TRANSFORMERS_WITH_INTERNAL_PARALLELISM[name] = cls

blacklisted_classifiers = []
for name, cls in ALL_CLASSIFIERS.items():
    if issubclass(cls, MetaEstimatorMixin):
        blacklisted_classifiers.append(name)
        continue
    if 'Dummy' in cls.__name__:
        blacklisted_classifiers.append(name)
        continue
    try:
        _estimator = cls()
    except Exception as e:
        pass
    else:
        if hasattr(_estimator, "n_jobs"):
            ALL_CLASSIFIERS_WITH_INTERNAL_PARALLELISM[name] = cls

for name in blacklisted_classifiers:
    ALL_CLASSIFIERS.pop(name)
