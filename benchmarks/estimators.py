import warnings
from sklearn.utils.testing import all_estimators
from sklearn.base import MetaEstimatorMixin

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    ALL_REGRESSORS = {
        k: v
        for k, v in all_estimators(type_filter="regressor")
        if not k.startswith("_")
    }
    ALL_CLASSIFIERS = {
        k: v
        for k, v in all_estimators(type_filter="classifier")
        if not k.startswith("_")
    }
    ALL_TRANSFORMERS = {
        k: v
        for k, v in all_estimators(type_filter="transformer")
        if not k.startswith("_")
    }

ALL_REGRESSORS_WITH_INTERNAL_PARALLELISM = {}
ALL_TRANSFORMERS_WITH_INTERNAL_PARALLELISM = {}
ALL_CLASSIFIERS_WITH_INTERNAL_PARALLELISM = {}
for name, cls in ALL_REGRESSORS.items():
    try:
        _estimator = cls()
    except Exception as e:
        pass
    else:
        if hasattr(_estimator, "n_jobs"):
            ALL_REGRESSORS_WITH_INTERNAL_PARALLELISM[name] = cls

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

blacklisted_estimators = []
for name, cls in ALL_CLASSIFIERS.items():
    if issubclass(cls, MetaEstimatorMixin):
        blacklisted_estimators.append(name)
        continue
    if 'Dummy' in cls.__name__:
        blacklisted_estimators.append(name)
        continue
    try:
        _estimator = cls()
    except Exception as e:
        pass
    else:
        if hasattr(_estimator, "n_jobs"):
            ALL_CLASSIFIERS_WITH_INTERNAL_PARALLELISM[name] = cls

for name in blacklisted_estimators:
    ALL_CLASSIFIERS.pop(name)

ALL_CLASSIFIERS.pop("ComplementNB")   # requires count data
ALL_CLASSIFIERS.pop("MultinomialNB")  # requires count data
ALL_CLASSIFIERS.pop("CheckingClassifier")  # nothing done during the fit?
