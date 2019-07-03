#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Common Utilities for joblib's benchmark suite
#
# Author: Pierre Glaser
"""Benchmark Suite for joblib


The benchmark routine of asv can be summarized as follow:
for b in benchmark:                        # loop 1
    for i in range(b.repeat):              # loop 2
        for args in params:                # loop 3
            for function in b:             # loop 4
                b.setup(*arg)
                for n in range(b.number):  # loop 5
                    function(*arg)
                b.teardown(*arg)

number and repeat attributes differ in the sense that a setup and teardown
call is run between two iterations of loop 2, in opposition with loop 5.

"""
import timeit
import warnings
from abc import ABC, ABCMeta, abstractproperty
from functools import wraps
from joblib import Memory, parallel_backend, Parallel, delayed
from sklearn.utils.testing import all_estimators
from sklearn.datasets import make_regression
from sklearn.base import clone

from benchmarks.profile_this import profile_this

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    ALL_REGRESSORS = {k: v for k, v in all_estimators(type_filter="regressor")}
    ALL_CLASSIFIERS = {
        k: v for k, v in all_estimators(type_filter="classifier")
    }
    ALL_TRANSFORMERS = {
        k: v for k, v in all_estimators(type_filter="transformer")
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

meta_estimators = []
for name, cls in ALL_CLASSIFIERS.items():
    try:
        _estimator = cls()
    except Exception as e:
        pass
    else:
        if hasattr(_estimator, "base_estimator"):
            meta_estimators.append(name)
            continue
        if hasattr(_estimator, "n_jobs"):
            ALL_CLASSIFIERS_WITH_INTERNAL_PARALLELISM[name] = cls

for name in meta_estimators:
    ALL_CLASSIFIERS.pop(name)

ALL_CLASSIFIERS.pop("ComplementNB")  # requires count data
ALL_CLASSIFIERS.pop("CheckingClassifier")  # nothing done during the fit?


def clone_and_fit(estimator, X, y):
    """clone and fit an estimator

    This function is performs a fitting process after cloning the estimator
    given as input. It can be safely called from within a Parallel loop with
    a shared memory backend, and is common to objects that implement a fit
    method (in opposition with cross_val_score, that requires scoring and
    therefore that cannot be used with transformers)
    """
    clone_and_fit_profiled(estimator, X, y)


@profile_this
def fit_estimator(estimator, X, y):
    """Fit an estimator.

    This function exists because we cannot profile directly methods of
    scikit-learn, as we need to decorate them in the source code
    """
    estimator.fit(X, y)


@profile_this
def clone_and_fit_profiled(estimator, X, y):
    # if we want to keep using pickle and not cloudpickle in those benchmarks,
    # we need to send pickleable fuctions. Directly decorating clone_and_fit
    # make it a nested function, thus not pickleable. Therefore, we make
    # clone_and_fit call a decorated function.
    cloned_estimator = clone(estimator)
    cloned_estimator.fit(X, y)


class SklearnBenchmark:
    processes = 1
    number = 1
    repeat = 1
    warmup_time = 0
    timer = timeit.default_timer
    timeout = 100

    # non-asv class attributes
    n_tasks = 10

    def setup(self, pickler):
        from joblib.externals.loky import set_loky_pickler

        set_loky_pickler(pickler)


class AbstractEstimatorBench(ABC, SklearnBenchmark):
    __metaclass__ = ABCMeta
    param_names = ["backend", "pickler", "n_jobs", "n_samples", "n_features"]
    params = (
        ["multiprocessing", "loky", "threading"][1:],
        ["pickle", "cloudpickle"][:1],
        [1, 2, 4][:2],
        ["auto"],
        ["auto"],
    )

    @abstractproperty
    def estimator_cls(self):
        raise NotImplementedError

    @abstractproperty
    def data_factory(self):
        raise NotImplementedError

    @abstractproperty
    def estimator_params(self):
        raise NotImplementedError

    @abstractproperty
    def default_n_samples(self):
        raise NotImplementedError

    @property
    def estimator_name(self):
        return self.estimator_cls.__name__

    def setup(self, backend, pickler, n_jobs, n_samples, n_features):
        super(AbstractEstimatorBench, self).setup(pickler)
        if n_samples == "auto":
            n_samples = self.default_n_samples

        if n_features == "auto":
            n_features = 10

        X, y = self.data_factory(n_samples, n_features)

        # warm up the executor to hide the process-creation overhead
        with parallel_backend(backend, n_jobs):
            Parallel()(delayed(id)(i) for i in range(10 * n_jobs))
        self.X = X
        self.y = y


class SingleFitParallelizationMixin:
    def time_single_fit_parallelization(
        self, backend, pickler, n_jobs, n_samples, n_features
    ):
        estimator = self.estimator_cls(**self.estimator_params)

        if "n_jobs" in estimator.get_params():
            estimator.set_params(n_jobs=n_jobs)
        else:
            print(
                "n_jobs is not an attribute of {}, not running the "
                " benchmark".format(self.estimator_name)
            )
            raise NotImplemented

        with parallel_backend(backend, n_jobs):
            fit_estimator(estimator, self.X, self.y)


class MultipleFitParallelizationMixin:
    def time_multiple_fit_parallelization(
        self, backend, pickler, n_jobs, n_samples, n_features
    ):
        estimator = self.estimator_cls(**self.estimator_params)

        if "n_jobs" in estimator.get_params():
            # avoid over subscription
            estimator.set_params(n_jobs=1)

        Parallel(backend=backend, n_jobs=n_jobs)(
            delayed(clone_and_fit)(estimator, self.X, self.y)
            for _ in range(self.n_tasks)
        )



class EstimatorWithLargeList:
    """simple estimator, with a large list as an attribute

    Instances of this class should take a long time to serizlize using
    cloudpickle, as large lists are typically very costly ot pickle"""

    def __init__(self):
        self.large_list = list(range(100000))
        self.best_estimator_ = self

    def get_params(self, *args, **kwargs):
        return dict()

    def set_params(self, *args, **kwargs):
        return self

    def fit(self, *args, **kwargs):
        pass

    def predict(self, X):
        return [0] * repeat(len(X))

    def score(self, *args, **kwargs):
        return 0
