#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Base benchmark classes.
#
# Author: Pierre Glaser
"""Base benchmark classes to study the impact of parallelism in sckit-learn.

------------------------------asv reminder------------------------------------
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
from functools import partial

from joblib import parallel_backend, Parallel, delayed
from sklearn.base import RegressorMixin, ClassifierMixin
from sklearn.datasets import make_regression, make_classification
from sklearn.utils.testing import all_estimators

from .config import N_SAMPLES, PARAMS, ESTIMATOR_BLACK_LIST
from .utils import has_parallelism, fit_estimator, clone_and_fit


class SklearnBenchmark:
    processes = 1
    number = 1
    repeat = 1
    warmup_time = 0
    timer = timeit.default_timer
    timeout = 50

    # non-asv class attributes
    n_tasks = 4

    def setup(self, pickler):
        from joblib.externals.loky import set_loky_pickler

        set_loky_pickler(pickler)


class AbstractEstimatorBench(ABC, SklearnBenchmark):
    __metaclass__ = ABCMeta
    param_names = ["backend", "pickler", "n_jobs", "n_samples", "n_features"]
    params = (
        ["multiprocessing", "loky", "threading"][1:],
        ["pickle", "cloudpickle"][:1],
        [1, 2, 4],
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


class AbstractRegressionBench(AbstractEstimatorBench):
    __metaclass__ = ABCMeta

    @property
    def data_factory(self):
        if "MultiTask" in self.estimator_name:
            return partial(make_regression, n_targets=4)
        else:
            return partial(make_regression, n_targets=1)


class AbstractClassificationBench(AbstractEstimatorBench):
    __metaclass__ = ABCMeta
    data_factory = partial(make_classification, n_classes=10, n_informative=8)


class SingleFitParallelizationMixin:
    def time_single_fit(self, backend, pickler, n_jobs, n_samples, n_features):
        estimator = self.estimator_cls(**self.estimator_params)
        estimator.set_params(n_jobs=n_jobs)

        with parallel_backend(backend, n_jobs):
            fit_estimator(estimator, self.X, self.y)


class MultipleFitParallelizationMixin:
    def time_multiple_fit(
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


def make_unique_bench_class(estimator, bench_type):
    estimator_name = estimator.__name__
    if bench_type == "single_fit":
        mixin = SingleFitParallelizationMixin
        bench_name = "{}SingleFitBench".format(estimator_name)
    elif bench_type == "multiple_fit":
        mixin = MultipleFitParallelizationMixin
        bench_name = "{}MultipleFitBench".format(estimator_name)
    else:
        raise ValueError(
            "acceptable benchmark types are single_fit and multiple_fit"
        )

    if issubclass(estimator, RegressorMixin):
        base = AbstractRegressionBench
    elif issubclass(estimator, ClassifierMixin):
        base = AbstractClassificationBench
    else:
        raise ValueError(
            "estimator {} must be a sklearn Classifier or Regressor".format(
                estimator_name
            )
        )

    bench_attrs = {
        "estimator_cls": estimator,
        "estimator_params": PARAMS.get(estimator_name, {}),
        "default_n_samples": N_SAMPLES[estimator_name],
    }

    bench_class = type(bench_name, (base, mixin), bench_attrs)
    return bench_name, bench_class


def make_all_bench_classes(type_filter):
    namespace = {}

    with warnings.catch_warnings():
        # listing all estimators imports all modules including
        # those who are deprecated
        warnings.filterwarnings("ignore", module="sklearn.externals.joblib")
        warnings.filterwarnings(
            "ignore", module="sklearn.utils.linear_assignment_"
        )
        estimators = all_estimators(type_filter=type_filter)

    for est_name, est_cls in estimators:
        if est_name in ESTIMATOR_BLACK_LIST:
            continue
        bench_name, bench_class = make_unique_bench_class(
            est_cls, "multiple_fit"
        )
        namespace[bench_name] = bench_class

        if has_parallelism(est_cls):
            bench_name, bench_class = make_unique_bench_class(
                est_cls, "single_fit"
            )
            namespace[bench_name] = bench_class
    return namespace
