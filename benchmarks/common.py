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
import os
import timeit
from functools import wraps
from joblib import Memory
from sklearn.utils.testing import all_estimators
from sklearn.datasets import make_regression
from sklearn.base import clone


memory = Memory('/tmp/pglaser/joblib')
ALL_REGRESSORS = {k: v for k, v in all_estimators(
    include_meta_estimators=False, type_filter='regressor')}
ALL_CLASSIFIERS = {k: v for k, v in all_estimators(
    include_meta_estimators=False, type_filter='classifier')}
ALL_TRANSFORMERS = {k: v for k, v in all_estimators(
    include_meta_estimators=False, type_filter='transformer')}


@wraps(make_regression)
@memory.cache
def make_regression_cached(*args, **kwargs):
    return make_regression(*args, **kwargs)


def clone_and_fit(estimator, X, y):
    """clone and fit an estimator

    This function is performs a fitting process after cloning the estimator
    given as input. It can be safely called from within a Parallel loop with
    a shared memory backend, and is common to objects that implement a fit
    method (in opposition with cross_val_score, that requires scoring and
    therefore that cannot be used with transformers)
    """
    cloned_estimator = clone(estimator)
    cloned_estimator.fit(X, y)


class SklearnBenchmark:
    processes = 1
    number = 1
    repeat = 1
    warmup_time = 0
    timer = timeit.default_timer
    timeout = 120

    # non-asv class attributes
    n_tasks = 10

    def setup(self, pickler):
        from joblib.externals.loky import set_loky_pickler
        set_loky_pickler(pickler)


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
        return [0]*repeat(len(X))

    def score(self, *args, **kwargs):
        return 0
