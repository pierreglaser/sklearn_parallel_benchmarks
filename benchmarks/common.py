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
import numpy as np


class SklearnBenchmark:
    processes = 1
    number = 1
    repeat = 1
    warmup_time = 0
    timer = timeit.default_timer
    timeout = 120

    # non-asv class attributes
    n_tasks = 10

    def setup(self, backend, pickler):
        # tell scikit-learn where to look for joblib
        os.environ['SKLEARN_SITE_JOBLIB'] = os.path.join(
                os.environ['ASV_ENV_DIR'], 'project')
        os.environ['LOKY_PICKLER'] = pickler


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
