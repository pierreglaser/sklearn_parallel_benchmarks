#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Benchmarking all in scikit-learn
#
# Author: Pierre Glaser
import os

import numpy as np

from benchmarks.common import EstimatorWithLargeList, SklearnBenchmark
from benchmarks.common import ALL_REGRESSORS
from benchmarks.common import clone_and_fit


class RegressionBench(SklearnBenchmark):
    param_names = ['estimator_name', 'backend', 'pickler', 'n_jobs',
                   'n_samples', 'n_features']
    params = (sorted(list(ALL_REGRESSORS.keys()))[:5],
              ['multiprocessing', 'loky', 'threading'][1:],
              ['pickle', 'cloudpickle'],
              [1, 2, 4],
              [10000, 30000][:1],
              [10])

    def setup(self, estimator_name, backend, pickler, n_jobs, n_samples,
              n_features):
        super(RegressionBench, self).setup(backend, pickler)
        from sklearn.datasets import make_regression
        X, y = make_regression(n_samples, n_features)
        self.X = X
        self.y = y

    def time_single_fit_parallelization(self, estimator_name, backend, pickler,
                                        n_jobs, n_samples, n_features):
        # ALL_REGRESSORS is a dict. The keys are the estimator class names, and
        # the values are the estimator classes
        cls = ALL_REGRESSORS[estimator_name]
        estimator = cls()
        if 'n_jobs' in estimator.get_params().keys():
            estimator.set_params(n_jobs=n_jobs)
        else:
            print('warning: n_jobs is not an attribute of {}'.format(
                estimator))

        from joblib import parallel_backend
        with parallel_backend(backend):
            estimator.fit(self.X, self.y)

    def time_multiple_fit_parallelization(self, estimator_name, backend,
                                          pickler, n_jobs, n_samples,
                                          n_features):
        cls = ALL_REGRESSORS[estimator_name]
        estimator = cls()
        from joblib import Parallel, delayed
        if 'n_jobs' in estimator.get_params().keys():
            # avoid over subscription
            estimator.set_params(n_jobs=1)

        Parallel(n_jobs=n_jobs)(delayed(clone_and_fit)(
            estimator, self.X, self.y) for _ in range(self.n_tasks))