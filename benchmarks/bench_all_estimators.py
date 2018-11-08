#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Benchmarking all in scikit-learn
#
# Author: Pierre Glaser
from benchmarks.common import SklearnBenchmark
from benchmarks.common import ALL_REGRESSORS
from benchmarks.common import clone_and_fit

from benchmarks.config import N_SAMPLES, benchmarks_results


class RegressionBench(SklearnBenchmark):
    param_names = ['estimator_name', 'backend', 'pickler', 'n_jobs',
                   'n_samples', 'n_features']
    params = (sorted(list(ALL_REGRESSORS.keys()))[:5],
              ['multiprocessing', 'loky', 'threading'][1:],
              ['pickle', 'cloudpickle'],
              [1, 2, 4],
              ['auto'],
              ['auto'])

    def setup(self, estimator_name, backend, pickler, n_jobs, n_samples,
              n_features):
        # we are currently trying to spot benchmarks that run too fast, because
        # then the actual fitting time is hidden in the workers
        # creation/communication overhead.
        # all benchmarks whose running time exceed a certain threshold are
        # considered valid for now, and we dont need to run them again, hence
        # we raise NotImplementedError, which causes the benchmark to be
        # skipped
        benchmark_times = benchmarks_results.xs(
                ['time_multiple_fit_parallelization', estimator_name, backend,
                    pickler, str(n_jobs), n_samples, n_features],
                level=['name', *self.param_names])
        avg_benchmark_time = benchmark_times.mean()
        if avg_benchmark_time > 5:
            raise NotImplementedError

        super(RegressionBench, self).setup(backend, pickler)
        from sklearn.datasets import make_regression

        if n_samples == 'auto':
            n_samples = N_SAMPLES[estimator_name]

        if n_features == 'auto':
            n_features = 10

        # For multitask estimators, generate multi-dimensional output
        if 'MultiTask' in estimator_name:
            X, y = make_regression(n_samples, n_features, n_targets=4)
        else:
            X, y = make_regression(n_samples, n_features, n_targets=1)

        X, y = make_regression(n_samples, n_features)
        self.X = X
        assert self.X.shape[0] == N_SAMPLES[estimator_name]
        self.y = y

    def time_single_fit_parallelization(self, estimator_name, backend, pickler,
                                        n_jobs, n_samples, n_features):
        # ALL_REGRESSORS is a dict. The keys are the estimator class names, and
        # the values are the estimator classes
        cls = ALL_REGRESSORS[estimator_name]
        estimator = cls()
        if 'n_jobs' in estimator.get_params():
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
        if 'n_jobs' in estimator.get_params():
            # avoid over subscription
            estimator.set_params(n_jobs=1)

        Parallel(backend=backend, n_jobs=n_jobs)(delayed(clone_and_fit)(
            estimator, self.X, self.y) for _ in range(self.n_tasks))
