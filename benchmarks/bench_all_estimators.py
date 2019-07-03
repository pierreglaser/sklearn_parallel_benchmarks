#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Benchmarking all in scikit-learn
#
# Author: Pierre Glaser

import os

import numpy
from joblib import Parallel, delayed, parallel_backend
from sklearn.datasets import make_classification

from benchmarks.common import ALL_REGRESSORS, SklearnBenchmark, clone_and_fit
from benchmarks.common import ALL_REGRESSORS_WITH_INTERNAL_PARALLELISM
from benchmarks.common import ALL_CLASSIFIERS_WITH_INTERNAL_PARALLELISM
from benchmarks.common import fit_estimator
from benchmarks.common import make_regression_cached
from benchmarks.config import N_SAMPLES, benchmarks_results


class RegressionBench(SklearnBenchmark):
    param_names = ['estimator_name', 'backend', 'pickler', 'n_jobs',
                   'n_samples', 'n_features']
    params = (sorted(list(ALL_REGRESSORS_WITH_INTERNAL_PARALLELISM.keys())),
              ['multiprocessing', 'loky', 'threading'][1:],
              ['pickle', 'cloudpickle'][:1],
              [1, 2, 4][:2],
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
        try:
            benchmark_times = benchmarks_results.xs(
                    ['time_multiple_fit_parallelization', estimator_name,
                     backend, pickler, str(n_jobs), n_samples, n_features],
                    level=['name', *self.param_names])
            avg_benchmark_time = benchmark_times.mean()
            if avg_benchmark_time > 5:
                raise NotImplementedError
        except (KeyError, AttributeError):
            pass

        super(RegressionBench, self).setup(pickler)
        if n_samples == 'auto':
            n_samples = N_SAMPLES[estimator_name]

        if n_features == 'auto':
            n_features = 10

        # For multitask estimators, generate multi-dimensional output
        if 'MultiTask' in estimator_name:
            X, y = make_regression_cached(n_samples, n_features, n_targets=4)
        else:
            X, y = make_regression_cached(n_samples, n_features, n_targets=1)

        # warm up the executor to hide the process-creation overhead
        with parallel_backend(backend, n_jobs):
            Parallel()(delayed(id)(i) for i in range(n_jobs))
        self.X = X
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
            print('n_jobs is not an attribute of {}, not running the '
                  ' benchmark'.format(
                  estimator_name))
            return NotImplemented

        if 'cv' in estimator.get_params():
            print("AHHHHA")
            estimator.set_params(cv=100)

        with parallel_backend(backend, n_jobs):
            fit_estimator(estimator, self.X, self.y)

    def time_multiple_fit_parallelization(self, estimator_name, backend,
                                          pickler, n_jobs, n_samples,
                                          n_features):
        cls = ALL_REGRESSORS[estimator_name]
        estimator = cls()
        if 'n_jobs' in estimator.get_params():
            # avoid over subscription
            estimator.set_params(n_jobs=1)

        Parallel(backend=backend, n_jobs=n_jobs)(delayed(clone_and_fit)(
            estimator, self.X, self.y) for _ in range(self.n_tasks))


class ClassificationBench(SklearnBenchmark):
    param_names = ['estimator_name', 'backend', 'pickler', 'n_jobs',
                   'n_samples', 'n_features']
    n = 9
    params = (sorted(list(ALL_CLASSIFIERS_WITH_INTERNAL_PARALLELISM.keys()))[n:n+1],
              ['multiprocessing', 'loky', 'threading'][1:],
              ['pickle', 'cloudpickle'][:1],
              [1, 2, 4][:2],
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
        try:
            benchmark_times = benchmarks_results.xs(
                    ['time_multiple_fit_parallelization', estimator_name,
                     backend, pickler, str(n_jobs), n_samples, n_features],
                    level=['name', *self.param_names])
            avg_benchmark_time = benchmark_times.mean()
            if avg_benchmark_time > 5:
                raise NotImplementedError
        except (KeyError, AttributeError):
            pass

        super(ClassificationBench, self).setup(pickler)
        if n_samples == 'auto':
            n_samples = N_SAMPLES[estimator_name]

        if n_features == 'auto':
            n_features = 10

        X, y = make_classification(n_samples, n_features, n_classes=10,
                                   n_informative=5)

        # warm up the executor to hide the process-creation overhead
        with parallel_backend(backend, n_jobs):
            Parallel()(delayed(id)(i) for i in range(n_jobs))
        self.X = X
        self.y = y

    def time_single_fit_parallelization(self, estimator_name, backend, pickler,
                                        n_jobs, n_samples, n_features):
        # ALL_REGRESSORS is a dict. The keys are the estimator class names, and
        # the values are the estimator classes
        cls = ALL_CLASSIFIERS_WITH_INTERNAL_PARALLELISM[estimator_name]
        estimator = cls()

        if estimator_name in ['LogisticRegression', 'LogisticRegressionCV']:
            estimator.set_params(solver='saga')

        if 'n_jobs' in estimator.get_params():
            estimator.set_params(n_jobs=n_jobs)
        else:
            print('n_jobs is not an attribute of {}, not running the '
                  ' benchmark'.format(estimator_name))
            return NotImplemented

        if 'cv' in estimator.get_params():
            estimator.set_params(cv=5)

        with parallel_backend(backend, n_jobs):
            fit_estimator(estimator, self.X, self.y)
