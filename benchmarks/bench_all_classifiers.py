#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Benchmarking all in scikit-learn
#
# Author: Pierre Glaser

from joblib import Parallel, delayed, parallel_backend
from sklearn.datasets import make_classification

from benchmarks.common import ALL_CLASSIFIERS_WITH_INTERNAL_PARALLELISM
from benchmarks.common import fit_estimator, SklearnBenchmark
from benchmarks.config import N_SAMPLES
from abc import ABCMeta, abstractproperty


class AbstractClassificationBench(SklearnBenchmark):
    __metaclass__ = ABCMeta
    param_names = ['backend', 'pickler', 'n_jobs',
                   'n_samples', 'n_features']
    n = 9
    params = (['multiprocessing', 'loky', 'threading'][1:],
              ['pickle', 'cloudpickle'][:1],
              [1, 2, 4][:2],
              ['auto'],
              ['auto'])

    @abstractproperty
    def estimator_cls(self):
        raise NotImplementedError

    def setup(self, backend, pickler, n_jobs, n_samples,
              n_features):
        # we are currently trying to spot benchmarks that run too fast, because
        # then the actual fitting time is hidden in the workers
        # creation/communication overhead.
        # all benchmarks whose running time exceed a certain threshold are
        # considered valid for now, and we dont need to run them again, hence
        # we raise NotImplementedError, which causes the benchmark to be
        # skipped
        super(AbstractClassificationBench, self).setup(pickler)
        if n_samples == 'auto':
            n_samples = N_SAMPLES[self.estimator_cls.__name__]

        if n_features == 'auto':
            n_features = 10

        X, y = make_classification(n_samples, n_features, n_classes=10,
                                   n_informative=5)

        # warm up the executor to hide the process-creation overhead
        with parallel_backend(backend, n_jobs):
            Parallel()(delayed(id)(i) for i in range(n_jobs))
        self.X = X
        self.y = y

    def time_single_fit_parallelization(self, backend, pickler,
                                        n_jobs, n_samples, n_features):
        # ALL_REGRESSORS is a dict. The keys are the estimator class names, and
        # the values are the estimator classes
        estimator_name = self.estimator_cls.__name__
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


ALL_BENCHMARKS = {}
for est_name, est_cls in ALL_CLASSIFIERS_WITH_INTERNAL_PARALLELISM.items():
    bench_name = "{}Bench".format(est_name)
    print(bench_name)
    bench_class = type(
        bench_name, (AbstractClassificationBench,), {"estimator_cls": est_cls})
    ALL_BENCHMARKS[est_name] = bench_class

globals().update(ALL_BENCHMARKS)
