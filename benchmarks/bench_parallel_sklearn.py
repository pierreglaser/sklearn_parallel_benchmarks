#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Benchmark of commonnly used ML pipelines in scikit-learn
#
# Author: Pierre Glaser
from joblib import parallel_backend, Parallel, delayed
from sklearn.datasets import fetch_20newsgroups, fetch_california_housing
from sklearn.datasets import make_regression
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.model_selection import GridSearchCV, ShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, PolynomialFeatures
from sklearn.preprocessing import StandardScaler

from benchmarks.common import EstimatorWithLargeList, SklearnBenchmark


class TwentyDataBench(SklearnBenchmark):
    param_names = ['backend', 'pickler', 'n_jobs']
    params = (['multiprocessing', 'loky', 'threading'][1:],
              ['pickle', 'cloudpickle'],
              [1, 2, 4])

    def setup(self, backend, pickler, n_jobs):
        super(TwentyDataBench, self).setup(pickler)
        data = fetch_20newsgroups()
        self.X = data.data
        self.y = data.target

    def time_text_vectorizer(self, backend, pickler, n_jobs):
        pipeline = Pipeline([('tfidf', TfidfVectorizer()),
                             ('clf', SGDClassifier())])
        cv = ShuffleSplit(n_splits=4, test_size=0.33)

        with parallel_backend(backend=backend):
            cross_val_score(pipeline, self.X, self.y, cv=cv,
                            n_jobs=n_jobs)


class CaliforniaHousingBench(SklearnBenchmark):
    param_names = ['backend', 'pickler', 'n_jobs']
    params = (['multiprocessing', 'loky', 'threading'][1:],
              ['pickle', 'cloudpickle'],
              [1, 2, 4])

    def setup(self, backend, pickler, n_jobs):
        super(CaliforniaHousingBench, self).setup(pickler)
        self.california_data = fetch_california_housing()

    def time_kbins_polynomial_pipeline(self, backend, pickler, n_jobs):
        pipeline = Pipeline([
            ('discretizer', KBinsDiscretizer(encode='onehot')),
            ('polynomial_features', PolynomialFeatures()),
            ('estimator', Ridge())])
        cv = ShuffleSplit(n_splits=4, test_size=0.3)

        with parallel_backend(backend=backend):
            cross_val_score(pipeline, self.california_data.data,
                            self.california_data.target, cv=cv,
                            n_jobs=n_jobs)


class MakeRegressionDataBench(SklearnBenchmark):
    param_names = ['backend', 'pickler', 'n_jobs', 'n_samples', 'n_features']
    params = (['multiprocessing', 'loky', 'threading'][1:],
              ['pickle', 'cloudpickle'],
              [1, 2, 4],
              [10000, 30000],
              [10])

    def setup(self, backend, pickler, n_jobs, n_samples, n_features):
        super(MakeRegressionDataBench, self).setup(pickler)
        X, y = make_regression(n_samples, n_features)
        self.X = X
        self.y = y

    def time_send_list(self, backend, pickler, n_jobs, n_samples, n_features):
        with parallel_backend(backend=backend):
            Parallel(n_jobs=n_jobs)(delayed(id)(
                list(range(100000))) for _ in range(self.n_tasks))

    def time_gridsearch_large_list(self, backend, pickler, n_jobs, n_samples,
                                   n_features):
        # Serializing object with big lists slow down cloudpickle-based
        # Picklers. If the benchmarks do not fulfill these expectations,
        # something wrong is going on, with the use of pickler of the vendoring
        # of joblib
        r = EstimatorWithLargeList()
        params = {'alpha': [1, 0.1, 0.001]}
        g = GridSearchCV(r, params, cv=4, n_jobs=n_jobs)

        with parallel_backend(backend):
            g.fit(self.X, self.y)

    def time_ridge_gridsearch(self, backend, pickler, n_jobs, n_samples,
                              n_features):
        params = {'alpha': [2**-i for i in range(1, 40)]}
        ridge = Ridge()
        # Use a large cv value because ridge is very fast
        with parallel_backend(backend=backend):
            rcv = GridSearchCV(ridge, params, cv=50, n_jobs=n_jobs)
            rcv.fit(self.X, self.y)

    def time_randomforest(self, backend, pickler, n_jobs, n_samples,
                          n_features):
        with parallel_backend(backend):
            rf = RandomForestRegressor(n_estimators=100, n_jobs=n_jobs)
            rf.fit(self.X, self.y)

    def time_scaler_kernelridge_pipeline(self, backend, pickle, n_jobs,
                                         n_samples, n_features):

        pipeline = Pipeline([('scaler', StandardScaler()),
                             ('estimator', KernelRidge())])

        cv = ShuffleSplit(n_splits=8, test_size=0.3)

        with parallel_backend(backend=backend):
            cross_val_score(pipeline, self.X, self.y, cv=cv, n_jobs=n_jobs)

    time_scaler_kernelridge_pipeline.param_names = [
            'backend', 'pickler', 'n_jobs', 'n_samples', 'n_features']
    time_scaler_kernelridge_pipeline.params = (
            ['multiprocessing', 'loky', 'threading'][1:],
            ['pickle', 'cloudpickle'],
            [1, 2, 4],
            [10000],
            [10])
