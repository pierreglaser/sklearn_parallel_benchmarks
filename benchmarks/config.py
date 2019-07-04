#!/usr/bin/env python
# -*- coding: utf-8 -*-
# default parameter values used for scikit-learn benchmarking
#
# Author: Pierre Glaser
import pandas as pd

N_SAMPLES = {
    'AdaBoostRegressor': -1,                 # meta estimator (30000)
    'AdaBoostClassifier': -1,                # meta estimator  (300)
    'ARDRegression': 400,                    # calibrated
    'BaggingRegressor': -1,                  # meta estimator (3000)
    'BaggingClassifier': -1,                 # meta estimator (3000)
    'BayesianRidge': -1,                     # too fast
    'BernoulliNB': -1,                       # require count data (1000)
    'CCA': -1,                               # cryptic warning
    'CalibratedClassifierCV': -1,            # meta estimator
    'DummyRegressor': -1,                    # not useful
    'HistGradientBoostingRegressor': 500000, # calibrated
    'HistGradientBoostingClassifier': 20000, # calibrated
    'GradientBoostingClassifier': -1,        # meta estimator
    'IsotonicRegression': -1,                # requires 1d data
    'MultiOutputRegressor': -1,              # meta estimator
    'MultiOutputClassifier': -1,             # meta estimator
    'RegressorChain': -1,                    # meta estimator
    'RegressorChain': -1,                    # meta estimator
    'ClassifierChain': -1,                   # meta estimator
    'VotingRegressor': -1,                   # meta estimator
    'VotingClassifier': -1,                  # meta estimator
    '_SigmoidCalibration': -1,               # private estimator
    'ComplementNB': -1,                      # require count data
    'CheckingClassifier': -1,                # too fast
    'DecisionTreeClassifier': 80000,         # calibrated
    'DecisionTreeRegressor': 150000,         # calibrated
    'ExtraTreeClassifier': 800000,           # calibrated
    'ExtraTreeRegressor': 700000,            # calibrated
    'ExtraTreesRegressor': -1,               # meta estimator (80000)
    'ExtraTreesClassifier': -1,              # meta estimator (80000)
    'GaussianNB': -1,                        # too fast
    'LinearDiscriminantAnalysis': -1,        # need no colinear variables?
    'QuadraticDiscriminantAnalysis': -1,     # need no colinear variables?
    'LinearSVC': 3000,                       # calibrated
    'LinearSVR': 1000000,                    # calibrated
    'NuSVC': 5000,                           # calibrated
    'NuSVR': 8000,                           # calibrated
    'SVC': 5000,                             # calibrated
    'SVR': 10000,                            # calibrated
    'MLPClassifier': 30000,                  # calibrated
    'MLPRegressor': 15000,                   # calibrated
    'MultinomialNB': -1,                     # need count data
    'NearestCentroid': -1,                   # too fast (100000)
    'RidgeClassifier': -1,                   # too fast (400000)
    'RidgeClassifierCV': -1,                 # too fast (400000)
    'DummyClassifier': -1,                   # not useful
    'OneVsOneClassifier': -1,                # meta estimator
    'OneVsRestClassifier': -1,               # meta estimator
    'OutputCodeClassifier': -1,              # meta estimator
    'ElasticNet': -1,                        # failing for now  see #14249 (500000)
    'ElasticNetCV': -1,                      # failing for now  see #14249 (20000)
    'GaussianProcessRegressor': 7000,        # calibrated
    'GaussianProcessClassifier': 1500,       # calibrated
    'GradientBoostingRegressor': -1,         # meta estimator
    'HuberRegressor': 300000,                # calibrated
    'KNeighborsRegressor': -1,               # fit does nothing
    'KNeighborsClassifier': -1,              # fit does nothing
    'KernelRidge': 5000,                     # calibrated
    'LabelPropagation': 8000,                # calibrated 20 # parallelism cause: knn, bottleneck: rbfkenrel  # noqa
    'LabelSpreading': 8000,                  # calibrated 21 # parallelism cause: knn, bottleneck: rbfkenrel  # noqa
    'Lars': -1,                              # too fast (300000)
    'LarsCV': 150000,                        # calibrated
    'LassoLars': 1000000,                    # calibrated
    'LassoLarsCV': 500000,                   # calibrated
    'LassoLarsIC': -1,                       # too fast (1000000)
    'Lasso': -1,                             # too fast (1000000)
    'MultiTaskLasso': -1,                    # too fast (100000)
    'MultiTaskLassoCV': 3000,                # calibrated
    'LassoCV': -1,                           # failing for now  see #14249
    'LinearRegression': -1,                  # too fast (1000000)
    'LogisticRegression': 200000,            # calibrated
    'LogisticRegressionCV': 10000,           # calibrated
    'MultiTaskElasticNetCV': 1000,           # failing for now  see #14249
    'MultiTaskElasticNet': 500000,           # calibrated
    'OrthogonalMatchingPursuit': -1,         # too fast (200000)
    'OrthogonalMatchingPursuitCV': 300000,   # calibrated
    'PassiveAggressiveRegressor': 1400000,   # calibrated
    'PassiveAggressiveClassifier': 500000,   # calibrated
    'Perceptron': 400000,                    # calibrated
    'PLSCanonical': -1,                      # cryptic warning
    'PLSRegression': 1000000,                # calibrated
    'RadiusNeighborsRegressor': -1,          # fit does nothing
    'RadiusNeighborsClassifier': -1,         # fit does nothing
    'RandomForestRegressor': -1,             # meta estimator
    'RandomForestClassifier': -1,            # meta estimator
    'RANSACRegressor': -1,                   # meta estimator
    'Ridge': -1,                             # too fast (1000000)
    'RidgeCV': 300000,                       # calibrated
    'SGDRegressor': 1000000,                 # calibrated
    'SGDClassifier': 300000,                 # calibrated
    'TheilSenRegressor': 10000,              # calibrated
    'TransformedTargetRegressor': -1,        # too fast (1000000)
 }

PARAMS = {
    'ExtraTreesRegressor': {'n_estimators': 20},
    'ExtraTreesClassifier': {'n_estimators': 20},
    'RandomForestClassifier': {'n_estimators': 20},
    'RandomForestRegressor': {'n_estimators': 20},
    'LarsCV': {'cv': 100},
    'ElasticNetCV': {'cv': 20},
    'LassoCV': {'cv': 20},
    'LassoLarsCV': {'cv': 20},
    'LogisticRegressionCV': {'cv': 20, 'multi_class': 'auto'},
    'LogisticRegression': {'solver': 'lbfgs'},
    'MultiTaskElasticNetCV': {'cv': 20},
    'MultiTaskLassoCV': {'cv': 20},
    'GradientBoostingClassifier': {'n_estimators': 1, 'min_sample_split':20},
    'HistGradientBoostingClassifier': {'max_iter': 30},
    'HistGradientBoostingRegressor': {'max_iter': 30},
    'OrthogonalMatchingPursuitCV': {'cv': 20},
    'RidgeCV': {'cv': 20},
    'LinearSVC': {'max_iter': 1000},
    'MLPClassifier': {'hidden_layer_sizes': (10,), 'max_iter': 1000, 'learning_rate_init':0.01, 'learning_rate': 'adaptive'},
    'MLPRegressor': {'hidden_layer_sizes': (10,), 'max_iter': 1000, 'learning_rate_init':0.01, 'learning_rate': 'adaptive'},
    'NuSVC': {'gamma': 'auto'},
    'SVC': {'gamma': 'auto'},
    'NuSVR': {'gamma': 'auto'},
    'SVR': {'gamma': 'auto'},
    'RidgeClassifier': {'solver': 'auto'},
    'LogisticRegression': {'multi_class': 'auto', 'solver': 'lbfgs'}
 }
