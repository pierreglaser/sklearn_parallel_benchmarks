#!/usr/bin/env python
# -*- coding: utf-8 -*-
# default parameter values used for scikit-learn benchmarking
#
# Author: Pierre Glaser
import pandas as pd

N_SAMPLES = {
    'AdaBoostRegressor': 30000,              # 0
    'ARDRegression': 500,                    # 1
    'BaggingRegressor': 3000,                # 2
    'BaggingClassifier': 100000,             # 3
    'BayesianRidge': 100000,                 # 4
    'CCA': 500000,                           # 5
    'DecisionTreeRegressor': 200000,         # 6
    'ElasticNet': 500000,                    # 7
    'ElasticNetCV': 20000,                   # 8
    'ExtraTreeRegressor': 5000000,           # 9
    'ExtraTreesRegressor': 200000,           # 10
    'ExtraTreesClassifier': 1000000,         # 11
    'GaussianProcessRegressor': 7000,        # 12
    'GaussianProcessClassifier': 2000,       # 13  # parallelism level: classes
    'GradientBoostingRegressor': 40000,      # 14
    'HuberRegressor': 300000,                # 15
    'KNeighborsRegressor': 2000000,          # 16
    'KNeighborsClassifier': 200,             # 17  # fit does nothing
    'KernelRidge': 5000,                     # 18
    'Lars': 300000,                          # 19
    'LabelPropagation': 8000,                # 20 # parallelism cause: knn, bottleneck: rbfkenrel  # noqa
    'LabelSpreading': 8000,                  # 21 # parallelism cause: knn, bottleneck: rbfkenrel  # noqa

    'LarsCV': 300000,                        # 22
    'Lasso': 1000000,                        # 23
    'LassoCV': 300000,                       # 24
    'LassoLars': 1000000,                    # 25
    'LassoLarsCV': 500000,                   # 26
    'LassoLarsIC': 1000000,                  # 27
    'LinearRegression': 1000000,             # 28
    'LogisticRegression': 100000,            # 29
    'LogisticRegressionCV': 10000,           # 30
    'LinearSVR': 1000000,                    # 31
    'MLPRegressor': 7000,                    # 32
    'MultiTaskElasticNet': 1000000,          # 33     # multidimensional y
    'MultiTaskElasticNetCV': 1000000,        # 34     # multidimensional y
    'MultiTaskLasso': 1000,                  # 35     # multidimensional y
    'MultiTaskLassoCV': 1000,                # 36     # multidimensional y
    'NuSVR': 10000,                          # 37
    'OrthogonalMatchingPursuit': 2000000,    # 38
    'OrthogonalMatchingPursuitCV': 1000000,  # 39
    'PassiveAggressiveRegressor': 1000000,   # 40
    'PassiveAggressiveClassifier': 500000,   # 41 # looks good
    'Perceptron': 800000,                    # 42 # looks good
    'PLSCanonical': 1000000,                 # 43
    'PLSRegression': 1000000,                # 44
    'RadiusNeighborsRegressor': 10000,       # 45  # fit does nothing
    'RadiusNeighborsClassifier': 10000,      # 46  # fit does nothing
    'RandomForestRegressor': 30000,          # 47
    'RandomForestClassifier': 100000,        # 48
    'RANSACRegressor': 1000000,              # 49
    'Ridge': 1000000,                        # 50
    'RidgeCV': 100000,                       # 51
    'SGDRegressor': 1000000,                 # 52
    'SGDClassifier': 500000,                 # 53
    'SVR': 10000,                            # 54
    'TheilSenRegressor': 10000,              # 55
    'TransformedTargetRegressor': 1000000    # 56
 }

PARAMS = {
    'ExtraTreesRegressor': {'n_estimators': 100},
    'ExtraTreesClassifier': {'n_estimators': 100},
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
    'OrthogonalMatchingPursuitCV': {'cv': 20},
    'RidgeCV': {'cv': 20},
 }
