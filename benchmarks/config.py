#!/usr/bin/env python
# -*- coding: utf-8 -*-
# default parameter values used for scikit-learn benchmarking
#
# Author: Pierre Glaser
import pandas as pd

N_SAMPLES = {
    'AdaBoostRegressor': 30000,              # 0
    'ARDRegression': 500,                    # 1
    'BaggingRegressor': 30000,               # 2
    'BaggingClassifier': 100000,             # 2
    'BayesianRidge': 100000,                 # 3
    'CCA': 500000,                           # 4
    'DecisionTreeRegressor': 200000,         # 5
    'ElasticNet': 500000,                    # 6
    'ElasticNetCV': 20000,                   # 7
    'ExtraTreeRegressor': 5000000,           # 8
    'ExtraTreesRegressor': 200000,           # 9
    'ExtraTreesClassifier': 1000000,         # 9
    'GaussianProcessRegressor': 7000,        # 10
    'GaussianProcessClassifier': 2000,       # 10  # parallelism level: classes
    'GradientBoostingRegressor': 40000,      # 11
    'HuberRegressor': 300000,                # 12
    'KNeighborsRegressor': 2000000,          # 13
    'KernelRidge': 5000,                     # 14
    'Lars': 300000,                          # 15
    'LabelPropagation': 8000,                # 15 # parallelism cause: knn, bottleneck: rbfkenrel
    'LabelSpreading': 8000,                  # 15 # parallelism cause: knn, bottleneck: rbfkenrel
    'LarsCV': 300000,                        # 16
    'Lasso': 1000000,                        # 17
    'LassoCV': 300000,                       # 18
    'LassoLars': 1000000,                    # 19
    'LassoLarsCV': 500000,                   # 20
    'LassoLarsIC': 1000000,                  # 21
    'LinearRegression': 1000000,             # 22
    'LogisticRegression': 100000,            # cython+ threading backend causing trouble
    'LogisticRegressionCV': 10000,           # cython+ threading backend causing trouble
    'LinearSVR': 1000000,                    # 23
    'MLPRegressor': 7000,                    # 24
    'MultiTaskElasticNet': 1000000,          # 25     # multidimensional y
    'MultiTaskElasticNetCV': 1000000,        # 26     # multidimensional y
    'MultiTaskLasso': 1000,                  # 27     # multidimensional y
    'MultiTaskLassoCV': 1000,                # 28     # multidimensional y
    'NuSVR': 10000,                          # 29
    'OrthogonalMatchingPursuit': 2000000,    # 30
    'OrthogonalMatchingPursuitCV': 1000000,  # 31
    'PassiveAggressiveRegressor': 1000000,   # 32
    'PassiveAggressiveClassifier': 500000,   # 32 # looks good
    'Perceptron': 800000,                    # 32 # looks good
    'PLSCanonical': 1000000,                 # 33
    'PLSRegression': 1000000,                # 34
    'RadiusNeighborsRegressor': 1000000,     # 35
    'RandomForestRegressor': 30000,          # 36
    'RANSACRegressor': 1000000,              # 37
    'Ridge': 1000000,                        # 38
    'RidgeCV': 100000,                       # 39
    'SGDRegressor': 1000000,                 # 40
    'SVR': 10000,                            # 41
    'TheilSenRegressor': 10000,              # 42
    'TransformedTargetRegressor': 1000000    # 43
 }

PARAMS = {
    'AdaBoostRegressor': 30000,
    'ARDRegression': 500,
    'BaggingRegressor': 30000,
    'BayesianRidge': 100000,
    'CCA': 500000,
    'DecisionTreeRegressor': 200000,
    'ElasticNet': 500000,
    'ElasticNetCV': 20000,
    'ExtraTreeRegressor': 300000,
    'ExtraTreesRegressor': {'n_estimators': 100},
    'GaussianProcessRegressor': 7000,
    'GradientBoostingRegressor': 40000,
    'HuberRegressor': 300000,
    'KNeighborsRegressor': 300000,
    'KernelRidge': 5000,
    'Lars': 300000,
    'LarsCV': {'cv': 100},
    'Lasso': 1000000,
    'LassoCV': 300000,
    'LassoLars': 1000000,
    'LassoLarsCV': 500000,
    'LassoLarsIC': 1000000,
    'LinearRegression': 1000000,
    'LinearSVR': 1000000,
    'MLPRegressor': 7000,
    'MultiTaskElasticNet': 1000,    # needs multidimensional y
    'MultiTaskElasticNetCV': 1000,  # needs multidimensional y
    'MultiTaskLasso': 1000,         # needs multidimensional y
    'MultiTaskLassoCV': 1000,       # needs multidimensional y
    'NuSVR': 10000,
    'OrthogonalMatchingPursuit': 2000000,
    'OrthogonalMatchingPursuitCV': 1000000,
    'PassiveAggressiveRegressor': 1000000,
    'PLSCanonical': 1000000,
    'PLSRegression': 1000000,
    'RadiusNeighborsRegressor': 1000000,
    'RandomForestRegressor': 30000,
    'RANSACRegressor': 1000000,
    'Ridge': 1000000,
    'RidgeCV': 100000,
    'SGDRegressor': 1000000,
    'SVR': 10000,
    'TheilSenRegressor': 10000,
    'TransformedTargetRegressor': 1000000
 }
