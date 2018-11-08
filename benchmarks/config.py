#!/usr/bin/env python
# -*- coding: utf-8 -*-
# default parameter values used for scikit-learn benchmarking
#
# Author: Pierre Glaser
from sum_up_results import create_benchmark_dataframe

N_SAMPLES = {
    'AdaBoostRegressor': 30000,
    'ARDRegression': 500,
    'BaggingRegressor': 30000,
    'BayesianRidge': 100000,
    'CCA': 500000,
    'DecisionTreeRegressor': 200000,
    'ElasticNet': 500000,
    'ElasticNetCV': 20000,
    'ExtraTreeRegressor': 5000000,
    'ExtraTreesRegressor': 200000,
    'GaussianProcessRegressor': 7000,
    'GradientBoostingRegressor': 40000,
    'HuberRegressor': 300000,
    'KNeighborsRegressor': 2000000,
    'KernelRidge': 5000,
    'Lars': 300000,
    'LarsCV': 300000,
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
    'LarsCV': 300000,
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

benchmarks_results = create_benchmark_dataframe(
        group_by='class')['RegressionBench']
benchmarks_results.fillna(0, inplace=True)
