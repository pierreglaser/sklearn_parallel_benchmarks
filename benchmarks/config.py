#!/usr/bin/env python
# -*- coding: utf-8 -*-
# default parameter values used for scikit-learn benchmarking
#
# Author: Pierre Glaser

# fmt: off
ESTIMATOR_BLACK_LIST = [
    'CalibratedClassifierCV',                 # meta estimator
    'ClassifierChain',                        # meta estimator
    'GradientBoostingClassifier',             # meta estimator
    'GradientBoostingRegressor',              # meta estimator
    'MultiOutputClassifier',                  # meta estimator
    'MultiOutputRegressor',                   # meta estimator
    'OneVsOneClassifier',                     # meta estimator
    'OneVsRestClassifier',                    # meta estimator
    'OutputCodeClassifier',                   # meta estimator
    'RANSACRegressor',                        # meta estimator
    'RandomForestClassifier',                 # meta estimator
    'RandomForestRegressor',                  # meta estimator
    'RegressorChain',                         # meta estimator
    'VotingClassifier',                       # meta estimator
    'VotingRegressor',                        # meta estimator
    'AdaBoostClassifier',                     # meta estimator  (300)
    'BaggingClassifier',                      # meta estimator (3000)
    'BaggingRegressor',                       # meta estimator (3000)
    'AdaBoostRegressor',                      # meta estimator (30000)
    'ExtraTreesClassifier',                   # meta estimator (80000)
    'ExtraTreesRegressor',                    # meta estimator (80000)
    'MultinomialNB',                          # need count data
    'LinearDiscriminantAnalysis',             # need no colinear variables?
    'QuadraticDiscriminantAnalysis',          # need no colinear variables?
    'DummyClassifier',                        # not useful
    'DummyRegressor',                         # not useful
    '_SigmoidCalibration',                    # private estimator
    'ComplementNB',                           # require count data
    'BernoulliNB',                            # require count data (1000)
    'IsotonicRegression',                     # requires 1d data
    'BayesianRidge',                          # too fast
    'CheckingClassifier',                     # too fast
    'GaussianNB',                             # too fast
    'MultiTaskLasso',                         # too fast (100000)
    'NearestCentroid',                        # too fast (100000)
    'Lasso',                                  # too fast (1000000)
    'LassoLarsIC',                            # too fast (1000000)
    'LinearRegression',                       # too fast (1000000)
    'TransformedTargetRegressor',             # too fast (1000000)
    'OrthogonalMatchingPursuit',              # too fast (200000)
    'Lars',                                   # too fast (300000)
    'RidgeClassifier',                        # too fast (400000)
    'RidgeClassifierCV',                      # too fast (400000)
    'CCA',                                    # cryptic warning
    'PLSCanonical',                           # cryptic warning
    'ElasticNetCV',                           # failing - see #14249 (20000)
    'MultiTaskElasticNetCV',                  # failing for now  see #14249
    'ElasticNet',                             # failing - see #14249 (500000)
    'LassoCV',                                # failing for now  see #14249
    'KNeighborsClassifier',                   # fit does nothing
    'KNeighborsRegressor',                    # fit does nothing
    'RadiusNeighborsClassifier',              # fit does nothing
    'RadiusNeighborsRegressor',               # fit does nothing
]

N_SAMPLES = {
    'Ridge': 100,                             # too fast (1000000)
    'LogisticRegressionCV': 10000,            # calibrated
    'SVR': 10000,                             # calibrated
    'TheilSenRegressor': 10000,               # calibrated
    'LassoLars': 1000000,                     # calibrated
    'LinearSVR': 1000000,                     # calibrated
    'PLSRegression': 1000000,                 # calibrated
    'SGDRegressor': 1000000,                  # calibrated
    'PassiveAggressiveRegressor': 1400000,    # calibrated
    'GaussianProcessClassifier': 1500,        # calibrated
    'MLPRegressor': 15000,                    # calibrated
    'DecisionTreeRegressor': 150000,          # calibrated
    'LarsCV': 150000,                         # calibrated
    'HistGradientBoostingClassifier': 20000,  # calibrated
    'LogisticRegression': 200000,             # calibrated
    'LinearSVC': 3000,                        # calibrated
    'MultiTaskLassoCV': 3000,                 # calibrated
    'MLPClassifier': 30000,                   # calibrated
    'HuberRegressor': 300000,                 # calibrated
    'OrthogonalMatchingPursuitCV': 300000,    # calibrated
    'RidgeCV': 300000,                        # calibrated
    'SGDClassifier': 300000,                  # calibrated
    'ARDRegression': 400,                     # calibrated
    'Perceptron': 400000,                     # calibrated
    'KernelRidge': 5000,                      # calibrated
    'NuSVC': 5000,                            # calibrated
    'SVC': 5000,                              # calibrated
    'HistGradientBoostingRegressor': 500000,  # calibrated
    'LassoLarsCV': 500000,                    # calibrated
    'MultiTaskElasticNet': 500000,            # calibrated
    'PassiveAggressiveClassifier': 500000,    # calibrated
    'GaussianProcessRegressor': 7000,         # calibrated
    'ExtraTreeRegressor': 700000,             # calibrated
    'LabelPropagation': 8000,                 # calibrated 20 # parallelism cause: knn, bottleneck: rbfkenrel  # noqa
    'LabelSpreading': 8000,                   # calibrated 21 # parallelism cause: knn, bottleneck: rbfkenrel  # noqa
    'NuSVR': 8000,                            # calibrated
    'DecisionTreeClassifier': 80000,          # calibrated
    'ExtraTreeClassifier': 800000,            # calibrated
 }
# fmt: on

PARAMS = {
    "ExtraTreesRegressor": {"n_estimators": 20},
    "ExtraTreesClassifier": {"n_estimators": 20},
    "RandomForestClassifier": {"n_estimators": 20},
    "RandomForestRegressor": {"n_estimators": 20},
    "LarsCV": {"cv": 100},
    "ElasticNetCV": {"cv": 20},
    "LassoCV": {"cv": 20},
    "LassoLarsCV": {"cv": 20},
    "LogisticRegressionCV": {"cv": 20, "multi_class": "auto"},
    "MultiTaskElasticNetCV": {"cv": 20},
    "MultiTaskLassoCV": {"cv": 20},
    "GradientBoostingClassifier": {"n_estimators": 1, "min_sample_split": 20},
    "HistGradientBoostingClassifier": {"max_iter": 30},
    "HistGradientBoostingRegressor": {"max_iter": 30},
    "OrthogonalMatchingPursuitCV": {"cv": 20},
    "RidgeCV": {"cv": 20},
    "LinearSVC": {"max_iter": 1000},
    "MLPClassifier": {
        "hidden_layer_sizes": (10,),
        "max_iter": 1000,
        "learning_rate_init": 0.01,
        "learning_rate": "adaptive",
    },
    "MLPRegressor": {
        "hidden_layer_sizes": (10,),
        "max_iter": 1000,
        "learning_rate_init": 0.01,
        "learning_rate": "adaptive",
    },
    "NuSVC": {"gamma": "auto"},
    "SVC": {"gamma": "auto"},
    "NuSVR": {"gamma": "auto"},
    "SVR": {"gamma": "auto"},
    "RidgeClassifier": {"solver": "auto"},
    "LogisticRegression": {"multi_class": "auto", "solver": "lbfgs"},
}
