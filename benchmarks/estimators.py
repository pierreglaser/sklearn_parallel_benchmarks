#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Set of custom scikit-learn compliant estimators
#
# Author: Pierre Glaser


class EstimatorWithLargeList:
    """simple estimator, with a large list as an attribute

    Instances of this class should take a long time to serizlize using
    cloudpickle, as large lists are typically very costly ot pickle.
    This estimator is used to validate that the pickling strategu during the
    benchmarks is set as expected."""

    def __init__(self):
        self.large_list = list(range(100000))
        self.best_estimator_ = self

    def get_params(self, *args, **kwargs):
        return dict()

    def set_params(self, *args, **kwargs):
        return self

    def fit(self, *args, **kwargs):
        pass

    def predict(self, X):
        return [0] * len(X)

    def score(self, *args, **kwargs):
        return 0
