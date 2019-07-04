#!/usr/bin/env python
# -*- coding: utf-8 -*- # Benchmarking all in scikit-learn
#
# Author: Pierre Glaser

from functools import partial
from abc import ABCMeta

from sklearn.datasets import make_classification

from benchmarks.common import (
    AbstractEstimatorBench,
    MultipleFitParallelizationMixin,
    SingleFitParallelizationMixin,
)
from benchmarks.estimators import (
    ALL_CLASSIFIERS,
    ALL_CLASSIFIERS_WITH_INTERNAL_PARALLELISM
)
from benchmarks.config import PARAMS, N_SAMPLES


class AbstractClassificationBench(AbstractEstimatorBench):
    __metaclass__ = ABCMeta
    data_factory = partial(make_classification, n_classes=10, n_informative=8)


ALL_BENCHMARKS = {}
for est_name, est_cls in ALL_CLASSIFIERS.items():
    bench_name = "{}MultipleFitBench".format(est_name)
    bench_attrs = {
        "estimator_cls": est_cls,
        "estimator_params": PARAMS.get(est_name, {}),
        "default_n_samples": N_SAMPLES[est_name],
    }

    multiple_fit_bench_class = type(
        bench_name,
        (AbstractClassificationBench, MultipleFitParallelizationMixin),
        bench_attrs
    )
    ALL_BENCHMARKS[bench_name] = multiple_fit_bench_class

    if est_name in ALL_CLASSIFIERS_WITH_INTERNAL_PARALLELISM:
        bench_name = "{}SingleFitBench".format(est_name)
        single_fit_bench_class = type(
            bench_name,
            (AbstractClassificationBench, SingleFitParallelizationMixin),
            bench_attrs
        )
        ALL_BENCHMARKS[bench_name] = single_fit_bench_class


globals().update(ALL_BENCHMARKS)
