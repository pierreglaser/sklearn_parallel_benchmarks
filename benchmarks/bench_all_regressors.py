from functools import partial
from abc import ABCMeta, abstractproperty, ABC

from sklearn.datasets import make_regression
from joblib import Parallel, delayed, parallel_backend

from benchmarks.common import (
    AbstractEstimatorBench,
    SingleFitParallelizationMixin,
    MultipleFitParallelizationMixin,
)
from benchmarks.estimators import (
    ALL_REGRESSORS,
    ALL_REGRESSORS_WITH_INTERNAL_PARALLELISM,
)
from benchmarks.config import N_SAMPLES, PARAMS


class AbstractRegressionBench(AbstractEstimatorBench):
    __metaclass__ = ABCMeta

    @property
    def data_factory(self):
        if "MultiTask" in self.estimator_name:
            return partial(make_regression, n_targets=4)
        else:
            return partial(make_regression, n_targets=1)


ALL_BENCHMARKS = {}
for est_name, est_cls in ALL_REGRESSORS.items():
    bench_name = "{}MultipleFitBench".format(est_name)
    bench_attrs = {
        "estimator_cls": est_cls,
        "estimator_params": PARAMS.get(est_name, {}),
        "default_n_samples": N_SAMPLES[est_name],
    }

    multiple_fit_bench_class = type(
        bench_name,
        (AbstractRegressionBench, MultipleFitParallelizationMixin),
        bench_attrs
    )
    ALL_BENCHMARKS[bench_name] = multiple_fit_bench_class

    if est_name in ALL_REGRESSORS_WITH_INTERNAL_PARALLELISM:
        bench_name = "{}SingleFitBench".format(est_name)
        single_fit_bench_class = type(
            bench_name,
            (AbstractRegressionBench, SingleFitParallelizationMixin),
            bench_attrs
        )
        ALL_BENCHMARKS[bench_name] = single_fit_bench_class

globals().update(ALL_BENCHMARKS)
