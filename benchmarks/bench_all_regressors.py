from abc import ABCMeta, abstractproperty, ABC

from sklearn.datasets import make_regression
from joblib import Parallel, delayed, parallel_backend

from benchmarks.common import (
    SklearnBenchmark,
    ALL_REGRESSORS_WITH_INTERNAL_PARALLELISM,
    fit_estimator,
    clone_and_fit,
)
from benchmarks.config import N_SAMPLES, PARAMS


class AbstractRegressionBench(ABC, SklearnBenchmark):
    __metaclass__ = ABCMeta
    param_names = ["backend", "pickler", "n_jobs", "n_samples", "n_features"]
    params = (
        ["multiprocessing", "loky", "threading"][1:],
        ["pickle", "cloudpickle"][:1],
        [1, 2, 4][:2],
        ["auto"],
        ["auto"],
    )

    @abstractproperty
    def estimator_cls(self):
        raise NotImplementedError

    @property
    def estimator_name(self):
        return self.estimator_cls.__name__

    def setup(self, backend, pickler, n_jobs, n_samples, n_features):
        super(AbstractRegressionBench, self).setup(pickler)
        if n_samples == "auto":
            n_samples = N_SAMPLES[self.estimator_name]

        if n_features == "auto":
            n_features = 10

        # For multitask estimators, generate multi-dimensional output
        if "MultiTask" in self.estimator_name:
            X, y = make_regression(n_samples, n_features, n_targets=4)
        else:
            X, y = make_regression(n_samples, n_features, n_targets=1)

        # warm up the executor to hide the process-creation overhead
        with parallel_backend(backend, n_jobs):
            Parallel()(delayed(id)(i) for i in range(n_jobs))
        self.X = X
        self.y = y

    def time_single_fit_parallelization(
        self, backend, pickler, n_jobs, n_samples, n_features
    ):
        cls = ALL_REGRESSORS_WITH_INTERNAL_PARALLELISM[self.estimator_name]
        estimator = cls()
        if "n_jobs" in estimator.get_params():
            estimator.set_params(n_jobs=n_jobs)
        else:
            print(
                "n_jobs is not an attribute of {}, not running the "
                " benchmark".format(self.estimator_name)
            )
            return NotImplemented

        if self.estimator_name in PARAMS:
            estimator.set_params(**PARAMS[self.estimator_name])

        if "cv" in estimator.get_params():
            estimator.set_params(cv=5)

        with parallel_backend(backend, n_jobs):
            fit_estimator(estimator, self.X, self.y)

    def time_multiple_fit_parallelization(
        self, backend, pickler, n_jobs, n_samples, n_features
    ):
        cls = ALL_REGRESSORS_WITH_INTERNAL_PARALLELISM[self.estimator_name]
        estimator = cls()
        if "n_jobs" in estimator.get_params():
            # avoid over subscription
            estimator.set_params(n_jobs=1)

        Parallel(backend=backend, n_jobs=n_jobs)(
            delayed(clone_and_fit)(estimator, self.X, self.y)
            for _ in range(self.n_tasks)
        )


ALL_BENCHMARKS = {}
for est_name, est_cls in ALL_REGRESSORS_WITH_INTERNAL_PARALLELISM.items():
    bench_name = "{}Bench".format(est_name)
    bench_class = type(
        bench_name, (AbstractRegressionBench,), {"estimator_cls": est_cls})
    ALL_BENCHMARKS[est_name] = bench_class

globals().update(ALL_BENCHMARKS)
