import inspect

from sklearn.base import clone
from sklearn.utils.testing import all_estimators


def has_parallelism(estimator):
    initargs = inspect.getargs(estimator.__init__.__code__)
    return 'n_jobs' in initargs.args


def fit_estimator(estimator, X, y):
    """Fit an estimator.

    This function exists because we cannot profile directly methods of
    scikit-learn, as we need to decorate them in the source code
    """
    estimator.fit(X, y)


def clone_and_fit(estimator, X, y):
    """clone and fit an estimator

    This function is performs a fitting process after cloning the estimator
    given as input. It can be safely called from within a Parallel loop with
    a shared memory backend, and is common to objects that implement a fit
    method (in opposition with cross_val_score, that requires scoring and
    therefore that cannot be used with transformers)
    """
    # if we want to keep using pickle and not cloudpickle in those benchmarks,
    # we need to send pickleable fuctions. Directly decorating clone_and_fit
    # make it a nested function, thus not pickleable. Therefore, we make
    # clone_and_fit call a decorated function.
    cloned_estimator = clone(estimator)
    cloned_estimator.fit(X, y)


