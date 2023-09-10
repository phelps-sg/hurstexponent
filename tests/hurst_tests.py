""" Tests for hurst estimators """
import functools
from collections.abc import Callable
from functools import lru_cache
from typing import List, Any, Iterable
from numpy.typing import NDArray


import numpy as np
import pytest
from typing_extensions import Tuple

from hurstexponent import standard_hurst, generalized_hurst
from util.generate_series import simple_series

Estimator = Callable[[NDArray[np.float64]], Tuple[float, float, Any]]


def estimators_with_all_fitting_methods(
    base_name: str,
    base_estimator: Callable,
    fitting_methods: Iterable[str] = ("mle", "least_squares", "OLS"),
) -> List[Tuple[str, Estimator]]:
    """Create estimators for all fitting methods."""
    return [
        (
            f"{base_name} [{fitting_method}]",
            functools.partial(base_estimator, fitting_method=fitting_method),
        )
        for fitting_method in fitting_methods
    ]


def estimators_with_initial_guesses(
    base_name: str,
    base_estimator: Callable,
    initial_guesses: Iterable[float] = (0.4, 0.6),
) -> List[Tuple[str, Estimator]]:
    """Create estimators for all initial guesses of H."""
    return [
        (
            f"{base_name} [mle initial_guess={guess}]",
            functools.partial(
                base_estimator, fitting_method="mle", initial_guess_H=guess
            ),
        )
        for guess in initial_guesses
    ]


@lru_cache
def all_estimators() -> List[Tuple[str, Estimator]]:
    """List of all estimators used for tests in the form [(estimator_name, estimator_fn), ...]"""
    return (
        estimators_with_all_fitting_methods("standard", standard_hurst)
        + estimators_with_all_fitting_methods("generalised", generalized_hurst)
        + estimators_with_initial_guesses("standard", standard_hurst)
    )


@lru_cache
def bootstrap(
    estimator: Estimator,
    reps: int = 10000,
    seed: int = 42,
    length: int = 2048,
    volatility: float = 0.00002,
) -> NDArray[np.float64]:
    """
    Perform a bootstrap where we compute the specified Hurst estimator for a
    white noise process (H=0.5) with the specified sample size and volatility.
    We compute each estimate iid. for the same data generation process,
    and fix the seed once ahead of generating the iid. samples.
    """
    np.random.seed(seed)
    return np.array(
        [
            estimator(simple_series(length=length, volatility=volatility))[0]
            for _repetition in range(reps)
        ]
    )


@pytest.mark.parametrize(["_estimator_name", "estimator"], all_estimators())
def test_unbiased_estimator(_estimator_name: str, estimator: Estimator):
    """Check whether the estimator gives an unbiased estimate of H=0.5 for white noise."""
    point_estimates = bootstrap(estimator)
    assert np.isclose(np.mean(point_estimates), 0.5, rtol=1e-2)


@pytest.mark.parametrize(["_estimator_name", "estimator"], all_estimators())
def test_within_limits(_estimator_name: str, estimator: Estimator):
    """Check whether the estimator gives valid point estimates within interval (0, 1)."""
    point_estimates = bootstrap(estimator)
    assert np.min(point_estimates) >= 0.0
    assert np.max(point_estimates) <= 1.0


@pytest.mark.parametrize(["_estimator_name", "estimator"], all_estimators())
def test_confidence_interval(_estimator_name: str, estimator: Estimator):
    """Check whether the estimator gives a 95% confidence interval whose bounds are within
    the worst-case reported in the literature.  See Weron, Rafał.
    "Estimating long-range dependence: finite sample properties and confidence intervals."
    Physica A: Statistical Mechanics and its Applications 312.1-2 (2002): 285-299.
    """
    point_estimates = bootstrap(estimator, length=2048)
    lower_ci = np.percentile(point_estimates, 2.5)
    upper_ci = np.percentile(point_estimates, 97.5)
    assert upper_ci < 0.75
    assert lower_ci > 0.25
