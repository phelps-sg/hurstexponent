from collections.abc import Callable
from functools import lru_cache
from typing import List, Any
from numpy.typing import NDArray


import numpy as np
import pytest
from typing_extensions import Tuple

from hurstexponent import standard_hurst, generalized_hurst
from util.generate_series import simple_series

Estimator = Callable[[Any], Tuple[float, float, Any]]

estimators: List[Tuple[str, Estimator]] = [
    ("standard (mle)", lambda s: standard_hurst(s, fitting_method="mle")),
    (
        "standard (mle, initial_guess=0.4)",
        lambda s: standard_hurst(s, fitting_method="mle", initial_guess_H=0.4),
    ),
    (
        "standard (mle, initial_guess=0.6)",
        lambda s: standard_hurst(s, fitting_method="mle", initial_guess_H=0.6),
    ),
    ("standard (OLS)", lambda s: standard_hurst(s, fitting_method="OLS")),
    (
        "standard (least_squares)",
        lambda s: standard_hurst(s, fitting_method="least_squares"),
    ),
    ("generalised (mle)", lambda s: generalized_hurst(s, fitting_method="mle")),
    ("generalised (OLS)", lambda s: generalized_hurst(s, fitting_method="OLS")),
    (
        "generalised (least_squares)",
        lambda s: generalized_hurst(s, fitting_method="least_squares"),
    ),
]


@lru_cache
def bootstrap(
    estimator: Estimator,
    reps: int = 1000,
    seed: int = 42,
    length: int = 2048,
    volatility: float = 0.00002,
) -> NDArray:
    """
    Perform a bootstrap where we compute the specified Hurst estimator for a
    white noise process with the specified sample size and volatility.
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


@pytest.mark.parametrize(["estimator_name", "estimator"], estimators)
def test_unbiased_estimator(estimator_name: str, estimator: Estimator):
    """Check whether the estimator gives an unbiased estimate of H=0.5 for white noise."""
    point_estimates = bootstrap(estimator)
    assert np.isclose(np.mean(point_estimates), 0.5, rtol=1e-2)


@pytest.mark.parametrize(["estimator_name", "estimator"], estimators)
def test_within_limits(estimator_name: str, estimator: Estimator):
    """Check whether the estimator gives valid point estimates within interval (0, 1)."""
    point_estimates = bootstrap(estimator)
    assert np.min(point_estimates) >= 0.0
    assert np.max(point_estimates) <= 1.0


@pytest.mark.parametrize(["estimator_name", "estimator"], estimators)
def test_confidence_interval(estimator_name: str, estimator: Estimator):
    """Check whether the estimator gives a confidence interval whose width is within
    the worst-case reported in the literature."""
    point_estimates = bootstrap(estimator, length=2048)
    lower_ci = np.percentile(point_estimates, 2.5)
    upper_ci = np.percentile(point_estimates, 97.5)
    assert upper_ci < 0.75
    assert lower_ci > 0.25
