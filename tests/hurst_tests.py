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
        "standard (mle, initial_guess=0.2)",
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
    np.random.seed(seed)
    return np.array(
        [
            estimator(simple_series(length=length, volatility=volatility))[0]
            for _repetition in range(reps)
        ]
    )


@pytest.mark.parametrize(["estimator_name", "estimator"], estimators)
def test_unbiased_estimator(estimator_name: str, estimator: Estimator):
    """Check whether estimator gives unbiased estimate of H=0.5 for white noise"""
    point_estimates = bootstrap(estimator)
    assert np.isclose(np.mean(point_estimates), 0.5, rtol=1e-2)


@pytest.mark.parametrize(["estimator_name", "estimator"], estimators)
def test_within_limits(estimator_name: str, estimator: Estimator):
    point_estimates = bootstrap(estimator)
    assert np.min(point_estimates) >= 0.0
    assert np.max(point_estimates) <= 1.0
