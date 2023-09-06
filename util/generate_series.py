import warnings
import numpy as np
from typing import List


# Synthetic data generators
def simple_series(length: int = 99999, initial_value: float = 1.0, volatility: float = 0.001, seed: int = None) -> np.ndarray:
    """
    Generate a geometric random walk with Gaussian innovations.

    Parameters
    ----------
    length : int, optional
        The length of the time series to generate. Default is 1000.
    initial_value : int
        The initial value of the series
    volatility: float, optional
        The volatility of returns
    seed : {None, int, array_like, BitGenerator}, optional Random seed used to initialize the pseudo-random number
    generator or an instantized BitGenerator.

    Returns
    -------
    numpy.ndarray
        The generated time series.
    """

    if seed is not None:
        np.random.seed(seed)
    returns = np.random.normal(size=length, scale=volatility)
    return initial_value * np.exp(np.cumsum(returns))


# Much of this function generalises Dmitry Motti's implementation of a random walk process, specifically around line 197
# at https://github.com/Mottl/hurst/blob/master/hurst/__init__.py. Key improvements around proba, which appears stale in
# the latter.
def stochastic_process(length: int = 99999, proba: float = 0.5, min_lag: int = 10, max_lag: int = 100, cumprod: bool = False, seed: int = None) -> List[float]:
    """
    Generates a stochastic process

    Parameters
    ----------
    length : int
        Length of the random walk series.
    proba : float, default 0.5
        The probability that the next increment will follow the trend.
        Set proba > 0.5 for the persistent random walk,
        set proba < 0.5 for the antipersistent one.
    min_lag : int, default 10
    max_lag : int, default 100
        Minimum and maximum lag sizes to calculate trend direction.
    cumprod : bool, default False
        Generate a random walk as a cumulative product instead of cumulative sum.
    seed : int, optional
        Random seed used to initialize the pseudo-random number generator.

    Returns
    -------
    series : List[float]
        Generated random walk series.
    """
    assert(min_lag >= 1)
    assert(max_lag >= min_lag)

    if max_lag > length:
        max_lag = length
        warnings.warn("max_lag has been set to the length of the series.")

    # Set the seed for the random number generator
    if seed is not None:
        np.random.seed(seed)

    series = np.zeros(length, dtype=float)
    series[0] = 1. if cumprod else 0.

    for i in range(1, length):
        if i < min_lag + 1:
            direction = np.sign(np.random.randn())
        else:
            lag = np.random.randint(min_lag, min(i-1, max_lag) + 1)
            direction = np.sign(series[i-1] / series[i-1-lag] - 1.) if cumprod else np.sign(series[i-1] - series[i-1-lag])
            direction *= np.sign(proba - np.random.uniform())

        increment = np.abs(np.random.randn())
        if cumprod:
            series[i] = series[i-1] * np.abs(1 + increment / 1000. * direction)
        else:
            series[i] = series[i-1] + increment * direction

    return series
