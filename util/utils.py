import warnings
import numpy as np
from scipy.stats import norm
from typing import Callable, Iterable, List


def simple_series(length: int = 1000, noise_pct_std: float = 0.001) -> np.ndarray:
    """
    Generate a synthetic time series using a random walk model with added Gaussian noise

    Parameters
    ----------
    length : int, optional
        The length of the time series to generate. Default is 1000.
    noise_pct_std : float, optional
        The standard deviation of the Gaussian noise added to the series, expressed as a percentage of the standard
        deviation of the original random walk. Default is 0.001.

    Returns
    -------
    numpy.ndarray
        The generated time series.
    """

    # Generate a series of random changes
    # np.random.seed(42)
    random_changes = 1 + np.random.randn(99999) / 1000

    # Create a non-stationary random walk series and then difference it to make it stationary
    series = np.cumprod(random_changes)  # create a random walk from random changes

    # Scale the random changes by the desired standard deviation as a percentage of the mean value
    series += np.random.randn(length) * noise_pct_std * np.std(series)

    return series


# Much of this function generalises Dmitry Motti's implementation of a random walk process, specifically around line 197
# at https://github.com/Mottl/hurst/blob/master/hurst/__init__.py. Key improvements sround proba, which appears stale in
# the latter, and the inclusion of noise_pct_std parameter
def stochastic_process(length: int, proba: float = 0.5, min_lag: int = 1, max_lag: int = 100, cumprod: bool = False) -> List[float]:
    """
    Generates a random walk series

    Parameters
    ----------
    length : int
        Length of the random walk series.
    proba : float, default 0.5
        The probability that the next increment will follow the trend.
        Set proba > 0.5 for the persistent random walk,
        set proba < 0.5 for the antipersistent one.
    min_lag : int, default 1
    max_lag : int, default 100
        Minimum and maximum lag sizes to calculate trend direction.
    cumprod : bool, default False
        Generate a random walk as a cumulative product instead of cumulative sum.

    Returns
    -------
    series : List[float]
        Generated random walk series.
    """
    assert(min_lag>=1)
    assert(max_lag>=min_lag)

    if max_lag > length:
        max_lag = length
        warnings.warn("max_lag has been set to the length of the eries.")

    series = np.zeros(length, dtype=float)
    series[0] = 1. if cumprod else 0.

    for i in range(1, length):
        if i < min_lag + 1:
            direction = np.sign(np.random.randn())
        else:
            lookback = np.random.randint(min_lag, min(i-1, max_lag)+1)
            direction = np.sign(series[i-1] / series[i-1-lookback] - 1.) if cumprod else np.sign(series[i-1] - series[i-1-lookback])
            direction *= np.sign(proba - np.random.uniform())

        increment = np.abs(np.random.randn())
        if cumprod:
            series[i] = series[i-1] * np.abs(1 + increment / 1000. * direction)
        else:
            series[i] = series[i-1] + increment * direction

    return series


# Helper functions
def neg_log_likelihood(params, x_values, y_values,func: Callable) -> float:
    """
        Compute the negative log-likelihood for a given model and parameters.

        The function calculates the negative log-likelihood between the observed y-values and the predicted y-values obtained from the specified function and parameters.

        Parameters
        ----------
        params : Iterable[float]
            The parameters of the model function. They are passed to the function 'func' for evaluation.
        x_values : Iterable[float]
            The independent variable values.
        y_values : Iterable[float]
            The observed dependent variable values corresponding to 'x_values'.
        func : Callable
            The model function that takes 'x_values' and '*params' as input and returns the predicted y-values.

        Returns
        -------
        float
            The negative log-likelihood value for the given model and parameters.

        """

    y_pred = func(x_values, *params)
    residuals = y_values - y_pred
    std_dev_residuals = np.std(residuals)
    loglikelihoods = norm.logpdf(residuals, loc=0, scale=std_dev_residuals)
    loglikelihood = -np.sum(loglikelihoods)  # sum over all data points

    return loglikelihood


def std_of_sums(ts: np.array, chunk_size: int) -> float:
    """
    Computes the standard deviation of sums of time series chunks of size chunk_size.

    Parameters
    ----------
    ts : np.array
        Time series data
    chunk_size : int
        The size of each chunk of the time series

    Returns
    -------
    std : float
        The standard deviation of the sums
    """
    sums = []
    for i in range(0, len(ts), chunk_size):  # Iterate over the time series with a step size of chunk_size
        chunk = ts[i: i + chunk_size]  # Get the next chunk of size chunk_size
        if len(chunk) == chunk_size:  # If we have a full chunk of size chunk_size
            sums.append(np.sum(chunk))  # Sum up the chunk and add to the list
    return np.std(sums)


def calculate_diffs(ts: np.array, lag: int) -> np.ndarray:
    """
    Calculate detrended differences at specified lag steps in the time series.

    Parameters
    ----------
    ts : np.array
        The time series data
    lag : int
        The step size to compute the differences

    Returns
    -------
    diffs : np.ndarray
        Detrended differences of the time series at specified lags
    """

    return ts[:-lag] - ts[lag:]


def structure_function(ts: np.array, moment: int, lag: int) -> float:
    """
    Calculate the structure function for a given moment and lag, defined as the mean of the absolute differences
    to the power of the specified moment divided by the mean of absolute ts to the power of the specified moment.

    Parameters
    ----------
    ts : np.array
        The time series data
    moment : int
        The moment for which the structure function is to be calculated
    lag : int
        The lag at which the structure function is to be calculated

    Returns
    -------
    S_q_tau : float
        The calculated structure function for the specified moment and lag.
        If the differences array is empty, it returns np.nan
    """
    diffs = np.abs(calculate_diffs(ts, lag))
    ts_abs_moment = np.abs(ts[:-lag]) ** moment
    if diffs.size != 0 and np.any(ts_abs_moment):
        return np.mean(diffs ** moment) / np.mean(ts_abs_moment)
    else:
        return np.nan


def interpret_hurst(H: float) -> str:
    """
    Interpretation of Hurst Exponent, which represents a measure of the long-term memory of time series.

    Parameters
    ----------
    H : float
        Hurst Exponent.

    Returns
    -------
    str
        Interpretation of Hurst Exponent.
    """
    # H = round(H, 2)

    if not 0 <= H <= 1:
        return "Hurst Exponent not in a valid range [0, 1], series may not be a long memory process"
    if H == 0.5:
        return "Perfect diffusivity: series is a geometric or Brownian random walk"
    if H < 0.5:
        return "Sub-diffusive: series demonstrates anti-persistent behavior"
    if H > 0.5:
        return "Super-diffusive: series demonstrates persistent long-range dependence"
    return "Invalid Hurst Exponent"
