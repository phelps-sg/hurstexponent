import numpy
import numpy as np
from Cython.Includes import numpy


# Helper functions
def std_of_sums(ts: np.array, chunk_size: int) -> float:
    """
    Calculates the standard deviation of sums of time series chunks of size chunk_size.

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
    Calculate the structure function for a given moment and lag.
    The structure function is defined as the mean of the absolute differences
    to the power of the specified moment divided by the mean of absolute ts
    to the power of the specified moment.

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


import numpy as np

import numpy as np


def generate_series(length=1000, noise_pct_std=0.001):

    # Generate a series of random changes
    # np.random.seed(42)
    random_changes = 1 + np.random.randn(99999) / 1000

    # Create a non-stationary random walk series and then difference it to make it stationary
    series = np.cumprod(random_changes)  # create a random walk from random changes


    # Scale the random changes by the desired standard deviation as a percentage of the mean value
    series += np.random.randn(length) * noise_pct_std * np.std(series)

    return series


def interpret_hurst(H: float) -> str:
    """
    Interpretation of Hurst Exponent.

    This function provides interpretation of the Hurst exponent which is a measure of the long-term memory of time series.

    Parameters
    ----------
    H : float
        Hurst Exponent.

    Returns
    -------
    str
        Interpretation of Hurst Exponent.

    Interpretations:
    ----------------
    If Hurst Exponent is not in the range [0, 1], it returns:
        "Hurst Exponent not in a valid range [0, 1], series may not be a long memory process".
    If Hurst Exponent equals 0.5, it returns:
        "Perfect diffusivity: series is a geometric or Brownian random walk".
    If Hurst Exponent is less than 0.5, it returns:
        "Sub-diffusive: series demonstrates anti-persistent behavior".
    If Hurst Exponent is greater than 0.5, it returns:
        "Super-diffusive: series demonstrates persistent long-range dependence".
    If none of the above conditions are met, it returns:
        "Invalid Hurst Exponent".
    """
    if not 0 <= H <= 1:
        return "Hurst Exponent not in a valid range [0, 1], series may not be a long memory process"
    if H == 0.5:
        return "Perfect diffusivity: series is a geometric or Brownian random walk"
    if H < 0.5:
        return "Sub-diffusive: series demonstrates anti-persistent behavior"
    if H > 0.5:
        return "Super-diffusive: series demonstrates persistent long-range dependence"
    return "Invalid Hurst Exponent"


def residual(params, N, y_values):
    """
    Calculate residuals of the predicted y_values from the actual ones.

    Parameters
    ----------
    params : list
        The list of parameters for the model, i.e., Hurst exponent and constant.
    N : array
        The list of valid lags.
    y_values : array
        The list of actual structure function values.

    Returns
    -------
    residuals : array
        The list of residuals between the actual and predicted structure function values.
    """
    H, c = params
    return y_values - (c * np.power(N, H))