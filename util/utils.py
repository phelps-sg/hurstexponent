import numpy as np
from scipy.stats import norm
from typing import Callable, Iterable


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


def std_of_sums(ts: np.array, lag_size: int) -> float:
    """
    Computes the standard deviation of sums of time series lags of size chunk_size.

    Parameters
    ----------
    ts : np.array
        Time series data
    lag_size : int
        The size of each chunk of the time series

    Returns
    -------
    std : float
        The standard deviation of the sums
    """
    if lag_size == 0:
        return np.nan

    # Reshape the array to have a size of (-1, chunk_size) and sum along the second axis
    lags = len(ts) // lag_size
    sums = ts[:lags * lag_size].reshape(-1, lag_size).sum(axis=1)

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
    H = round(H, 2)

    if not 0 <= H <= 1:
        return "Hurst Exponent not in a valid range [0, 1].  Series may not be a long memory process"
    if H == 0.5:
        return "Perfect diffusivity: series is a geometric or Brownian random walk"
    if H < 0.5:
        return "Sub-diffusive: series demonstrates anti-persistent behaviour"
    if H > 0.5:
        return "Super-diffusive: series demonstrates persistent long-range dependence"
    return "Invalid Hurst Exponent"
