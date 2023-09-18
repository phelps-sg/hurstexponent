import numpy as np


# Helper functions


# def std_of_sums(ts: np.array, chunk_size: int) -> float:
#     """
#     Computes the standard deviation of sums of time series chunks of size chunk_size.
#     std : float
#         The standard deviation of the sums
#     """
#     sums = []
#     for i in range(0, len(ts), chunk_size):  # Iterate over the time series with a step size of chunk_size
#         chunk = ts[i: i + chunk_size]  # Get the next chunk of size chunk_size
#         if len(chunk) == chunk_size:  # If we have a full chunk of size chunk_size
#             sums.append(np.sum(chunk))  # Sum up the chunk and add to the list
#     return np.std(sums)
def get_sums_of_chunks(series: np.array, N: int) -> np.array:
    """
    Reshapes a series into chunks of size N and sums each chunk.

    Parameters
    ----------
    series : np.array
        The time series to process
    N : int
        Chunk size

    Returns
    -------
    np.array
        Summed values of each chunk
    """
    reshaped_series = series[: len(series) // N * N].reshape(-1, N)
    return np.sum(reshaped_series, axis=1)


def std_of_sums(ts: np.array, lag_size: int) -> float:
    """
    Computes the standard deviation of sums of time series lags of size lag_size.

    .. math::

        TODO: define standard deviation of sums

    Parameters
    ----------
    ts : np.array
        Time series data
    lag_size : int
        The size of each lag of the time series

    Returns
    -------
    std : float
        The standard deviation of the sums
    """
    if lag_size == 0:
        return np.nan

    sums = get_sums_of_chunks(ts, lag_size)
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
    to the power of the specified moment.

    .. math::

        S_q(lag) = < | x(t + lag) - x(t) |^q >_t \sim lag^{qH(q)}

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
        return np.mean(diffs**moment)
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
    if not 0 <= H <= 1:
        return "Hurst Exponent not in a valid range [0, 1].  Series may not be a long memory process"
    if np.isclose(H, 0.5):
        return "Perfect diffusivity: series is a Geometric or Brownian random walk"
    if H < 0.5:
        return "Sub-diffusive: series demonstrates anti-persistent behaviour"
    if H > 0.5:
        return "Super-diffusive: series demonstrates persistent long-range dependence"
    return "Invalid Hurst Exponent"
