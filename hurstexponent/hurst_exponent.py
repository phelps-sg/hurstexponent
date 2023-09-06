import numpy as np
import pandas as pd
from scipy.optimize import minimize
from matplotlib import pyplot as plt
from scipy.optimize import least_squares
from typing import List, Tuple
from util.utils import std_of_sums, structure_function, interpret_hurst, neg_log_likelihood


def standard_hurst(series: np.array, fitting_method: str = 'mle', min_lag: int = 10,
                   max_lag: int = 1000) -> Tuple[float, float, List[float]]:
    """
    Estiamte the Hurst exponent of a time series from the standard deviation of sums of N successive events using
    the specified fitting method.

    Parameters
    ----------
    series: list or array-like series
            Represents time-series data
    fitting_method: str, optional
        The method to use to estimate the Hurst exponent. Options include:
        - 'OLS': Log-log OLS regression fitting method
        - 'least_squares': Direct fitting using Nonlinear Least-squares
        - 'mle': Maximum Likelihood Estimation (MLE)
        Default is 'OLS'.
    max_lag: int, optional
        The maximum consecutive lag (windows, bins, chunks) to use in the calculation of H. Default is 500.

    Returns
    -------
     A tuple containing:
             - H: The Hurst exponent
             - c: A constant
             - [list(lag_sizes), y_values]: A list containing lag sizes and corresponding y_values
    interpretation: str
        Interpretation of Hurst Exponent.

    Raises
    ------
    ValueError
        If an invalid fitting_method is provided.
    """

    series = series
    if len(series) < 100:
        raise ValueError("Length of series cannot be less than 100")

    # Convert to series to array-like if series is not numpy array or pandas Series and handle zeros
    if not isinstance(series, (np.ndarray, pd.Series)):
        series = np.array(series, dtype=float)
    replace_zero = 1e-10
    series[series == 0] = replace_zero


    # Check for invalid values in the time series
    if np.any(np.isnan(series)) or np.any(np.isinf(series)):
        raise ValueError("Time series contains NaN or Inf values")

    # Ensure the fitting_method is valid
    if fitting_method not in ['mle', 'least_squares', 'OLS']:
        raise ValueError(f"Unknown method: {fitting_method}. Expected 'mle' or 'least_squares' 'OLS'.")

    # Check if the time series is stationary
    mean = np.mean(series)
    if mean != 0:
        # Convert noise like time series to random walk like time series
        series = (series - mean)  # If not, subtract the mean to center the data around zero
        series = np.diff(series)  # Diff each observation making a series stationary

    # Hurst exponent
    def _hurst_function(N, H, c):
        return c * N ** H

    # Compute the valid lags and corresponding values
    min_lag = min_lag
    max_lag = min(max_lag, len(series))
    lag_sizes = range(min_lag, max_lag)

    valid_lags_and_values = [(lag_size, std_of_sums(series, lag_size)) for lag_size in lag_sizes
                             if std_of_sums(series, lag_size) is not None and
                             std_of_sums(series, lag_size) != 0]
    valid_lags, y_values = zip(*valid_lags_and_values)
    if not valid_lags or not y_values:
        return np.nan, np.nan, [[], []]

    # Perform fitting based on the selected method
    if fitting_method == 'mle':
        bounds = [(0, 1), (-np.inf, np.inf)]
        initial_guess = [0.5, 0]  # Guess for both H and c
        result = minimize(neg_log_likelihood, initial_guess,
                          args=(np.array(valid_lags), np.array(y_values), _hurst_function), bounds = bounds)
        H, c = result.x
    elif fitting_method == 'least_squares':
        lower_bounds = [0, -np.inf]
        upper_bounds = [1, np.inf]
        bounds = (lower_bounds, upper_bounds)
        initial_guess = [0.5, 0]
        _residuals = lambda params, N, y_values: y_values - _hurst_function(N, params[0], params[1])
        result = least_squares(_residuals, initial_guess,args=(valid_lags, y_values), bounds = bounds)
        H, c = result.x
    else:  # Log-log OLS regression fitting method
        log_valid_lags = np.log(valid_lags)
        log_y_values = np.log(y_values)
        A = np.vstack([log_valid_lags, np.ones(len(log_valid_lags))]).T
        H, log_c = np.linalg.lstsq(A, log_y_values, rcond=None)[0]
        c = np.exp(log_c)

    interpretation = interpret_hurst(H)

    return H, c, [list(valid_lags), y_values], interpretation


def generalized_hurst(series: np.array, moment: int = 1, fitting_method: str = 'mle', min_lag: int = 10,
                      max_lag: int = 1000) -> Tuple[float, float, List[List[float]]]:
    """
    Estimate the generalized Hurst exponent of a time series using the specified method.

    Parameters
    ----------
    series : list or array-like series
            Represents time-series data
    moment : int
        The moment to use in the calculation. Defaults to 1st moment, the mean.
    fitting_method : str, optional
        The method to use to estimate the Hurst exponent. Options include:
        - 'OLS': Log-log OLS regression fitting method
        - 'least_squares': Direct fitting using Nonlinear Least-squares
        - 'mle': Maximum Likelihood Estimation (MLE)
        Default is 'OLS'.
    max_lag : int, optional
        The maximum consecutive lag (windows, bins, chunks) to use in the calculation of H. Default is 500.

    Returns
    -------
    Tuple[float, float, List[List[float]]]
        The estimated Hurst exponent, the constant c, and the list of valid lags and their corresponding S_q_tau values.

    Raises
    ------
    ValueError
        If an invalid fitting_method is provided.
    """

    series = series
    if len(series) < 100:
        raise ValueError("Length of series cannot be less than 100")

    # Convert to series to array-like if series is not numpy array or pandas Series and handle zeros
    if not isinstance(series, (np.ndarray, pd.Series)):
        series = np.array(series, dtype=float)
    replace_zero = 1e-10
    series[series == 0] = replace_zero

    # Check for invalid values in the time series
    if np.any(np.isnan(series)) or np.any(np.isinf(series)):
        raise ValueError("Time series contains NaN or Inf values")

    # Ensure the fitting_method is valid
    if fitting_method not in ['mle', 'least_squares', 'OLS']:
        raise ValueError(f"Unknown method: {fitting_method}. Expected 'mle' or 'least_squares' 'OLS'.")

    # Subtract the mean to center the data around zero
    mean = np.mean(series)
    if mean != 0:
        series = (series - mean)

    # Generalised Hurst
    def _generalized_function(lag, H_q, c):
        return c * (lag ** H_q)

    # Compute the S_q_tau values and valid lags
    min_lag = min_lag
    max_lag = min(max_lag, len(series))
    lag_sizes = range(min_lag, max_lag)

    S_q_tau_values, valid_lags = zip(
        *[(structure_function(series, moment, lag), lag) for lag in lag_sizes
          if np.isfinite(structure_function(series, moment, lag))])
    if not valid_lags or not S_q_tau_values:
        return np.nan, np.nan, [[], []]

    # Perform fitting based on the selected method
    if fitting_method == 'mle':
        initial_guess = [0.5, 0]
        bounds = [(0, 1), (-np.inf, np.inf)]
        result = minimize(neg_log_likelihood, initial_guess,
                          args=(np.array(valid_lags), np.array(S_q_tau_values), _generalized_function), bounds = bounds)
        H_q, c = result.x
    elif fitting_method == 'least_squares':
        initial_guess = [0.5, 0]
        lower_bounds = [0, -np.inf]
        upper_bounds = [1, np.inf]
        bounds = (lower_bounds, upper_bounds)
        _residuals = lambda params, lag, S_q_tau_values: S_q_tau_values - _generalized_function(lag, params[0], params[1])
        result = least_squares(_residuals, initial_guess, args=(valid_lags, S_q_tau_values), bounds = bounds)
        H_q, c = result.x
    else:  # Log-log OLS regression fitting method
        # Log-log regression fitting method
        log_tau = np.log(valid_lags)
        log_S_q_tau = np.log(S_q_tau_values)
        A = np.vstack([log_tau, np.ones(len(log_tau))]).T
        H_q, log_c = np.linalg.lstsq(A, log_S_q_tau / moment, rcond=None)[0]
        c = np.exp(log_c)

    interpretation = interpret_hurst(H_q)

    return H_q, c, [valid_lags, S_q_tau_values], interpretation


if __name__ == '__main__':

    # Generate simple random walk series
    from util.generate_series import simple_series
    series = simple_series(length=99999, noise_pct_std=0.02, seed=70) # avg. daily market volatility

    # Genereate stochastic process with specific long-range properties
    # from util.generate_series import stochastic_process
    # series = stochastic_process(length=99999, proba=.5, cumprod=False, seed=40)

    # Hurst
    H, D, data, interpretation = standard_hurst(series)
    print(f"Hurst Estimate via Standard Hurst: {H}, D constant: {D if D is not None else 'N/A'}, ({interpretation})")

    # Generalized Hurst
    H, c, data, interpretation = generalized_hurst(series)
    print(f"Hurst Estimate via Generalized Hurst: {H}, D constant: {D if D is not None else 'N/A'}, ({interpretation})")

    # Hurst from Rescaled Range
    from hurst import compute_Hc
    H, c, data = compute_Hc(series)
    print(f"Hurst Estimate via R/S: {H}, c constant: {c if c is not None else 'N/A'}, ({interpret_hurst(H)})")


    # Plot raw series
    plt.figure(figsize=(10, 6))
    plt.plot(series)
    plt.title('Raw Series')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.savefig('../plots/random_walk.png', bbox_inches='tight')
    plt.show()

    # Plotting Hurst
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    # Standard Hurst
    H, c, data, _ = standard_hurst(series)
    lag_sizes, y_values = data
    axs[0].plot(lag_sizes, y_values, 'b.', label='Observed Values')
    axs[0].plot(lag_sizes, D * np.array(lag_sizes) ** H, "g--", label=f' Standard Hurst (H={H:.2f})')
    axs[0].loglog()
    axs[0].set_xlabel('Lag')
    axs[0].set_ylabel('Standard deviation of sums')
    axs[0].legend(frameon=False)
    axs[0].grid(False)

    # Generalized Hurst
    H, c, data, _ = generalized_hurst(series)
    tau, S_q_tau = data  # change to raw values instead of logarithmic
    log_tau = np.log10(tau)  # calculate log_tau
    c = np.mean(np.log10(S_q_tau)) - H * np.mean(log_tau)  # calculate constant with log_S_q_tau
    axs[1].plot(tau, S_q_tau, 'b.', label='Observed Values')  # plot with raw values
    axs[1].plot(tau, 10 ** (c + np.array(log_tau) * H), 'g--',
                label=f'Generalized Hurst (H={H:.2f})')  # plot using line equation with base 10 exponent
    axs[1].loglog()
    axs[1].set_xlabel('Lag')
    axs[1].set_ylabel('Structure Function')
    axs[1].legend(frameon=False)
    axs[1].grid(False)

    # Rescaled Range
    H, c, data = compute_Hc(series)
    axs[2].plot(data[0], data[1], 'b.', label='(Observed Values)')
    axs[2].plot(data[0], c * data[0] ** H, 'g--', label=f'R/S Hurst (H={H:.2f})')
    axs[2].loglog()
    axs[2].set_xlabel('Lag')
    axs[2].set_ylabel('R/S ratio')
    axs[2].legend(frameon=False)
    axs[2].grid(False)

    plt.savefig('../plots/hurst.png', bbox_inches='tight')
    plt.show()

    # Estimate the Hurst exponent using the standard method and the bootstrap technique
    from arch.bootstrap import MovingBlockBootstrap, StationaryBootstrap
    print('\n')
    print('Confidence:')

    # Create a bootstrap object with block size, e.g., 100
    bs = MovingBlockBootstrap(1000, series)

    # Apply the function to the bootstrap object
    results = bs.apply(lambda s: standard_hurst(s)[0], reps=1000)

    mean_hurst = np.mean(results)
    std_dev_hurst = np.std(results)

    # Standard 1.96 multiplier for a 95% confidence interval under the assumption of a normality
    lower_ci = mean_hurst - 1.96 * std_dev_hurst
    upper_ci = mean_hurst + 1.96 * std_dev_hurst

    print("Bootstrap Mean:", mean_hurst)
    print("Bootstrap Standard Deviation:", std_dev_hurst)
    print("95% Confidence Interval:", (lower_ci, upper_ci))
