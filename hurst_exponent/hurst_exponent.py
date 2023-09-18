import numpy as np
import pandas as pd
from powerlaw_function import Fit
from matplotlib import pyplot as plt
from typing import List, Tuple, Callable, Any
from stochastic.processes.continuous import FractionalBrownianMotion, GeometricBrownianMotion

from util.utils import (
    std_of_sums,
    structure_function,
    interpret_hurst,
)


def standard_hurst(
    series: np.array, fitting_method: str = "MLE", min_lag: int = 10, max_lag: int = 100
) -> Tuple[float, float, List[float]]:
    """
    Estiamte the Hurst exponent of a time series from the standard deviation of sums of N successive events using
    the specified fitting method.

    Parameters
    ----------
    series: list or array-like series
            Represents time-series data
    fitting_method: str, optional
        The method to use to estimate the Hurst exponent. Options include:
        - 'Least_squares': Direct fitting using Nonlinear Least-squares
        - 'MLE': Maximum Likelihood Estimation (MLE)
        Default is 'MLE'.
    max_lag: int, optional
        The maximum consecutive lag (windows, bins, chunks) to use in the calculation of H. Default is 100. Fit process
        is highly sensitive to this hyperparameter – currently set with respect to series length and heuristically given
        problem domain.

    Returns
    -------
     fit_results object containing:
            - Params:
             - H: The Hurst exponent
             - c: A constant

            - FitResult: Represents the result of a fitting procedure.

    Raises
    ------
    ValueError
        If an invalid fitting_method is provided.
    """

    # Data checks
    series = series
    if len(series) < 100:
        raise ValueError("Length of series cannot be less than 100")

    if not isinstance(series, (np.ndarray, pd.Series)):
        series = np.array(series, dtype=float)
    replace_zero = 1e-10
    series[series == 0] = replace_zero

    if np.any(np.isnan(series)) or np.any(np.isinf(series)):
        raise ValueError("Time series contains NaN or Inf values")

    if fitting_method not in ["MLE", "Least_squares"]:
        raise ValueError(f"Unknown method: {fitting_method}. Expected 'MLE' or 'Least_squares'.")

    # Check if the time series is stationary
    mean = np.mean(series)
    if not np.isclose(mean, 0.0):
        series = series - mean  # Subtracting mean to center data around zero
        series = np.diff(series)

    # Find valid lags and corresponding values
    min_lag = min_lag
    max_lag = min(max_lag, len(series))
    num_lags = int(np.sqrt(len(series)))
    lag_sizes = np.linspace(min_lag, max_lag, num=num_lags, dtype=int)

    # Compute standard deviation of sums
    y_values, valid_lags = zip(
        *[(std_of_sums(series, lag), lag) for lag in lag_sizes if np.isfinite(std_of_sums(series, lag))]
    )
    if not valid_lags or not y_values:
        return np.nan, np.nan, [[], []]

    # Perform fitting based on the selected method and return fitted object
    xy_df = pd.DataFrame({"x_values": valid_lags, "y_values": y_values})

    if fitting_method == "MLE":
        fit_results = Fit(xy_df, xmin_distance="BIC")
    else:
        fit_results = Fit(xy_df, nonlinear_fit_method=fitting_method, xmin_distance="BIC")

    H = fit_results.powerlaw.params.alpha

    return H, fit_results


def generalized_hurst(
    series: np.array, moment: int = 1, fitting_method: str = "MLE", min_lag: int = 10, max_lag: int = 500
) -> Tuple[float, float, List[List[float]]]:
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
        - 'Least_squares': Direct fitting using Nonlinear Least-squares
        - 'MLE': Maximum Likelihood Estimation (MLE)
        Default is 'MLE'.
    max_lag : int, optional
        The maximum consecutive lag (windows, bins, chunks) to use in the calculation of H. Default is 500. Hyperparameter
        greatly influence the result of the fit process – currently set with respect to series length and heuristically given
        problem domain.

    Returns
    -------
     standard_hurst object containing:
            - Params:
             - H: The Hurst exponent
             - c: A constant

            - FitResult: Represents the result of a fitting procedure.

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
    if fitting_method not in ["MLE", "Least_squares"]:
        raise ValueError(f"Unknown method: {fitting_method}. Expected 'MLE' or 'Least_squares'.")

    # Subtract the mean to center the data around zero
    mean = np.mean(series)
    if mean != 0:
        series = series - mean

    # Compute the S_q_tau values and valid lags
    min_lag = min_lag
    max_lag = min(max_lag, len(series))
    # lag_sizes = range(min_lag, max_lag)
    num_lags = int(np.sqrt(len(series)))
    lag_sizes = np.linspace(min_lag, max_lag, num=num_lags, dtype=int)

    # Compute structure function
    S_q_tau_values = []
    valid_lags = []
    for lag in lag_sizes:
        S_q_tau = structure_function(series, moment, lag)
        if np.isfinite(S_q_tau):
            S_q_tau_values.append(S_q_tau)
            valid_lags.append(lag)
    if not valid_lags or not S_q_tau_values:
        return np.nan, np.nan, [[], []]

    # Fit
    xy_df = pd.DataFrame({"x_values": valid_lags, "y_values": S_q_tau_values})

    if fitting_method == "MLE":
        fit_results = Fit(xy_df, xmin_distance="BIC")
    else:
        fit_results = Fit(xy_df, nonlinear_fit_method=fitting_method, xmin_distance="BIC")

    H = fit_results.powerlaw.params.alpha

    return H, fit_results


if __name__ == "__main__":
    # Fractal BM
    fbm = FractionalBrownianMotion(hurst=0.5, t=1)
    fbm_series = fbm.sample(10000)
    lags = fbm.times(10000)
    plt.plot(lags, fbm_series)
    plt.show()

    # GBM
    gbm = GeometricBrownianMotion(drift=2, t=1)
    gbm_series = fbm.sample(10000)
    lags = fbm.times(10000)
    plt.plot(lags, fbm_series)
    plt.show()

    # Standard Hurst
    print("Standard Hurst Exponent")
    H, fit_results = standard_hurst(gbm_series)  # fitting_method='Least_squares'
    fit_results.powerlaw.print_fitted_results()
    fit_results.powerlaw.plot_fit()
    print(f"Hurst Estimate via Standard deviation of sums: H = {H}, ({interpret_hurst(H)})")
    print("\n")

    # Generalized Hurst
    print("Generalized Hurst Exponent")
    H, fit_results = generalized_hurst(gbm_series, max_lag=100)
    fit_results.powerlaw.print_fitted_results()
    fit_results.powerlaw.plot_fit()
    print(f"Hurst Estimate via Standard deviation of sums: H = {H}, ({interpret_hurst(H)})")

    # Boostrap
    def bootstrap(
        estimator: Callable[[Any], Tuple[float, Any]],  # Adjusted the expected return type of the estimator
        reps: int,
        seed: int,
    ) -> np.array:
        np.random.seed(seed)
        return np.array([estimator(fbm.sample(1000))[0] for _repetition in range(reps)])  # Extract the Hurst exponent

    results = bootstrap(standard_hurst, reps=1000, seed=42)

    lower_ci = np.percentile(results, 5)
    upper_ci = np.percentile(results, 95)

    print(pd.DataFrame(results, columns=["^H"]).describe())
    print()
    print("95% Confidence Interval:", (lower_ci, upper_ci))
