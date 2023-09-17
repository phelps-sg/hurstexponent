import numpy as np
import pandas as pd
from typing import List, Tuple
from powerlaw_function import Fit
from matplotlib import pyplot as plt
from util.utils import hurst_exponent, std_of_sums, structure_function, interpret_hurst


def standard_hurst(series: np.array, fitting_method: str = 'MLE', min_lag: int = 10,
                   max_lag: int = 100) -> Tuple[float, float, List[float]]:
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

    if fitting_method not in ['MLE', 'Least_squares']:
        raise ValueError(f"Unknown method: {fitting_method}. Expected 'MLE' or 'Least_squares'.")

    # Check if the time series is stationary
    mean = np.mean(series)
    if not np.isclose(mean, 0.0):
        # Subtracting mean to center data around zero
        series = (series - mean)

        # Diff each observation making a series stationary
        series = np.diff(series)

    # Find valid lags and corresponding values
    min_lag = min_lag
    max_lag = min(max_lag, len(series))
    num_lags = int(np.sqrt(len(series)))
    lag_sizes = np.linspace(min_lag, max_lag, num=num_lags, dtype=int)

    # Compute standard deviation of sums
    y_values, valid_lags = zip(
        *[(std_of_sums(series, lag), lag) for lag in lag_sizes
          if np.isfinite(std_of_sums(series, lag))])
    if not valid_lags or not y_values:
        return np.nan, np.nan, [[], []]

    # Perform fitting based on the selected method and return fitted object
    xy_df = pd.DataFrame({
        'x_values': valid_lags,
        'y_values': y_values
    })

    if fitting_method == 'MLE':
        standard_hurst = Fit(xy_df, xmin_distance='BIC')
    else:
        standard_hurst = Fit(xy_df, nonlinear_fit_method=fitting_method, xmin_distance='BIC')

    # TODO: Remove below
    # Fit custom function
    # custom_powerlaw = {
    #     'generalized_hurst': hurst_exponent
    # }
    #
    # generalized_hurst.fit_powerlaw_function(custom_powerlaw)

    return standard_hurst


def generalized_hurst(series: np.array, moment: int = 1, fitting_method: str = 'MLE', min_lag: int = 10,
                      max_lag: int = 500) -> Tuple[float, float, List[List[float]]]:
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
    if fitting_method not in ['MLE', 'Least_squares']:
        raise ValueError(f"Unknown method: {fitting_method}. Expected 'MLE' or 'Least_squares'.")

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

    # Perform fitting based on the selected method and return fitted object
    xy_df = pd.DataFrame({
        'x_values': valid_lags,
        'y_values': S_q_tau_values
    })

    if fitting_method == 'MLE':
        generalized_hurst = Fit(xy_df, xmin_distance='BIC')
    else:
        generalized_hurst = Fit(xy_df, nonlinear_fit_method=fitting_method, xmin_distance='BIC')

    return generalized_hurst



if __name__ == '__main__':

    # Generate simple random walk series
    # from util.generate_series import simple_series
    # series = simple_series(length=99999, seed=70)

    # Genereate stochastic process with specific long-range properties
    from util.generate_series import stochastic_process
    series = stochastic_process(length=99999, proba=.50, cumprod=True, seed=50)

    # def acf(series: pd.Series, lags: int) -> List:
    #     """
    #     Returns a list of autocorrelation values for each of the lags from 0 to `lags`
    #     """
    #     acl_ = []
    #     for i in range(lags):
    #         ac = series.autocorr(lag=i)
    #         acl_.append(ac)
    #     return acl_
    #
    # # Load sample data – TSLA stock trade signs.
    # import os
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    # csv_path = os.path.join(current_dir, '..', 'datasets', 'stock_tsla.csv')
    # sample = pd.read_csv(csv_path, header=0, index_col=0)
    #
    # # Series generated from a function, in this example, autocorrelation function (ACF)
    # ACF_RANGE = 1001
    # series = acf(sample['trade_sign'], ACF_RANGE)[1:]
    # series = np.array(series)


    # Plot raw series
    plt.figure(figsize=(10, 6))
    plt.plot(series)
    plt.title('Raw Series')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.savefig('../plots/random_walk.png', bbox_inches='tight')
    plt.show()

    # Standard Hurst
    print('Standard Hurst Exponent')
    hurst = standard_hurst(series) # fitting_method='Least_squares'
    hurst.powerlaw.print_fitted_results()
    hurst.powerlaw.plot_fit()
    interpretation = interpret_hurst(hurst.powerlaw.params.alpha)
    print(f'Hurst Estimate via Standard deviation of sums: H = {hurst.powerlaw.params.alpha}, ({interpretation})')
    print('\n')

    # Generalized Hurst
    print('Generalized Hurst Exponent')
    generalized_hurst = generalized_hurst(series, max_lag=1000)
    generalized_hurst.powerlaw.print_fitted_results()
    generalized_hurst.powerlaw.plot_fit()
    interpretation = interpret_hurst(generalized_hurst.powerlaw.params.alpha)
    print(f"Hurst Estimate via Generalized Hurst: H = {generalized_hurst.powerlaw.params.alpha}, ({interpretation})")


