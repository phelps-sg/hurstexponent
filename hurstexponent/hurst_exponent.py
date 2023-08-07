import math
import numpy as np
import pandas as pd
from sys import float_info
from hurst import compute_Hc
from matplotlib import pyplot as plt
from scipy.optimize import least_squares
from typing import List, Optional, Tuple
from arch.bootstrap import StationaryBootstrap
from util.utils import std_of_sums, structure_function, interpret_hurst, generate_series, residual

from scipy.optimize import minimize
from scipy.stats import norm

# def neg_log_likelihood(params, N, y_values, function):
#     y_pred = function(N, *params)
#     residuals = y_values - y_pred
#     return -np.sum(norm.logpdf(residuals, loc=0, scale=np.std(residuals)))

def neg_log_likelihood(params, x_values, y_values, func):
    y_pred = func(x_values, *params)
    residuals = y_values - y_pred
    std_dev_residuals = np.std(residuals)
    loglikelihoods = norm.logpdf(residuals, loc=0, scale=std_dev_residuals)
    loglikelihood = -np.sum(loglikelihoods)  # sum over all data points

    return loglikelihood


class HurstEstimator:
    """
    This class is used to estimate the Hurst exponent of a time series. The Hurst exponent is a statistical measure
    used to classify time series particularly in the field of finance, hydrology, etc. It helps in identifying whether
    a time series is a random walk (H=0.5), or has some memory (H>0.5 or H<0.5).

    Attributes
    ----------
    ts : np.array
        A pre-processed time series

    Methods
    -------
    standard_hurst():
        Estimates the Hurst exponent using standard Hurst method.
    generalized_hurst(moment: int, max_lag: int = 19):
        Estimates the Hurst exponent using generalized Hurst method.
    rescaled_range(kind: str = 'random_walk'):
        Estimates the Hurst exponent using rescaled range method.
    hurst_from_alpha(alpha: float):
        Estimates the Hurst exponent using the given alpha value.
    confidence_interval(moment: int = 1, method: str = 'bootstrap', hurst_approach: str = 'generalized_hurst',
                        block_size: int = 10, n_iterations: int = 1000, hurst_params: dict = {}):
        Estimates Hurst exponent and calculates its confidence interval.
    estimate(method: str = 'hurst', **kwargs):
        Estimates the Hurst exponent using the given method.
    """

    def __init__(self, ts):
        """
        Initializes the HurstEstimator object with the provided time series data.

        Parameters
        ----------
        ts : list or np.array
            Time series data
        """

        # Check if time series is provided
        if ts is None:
            raise ValueError("Time series can't be None")

        # Convert to numpy array and handle zeros
        self.ts = np.where(np.array(ts, dtype=float) == 0, 1e-10, ts)

        # Check for invalid values in the time series
        if np.any(np.isnan(self.ts)) or np.any(np.isinf(self.ts)):
            raise ValueError("Time series contains NaN or Inf values")


    def standard_hurst(self, fitting_method: str = 'mle', min_lag: int = 10, max_lag: int = 1000) -> Tuple[float, float, List[float]]:
        """
        Calculate the Hurst exponent of a time series from the standard deviation of sums of N successive events using
        the specified fitting method.

        fitting_method : str, optional
            The method to use to estimate the Hurst exponent. Options include:
            - 'log_log': Log-log regression fitting method
            - 'least_squares': Direct fitting using Least-squares
            - 'mle': Maximum Likelihood Estimation (MLE)
            Default is 'log_log'.
        max_lag : int, optional
            The maximum lag to use in the calculation. Default is 1000.

        :return: A tuple containing:
                 - H: The Hurst exponent
                 - c: A constant
                 - [list(lag_sizes), y_values]: A list containing lag sizes and corresponding y_values

        :raises ValueError: If an invalid fitting_method is provided.
        """

        # Hurst exponent
        def _hurst_function(N, H, c):
            return c * N ** H

        # Ensure the fitting_method is valid
        if fitting_method not in ['mle', 'least_squares', 'log_log']:
            raise ValueError(f"Unknown method: {fitting_method}. Expected 'mle' or 'least_squares' 'log_log'.")

        # Check if the time series is zero-mean
        mean = np.mean(self.ts)
        series = self.ts
        if mean != 0:
            # If not, subtract the mean from each observation to make it zero-mean
            series = np.diff(self.ts)

        # Lag sizes
        min_lag = min_lag
        max_lag = min(max_lag, len(self.ts))

        #Compute the lag sizes based on the fitting method
        # if fitting_method == 'log_log':
        #     lag_sizes = [int(10 ** x) for x in np.arange(math.log10(min_lag), math.log10(max_lag), 0.25)]
        #     lag_sizes.append(len(series))
        # else:
        #     lag_sizes = range(min_lag, max_lag)
        lag_sizes = range(min_lag, max_lag)

        # Compute the valid lags and corresponding values
        valid_lags_and_values = [(lag_size, std_of_sums(series, lag_size)) for lag_size in lag_sizes
                                 if std_of_sums(series, lag_size) is not None and
                                 std_of_sums(series, lag_size) != 0]
        valid_lags, y_values = zip(*valid_lags_and_values)
        if not valid_lags or not y_values:
            return np.nan, np.nan, [[], []]

        # Perform fitting based on the selected method
        if fitting_method == 'mle':
            initial_guess = [np.mean(y_values), 0]  # Guess for both H_q and c
            result = minimize(neg_log_likelihood, initial_guess,
                              args=(np.array(valid_lags), np.array(y_values), _hurst_function))
            H, c = result.x
        elif fitting_method == 'least_squares':
            _residuals = lambda params, N, y_values: y_values - _hurst_function(N, params[0], params[1])
            initial_guess = [np.mean(y_values), 0]
            result = least_squares(_residuals, initial_guess, args=(valid_lags, y_values))  # result = least_squares(residual
            H, c = result.x
        else:  # Log-log regression fitting method
            log_valid_lags = np.log(valid_lags)
            log_y_values = np.log(y_values)
            A = np.vstack([log_valid_lags, np.ones(len(log_valid_lags))]).T
            H, log_c = np.linalg.lstsq(A, log_y_values, rcond=None)[0]
            c = np.exp(log_c)

        return H, c, [list(valid_lags), y_values]


    def generalized_hurst(self, moment: int = 1, fitting_method: str = 'mle', min_lag: int = 10, max_lag: int = 1000) -> Tuple[
        float, float, List[List[float]]]:
        """
        Estimate the generalized Hurst exponent of a time series using the specified method.

        Parameters
        ----------
        moment : int
            The moment to use in the calculation. Defaults to 1st moment, the mean.
        fitting_method : str, optional
            The method to use to estimate the Hurst exponent. Options include:
            - 'log_log': Log-log regression fitting method
            - 'least_squares': Direct fitting using Least-squares
            - 'mle': Maximum Likelihood Estimation (MLE)
            Default is 'log_log'.
        max_lag : int, optional
            The maximum lag to use in the calculation. Default is 1000.

        Returns
        -------
        Tuple[float, float, List[List[float]]]
            The estimated Hurst exponent, the constant c, and the list of valid lags and their corresponding S_q_tau values.

        Raises
        ------
        ValueError
            If an invalid fitting_method is provided.
        """

        series = self.ts

        # Generalised Hurst
        def _generalized_function(lag, H_q, c):
            return c * (lag ** H_q)

        # Ensure the fitting_method is valid
        if fitting_method not in ['mle', 'least_squares', 'log_log']:
            raise ValueError(f"Unknown method: {fitting_method}. Expected 'mle' or 'least_squares' 'log_log'.")

        min_lag = min_lag
        max_lag = min(max_lag, len(series))

        # Compute the S_q_tau values and valid lags
        S_q_tau_values, valid_lags = zip(
            *[(structure_function(series, moment, lag), lag) for lag in range(min_lag, max_lag + 1)
              if np.isfinite(structure_function(series, moment, lag))])
        if not valid_lags or not S_q_tau_values:
            return np.nan, np.nan, [[], []]

        # Perform fitting based on the selected method
        if fitting_method == 'mle':
            initial_guess = [np.mean(S_q_tau_values), 0]  # Guess for both H_q and c np.mean(S_q_tau_values)
            result = minimize(neg_log_likelihood, initial_guess,
                              args=(np.array(valid_lags), np.array(S_q_tau_values), _generalized_function))
            H_q, c = result.x
        elif fitting_method == 'least_squares':
            _residuals = lambda params, lag, S_q_tau_values: S_q_tau_values - _generalized_function(lag, params[0], params[1])
            initial_guess = [np.mean(S_q_tau_values), 0]
            result = least_squares(_residuals, initial_guess, args=(valid_lags, S_q_tau_values))
            H_q, c = result.x
        else:  # 'log_log'
            # Log-log regression fitting method
            log_tau = np.log(valid_lags)
            log_S_q_tau = np.log(S_q_tau_values)
            A = np.vstack([log_tau, np.ones(len(log_tau))]).T
            H_q, log_c = np.linalg.lstsq(A, log_S_q_tau / moment, rcond=None)[0]
            c = np.exp(log_c)

        return H_q, c, [valid_lags, S_q_tau_values]

    def rescaled_range(self, kind: str = 'random_walk') -> Tuple[float, float, List[float]]:
        """Computes the rescaled range of the time series.

        Parameters
        ----------
        kind: str, optional
            Kind of series for the calculation. Default is 'random_walk'.

        Returns
        -------
        Tuple[float, float, List[float]]
            Tuple containing the Hurst exponent, constant 'c', and data.
        """
        series = self.ts

        return compute_Hc(series, kind=kind, simplified=False)


    @staticmethod
    def hurst_from_alpha(alpha: float) -> Tuple[float, Optional[float]]:
        """Estimates the Hurst exponent using alpha.

        Parameters
        ----------
        alpha: float
            The alpha value.

        Returns
        -------
        Tuple[float, Optional[float]]
            Tuple containing the Hurst exponent and None.
        """
        return 1 - alpha / 2, None


    def confidence_interval(self, moment: int = 1, hurst_approach: str = 'standard_hurst',
                            block_size: int = 100, n_iterations: int = 10000, hurst_params: dict = {}, **kwargs) -> \
    Tuple[
        float, float, float, float, str]:
        """
        Estimate Hurst exponent and calculate its confidence interval.

        Parameters
        ----------
        ts: array_like
            Time series data
        moment : int
            The moment for which the structure function is to be calculated. Defaults to 1st moment, the mean.
        hurst_approach: str
            Hurst method to use for Hurst exponent estimation. Options are 'rescaled_range', 'hurst_from_alpha', 'standard_hurst', 'generalized_hurst'.
        block_size: int
            Block size to use for bootstrapping.
        n_iterations: int
            Number of iterations for bootstrapping.
        hurst_params: dict
            Dictionary of parameters for the chosen Hurst method.

        Returns
        -------
        Hurst exponent, lower and upper confidence intervals, standard deviation and its interpretation as a tuple.
        """

        hurst_values = []

        bs = StationaryBootstrap(block_size, self.ts)
        for data in bs.bootstrap(n_iterations):
            ts_resample = data[0][0]
            hurst_estimator = HurstEstimator(ts_resample)
            if hurst_approach == 'generalized_hurst':
                H, _, _, _ = hurst_estimator.estimate(hurst_approach, moment=moment, **hurst_params, **kwargs)
            else:
                H, _, _, _ = hurst_estimator.estimate(hurst_approach, **hurst_params, **kwargs)
            hurst_values.append(H)

        confidence_lower = np.percentile(hurst_values, 2.5)
        confidence_upper = np.percentile(hurst_values, 97.5)
        H = np.mean(hurst_values)
        std_dev = np.std([confidence_lower, confidence_upper])  # Compute the standard deviation

        return H, confidence_lower, confidence_upper, std_dev, interpret_hurst(H)

    def estimate(self, method: str = 'standard_hurst', **kwargs) -> Tuple[float, float, pd.DataFrame, str]:
        if method not in ['rescaled_range', 'hurst_from_alpha', 'standard_hurst', 'generalized_hurst']:
            raise ValueError(f"Unknown method: {method}")

        const = None
        data = None
        if method == 'rescaled_range':
            H, const, data = self.rescaled_range(**kwargs)
        elif method == 'standard_hurst':
            H, const, data = self.standard_hurst(**kwargs)
        elif method == 'generalized_hurst':
            H, const, data = self.generalized_hurst(**kwargs)
        elif method == 'hurst_from_alpha':
            H, const = HurstEstimator.hurst_from_alpha(**kwargs)
            data = None

        interpretation = interpret_hurst(H)
        return H, const, data, interpretation


if __name__ == '__main__':

    random_changes = generate_series(length=99999, noise_pct_std=0.02) # avg. daily market volatility

    # Create an instance of HurstEstimator
    hurst_estimator = HurstEstimator(random_changes)

    # Hurst
    H, D, data, interpretation = hurst_estimator.estimate('standard_hurst')
    print(f"Hurst Estimate via Standard Hurst: {H}, D constant: {D if D is not None else 'N/A'}, ({interpretation})")

    # Generalized Hurst
    moment = 1
    H, c, data, interpretation = hurst_estimator.estimate('generalized_hurst', moment=moment)
    print(f"Hurst Estimate via generalized_hurst: {H}, c constant: {c if c is not None else 'N/A'} ({interpretation})")

    # Rescaled Range
    H, c, data, interpretation = hurst_estimator.estimate('rescaled_range', kind='random_walk')
    print(f"Hurst Estimate via R/S: {H}, c constant: {c if c is not None else 'N/A'}, ({interpretation})")


    # # From Alpha
    # alpha = 0.65
    # H, _, _, interpretation = hurst_estimator.estimate('hurst_from_alpha', alpha=alpha)
    # print(f"Hurst Estimate from alpha: {H}, ({interpretation})")

    # # Confidence
    # print('\n')
    # print(f'Confidence interval for standard_hurst')
    # H, lower_ci, upper_ci, std_dev, interpretation = hurst_estimator.confidence_interval(hurst_approach='standard_hurst', n_iterations=1000)
    #
    # print(f'Hurst: {H}, Upper CI {upper_ci}, Lower CI: {lower_ci}, Standard dev: {std_dev}. {interpretation}')


    # Plotting

    # Plot series
    plt.figure(figsize=(10, 6))
    plt.plot(random_changes)
    plt.title('Raw Series')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.show()

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))  # Adjusted for 3 subplots

    # Standard Hurst
    H, D, data, interpretation = hurst_estimator.estimate('standard_hurst')
    lag_sizes, y_values = data
    axs[0].plot(lag_sizes, y_values, 'b.', label='Observed Values')
    axs[0].plot(lag_sizes, D * np.array(lag_sizes) ** H, "g--", label=f' Standard Hurst (H={H:.2f})')
    axs[0].loglog()
    axs[0].set_xlabel('Lag')
    axs[0].set_ylabel('Standard deviation of sums')
    axs[0].legend(frameon=False)
    axs[0].grid(False)

    # Generalized Hurst
    moment = 1
    H, c, data, _ = hurst_estimator.estimate('generalized_hurst', moment=moment)
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
    H, c, data, interpretation = hurst_estimator.estimate('rescaled_range', kind='random_walk')
    axs[2].plot(data[0], data[1], 'b.', label='(Observed Values)')
    axs[2].plot(data[0], c * data[0] ** H, 'g--', label=f'R/S Hurst (H={H:.2f})')
    axs[2].loglog()
    axs[2].set_xlabel('Lag')
    axs[2].set_ylabel('R/S ratio')
    axs[2].legend(frameon=False)
    axs[2].grid(False)

    plt.savefig('../plots/hurst.png', bbox_inches='tight')
    plt.show()



