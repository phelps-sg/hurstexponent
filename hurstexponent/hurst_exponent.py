import numpy as np
import pandas as pd
import statsmodels.api as sm
from hurst import compute_Hc
from matplotlib import pyplot as plt
from scipy.optimize import least_squares
from typing import List, Optional, Tuple


class HurstEstimator:
    """
    A class used to estimate the Hurst exponent of a time series.

    Attributes
    ----------
    ts : np.array
        A pre-processed time series

    Methods
    -------
    _std_of_sums(chunk_size):
        Calculates the standard deviation of sums of time series chunks of size chunk_size.
    generalized_hurst(fitting_method: str ='direct_fit') -> Tuple[float, float, List[float]]:
        Computes the Hurst exponent of the time series using the given fitting method.
    hurst_from_alpha(alpha: float) -> Tuple[float, Optional[float]]:
        Estimates the Hurst exponent using alpha.
    rescaled_range(kind: str = 'random_walk') -> Tuple[float, float, List[float]]:
        Computes the rescaled range of the time series.
    estimate(method: str = 'generalized_hurst', **kwargs) -> Tuple[float, float, pd.DataFrame, str]:
        Estimates the Hurst exponent using the given method.
    """

    def __init__(self, ts):
        """
        Constructs all the necessary attributes for the HurstEstimator object.

        Parameters
        ----------
            ts : list or np.array
                Time series data
        """

        # input check - cannot proceed without a time series
        if ts is None:
            raise ValueError("Time series can't be None")

        # Pre-processing checks
        ts = np.array(ts, dtype=float)

        # input check - NaNs or infs will cause problems later
        if np.any(np.isnan(ts)) or np.any(np.isinf(ts)):
            raise ValueError("Time series contains NaN or Inf values")

        # pre-processing step - zero values will cause problems with log-log regression
        ts = np.where(ts == 0, 1e-10, ts)

        # Store the time series
        self.ts = ts

    # Helper functions
    def _interpret_hurst(self, H: float) -> str:
        """ Interpretation of Hurst Exponent """
        if not 0 <= H <= 1:
            return "Hurst Exponent not in a valid range [0, 1], series may not be a long memory process"
        if H == 0.5:
            return "Perfect diffusivity: series is a geometric or Brownian random walk"
        if H < 0.5:
            return "Sub-diffusive: series demonstrates anti-persistent behavior"
        if H > 0.5:
            return "Super-diffusive: series demonstrates persistent long-range dependence"
        return "Invalid Hurst Exponent"

    def _std_of_sums(self, chunk_size):
        """
        Calculates the standard deviation of sums of time series chunks of size chunk_size.

        Parameters
        ----------
            chunk_size : int
                The size of each chunk of the time series

        Returns
        -------
            std : float
                The standard deviation of the sums
        """

        sums = []
        for i in range(0, len(self.ts), chunk_size):  # Iterate over the time series with a step size of chunk_size
            chunk = self.ts[i: i + chunk_size]  # Get the next chunk of size chunk_size
            if len(chunk) == chunk_size:  # If we have a full chunk of size chunk_size
                sums.append(np.sum(chunk))  # Sum up the chunk and add to the list
        return np.std(sums)  # Return the standard deviation of the sums

    # Generalized Hurst
    def generalized_hurst(self, fitting_method: str = 'direct_fit') -> Tuple[float, float, List[float]]:
        if fitting_method not in ['log_log', 'direct_fit']:
            raise ValueError(f"Unknown method: {fitting_method}. Expected 'log_log' or 'direct_fit'.")

        # Default behaviour - Hurst exponent over a range of chunk sizes, starting from 5 and going up to (
        # approximately) half the length of the time series, in steps of 5
        chunk_sizes = range(5, len(self.ts) // 2, 5)

        # Define Hurst function and residuals for optimization
        _hurst_function = lambda N, D, H: D * N ** H
        _residuals = lambda params, N, y_values: y_values - _hurst_function(N, *params)

        # Calculate y_values only for valid chunk sizes
        y_values = []
        valid_chunk_sizes = []
        for chunk_size in chunk_sizes:
            result = self._std_of_sums(chunk_size)
            if result is not None and result != 0:  # Or some other condition for valid results
                y_values.append(result)
                valid_chunk_sizes.append(chunk_size)

        # Only keep chunk sizes for which _std_of_sums gave a valid result
        chunk_sizes = valid_chunk_sizes

        if fitting_method == 'direct_fit':
            result = least_squares(_residuals, [0.5, 0.5], loss='soft_l1',
                                   args=(np.array(chunk_sizes), np.array(y_values)))
            D, H = result.x
        else:  # 'log_log'
            log_chunk_sizes = np.log(chunk_sizes)
            log_stds = np.log(y_values)
            X = sm.add_constant(log_chunk_sizes)
            model = sm.RLM(log_stds, X, M=sm.robust.norms.HuberT())
            results = model.fit()
            H = results.params[1]
            D = np.exp(results.params[0])

        # Return the chunk sizes and y_values along with the Hurst exponent and constant D
        return H, D, [list(chunk_sizes), y_values]

    # R/S Range
    def rescaled_range(self, kind: str = 'random_walk') -> Tuple[float, float, List[float]]:
        H, c, data = compute_Hc(self.ts, kind=kind)
        return H, c, data

    # From alpha
    @staticmethod
    def hurst_from_alpha(alpha: float) -> Tuple[float, Optional[float]]:
        H = 1 - alpha / 2
        return H, None

    def estimate(self, method: str = 'generalized_hurst', **kwargs) -> Tuple[float, float, pd.DataFrame, str]:
        if method not in ['rescaled_range', 'hurst_from_alpha', 'generalized_hurst']:
            raise ValueError(f"Unknown method: {method}")

        data = None
        if method == 'rescaled_range' or method == 'generalized_hurst':
            H, const, data = getattr(self, method)(**kwargs)
        else:  # if method == 'hurst_from_alpha'
            H, const = getattr(HurstEstimator, method)(**kwargs)

        interpretation = self._interpret_hurst(H)
        return H, const, data, interpretation


if __name__ == '__main__':
    np.random.seed(42)
    random_changes = 1. + np.random.randn(99999) / 1000.
    series = np.cumprod(random_changes)

    # Create an instance of HurstEstimator
    hurst_estimator = HurstEstimator(series)

    # Generalized Hurst
    H, D, data, interpretation = hurst_estimator.estimate('generalized_hurst', fitting_method='direct_fit')
    print(f"Hurst Estimate via Generalised Hurst: {H}, D constant: {D if D is not None else 'N/A'} ({interpretation})")

    # Rescaled Range
    H, c, data, interpretation = hurst_estimator.estimate('rescaled_range', kind='random_walk')
    print(f"Hurst Estimate via R/S: {H}, c constant: {c if c is not None else 'N/A'} ({interpretation})")

    # From Alpha
    alpha = 0.65
    H, _, _, interpretation = hurst_estimator.estimate('hurst_from_alpha', alpha=alpha)
    print(f"Hurst Estimate from alpha: {H} ({interpretation})")

    # Plotting
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    # Generalized Hurst
    H, D, data, interpretation = hurst_estimator.estimate('generalized_hurst', fitting_method='direct_fit')
    chunk_sizes, y_values = data
    axs[0].plot(chunk_sizes, y_values, 'b.', label='Observed Values')
    axs[0].plot(chunk_sizes, D * np.array(chunk_sizes) ** H, "g--", label=f'Generalized Hurst (H={H:.2f})')
    axs[0].loglog()
    axs[0].set_xlabel('Lag')
    axs[0].set_ylabel('Standard deviation of sums')
    axs[0].legend(frameon=False)
    axs[0].grid(False)

    # Rescaled Range
    H, c, data, interpretation = hurst_estimator.estimate('rescaled_range', kind='random_walk')
    axs[1].plot(data[0], data[1], 'b.', label='(Lag, R/S value)')
    axs[1].plot(data[0], c * data[0] ** H, 'g--', label=f'Hurst (H={H:.2f})')
    axs[1].loglog()
    axs[1].set_xlabel('Lag')
    axs[1].set_ylabel('R/S ratio')
    axs[1].legend(frameon=False)
    axs[1].grid(False)
    plt.savefig('../plots/hurst.png', bbox_inches='tight')

    plt.show()
