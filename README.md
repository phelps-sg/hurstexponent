# Hurst Estimator

Estimate the Hurst exponent, a statistical measure of the long-term memory of a stochastic process using robust statistical methods. Our package provides methods to compute the Hurst exponent using both the standard deviation of sums and a generalized method through the structure function. 

## Features:

  - Support for both standard and generalized Hurst exponent calculations.
  - Goodness of fit metrics.
  - Interpretation of Hurst exponent values.
  - Ability to plot and visualize fit results.
  - Bootstrap functionality for additional statistics.
  - Test suite test suite is meant to validate the functionality of Hurst exponent estimators.

This repository  is actively being developed and any tickets will be addressed in order of importance. Feel free to raise an issue if you find a problem. Other hurst estimation methods will be supported in the near future.

## Installation 

To get started;

`pip install hurst_exponent`


### Dependencies
Ensure the following dependencies are installed before utilizing the package.

  - numpy
  - pandas
  - powerlaw_function
  - stochastic

## Basic Usage 

This tells you everything you need to know for the simplest, typical use cases:
  
~~~python
from hurst_exponent import standard_hurst, generalized_hurst

# Define your time series data
series = [...]

# Estimate Hurst Exponent using standard method
hurst_std, fit_std = standard_hurst(series)

# Estimate Hurst Exponent using generalized method
hurst_gen, fit_gen = generalized_hurst(series)

# Print results
print(f"Standard Hurst Exponent: {hurst_std}")
print(f"Generalized Hurst Exponent: {hurst_gen}")
~~~

## Documentation

### Main Functions
  ~~~python
  standard_hurst(series: np.array, ...):
  ~~~
  Compute the Hurst exponent using the standard deviation of sums.

  ~~~python
  generalized_hurst(series: np.array, ...):
  ~~~
  Computes the generalized Hurst exponent using the structure function method.


### Utils

  `bootstrap(estimator: Callable, ...)`: Generates bootstrap samples.
  
  `std_of_sums(ts: np.array, lag_size: int)`: Computes the standard deviation of sums of time series lags of size lag_size.

  `structure_function(ts: np.array, moment: int, lag: int)`: Calculate the structure function for a given moment and lag.

  `interpret_hurst(H: float)`: Provides an interpretation for the given Hurst Exponent.


### Hurst Estimators Test Suite

Our Hurst Estimators Test Suite ensures the robustness and accuracy of the  standard_hurst and generalized_hurst estimators in the context of time series analysis.

#### Highlights:

Tests various hyperparameter combinations for both the generalized and standard Hurst estimators.

  ##### Estimators;
  
    - Simulation: Uses Geometric Brownian Motion (GBM) to simulate data for testing, representing a standard model for stock price movements with a known Hurst of 0.5.

    - Bootstrapping: Repeated sampling is employed to create a distribution of Hurst estimates for statistical validation.

  ##### Core Tests;

    - Unbiasedness: Checks if the estimator accurately identifies a random walk in GBM data.
    - Validity: Ensures estimates fall within the [0, 1] range.
    - Confidence Intervals: Validates that the 95% CI aligns with bounds cited in major literature.
  
To delve into the specifics, review the test suite source code.

# License
This project is licensed under the MIT License. See the LICENSE.md file for details.



