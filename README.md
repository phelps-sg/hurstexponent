# Hurst Estimator
A simple statistical package for estimating the long-term memory of time series data.  

#
This repository contains a Python Package for estimating the Hurst exponent of a time series. The Hurst exponent is used as a measure of long-term memory of time series and relates to both the scaling of the standard deviation of sums of N successive events and the autocorrelations of the time series given the rate at which these decrease as the lag between pairs of values increases.

Feel free to raise an issue if you find a problem; this repository is actively being developed and any tickets will be addressed in order of importance.

# Table of Contents
[Installation](#Installation)</b>

[Basic Usage](#Usage)</b>

[Example](#Example)</b>

## Installation 

We recommend conda for managing Python packages; pip for everything else. To get started, `pip install hurstexponent` ensuring the following dependencies:

  `pip install scipy numpy pandas statsmodels hurst typing matplotlib`


## Basic Usage 

This tells you everything you need to know for the simplest, typical use cases:
  
~~~python
# Generate simple random walk series
from util.generate_series import simple_series
series = simple_series(length=99999, noise_pct_std=0.02, seed=70) # avg. daily market volatility

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
hurst = standard_hurst(series) # fit_method='Least_squares'
hurst.powerlaw.print_fitted_results()
interpretation = interpret_hurst(hurst.powerlaw.params.alpha)
print(f'Hurst Estimate via Standard deviation of sums: H = {hurst.powerlaw.params.alpha}, ({interpretation})')
print('\n')

# Generalized Hurst
print('Generalized Hurst Exponent')
generalized_hurst = generalized_hurst(series)
generalized_hurst.powerlaw.print_fitted_results()
interpretation = interpret_hurst(generalized_hurst.powerlaw.params.alpha)
print(f"Hurst Estimate via Generalized Hurst: H = {generalized_hurst.powerlaw.params.alpha}, ({interpretation})")
~~~

## Advanced Usage 

~~~python
def acf(series: pd.Series, lags: int) -> List:
    """
    Returns a list of autocorrelation values for each of the lags from 0 to `lags`
    """
    acl_ = []
    for i in range(lags):
        ac = series.autocorr(lag=i)
        acl_.append(ac)
    return acl_


# Load sample data – TSLA stock trade signs.
sample = pd.read_csv('../datasets/stock_tsla.csv', header=0, index_col=0)
# Series generated from a function, in this example, autocorrelation function (ACF)
ACF_RANGE = 1001
series = acf(sample['trade_sign'], ACF_RANGE)[1:]
series = np.array(series)

# Generalized Hurst
print('Generalized Hurst Exponent')
generalized_hurst = generalized_hurst(series, fitting_method='Least_squares', max_lag: int = 100) # select fitting parameters appropriate for problem domain
generalized_hurst.powerlaw.print_fitted_results()
generalized_hurst.powerlaw.plot_fit()
interpretation = interpret_hurst(generalized_hurst.powerlaw.params.alpha)
print(f"Hurst Estimate via Generalized Hurst: H = {generalized_hurst.powerlaw.params.alpha}, ({interpretation})")
~~~


# Results

~~~
Standard Hurst Exponent

For powerlaw fitted using MLE;

Pre-fitting parameters:
xmin: 10.0

Fitting parameters:
param_names = ['C', 'alpha']
C = 0.0008792925582104561
alpha = 0.5121080536945518

Goodness of fit to data:
D = 0.031746031746031744
bic = -6004.969035377474
Adjusted R-squared = 0.9984213733623283
Hurst Estimate via Standard deviation of sums: H = 0.5121080536945518, (Super-diffusive: series demonstrates persistent long-range dependence)

Generalized Hurst Exponent

For powerlaw fitted using MLE;

Pre-fitting parameters:
xmin: 13.0

Fitting parameters:
param_names = ['C', 'alpha']
C = 0.0007479492781326126
alpha = 0.4998855852717425

Goodness of fit to data:
D = 0.01904761904761905
bic = -5864.657061268197
Adjusted R-squared = 0.999732241890803
Hurst Estimate via Generalized Hurst: H = 0.4998855852717425, (Sub-diffusive: series demonstrates anti-persistent behaviour)
~~~
