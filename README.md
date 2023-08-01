# Hurst Estimator
A simple statistical package for estimating the long-term memory of time series data.  

#
This repository contains a Python class for estimating the Hurst exponent of a time series. The Hurst exponent is used as a measure of long-term memory of time series and relates to both the scaling of the standard deviation of sums of N successive events and the autocorrelations of the time series given the rate at which these decrease as the lag between pairs of values increases.

Feel free to raise an issue if you find a problem; this repository is actively being developed and any tickets will be addressed in order of importance.

# Table of Contents
[Installation](#Installation)</b>

[Basic Usage](#Usage)</b>

[Example](#Example)</b>

## Installation 

We recommend conda for managing Python packages; pip for everything else. To get started, `pip install hurstexponent` ensuring the following dependencies:

  `pip install scipy numpy pandas statsmodels hurst typing matplotlib`

## Usage 

This tells you everything you need to know for the simplest, typical use cases:
  
	from hurstexponent.hurst_exponent import HurstEstimator

  	# Initialize the HurstEstimator with a time series
	np.random.seed(42)
	random_changes = 1. + np.random.randn(99999) / 1000.
	series = np.cumprod(random_changes)
	
	hurst_estimator = HurstEstimator(series)

	# Estimate the Hurst exponent via generalised hurst (default methods)
 	H, D, data, interpretation = hurst_estimator.estimate()

  	# Estimate the Hurst exponent via Rescaled Range
 	H, c, data, interpretation = hurst_estimator.estimate('rescaled_range', kind='random_walk')

 	# Estimate the Hurst exponent via alpha found from fitting autocorrelation function (ACF)
  	H, _, _, interpretation = hurst_returns_acf.estimate('hurst_from_alpha', alpha=alpha)
  	
## Example 

	from hurstexponent.hurst_exponent import HurstEstimator
 
	np.random.seed(42)
	random_changes = 1. + np.random.randn(99999) / 1000.
	series = np.cumprod(random_changes)
	
	# Create an instance of HurstEstimator
	hurst_estimator = HurstEstimator(series)
	
	# Generalized Hurst
	H, D, data, interpretation = hurst_estimator.estimate('generalized_hurst')
	print(f"Hurst Estimate via Generalised Hurst: {H}, D constant: {D if D is not None else 'N/A'} ({interpretation})")
	
	# Rescaled Range
	H, c, data, interpretation = hurst_estimator.estimate('rescaled_range', kind='random_walk')
	print(f"Hurst Estimate via R/S: {H}, c constant: {c if c is not None else 'N/A'} ({interpretation})")
	
	# Plotting
	fig, axs = plt.subplots(1, 2, figsize=(10, 4))
	
	# Generalized Hurst
	H, D, data, interpretation = hurst_estimator.estimate('generalized_hurst')
	chunk_sizes, y_values = data
	axs[0].plot(chunk_sizes, D*np.array(chunk_sizes)**H, "g--", label=f'Generalized Hurst (H={H:.2f})')
	axs[0].plot(chunk_sizes, y_values, 'b.',  label='Observed Values')
	axs[0].loglog()
	axs[0].set_xlabel('Lag')
	axs[0].set_ylabel('Standard deviation of sums')
	axs[0].legend(frameon=False)
	axs[0].grid(False)
	
	# Rescaled Range
	H, c, data, interpretation = hurst_estimator.estimate('rescaled_range', kind='random_walk')
	axs[1].plot(data[0], c*data[0]**H, 'g--', label=f'Hurst (H={H:.2f})')
	axs[1].plot(data[0], data[1],  'b.', label='(Lag, R/S value)')
	axs[1].loglog()
	axs[1].set_xlabel('Lag')
	axs[1].set_ylabel('R/S ratio')
	axs[1].legend(frameon=False)
	axs[1].grid(False)
	
	plt.show()
 

![Hurst, generalised and r/s hurst](/plots/hurst.png)

 # Acknowledgement 

 The Package makes use of https://github.com/Mottl/hurst to make use of their computation of the Hurst from the R/S method. 

