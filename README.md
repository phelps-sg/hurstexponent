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


## Basic Usage 

This tells you everything you need to know for the simplest, typical use cases:
  
	# Generate simple random walk series
	from util.generate_series import simple_series
	series = simple_series(length=99999, noise_pct_std=0.02, seed=70) # avg. daily market volatility

	# Genereate stochastic process with specific long-range properties
	# from util.utils import stochastic_process
	# series = stochastic_process(length=99999, proba=.5, cumprod=False, seed=50)

	# Plot raw series
	plt.figure(figsize=(10, 6))
	plt.plot(series)
	plt.title('Raw Series')
	plt.xlabel('Time')
	plt.ylabel('Value')
	plt.savefig('../plots/random_walk.png', bbox_inches='tight')
	plt.show()

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


## Advanced Usage 

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

	# Hurst
	H, D, data, interpretation = standard_hurst(series)
	print(f"Hurst Estimate via Standard Hurst: {H}, D constant: {D if D is not None else 'N/A'}, ({interpretation})")

	# Generalized Hurst
	H, c, data, interpretation = generalized_hurst(series)
	print(f"Hurst Estimate via Generalized Hurst: {H}, D constant: {D if D is not None else 'N/A'}, ({interpretation})")

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
	
	    plt.show()
	 

### Bootstrap confidence interval

	# Estimate the Hurst exponent using the standard method and the bootstrap technique
	from arch.bootstrap import MovingBlockBootstrap
	
	def hurst_wrapper(series):
	H, _, _, _ = standard_hurst(series)
	# H, _, _, _ = generalized_hurst(series)
	
	return H
	
	# Create a bootstrap object with block size, e.g., 10
	bs = MovingBlockBootstrap(1000, series)
	
	# Apply the function to the bootstrap object
	results = bs.apply(hurst_wrapper, 1000) # number of repetitions
	
	mean_hurst = np.mean(results)
	std_dev_hurst = np.std(results)
	
	# Standard 1.96 multiplier for a 95% confidence interval under the assumption of a normal distribution
	lower_ci = mean_hurst - 1.96 * std_dev_hurst
	upper_ci = mean_hurst + 1.96 * std_dev_hurst
	
	print("Bootstrap Mean:", mean_hurst)
	print("Bootstrap Standard Deviation:", std_dev_hurst)
	print("95% Confidence Interval:", (lower_ci, upper_ci))
 

 # Results

	Hurst Estimate via Standard Hurst: 0.4970432488504442, D constant: 0.0009524431301358832, (Perfect diffusivity: series is a geometric or Brownian random walk)
	Hurst Estimate via Generalized Hurst: 0.5, D constant: 0.0009524431301358832, (Perfect diffusivity: series is a geometric or Brownian random walk)
	Hurst Estimate via R/S: 0.5035988483688979, c constant: 1.4471624961034133, (Perfect diffusivity: series is a geometric or Brownian random walk)

	
	Confidence:
	Bootstrap Mean: 0.48938380943505
	Bootstrap Standard Deviation: 0.00976463430548988
	95% Confidence Interval: (0.47024512619628983, 0.5085224926738101)


![Hurst, generalised and r/s hurst](/plots/random_walk.png)

![Hurst, generalised and r/s hurst](/plots/hurst.png)
