from hurst_exponent.hurst_exponent import *

if __name__ == "__main__":
    # Fractal BM
    from stochastic.processes.continuous import FractionalBrownianMotion
    from matplotlib import pyplot as plt

    fbm = FractionalBrownianMotion(hurst=0.5, t=1)
    fbm_series = fbm.sample(10000)
    lags = fbm.times(10000)
    plt.plot(lags, fbm_series)
    plt.show()

    # Estimate Hurst Exponent using both methods
    hurst_std, fit_std = standard_hurst(fbm_series)
    hurst_gen, fit_gen = generalized_hurst(fbm_series)

    # Print fitting results
    fit_std.powerlaw.print_fitted_results()
    fit_gen.powerlaw.print_fitted_results()

    # Interpret and display the results
    fit_std.powerlaw.plot_fit()
    fit_gen.powerlaw.plot_fit()
    print(f"Standard Hurst Exponent: {hurst_std} ({interpret_hurst(hurst_std)})")
    print(f"Generalized Hurst Exponent: {hurst_gen} ({interpret_hurst(hurst_gen)})")

    # Bootstrap
    results = bootstrap(generalized_hurst, reps=1000, seed=50)

    lower_ci = np.percentile(results, 5)
    upper_ci = np.percentile(results, 95)

    print(pd.DataFrame(results, columns=["H"]).describe())
    print()
    print("95% Confidence Interval:", (lower_ci, upper_ci))