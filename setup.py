import setuptools

with open("README.md", "r") as file:
    long_description = file.read()

setuptools.setup(
    name="hurstexponent",
    version="0.1.1",
    author="Anastasia Bugeenko",
    author_email="anabugaenko@gmail.com",
    license="MIT",
    description="Hurst exponent estimator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/anabugaenko/hurstexponent",
    install_requires=['scipy', 'numpy', 'pandas', 'typing', 'statsmodels', 'hurst', 'matplotlib'],
    keywords="hurst autocorrelation time-series fractals",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Intended Audience :: Science/Research"
    ]
)