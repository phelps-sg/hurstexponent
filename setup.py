from os import path
from setuptools import setup, find_packages


here = path.abspath(path.dirname(__file__))
with open(path.join(here, "README.rst"), encoding="utf-8") as readme_file:
    readme = readme_file.read()

setup(
    name="hurst-exponent",
    version="0.1.1",
    packages=find_packages(),
    url="https://github.com/anabugaenko/hurst_exponent",
    license="MIT LICENSE",
    author="Anastasia Bugaenko",
    author_email="anabugaenko@gmail.com",
    description="Hurst exponent estimator",
    long_description=readme,
)
