from setuptools import setup

setup(
    name="Thana_strat_tool",
    version="2.0.1",
    description="Thana B. indicators and backtester as extension",
    author="Thna B.",
    packages=[],
    install_requires=[
        "matplotlib",
        "numpy",
        "pandas",
        "pycodestyle",
        "varname",
        "statsmodels",
        "TA-Lib",
        "xarray",
        "black",
	"numba"
    ],
)
