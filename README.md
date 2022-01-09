
# SAMPLE_Thana_time-series_data_analysis
This repository contains trading strategy, backtest and indicators that can be used as educational materials/references for quantitative analysis trading strategy research coding.

For those who are interested in data analysis for time-series, you can visit SAMPLE_Thana_time-series_data_analysis\src\Thana_strat_tool\indicators. It contains a large number of statistical analysis, and filters. Those functions have been improved to speed up the calculations. I used some of those analysis tools to generate more than 5 % of active return in Thai equity market in 2021.

## Prerequisites
``` 
dataframe_image==0.1.1
matplotlib==3.3.2
numba==0.51.2
numpy==1.19.5
pandas==1.1.4
Pillow==9.0.0
requests==2.26.0
scikit_learn==1.0.2
scipy==1.5.2
seaborn==0.11.0
statsmodels==0.13.1
TA_Lib==0.4.19
``` 
## Installing

1. Clone it from GitHub.
2. Install the package by the following code. It will run SAMPLE_Thana_time-series_data_analysis\setup.py for you and install the package with prerequisities.
3. DONE. you can use SAMPLE_Thana_time-series_data_analysis\src\Thana_strat_tool\indicator for your time-series analysis right away.
``` Terminal
pip install .
```
REMARK: if you have a problem with installing a library called "talib", please manually install it from the following address.
https://www.lfd.uci.edu/~gohlke/pythonlibs/
## Author
* Thana Burana-amorn （ブーラナアモン タナ）

## Acknowledgments
* Thank Pimol Burana-amorn for supporting my dream.
* Thank Quantnet for giving me inspriation.
