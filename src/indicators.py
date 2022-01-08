import pandas as pd
import talib as ta
import numpy as np
from datetime import datetime
import statsmodels.api as sm
from statsmodels import regression
import timeit
import matplotlib.pyplot as plt

import xarray as xr

# %% reference from website quantnet


"""

    attention: !!
    install the following to boost the speed of numba to the next level (x2)
    conda install -c numba icc_rt
    
    2021-08-16 update
        add low pass filter
    
    2021-08-17 update
        add median_ts
"""

import typing as tp

NdType = tp.Union[np.ndarray, pd.DataFrame, xr.DataArray, pd.Series]
NdTupleType = tp.Union[
    tp.Tuple[NdType],
    tp.Tuple[NdType, NdType],
    tp.Tuple[NdType, NdType, NdType],
    tp.Tuple[NdType, NdType, NdType, NdType],
]

XR_TIME_DIMENSION = "time"

import numpy as np
import pandas as pd
import xarray as xr
import typing as tp

NdType = tp.Union[np.ndarray, pd.DataFrame, xr.DataArray, pd.Series]
NdTupleType = tp.Union[
    tp.Tuple[NdType],
    tp.Tuple[NdType, NdType],
    tp.Tuple[NdType, NdType, NdType],
    tp.Tuple[NdType, NdType, NdType, NdType],
]

XR_TIME_DIMENSION = "time"


def nd_universal_adapter(
    d1_function, nd_args: NdTupleType, plain_args: tuple
) -> NdType:
    if isinstance(nd_args[0], np.ndarray):
        return nd_np_adapter(d1_function, nd_args, plain_args)
    if isinstance(nd_args[0], pd.DataFrame):
        return nd_pd_df_adapter(d1_function, nd_args, plain_args)
    if isinstance(nd_args[0], pd.Series):
        return nd_pd_s_adapter(d1_function, nd_args, plain_args)
    if isinstance(nd_args[0], xr.DataArray):
        return nd_xr_da_adapter(d1_function, nd_args, plain_args)
    raise Exception("unsupported")


def nd_np_adapter(
    d1_function, nd_args: tp.Tuple[np.ndarray], plain_args: tuple
) -> np.ndarray:
    shape = nd_args[0].shape
    if len(shape) == 1:
        args = nd_args + plain_args
        return d1_function(*args)
    nd_args_2d = tuple(a.reshape(-1, shape[-1]) for a in nd_args)
    result2d = np.empty_like(
        nd_args_2d[0],
    )
    for i in range(nd_args_2d[0].shape[0]):
        slices = tuple(a[i] for a in nd_args_2d)
        args = slices + plain_args
        result2d[i] = d1_function(*args)
    return result2d.reshape(shape)


def nd_pd_df_adapter(
    d1_function, nd_args: tp.Tuple[pd.DataFrame], plain_args: tuple
) -> pd.DataFrame:
    np_nd_args = tuple(a.to_numpy().transpose() for a in nd_args)
    np_result = nd_np_adapter(d1_function, np_nd_args, plain_args)
    np_result = np_result.transpose()
    return pd.DataFrame(np_result, columns=nd_args[0].columns, index=nd_args[0].index)


def nd_pd_s_adapter(
    d1_function, nd_args: tp.Tuple[pd.Series], plain_args: tuple
) -> pd.Series:
    np_nd_args = tuple(a.to_numpy() for a in nd_args)
    np_result = nd_np_adapter(d1_function, np_nd_args, plain_args)
    np_result = np_result.transpose()
    return pd.Series(np_result, nd_args[0].index)


def nd_xr_da_adapter(
    d1_function, nd_args: tp.Tuple[xr.DataArray], plain_args: tuple
) -> xr.DataArray:
    origin_dims = nd_args[0].dims
    transpose_dims = tuple(i for i in origin_dims if i != XR_TIME_DIMENSION) + (
        XR_TIME_DIMENSION,
    )
    np_nd_args = tuple(a.transpose(*transpose_dims).values for a in nd_args)
    np_result = nd_np_adapter(d1_function, np_nd_args, plain_args)
    return xr.DataArray(
        np_result, dims=transpose_dims, coords=nd_args[0].coords
    ).transpose(*origin_dims)


def nd_to_1d_universal_adapter(
    np_function, nd_args: NdTupleType, plain_args: tuple
) -> NdType:
    if isinstance(nd_args[0], np.ndarray):
        return nd_to_1d_np_adapter(nd_args, plain_args)
    if isinstance(nd_args[0], pd.DataFrame):
        return nd_to_1d_pd_df_adapter(np_function, nd_args, plain_args)
    if isinstance(nd_args[0], xr.DataArray):
        return nd_to_1d_xr_da_adapter(np_function, nd_args, plain_args)
    raise Exception("unsupported")


def nd_to_1d_np_adapter(
    np_function, nd_args: tp.Tuple[np.ndarray], plain_args: tuple
) -> np.ndarray:
    args = nd_args + plain_args
    return np_function(*args)


def nd_to_1d_pd_df_adapter(
    np_function, nd_args: tp.Tuple[pd.DataFrame], plain_args: tuple
) -> pd.Series:
    np_nd_args = tuple(a.to_numpy().transpose() for a in nd_args)
    np_result = nd_to_1d_np_adapter(np_function, np_nd_args, plain_args)
    np_result = np_result.transpose()
    return pd.Series(np_result, index=nd_args[0].index)


def nd_to_1d_xr_da_adapter(
    np_function, nd_args: tp.Tuple[xr.DataArray], plain_args: tuple
) -> xr.DataArray:
    origin_dims = nd_args[0].dims
    transpose_dims = tuple(i for i in origin_dims if i != XR_TIME_DIMENSION) + (
        XR_TIME_DIMENSION,
    )
    np_nd_args = tuple(a.transpose(*transpose_dims).values for a in nd_args)
    np_result = nd_to_1d_np_adapter(np_function, np_nd_args, plain_args)
    return xr.DataArray(
        np_result,
        dims=[XR_TIME_DIMENSION],
        coords=[nd_args[0].coords[XR_TIME_DIMENSION]],
    )


import numpy as np
import numba as nb
import pandas as pd


@nb.jit(nb.float64[:](nb.float64[:], nb.int64), nopython=True)
def sma_np_1d(series: np.ndarray, periods: int) -> np.ndarray:
    tail = np.empty((periods,), dtype=np.double)
    not_nan_cnt = 0
    result = np.full(series.shape, np.nan, dtype=np.double)
    for i in range(series.shape[0]):
        if not np.isnan(series[i]):
            idx = not_nan_cnt % periods
            tail[idx] = series[i]
            not_nan_cnt += 1
            if not_nan_cnt >= periods:
                result[i] = tail.mean()
    return result


def MA(input_: NdType, periods: int = 20) -> NdType:
    exist = data_exist_1_nan(input_)
    exist = exist * (exist.shift(periods))
    target_ = input_
    target_clean = clean_target(target_).astype(np.float64)
    return nd_universal_adapter(sma_np_1d, (target_clean,), (periods,)) * exist


@nb.jit(nb.float64[:](nb.float64[:], nb.int64), nopython=True)
def shift_np_1d(series: np.ndarray, periods: int) -> np.ndarray:
    if periods < 0:
        return np.flip(shift_np_1d(np.flip(series), -periods))
    tail = np.empty((periods + 1,), dtype=np.double)
    not_nan_cnt = 0
    result = np.full(series.shape, np.nan, dtype=np.double)
    for i in range(series.shape[0]):
        if not np.isnan(series[i]):
            idx = not_nan_cnt % tail.shape[0]
            tail[idx] = series[i]
            if not_nan_cnt >= periods:
                result[i] = tail[idx - periods]
            not_nan_cnt += 1
    return result


def shift(series: NdType, periods: int = 1) -> NdType:
    return nd_universal_adapter(shift_np_1d, (series,), (periods,))


def ROC(series: NdType, periods: int = 7) -> NdType:
    """
    Rate of change
    """
    exist = data_exist_1_nan(series)
    exist = exist * (exist.shift(periods))
    target_ = series
    target_clean = clean_target(target_)
    shifted = shift(target_clean, periods)
    return 100 * (target_clean / shifted - 1) * exist


@nb.jit(nb.float64[:](nb.float64[:], nb.int64, nb.int64), nopython=True)
def ema_np_1d(series: np.ndarray, periods: int, warm_periods: int) -> np.ndarray:
    result = np.full(series.shape, np.nan, dtype=np.double)
    k = 2 / (periods + 1)
    not_nan_cnt = 0
    value = 0
    for i in range(series.shape[0]):
        if not np.isnan(series[i]):
            not_nan_cnt += 1
            if not_nan_cnt <= warm_periods:
                value += series[i] / warm_periods
            else:
                value = (series[i] - value) * k + value
            if not_nan_cnt >= warm_periods:
                result[i] = value
    return result


def EMA(
    series: NdType, periods: int = 20, warm_periods: tp.Union[int, None] = None
) -> NdType:
    """
    Exponential moving average
    """
    if warm_periods is None:
        warm_periods = periods
    return nd_universal_adapter(
        ema_np_1d,
        (series,),
        (
            periods,
            warm_periods,
        ),
    )


def tr(high: NdType, low: NdType, close: NdType) -> NdType:
    prev_close = shift(close, 1)
    return np.maximum(high, prev_close) - np.minimum(low, prev_close)


def wilder_ma(
    series: NdType, periods: int = 14, warm_periods: tp.Union[int, None] = None
):
    """
    Wilder's Moving Average
    """
    if warm_periods is None:
        warm_periods = periods
    return EMA(series, periods * 2 - 1, warm_periods)


def ATR(high: NdType, low: NdType, close: NdType, ma: tp = 14) -> NdType:
    if isinstance(ma, int):
        ma_period = ma
        ma = lambda s: wilder_ma(s, ma_period)
    return ma(tr(high, low, close))


@nb.jit(nb.float64[:](nb.float64[:], nb.int64), nopython=True)
def variance_np_1d(series: np.ndarray, periods: int) -> np.ndarray:
    tail = np.empty((periods,), dtype=np.double)
    not_nan_cnt = 0
    result = np.full(series.shape, np.nan, dtype=np.double)
    for i in range(series.shape[0]):
        if not np.isnan(series[i]):
            idx = not_nan_cnt % tail.shape[0]
            tail[idx] = series[i]
            if not_nan_cnt >= periods:
                result[i] = np.var(tail)
            not_nan_cnt += 1
    return result


@nb.jit(nb.float64[:](nb.float64[:], nb.int64), nopython=True)
def std_np_1d(series: np.ndarray, periods: int) -> np.ndarray:
    tail = np.empty((periods,), dtype=np.double)
    not_nan_cnt = 0
    result = np.full(series.shape, np.nan, dtype=np.double)
    for i in range(series.shape[0]):
        if not np.isnan(series[i]):
            idx = not_nan_cnt % tail.shape[0]
            tail[idx] = series[i]
            if not_nan_cnt >= periods:
                result[i] = np.std(tail)
            not_nan_cnt += 1
    return result


@nb.jit(nb.float64[:](nb.float64[:], nb.float64[:], nb.int64), nopython=True)
def covariance_np_1d(x: np.ndarray, y: np.ndarray, periods: int) -> np.ndarray:
    tail_x = np.empty((periods,), dtype=np.double)
    tail_y = np.empty((periods,), dtype=np.double)
    not_nan_cnt = 0
    result = np.full(x.shape, np.nan, dtype=np.double)
    for i in range(x.shape[0]):
        if not np.isnan(x[i]) and not np.isnan(y[i]):
            idx = not_nan_cnt % tail_x.shape[0]
            tail_x[idx] = x[i]
            tail_y[idx] = y[i]
            if not_nan_cnt >= periods:
                mx = np.mean(tail_x)
                vx = tail_x - mx
                my = np.mean(tail_y)
                vy = tail_y - my
                cov = np.mean(vx * vy)
                result[i] = cov
            not_nan_cnt += 1
    return result


def variance(series: NdType, periods: int = 1) -> NdType:
    return nd_universal_adapter(variance_np_1d, (series,), (periods,))


def std(series: NdType, periods: int = 1) -> NdType:
    exist = data_exist_1_nan(series)
    exist = exist * (exist.shift(periods))
    target_ = series
    target_clean = clean_target(target_)
    return nd_universal_adapter(std_np_1d, (target_clean,), (periods,)) * exist


def STDDEV(series: NdType, periods: int = 1) -> NdType:
    return std(series, periods)


def covariance(x: NdType, y: NdType, periods: int = 1) -> NdType:
    return nd_universal_adapter(covariance_np_1d, (x, y), (periods,))


def beta(price_x, price_y, periods=252):
    rx = ROC(price_x, 1) / 100
    ry = ROC(price_y, 1) / 100
    return covariance(rx, ry, periods) / variance(ry, periods)


def MACD(
    series: NdType, fast_ma: tp.Any = 12, slow_ma: tp.Any = 26, signal_ma: tp.Any = 9
) -> tp.Tuple[NdType, NdType, NdType]:
    """
    MACD
    :return: (macd_line, signal_line, histogram)
    """
    if isinstance(fast_ma, int):
        fast_ma_period = fast_ma
        fast_ma = lambda s: EMA(s, fast_ma_period)
    if isinstance(slow_ma, int):
        slow_ma_period = slow_ma
        slow_ma = lambda s: EMA(s, slow_ma_period)
    if isinstance(signal_ma, int):
        signal_ma_period = signal_ma
        signal_ma = lambda s: EMA(s, signal_ma_period)
    fast = fast_ma(series)
    slow = slow_ma(series)
    macd_line = fast - slow
    signal_line = signal_ma(macd_line)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


@nb.jit(nb.float64[:](nb.float64[:], nb.int64), nopython=True)
def change_np_1d(series: np.ndarray, periods: int) -> np.ndarray:
    shifted = shift_np_1d(series, periods)
    return series - shifted


def change(series: NdType, periods: int = 1) -> np.ndarray:
    shifted = shift(series, periods)
    return series - shifted


def RSI(
    series: NdType, ma: tp.Any = 14  # lambda series: wilder_ma(series, 14)
) -> NdType:
    if isinstance(ma, int):
        ma_period = ma
        ma = lambda series: wilder_ma(series, ma_period)

    ch = change(series)
    up = np.maximum(ch, 0)  # positive changes
    down = -np.minimum(ch, 0)  # negative changes (inverted)

    up = ma(up)
    down = ma(down)

    rs = up / down
    rsi = 100 * (1 - 1 / (1 + rs))

    return rsi


@nb.jit(
    nb.float64[:](nb.float64[:], nb.float64[:], nb.float64[:], nb.int64), nopython=True
)
def k_np_1d(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, periods: int
) -> np.ndarray:
    tail_low = np.empty((periods,), dtype=np.double)
    tail_high = np.empty((periods,), dtype=np.double)
    not_nan_cnt = 0
    result = np.full(close.shape, np.nan, dtype=np.double)
    for i in range(close.shape[0]):
        if not np.isnan(high[i]) and not np.isnan(low[i]) and not np.isnan(close[i]):
            idx = not_nan_cnt % periods
            tail_low[idx] = low[i]
            tail_high[idx] = high[i]
            not_nan_cnt += 1
            if not_nan_cnt >= periods:
                highest = tail_high.max()
                lowest = tail_low.min()
                if highest > lowest:
                    result[i] = 100 * (close[i] - lowest) / (highest - lowest)
    return result


@nb.jit(nb.float64[:](nb.float64[:], nb.int64), nopython=True)
def lwma_np_1d(series: np.ndarray, periods: int) -> np.ndarray:
    tail = np.empty((periods,), dtype=np.double)
    not_nan_cnt = 0
    result = np.full(series.shape, np.nan, dtype=np.double)
    w_sum = periods * (periods + 1) / 2
    for i in range(series.shape[0]):
        if not np.isnan(series[i]):
            idx = not_nan_cnt % periods
            tail[idx] = series[i]
            not_nan_cnt += 1
            if not_nan_cnt >= periods:
                sum = 0
                for j in range(periods):
                    w = periods - j
                    sum += tail[idx - j] * w
                result[i] = sum / w_sum
    return result


@nb.jit(nb.float64[:](nb.float64[:], nb.float64[:]), nopython=True, error_model="numpy")
def wma_np_1d(series: np.ndarray, weights: np.ndarray) -> np.ndarray:
    periods = len(weights)
    tail = np.empty((periods,), dtype=np.double)
    not_nan_cnt = 0
    result = np.full(series.shape, np.nan, dtype=np.double)
    w_sum = weights.sum()
    for i in range(series.shape[0]):
        if not np.isnan(series[i]):
            idx = not_nan_cnt % periods
            tail[idx] = series[i]
            not_nan_cnt += 1
            if not_nan_cnt >= periods:
                sum = 0
                for j in range(periods):
                    sum += tail[idx - j] * weights[j]
                result[i] = sum / w_sum
    return result


@nb.jit(
    nb.float64[:](nb.float64[:], nb.float64[:], nb.int64),
    nopython=True,
    error_model="numpy",
)
def vwma_np_1d(price: np.ndarray, volume: np.ndarray, periods: int) -> np.ndarray:
    price_tail = np.empty((periods,), dtype=np.double)
    volume_tail = np.empty((periods,), dtype=np.double)
    not_nan_cnt = 0
    result = np.full(price.shape, np.nan, dtype=np.double)
    for i in range(price.shape[0]):
        if not np.isnan(price[i]) and not np.isnan(volume[i]):
            idx = not_nan_cnt % periods
            price_tail[idx] = price[i]
            volume_tail[idx] = volume[i]
            not_nan_cnt += 1
            if not_nan_cnt >= periods:
                result[i] = (price_tail * volume_tail).sum() / volume_tail.sum()
    return result


last_alert = 0

import time


def lwma(series: NdType, periods: int = 20):
    return nd_universal_adapter(lwma_np_1d, (series,), (periods,))


def WMA(series: NdType, weights: tp.Union[tp.List[float], np.ndarray] = None) -> NdType:
    """
    :param weights: weights in decreasing order. lwma(series, 3) == wma(series, [3,2,1])
    """

    global last_alert
    if weights is None or type(weights) is int:
        if time.time() - last_alert > 60:
            last_alert = time.time()

        return lwma(series, weights)
    if type(weights) is list:
        weights = np.array(weights, np.float64)
    return nd_universal_adapter(wma_np_1d, (series,), (weights,))


def VWMA(price: NdType, volume: NdType, periods: int = 20):
    exist = data_exist_1_nan(price)
    exist = exist * (exist.shift(periods))
    target_ = price
    target_clean = clean_target(target_)
    return nd_universal_adapter(vwma_np_1d, (target_clean, volume), (periods,))


def VMA(price: NdType, volume: NdType, periods: int = 20):
    return VWMA(price, volume, periods)


"""
    Thana original numby-based function

"""


@nb.jit(nb.float64[:](nb.float64[:], nb.int64, nb.float64), nopython=True)
def MA_trim_series(target_np: np.array, period_: int, trim_: float):

    period_ = int(period_)

    bar_num = target_np.shape[0]

    start_include = round(period_ * trim_)
    end_include = period_ - round(period_ * trim_)

    output_np = np.empty(target_np.shape)
    output_np[:] = np.nan

    for row in range(period_ - 1, bar_num, 1):

        sliced_np = target_np[row - (period_ - 1) : row + 1]

        if np.isnan(sliced_np).sum() < 1:

            sliced_np = np.sort(sliced_np)
            output_pre = np.nanmean(sliced_np[start_include:end_include])

            output_np[row] = output_pre

    return output_np


def MA_trim(input_: NdType, periods_, trim_=0.05):
    exist = data_exist_1_nan(input_)
    exist = exist * (exist.shift(periods_))
    target_ = input_
    target_clean = clean_target(target_)
    output = nd_universal_adapter(
        MA_trim_series,
        (target_clean,),
        (
            periods_,
            trim_,
        ),
    )
    return output * exist


@nb.jit(nb.float64[:](nb.float64[:], nb.int64), nopython=True)
def downside_deviation_series(target_np: np.array, period_: int):

    period_ = int(period_)

    def cal_downside_deviation(target_):
        mean = np.mean(target_)
        relative = target_ - mean
        smaller_than_mean = relative < 0
        downside = np.zeros(target_.shape)
        downside = smaller_than_mean * relative * relative
        downside = downside.sum() / (target_.shape[0])
        downside = np.sqrt(downside)
        return downside

    bar_num = target_np.shape[0]

    output_np = np.empty(target_np.shape)
    output_np[:] = np.nan

    for row in range(period_ - 1, bar_num, 1):
        sliced_np = target_np[row - (period_ - 1) : row + 1]
        if np.isnan(sliced_np).sum() < 1:
            output_np[row] = cal_downside_deviation(sliced_np)

    return output_np


def get_downside_deviation(input_: NdType, periods_=60):
    exist = data_exist_1_nan(input_)
    exist = exist * (exist.shift(periods_))
    target_ = input_
    target_clean = clean_target(target_)
    output = nd_universal_adapter(
        downside_deviation_series, (target_clean,), (periods_,)
    )
    return output * exist


def rank_ts_asc_pct_series(target_np: np.array, period_: int):
    # https://www.delftstack.com/howto/numpy/python-numpy-rank/
    period_ = int(period_)

    bar_num = target_np.shape[0]

    output_np = np.empty(target_np.shape)
    output_np[:] = np.nan

    for row in range(period_ - 1, bar_num, 1):

        sliced_np = target_np[row - (period_ - 1) : row + 1]

        if np.isnan(sliced_np).sum() < 1:

            # sliced_np = np.sort(sliced_np)
            # output_pre = np.nanmean(sliced_np[start_include: end_include])
            temp = sliced_np.argsort()

            ranks = np.empty_like(temp)

            ranks[temp] = np.arange(len(sliced_np))

            rank_max = ranks.max()
            rank_min = ranks.min()
            output_np[row] = (rank_max - ranks[-1]) / (rank_max - rank_min)

    return output_np


def rank_ts_asc_pct(input_: NdType, periods_, period_=20):
    """
    pseudo rank:
        it ranges from 0 to 1
    """
    exist = data_exist_1_nan(input_)
    exist = exist * (exist.shift(periods_))
    target_ = input_
    target_clean = clean_target(target_)
    output = nd_universal_adapter(rank_ts_asc_pct_series, (target_clean,), (periods_,))
    return output * exist


def rank_ts_dsc_pct_series(target_np: np.array, period_: int):
    # https://www.delftstack.com/howto/numpy/python-numpy-rank/
    period_ = int(period_)

    bar_num = target_np.shape[0]

    output_np = np.empty(target_np.shape)
    output_np[:] = np.nan

    for row in range(period_ - 1, bar_num, 1):

        sliced_np = target_np[row - (period_ - 1) : row + 1]

        if np.isnan(sliced_np).sum() < 1:

            # sliced_np = np.sort(sliced_np)
            # output_pre = np.nanmean(sliced_np[start_include: end_include])
            temp = sliced_np.argsort()

            ranks = np.empty_like(temp)

            ranks[temp] = np.arange(len(sliced_np))
            rank_max = ranks.max()
            rank_min = ranks.min()
            output_np[row] = 1 - (rank_max - ranks[-1]) / (rank_max - rank_min)

    return output_np


def rank_ts_dsc_pct(input_: NdType, periods_, period_=20):
    """
    pseudo rank:
        it ranges from 0 to 1
    """
    exist = data_exist_1_nan(input_)
    exist = exist * (exist.shift(periods_))
    target_ = input_
    target_clean = clean_target(target_)
    output = nd_universal_adapter(rank_ts_dsc_pct_series, (target_clean,), (periods_,))
    return output * exist


# %%
""" joice example
def apply_each_series(x, name):
    print(name, x.rank(pct=True).iloc[-1])
    return x.rank(pct=True).iloc[-1]

def apply_each_col(x, window):
    return x.rolling(window).apply(apply_each_series, args=("joice",))

df.apply(apply_each_col, axis=0, args=(3,))
"""


def apply_each_col_median(x, window):
    return x.rolling(window).median()


def median_ts(target_, period_):
    exist = data_exist_1_nan(target_)
    target_ = clean_target(target_)
    return target_.apply(apply_each_col_median, axis=0, args=(period_,)) * exist


from statsmodels.tsa.seasonal import STL


def row_low_pass_filter_deseason(row_, period_=14):
    return STL(row_, period=period_, robust=False).fit().trend[-1]


def col_low_pass_filter_deseason(col_, window_, period_):
    return col_.rolling(window_).apply(row_low_pass_filter_deseason, args=(period_,))


def low_pass_filter_deseason(target_, train_cycle=2, period_=14):
    exist = data_exist_1_nan(target_)
    exist = exist * (exist.shift(period_))
    target_clean = clean_target(target_)
    train_window = train_cycle * period_
    return (
        target_clean.apply(
            col_low_pass_filter_deseason,
            axis=0,
            args=(
                train_window,
                period_,
            ),
        )
        * exist
    )


def row_low_pass_filter_detrend(row_, period_=14):
    return STL(row_, period=period_, robust=False).fit().seasonal[-1]


def col_low_pass_filter_detrend(col_, window_, period_):
    return col_.rolling(window_).apply(row_low_pass_filter_detrend, args=(period_,))


def low_pass_filter_detrend(target_, train_cycle=2, period_=14):
    exist = data_exist_1_nan(target_)
    exist = exist * (exist.shift(period_))
    target_clean = clean_target(target_)
    train_window = train_cycle * period_
    return (
        target_clean.apply(
            col_low_pass_filter_detrend,
            axis=0,
            args=(
                train_window,
                period_,
            ),
        )
        * exist
    )


def rank_ts_old(data_df, window):
    return data_df.apply(
        lambda s: s.rolling(window).apply(lambda x: x.rank(pct=True).iloc[-1]), axis=0
    )


def correlation(target_: pd.DataFrame, SET_: pd.DataFrame, period_=60):

    period_ = int(period_)
    target_np = target_.ffill(limit=5)
    temp_concat = pd.concat([target_, SET_.iloc[:, 0]], axis=1)
    input_matrix = np.array(temp_concat)

    correlation_result_np = np.zeros(target_np.shape, dtype=np.float64)

    target_np = target_.ffill(limit=5).to_numpy()
    bar_num = target_np.shape[0]
    correlation_result_ = pd.DataFrame(index=target_.index, columns=target_.columns)

    for row in range(period_ - 1, bar_num, 1):

        sliced_matrix = input_matrix[row - (period_ - 1) : row + 1, :]
        cor_np = np.corrcoef(np.transpose(sliced_matrix))
        cor_np = np.transpose(cor_np)

        correlation_result_np[row, :] = cor_np[-1, :][:-1]
        if row % 100 == 0:
            print(row)
    correlation_result_ = pd.DataFrame(
        index=target_.index, columns=target_.columns, data=correlation_result_np
    )
    return correlation_result_


def init_df(mother_df_: pd.DataFrame, value_):
    return pd.DataFrame(columns=mother_df_.columns, index=mother_df_.index, data=value_)


def series_to_df(mother_: pd.Series, df_blueprint_: pd.DataFrame):
    # dubplicated
    extended_df_ = pd.DataFrame(
        index=df_blueprint_.index, columns=df_blueprint_.columns
    )
    extended_df_ = extended_df_.apply(lambda col: mother_.values)
    return extended_df_


def clean_target(target_: pd.DataFrame):

    clean_target_ = target_.copy()
    clean_target_.ffill(inplace=True)
    clean_target_.bfill(inplace=True)
    clean_target_.replace(np.nan, 0, inplace=True)
    clean_target_.replace(np.inf, 0, inplace=True)
    return clean_target_


# Define a function called plot_timeseries
def plot_timeseries(axes, x, y, color, xlabel, ylabel):

    # Plot the inputs x,y in the provided color
    axes.plot(x, y, color=color)

    # Set the x-axis label
    axes.set_xlable(xlabel)

    # Set the y-axis label
    axes.set_ylable(ylabel, color=color)

    # Set the colors tick params for y-axis
    axes.tick_params("y", colors=color)


def one_zero_gen_frm_BUYSELL(
    BUY_bool: pd.DataFrame, SELL_bool: pd.DataFrame, universe: pd.DataFrame
) -> pd.DataFrame:

    BOOLEAN_one_zero = init_df(BUY_bool, np.nan)

    BOOLEAN_one_zero[BUY_bool] = 1.0
    BOOLEAN_one_zero[SELL_bool] = 0.0

    BOOLEAN_one_zero.ffill(inplace=True)
    BOOLEAN_one_zero = BOOLEAN_one_zero * universe
    BOOLEAN_one_zero.replace(np.inf, 0, inplace=True)
    BOOLEAN_one_zero.replace(np.nan, 0, inplace=True)
    return BOOLEAN_one_zero


def ranking_ASC_within_universe(
    input_df: pd.DataFrame, universe: pd.DataFrame
) -> pd.DataFrame:
    universe.replace(0, np.nan, inplace=True)
    target_ = input_df.copy()
    target_ = target_ * universe
    output_ = target_.rank(axis=1)
    return output_


def ranking_DSC_within_universe(
    input_df: pd.DataFrame, universe: pd.DataFrame
) -> pd.DataFrame:
    universe.replace(0, np.nan, inplace=True)
    target_ = input_df.copy()
    target_ = target_ * universe
    output_ = target_.rank(axis=1, ascending=False)
    return output_


def ranking_ASC_within_universe_pct(
    input_df: pd.DataFrame, universe: pd.DataFrame
) -> pd.DataFrame:
    universe.replace(0, np.nan, inplace=True)
    target_ = input_df.copy()
    target_ = target_ * universe
    output_ = target_.rank(axis=1, pct=True)
    return output_


def ranking_DSC_within_universe_pct(
    input_df: pd.DataFrame, universe: pd.DataFrame
) -> pd.DataFrame:
    universe.replace(0, np.nan, inplace=True)
    target_ = input_df.copy()
    target_ = target_ * universe
    output_ = target_.rank(axis=1, ascending=False, pct=True)
    return output_


from potter import backtester as bt
from potter.backtester import factor as fc


def lookback_q(fund_var, q):
    return bt.shift(fund_var, q, True, None)


def sum_q(fund_var, q):
    return bt.sum(fund_var, q, True)


def universe_smoothold(universe_input, num_of_SET_member_series_, threshold_perc=0.025):
    universe_output = init_df(universe_input, 0)
    rebalace_thres = threshold_perc * (num_of_SET_member_series_.copy())
    bar_num = universe_input.shape[0]
    for row_index in range(1, bar_num):
        if row_index % 10 == 0:
            print(row_index)
        universe_output.iloc[row_index, :] = universe_output.iloc[row_index - 1, :]
        diff_series = (
            universe_input.iloc[row_index, :] != universe_output.iloc[row_index, :]
        ).astype(np.float64)
        diff_num = diff_series.sum()
        # print(diff_num)
        if diff_num > rebalace_thres.iloc[row_index]:
            display = diff_series.replace(0, np.nan).dropna()
            # print(display)
            universe_output.iloc[row_index, :] = universe_input.iloc[row_index, :]
    return universe_output.replace(0, np.nan)


def universe_smooth(universe_input, num_of_SET_member_series_, threshold_perc=0.025):

    universe_input_np = np.array(universe_input)
    universe_output_np = np.zeros(universe_input.shape, dtype=np.int64)
    num_of_SET_member_series_np = np.array(num_of_SET_member_series_)
    rebalace_thres = threshold_perc * (num_of_SET_member_series_.copy())
    bar_num = universe_input_np.shape[0]
    for row_index in range(1, bar_num):
        # if(row_index%1000 == 0):
        #     print(row_index)
        universe_output_np[row_index, :] = universe_output_np[row_index - 1, :]
        diff_series = (
            universe_input_np[row_index, :] != universe_output_np[row_index, :]
        ).astype(np.float64)
        diff_num = np.sum(diff_series)
        # print(diff_num)
        if diff_num > rebalace_thres[row_index]:

            universe_output_np[row_index, :] = universe_input_np[row_index, :]

    universe_output = pd.DataFrame(
        index=universe_input.index,
        columns=universe_input.columns,
        data=universe_output_np,
    )
    return universe_output.replace(0, np.nan)


def MAold(target_: pd.DataFrame, period_: int) -> pd.DataFrame:
    period_ = int(period_)
    data_exist = (target_.isnull() == False).astype(np.float64).replace(0, np.nan)
    data_exist = data_exist * (data_exist.shift(period_))
    target_ = clean_target(target_)
    SMA_ = target_.apply(lambda col: ta.SMA(col.astype(np.float64), period_))
    return SMA_


def SUM(target_: pd.DataFrame, period_: int) -> pd.DataFrame:
    period_ = int(period_)
    data_exist = (target_.isnull() == False).astype(np.float64).replace(0, np.nan)

    clean_target_ = target_.copy()
    clean_target_.ffill(inplace=True)
    clean_target_.replace(np.nan, 0, inplace=True)
    clean_target_.replace(np.inf, 0, inplace=True)

    target_np = np.array(clean_target_).astype(np.float64)
    output_np = np.apply_along_axis(ta.SUM, 0, target_np, period_)
    output_df = pd.DataFrame(output_np, index=target_.index, columns=target_.columns)
    return output_df.astype(np.float64) * data_exist


def SUMold(target_: pd.DataFrame, period_: int) -> pd.DataFrame:
    period_ = int(period_)
    data_exist = (target_.isnull() == False).astype(np.float64).replace(0, np.nan)
    target_ = clean_target(target_)
    SUM_ = target_.apply(lambda col: ta.SUM(col.astype(np.float64), period_))
    SUM_ = SUM_ * data_exist
    # SUM_ = SUM_*target_temp_02
    return SUM_.astype(np.float64)


def VMAold(
    target_: pd.DataFrame, target_v_: pd.DataFrame, period_: int
) -> pd.DataFrame:
    period_ = int(period_)
    data_exist = (target_.isnull() == False).astype(np.float64).replace(0, np.nan)
    data_exist = data_exist * (data_exist.shift(period_))
    target_ = clean_target(target_)
    target_v_ = clean_target(target_v_)
    PV = target_ * target_v_
    PV_period_sum = PV.apply(
        lambda col: ta.SUM(col.astype(np.float64), period_)
    ).astype(np.float64)
    V_period_sum = target_v_.apply(
        lambda col: ta.SUM(col.astype(np.float64), period_)
    ).astype(np.float64)
    VMA_ = PV_period_sum / V_period_sum
    VMA_ = VMA_ * data_exist
    return VMA_


def powerMA(target_: pd.DataFrame, last_step_=0.75, period_=10):
    data_exist = (target_.isnull() == False).astype(np.float64).replace(0, np.nan)
    data_exist = data_exist * (data_exist.shift(period_))
    target_ = clean_target(target_)
    base_ = 1 / last_step_
    period_ = int(period_)
    powerMA_ = init_df(target_, 0.0)
    for period_j in range(0, period_):
        # print("period_j : " + str(period_j))
        # print("period_-period_j : " + str(period_-period_j))
        if period_j % 50 == 0:
            print(period_j)
        component_p = (target_.shift(period_j) - target_.shift(period_j + 1)) / (
            base_ ** (period_ - period_j)
        )
        powerMA_ = powerMA_ + component_p

    powerMA_ = powerMA_ + target_.shift(period_)
    powerMA_ = powerMA_ * data_exist
    return powerMA_


def STDDEVold(target_: pd.DataFrame, period_: int) -> pd.DataFrame:
    # Sample standard deviation : divided by n - 1
    period_ = int(period_)
    data_exist = (target_.isnull() == False).astype(np.float64).replace(0, np.nan)
    data_exist = data_exist * (data_exist.shift(period_))
    target_ = clean_target(target_)
    STDDEV_ = target_.apply(lambda col: ta.STDDEV(col, period_)).astype(np.float64)
    STDDEV_.replace(np.nan, 0, inplace=True)
    STDDEV_.replace(np.inf, 0, inplace=True)
    STDDEV_ = STDDEV_ * data_exist
    return STDDEV_


def z_score(target_: pd.DataFrame, period_: int) -> pd.DataFrame:
    period_ = int(period_)
    data_exist = (target_.isnull() == False).astype(np.float64).replace(0, np.nan)
    data_exist = data_exist * (data_exist.shift(period_))
    target_ = clean_target(target_)
    SMA_tar = MA(target_, period_)
    STDDEV_tar = STDDEV(target_, period_)
    z_score_ = (target_ - SMA_tar) / STDDEV_tar
    z_score_.replace(np.inf, np.nan, inplace=True)
    z_score_.replace(np.nan, 0, inplace=True)
    z_score_ = z_score_ * data_exist
    return z_score_


def z_score_sm(target_: pd.DataFrame, period_sm_: int, period_: int) -> pd.DataFrame:
    period_ = int(period_)
    period_sm_ = int(period_sm_)
    data_exist = (target_.isnull() == False).astype(np.float64).replace(0, np.nan)
    data_exist = data_exist * (data_exist.shift(period_ + period_sm_))
    target_ = clean_target(target_)
    SMA_tar = MA(target_, period_)
    STDDEV_tar = STDDEV(target_, period_)
    z_score_ = (MA(target_, period_sm_) - SMA_tar) / STDDEV_tar
    z_score_.replace(np.inf, np.nan, inplace=True)
    z_score_.replace(np.nan, 0, inplace=True)
    z_score_ = z_score_ * data_exist
    return z_score_


def MA_trimold(target_: pd.DataFrame, period_=60, trim_=0.05):
    from scipy import stats

    data_exist = (target_.isnull() == False).astype(np.float64).replace(0, np.nan)
    data_exist = data_exist * (data_exist.shift(period_))
    target_ = clean_target(target_)
    period_ = int(period_)
    target_np = target_.ffill(limit=5).to_numpy()

    stock_num = target_np.shape[1]
    bar_num = target_np.shape[0]

    output_np = np.empty(target_np.shape)
    output_np[:] = np.nan

    for stock in range(stock_num):
        price_of_a_stock = target_np[:, stock]
        if stock % 100 == 0:
            print(stock)
        for row in range(period_ - 1, bar_num, 1):
            sliced_np = price_of_a_stock[row - (period_ - 1) : row + 1]
            if np.isnan(sliced_np).sum() < 1:
                output_np[row, stock] = stats.trim_mean(sliced_np, trim_)

    output_ = pd.DataFrame(index=target_.index, columns=target_.columns, data=output_np)
    output_ = output_ * data_exist
    return output_


def z_score_trim_sm(
    target_: pd.DataFrame, period_sm_: int, period_: int, trim=0.05
) -> pd.DataFrame:
    period_ = int(period_)
    period_sm_ = int(period_sm_)
    data_exist = (target_.isnull() == False).astype(np.float64).replace(0, np.nan)
    data_exist = data_exist * (data_exist.shift(period_ + period_sm_))
    target_ = clean_target(target_)
    SMA_tar = MA_trim(
        target_,
        period_,
    )
    STDDEV_tar = STDDEV(target_, period_)
    z_score_ = (MA(target_, period_sm_) - SMA_tar) / STDDEV_tar
    z_score_.replace(np.inf, np.nan, inplace=True)
    z_score_.replace(np.nan, 0, inplace=True)
    z_score_ = z_score_ * data_exist
    return z_score_


def SQRT(target_: pd.DataFrame) -> pd.DataFrame:
    data_exist = (target_.isnull() == False).astype(np.float64).replace(0, np.nan)
    data_exist = data_exist
    target_np = np.array(clean_target(target_)).astype(np.float64)
    output_np = np.apply_along_axis(ta.SQRT, 0, target_np)
    output_df = pd.DataFrame(output_np, index=target_.index, columns=target_.columns)
    return output_df


def SQRTold(target_: pd.DataFrame) -> pd.DataFrame:
    data_exist = (target_.isnull() == False).astype(np.float64).replace(0, np.nan)
    target_ = clean_target(target_)
    SQRT_ = target_.apply(lambda col: ta.SQRT(col)).astype(np.float64)
    SQRT_ = SQRT_ * data_exist
    return SQRT_


def EMAold(target_: pd.DataFrame, period_: int) -> pd.DataFrame:
    period_ = int(period_)
    data_exist = (target_.isnull() == False).astype(np.float64).replace(0, np.nan)
    data_exist = data_exist * (data_exist.shift(period_ + 240))
    target_ = clean_target(target_)
    EMA_ = target_.apply(lambda col: ta.EMA(col, period_)).astype(np.float64)
    EMA_ = EMA_ * data_exist
    return EMA_


def WMAold(target_: pd.DataFrame, period_: int) -> pd.DataFrame:
    period_ = int(period_)
    data_exist = (target_.isnull() == False).astype(np.float64).replace(0, np.nan)
    data_exist = data_exist * (data_exist.shift(period_))
    target_ = clean_target(target_)
    WMA_ = target_.apply(lambda col: ta.WMA(col, period_)).astype(np.float64)
    WMA_ = WMA_ * data_exist
    return WMA_


def change_by_step(target_, step_: float):
    return round(target_ / step_) * step_


# c_MA_20 = c.apply(lambda col: ta.SMA(col,20)).astype(np.float64)
def normalize_shares(shares_: pd.DataFrame) -> pd.DataFrame:
    shares_ = clean_target(shares_)
    shares_ = (round(round(shares_ / 100.0) * 100.0, 0)).astype(np.float64)
    shares_.replace(np.nan, 0, inplace=True)
    return shares_


def ROCold(target_: pd.DataFrame, period_: int) -> pd.DataFrame:
    period_ = int(period_)
    data_exist = (target_.isnull() == False).astype(np.float64).replace(0, np.nan)
    data_exist = data_exist * (data_exist.shift(period_))
    target_ = clean_target(target_)
    ROC_ = target_.apply(lambda col: ta.ROC(col, period_)).astype(np.float64)
    ROC_.replace(np.inf, 0, inplace=True)
    ROC_ = ROC_ * data_exist
    return ROC_


def loge(target_: pd.DataFrame) -> pd.DataFrame:
    data_exist = (target_.isnull() == False).astype(np.float64).replace(0, np.nan)
    target_ = clean_target(target_)
    LN_ = target_.apply(lambda col: ta.LN(col)).astype(np.float64)
    LN_.replace(np.inf, 0, inplace=True)
    LN_ = LN_ * data_exist
    return LN_


def log10(target_: pd.DataFrame) -> pd.DataFrame:
    data_exist = (target_.isnull() == False).astype(np.float64).replace(0, np.nan)
    target_ = clean_target(target_)
    LOG10_ = target_.apply(lambda col: ta.LOG10(col)).astype(np.float64)
    LOG10_.replace(np.inf, 0, inplace=True)
    LOG10_ = LOG10_ * data_exist
    return LOG10_


def unlog(input_, base):
    base_df = init_df(input_, base)
    return base_df.pow(input_)


def ADX(
    h_: pd.DataFrame, l_: pd.DataFrame, c_: pd.DataFrame, period_: int
) -> pd.DataFrame:
    period_ = int(period_)
    data_exist = (c_.isnull() == False).astype(np.float64).replace(0, np.nan)
    data_exist = data_exist * (data_exist.shift(20 * period_ + 5))
    h_ = clean_target(h_)
    l_ = clean_target(l_)
    c_ = clean_target(c_)
    ADX_ = pd.DataFrame(index=h_.index, columns=h_.columns, data=0)
    ADX_ = ADX_.apply(
        lambda col: ta.ADX(h_[col.name], l_[col.name], c_[col.name], period_)
    ).astype(np.float64)
    ADX_.replace(np.inf, 0, inplace=True)
    ADX_.replace(np.nan, 0, inplace=True)
    ADX_ = ADX_ * data_exist
    return ADX_


def PDI(
    h_: pd.DataFrame, l_: pd.DataFrame, c_: pd.DataFrame, period_: int
) -> pd.DataFrame:
    period_ = int(period_)
    data_exist = (c_.isnull() == False).astype(np.float64).replace(0, np.nan)
    data_exist = data_exist * (data_exist.shift(6 * period_ + 5))
    h_ = clean_target(h_)
    l_ = clean_target(l_)
    c_ = clean_target(c_)
    PDI_ = pd.DataFrame(index=h_.index, columns=h_.columns, data=0)
    PDI_ = PDI_.apply(
        lambda col: ta.PLUS_DI(h_[col.name], l_[col.name], c_[col.name], period_)
    ).astype(np.float64)
    PDI_.replace(np.inf, 0, inplace=True)
    PDI_ = PDI_ * data_exist
    return PDI_


def bollinger_band(
    target_: pd.DataFrame, period_: int, sd_up_, sd_down_: float
) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    period_ = int(period_)
    data_exist = (target_.isnull() == False).astype(np.float64).replace(0, np.nan)
    data_exist = data_exist * (data_exist.shift(period_))
    target_ = clean_target(target_)
    upper = target_.apply(
        lambda col: ta.BBANDS(col.astype(np.float64), period_, sd_up_, sd_down_, 0)[0]
    ).astype(np.float64)
    middle = target_.apply(
        lambda col: ta.BBANDS(col.astype(np.float64), period_, sd_up_, sd_down_, 0)[1]
    ).astype(np.float64)
    lower = target_.apply(
        lambda col: ta.BBANDS(col.astype(np.float64), period_, sd_up_, sd_down_, 0)[2]
    ).astype(np.float64)
    upper = upper * data_exist
    middle = middle * data_exist
    lower = lower * data_exist
    return upper, middle, lower


def MDI(
    h_: pd.DataFrame, l_: pd.DataFrame, c_: pd.DataFrame, period_: int
) -> pd.DataFrame:
    period_ = int(period_)
    data_exist = (c_.isnull() == False).astype(np.float64).replace(0, np.nan)
    data_exist = data_exist * (data_exist.shift(6 * period_ + 5))
    h_ = clean_target(h_)
    l_ = clean_target(l_)
    c_ = clean_target(c_)
    MDI_ = pd.DataFrame(index=h_.index, columns=h_.columns, data=0)
    MDI_ = MDI_.apply(
        lambda col: ta.MINUS_DI(h_[col.name], l_[col.name], c_[col.name], period_)
    ).astype(np.float64)
    MDI_.replace(np.inf, 0, inplace=True)
    MDI_ = MDI_ * data_exist
    return MDI_


def MACDold(
    target_: pd.DataFrame, fast_period_: int, slow_period_: int
) -> pd.DataFrame:
    fast_period_ = int(fast_period_)
    slow_period_ = int(slow_period_)
    data_exist = (target_.isnull() == False).astype(np.float64).replace(0, np.nan)
    data_exist = data_exist * (data_exist.shift(slow_period_))
    target_ = clean_target(target_)
    MACD_ = target_.apply(
        lambda col: ta.MACD(col, fast_period_, slow_period_, 9)[0]
    ).astype(np.float64)
    MACD_.replace(np.inf, 0, inplace=True)
    MACD_ = MACD_ * data_exist
    return MACD_


def MFI(
    h_: pd.DataFrame, l_: pd.DataFrame, c_: pd.DataFrame, v_: pd.DataFrame, period_: int
) -> pd.DataFrame:
    period_ = int(period_)
    data_exist = (c_.isnull() == False).astype(np.float64).replace(0, np.nan)
    data_exist = data_exist * (data_exist.shift(period_))
    h_ = clean_target(h_)
    l_ = clean_target(l_)
    c_ = clean_target(c_)
    v_ = clean_target(v_)
    MFI_ = pd.DataFrame(index=h_.index, columns=h_.columns, data=0)
    MFI_ = MFI_.apply(
        lambda col: ta.MFI(
            h_[col.name], l_[col.name], c_[col.name], v_[col.name], period_
        )
    ).astype(np.float64)
    MFI_.replace(np.inf, 0, inplace=True)
    MFI_.replace(np.nan, 0, inplace=True)
    MFI_ = MFI_ * data_exist
    return MFI_


def RSIold(target_: pd.DataFrame, period_: int) -> pd.DataFrame:
    period_ = int(period_)
    target_ = clean_target(target_)
    data_exist = (target_.isnull() == False).astype(np.float64).replace(0, np.nan)
    data_exist = data_exist * (data_exist.shift(period_))
    RSI_ = target_.apply(lambda col: ta.RSI(col, period_)).astype(np.float64)
    RSI_.replace(np.inf, 0, inplace=True)
    RSI_.replace(np.nan, 0, inplace=True)
    RSI_ = RSI_ * data_exist
    return RSI_


def perK(
    h_: pd.DataFrame,
    l_: pd.DataFrame,
    target_: pd.DataFrame,
    range_: int,
    SMA_smooth: int,
) -> pd.DataFrame:
    range_ = int(range_)
    SMA_smooth = int(SMA_smooth)
    h_ = clean_target(h_)
    l_ = clean_target(l_)
    data_exist = (target_.isnull() == False).astype(np.float64).replace(0, np.nan)
    data_exist = data_exist * (data_exist.shift(range_ + SMA_smooth))
    target_ = clean_target(target_)
    perK_ = pd.DataFrame(index=h_.index, columns=h_.columns, data=0)
    perK_ = perK_.apply(
        lambda col: ta.STOCH(
            h_[col.name], l_[col.name], target_[col.name], range_, SMA_smooth, 0, 3, 0
        )[0]
    ).astype(np.float64)
    perK_.replace(np.inf, 0, inplace=True)
    perK_.replace(np.nan, 0, inplace=True)
    perK_ = perK_ * data_exist
    return perK_


def HHV(target_: pd.DataFrame, period_: int) -> pd.DataFrame:
    period_ = int(period_)
    data_exist = (target_.isnull() == False).astype(np.float64).replace(0, np.nan)
    target_ = clean_target(target_)
    HHV_ = target_.apply(lambda col: ta.MAX(col, period_)).astype(np.float64)
    HHV_.replace(np.inf, 0, inplace=True)
    HHV_.replace(np.nan, 0, inplace=True)
    HHV_ = HHV_ * data_exist
    return HHV_


def prev_HHV(target_: pd.DataFrame, period_: int) -> pd.DataFrame:
    period_ = int(period_)
    return HHV(target_, period_).shift()


def HHVbars(target_: pd.DataFrame, period_: int) -> pd.DataFrame:
    period_ = int(period_)
    data_exist = (target_.isnull() == False).astype(np.float64).replace(0, np.nan)
    target_ = clean_target(target_)
    HHV_index = pd.DataFrame(index=target_.index, columns=target_.columns, data=0)
    HHV_index = target_.apply(lambda col: ta.MAXINDEX(col, period_)).astype(np.float64)
    index_np = np.arange(target_.index.size)
    index_df = pd.DataFrame(index=target_.index, columns=target_.columns, data=0)
    index_df = index_df.apply(lambda col: index_np)
    HHVbars_ = pd.DataFrame(index=target_.index, columns=target_.columns, data=0)
    HHVbars_ = (index_df - HHV_index) * data_exist
    return HHVbars_


def LLV(target_: pd.DataFrame, period_: int) -> pd.DataFrame:
    period_ = int(period_)
    data_exist = (target_.isnull() == False).astype(np.float64).replace(0, np.nan)
    target_ = clean_target(target_)
    LLV_ = target_.apply(lambda col: ta.MIN(col, period_)).astype(np.float64)
    LLV_.replace(np.inf, 0, inplace=True)
    LLV_.replace(np.nan, 0, inplace=True)
    LLV_ = LLV_ * data_exist
    return LLV_


def prev_LLV(target_: pd.DataFrame, period_: int) -> pd.DataFrame:
    period_ = int(period_)
    return LLV(target_, period_).shift()


def LLVbars(target_: pd.DataFrame, period_: int) -> pd.DataFrame:
    period_ = int(period_)
    data_exist = (target_.isnull() == False).astype(np.float64).replace(0, np.nan)
    target_.bfill(inplace=True)
    target_ = clean_target(target_)
    LLV_index = pd.DataFrame(index=target_.index, columns=target_.columns, data=0)
    LLV_index = target_.apply(lambda col: ta.MININDEX(col, period_)).astype(np.float64)
    index_np = np.arange(target_.index.size)
    index_df = pd.DataFrame(index=target_.index, columns=target_.columns, data=0)
    index_df = index_df.apply(lambda col: index_np)
    LLVbars_ = pd.DataFrame(index=target_.index, columns=target_.columns, data=0)
    LLVbars_ = (index_df - LLV_index) * data_exist
    return LLVbars_


def BarsSince_latest_trg(condition: pd.DataFrame) -> pd.DataFrame:
    condition = clean_target(condition)
    index_np = np.arange(condition.index.size)
    index_df = pd.DataFrame(index=condition.index, columns=condition.columns, data=0)
    index_df = index_df.apply(lambda col: index_np)

    BarsSince_latest_trg_index_ = pd.DataFrame(
        index=condition.index, columns=condition.columns, data=np.nan
    )

    BarsSince_latest_trg_index_[(condition == True)] = 1.0

    BarsSince_latest_trg_index_ = BarsSince_latest_trg_index_ * index_df

    BarsSince_latest_trg_index_.ffill(inplace=True)
    BarsSince_latest_trg_index_.fillna(value=0, inplace=True)

    BarsSince_latest_trg_ = index_df - BarsSince_latest_trg_index_

    return BarsSince_latest_trg_


def BarsSince(condition: pd.DataFrame) -> pd.DataFrame:
    return BarsSince_latest_trg(condition)


def BarsSince_first_trg(condition: pd.DataFrame) -> pd.DataFrame:
    condition = clean_target(condition)
    index_np = np.arange(condition.index.size)
    index_df = pd.DataFrame(index=condition.index, columns=condition.columns, data=0)
    index_df = index_df.apply(lambda col: index_np)

    BarsSince_first_trg_index_ = pd.DataFrame(
        index=condition.index, columns=condition.columns, data=np.nan
    )

    BarsSince_first_trg_index_[
        (condition == True) & (condition.shift().replace(np.nan, False) == False)
    ] = 1.0

    BarsSince_first_trg_index_ = BarsSince_first_trg_index_ * index_df

    BarsSince_first_trg_index_.ffill(inplace=True)
    BarsSince_first_trg_index_.fillna(value=0, inplace=True)

    BarsSince_first_trg_ = index_df - BarsSince_first_trg_index_

    return BarsSince_first_trg_


def BarsSince_since_first_buy_to_sell(
    BUY_: pd.DataFrame, SELL_: pd.DataFrame
) -> pd.DataFrame:

    BUY_SELL_one_zero = one_zero_gen_frm_BUYSELL(BUY_, SELL_)

    index_np = np.arange(BUY_SELL_one_zero.index.size)
    index_df = pd.DataFrame(
        index=BUY_SELL_one_zero.index, columns=BUY_SELL_one_zero.columns, data=0
    )
    index_df = index_df.apply(lambda col: index_np)

    BarsSince_first_trg_index_ = pd.DataFrame(
        index=BUY_SELL_one_zero.index, columns=BUY_SELL_one_zero.columns, data=np.nan
    )

    BarsSince_first_trg_index_[
        (BUY_SELL_one_zero == 1) & (BUY_SELL_one_zero.shift() == 0)
    ] = 1.0
    BarsSince_first_trg_index_[
        (BUY_SELL_one_zero == 0) & (BUY_SELL_one_zero.shift() == 1)
    ] = 0.0

    BarsSince_first_trg_index_ = BarsSince_first_trg_index_ * index_df

    BarsSince_first_trg_index_.ffill(inplace=True)
    BarsSince_first_trg_index_.fillna(value=0, inplace=True)

    BarsSince_first_trg_ = (
        index_df - BarsSince_first_trg_index_ + 1
    )  # avoid bug dsc_oldest_ranking_DSC_within_universe

    return BarsSince_first_trg_


def mean_trim_within_universe(
    target_: pd.DataFrame, universe: pd.DataFrame, trim_: float = 0.10
) -> pd.DataFrame:

    from scipy import stats

    target_within_universe = target_ * universe

    mean_ = target_within_universe.apply(
        lambda row: (stats.trim_mean(row[~np.isnan(row.astype(np.float64))], trim_)),
        axis=1,
    )
    mean_within_universe_ = pd.DataFrame(index=mean_.index, columns=target_.columns)
    mean_within_universe_ = mean_within_universe_.apply(lambda col: mean_.values).ffill(
        limit=20
    )
    # mean_within_universe_ = mean_within_universe_*universe
    return mean_within_universe_.astype(np.float64)


def mean_within_universe(target_: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    target_within_universe = target_ * universe

    mean_ = target_within_universe.apply(
        lambda row: (row[~np.isnan(row.astype(np.float64))].mean()), axis=1
    )
    mean_within_universe_ = pd.DataFrame(index=mean_.index, columns=target_.columns)
    mean_within_universe_ = mean_within_universe_.apply(lambda col: mean_.values).ffill(
        limit=20
    )
    # mean_within_universe_ = mean_within_universe_*universe
    return mean_within_universe_.astype(np.float64)


def remove_odds_from_row(target_: pd.Series, trim_=0.025) -> pd.Series:
    winsor_pecent_ = float(trim_)

    target_dict_ = target_.dropna().to_dict()
    winsor_num = int(len(target_dict_) * winsor_pecent_)
    top = sorted(target_dict_, key=target_dict_.get, reverse=True)[: (winsor_num + 1)]
    bot = sorted(target_dict_, key=target_dict_.get, reverse=False)[: (winsor_num + 1)]
    for symbol in top:
        target_dict_[symbol] = np.nan
    for symbol in bot:
        target_dict_[symbol] = np.nan
    output = pd.Series(data=target_dict_, index=target_.index)
    # output = clean_target(output)
    return output.astype(np.float64)


def universe_excluding_odds(
    score_: pd.DataFrame, universe: pd.DataFrame, trim_=0.025
) -> pd.DataFrame:
    universe.replace(0, np.nan, inplace=True)
    score_within_universe = score_ * universe
    odd_removed_universe = score_within_universe.apply(
        lambda row: remove_odds_from_row(row, trim_), axis=1
    )
    odd_removed_universe = data_exist_1_nan(odd_removed_universe)
    return odd_removed_universe.astype(np.float64)


def winsorizing(target_: pd.Series, winsor_pecent_=0.025) -> pd.Series:
    winsor_pecent_ = float(winsor_pecent_)

    target_dict_ = target_.dropna().to_dict()
    winsor_num = int(len(target_dict_) * winsor_pecent_)
    top = sorted(target_dict_, key=target_dict_.get, reverse=True)[: (winsor_num + 1)]
    bot = sorted(target_dict_, key=target_dict_.get, reverse=False)[: (winsor_num + 1)]
    for symbol in top:
        target_dict_[symbol] = target_dict_[top[-1]]
    for symbol in bot:
        target_dict_[symbol] = target_dict_[bot[-1]]
    output = pd.Series(data=target_dict_, index=target_.index)
    # output = clean_target(output)
    return output.astype(np.float64)


def winsorizing_within_universe(
    target_: pd.DataFrame, universe: pd.DataFrame, winsor_pecent_=0.025
) -> pd.DataFrame:
    universe.replace(0, np.nan, inplace=True)
    target_within_universe = target_ * universe
    winsorizing_ = target_within_universe.apply(
        lambda row: winsorizing(row, winsor_pecent_), axis=1
    )
    return winsorizing_.astype(np.float64)


def sd_within_universe(target_: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    # Sample standard deviation : divided by n - 1
    target_within_universe = target_ * universe
    sd_ = target_within_universe.apply(
        lambda row: (row[~np.isnan(row.astype(np.float64))].std()), axis=1
    )
    sd_within_universe_ = pd.DataFrame(index=sd_.index, columns=target_.columns)
    sd_within_universe_ = (
        sd_within_universe_.apply(lambda col: sd_.values).ffill().fillna(0)
    )
    # sd_within_universe_ = sd_within_universe_*universe
    return sd_within_universe_.astype(np.float64)


def z_score_within_universe(
    target_: pd.DataFrame, universe: pd.DataFrame
) -> pd.DataFrame:
    mean_ = mean_within_universe(target_, universe)
    sd_ = sd_within_universe(target_, universe)
    z_score = ((target_ - mean_) / sd_) * universe
    z_score.ffill(limit=5, inplace=True)
    return z_score.astype(np.float64)


def stability_within_universe(
    target_: pd.DataFrame, universe: pd.DataFrame
) -> pd.DataFrame:
    sd = sd_within_universe(target_, universe)
    sd = clean_target(sd)
    mean = mean_within_universe(target_, universe)
    mean = clean_target(mean)
    output = clean_target(mean / sd)
    output = output * universe
    return output.astype(np.float64)


def data_exist_1_nan(target_: pd.DataFrame):
    return (target_.isnull() == False).astype(np.float64).replace(0, np.nan)


def stability(target_: pd.DataFrame, period_: int) -> pd.DataFrame:

    period_ = int(period_)
    data_exist = (target_.isnull() == False).astype(np.float64).replace(0, np.nan)
    data_exist = data_exist * (data_exist.shift(period_).ffill())

    sd = STDDEV(target_, period_)
    sd = clean_target(sd)
    mean = MA(target_, period_)
    mean = clean_target(mean)
    output = clean_target(mean / sd)
    output = output * data_exist
    return output


######################


def ATRold(
    h_: pd.DataFrame, l_: pd.DataFrame, c_: pd.DataFrame, period_: int
) -> pd.DataFrame:
    period_ = int(period_)
    h_ = clean_target(h_)
    l_ = clean_target(l_)
    c_ = clean_target(c_)
    data_exist = (c_.isnull() == False).astype(np.float64).replace(0, np.nan)
    data_exist = data_exist * (data_exist.shift(2 * period_))
    ATR_ = pd.DataFrame(index=h_.index, columns=h_.columns, data=0)
    ATR_ = ATR_.apply(
        lambda col: ta.ATR(h_[col.name], l_[col.name], c_[col.name], period_)
    ).astype(np.float64)
    ATR_.replace(np.inf, 0, inplace=True)
    ATR_ = ATR_ * data_exist
    return ATR_


# def clean_shooting_dropping(target_, mul_, period_ATR_, period_MA_): :
#     target_ = clean_target(target_)
#     mean_ = MA(target_, period_MA_).shift()
#     KValue = mul_*ATR(target_, target_, target_, period_ATR_)
#     Ktop = mean_ + KValue
#     Kbot = mean_ - KValue
#     shooting = target_ > Ktop
#     dropping = target_ < Kbot
#     clean_shooting_dropping_ = target_.copy()
#     clean_shooting_dropping_[shooting] = Ktop
#     clean_shooting_dropping_[dropping] = Kbot
#     clean_shooting_dropping_.replace(np.nan, 0, inplace = True)
#     clean_shooting_dropping_.replace(np.inf, 0, inplace = True)
#     return clean_shooting_dropping_


def clean_shooting_dropping(
    target_: pd.DataFrame, mul_: float, period_ATR_: int, period_MA_: int
) -> pd.DataFrame:
    period_ATR_ = int(period_ATR_)
    period_MA_ = int(period_MA_)
    target_ = clean_target(target_)
    mean_ = MA(target_, period_MA_).shift()
    KValue = mul_ * ATR(target_, target_, target_, period_ATR_)
    Ktop = mean_ + KValue
    Kbot = mean_ - KValue
    shooting = target_ > Ktop
    dropping = target_ < Kbot
    clean_shooting_dropping_ = target_.copy()
    clean_shooting_dropping_[shooting] = Ktop
    clean_shooting_dropping_[dropping] = Kbot
    clean_shooting_dropping_.replace(np.nan, 0, inplace=True)
    clean_shooting_dropping_.replace(np.inf, 0, inplace=True)
    return clean_shooting_dropping_


def delay_out(target_: pd.DataFrame, step_: float, period_: int) -> pd.DataFrame:
    period_ = int(period_)
    target_ = clean_target(target_)
    target_ = MA(target_, period_) / step_
    target_ = target_.apply(np.ceil) * step_
    return target_.astype(np.float64)


def delay_out_new(target_: pd.DataFrame, step_: float, period_: int) -> pd.DataFrame:
    period_ = int(period_)
    target_ = clean_target(target_)
    target_slow = MA(target_, period_) / step_
    target_fast = WMA(target_, period_) / step_
    target_ = target_.apply(
        lambda col: np.maximum(target_slow[col.name], target_fast[col.name])
    )
    target_ = target_.apply(np.ceil) * step_
    return target_.astype(np.float64)


def dsc_oldest_ranking_DSC_within_universe(
    target_: pd.DataFrame,
    universe: pd.DataFrame,
    buy_top_largest: int,
    sell_drop_from_top: int,
    weight_recent_ranking=0.11,
) -> pd.DataFrame:
    # buy top smallest

    target_dsc_ranking = ranking_DSC_within_universe(target_, universe)
    BUY = (target_dsc_ranking <= buy_top_largest) & (
        target_dsc_ranking.shift() > buy_top_largest
    )
    SELL = (target_dsc_ranking >= sell_drop_from_top) & (
        target_dsc_ranking.shift() < sell_drop_from_top
    )
    universe_frm_ranking = one_zero_gen_frm_BUYSELL(BUY, SELL) * universe
    universe_frm_ranking.replace(0, np.nan, inplace=True)
    bars_since_buy = BarsSince_since_first_buy_to_sell(BUY, SELL) * universe_frm_ranking
    # bars_since_buy= BarsSince(BUY)*universe_frm_ranking
    bars_since_buy_ranking_dsc = ranking_DSC_within_universe(
        bars_since_buy, universe_frm_ranking
    ).astype(np.float64)
    # include the following thing to avoid same ranking number
    weight_recent_ranking = weight_recent_ranking + 0.001
    adj_bars_since_buy_ranking_val_asc = (
        bars_since_buy_ranking_dsc + weight_recent_ranking * target_dsc_ranking
    )
    adj_bars_since_buy_ranking_asc = ranking_ASC_within_universe(
        adj_bars_since_buy_ranking_val_asc, universe_frm_ranking
    ).astype(np.float64)
    # adj_bars_since_buy_ranking_asc = adj_bars_since_buy_ranking_asc[adj_bars_since_buy_ranking_asc<=buy_top_largest]
    return adj_bars_since_buy_ranking_asc


def sig_ranking_DSC_within_universe(
    target_: pd.DataFrame,
    universe: pd.DataFrame,
    buy_top_largest: int,
    sell_drop_from_top: int,
    weight_recent_ranking=0.11,
) -> pd.DataFrame:

    dsc_oldest_ranking_DSC_within_universe_ = dsc_oldest_ranking_DSC_within_universe(
        target_, universe, buy_top_largest, sell_drop_from_top, weight_recent_ranking
    )

    adj_bars_since_buy_ranking_asc = dsc_oldest_ranking_DSC_within_universe_[
        dsc_oldest_ranking_DSC_within_universe_ <= buy_top_largest
    ]

    return adj_bars_since_buy_ranking_asc


def dsc_oldest_ranking_ASC_within_universe(
    target_: pd.DataFrame,
    universe: pd.DataFrame,
    buy_top_smallest: int,
    sell_drop_from_top: int,
    weight_recent_ranking=0.01,
) -> pd.DataFrame:
    # buy top smallest

    target_asc_ranking = ranking_ASC_within_universe(target_, universe)
    BUY = (target_asc_ranking <= buy_top_smallest) & (
        target_asc_ranking.shift() > buy_top_smallest
    )
    SELL = (target_asc_ranking >= sell_drop_from_top) & (
        target_asc_ranking.shift() < sell_drop_from_top
    )
    universe_frm_ranking = one_zero_gen_frm_BUYSELL(BUY, SELL) * universe
    universe_frm_ranking.replace(0, np.nan, inplace=True)

    bars_since_buy = BarsSince_since_first_buy_to_sell(BUY, SELL) * universe_frm_ranking
    # bars_since_buy= BarsSince(BUY)*universe_frm_ranking
    bars_since_buy_ranking_dsc = ranking_DSC_within_universe(
        bars_since_buy, universe_frm_ranking
    ).astype(np.float64)
    # include the following thing to avoid same ranking number
    weight_recent_ranking = weight_recent_ranking + 0.001
    adj_bars_since_buy_ranking_val_asc = (
        bars_since_buy_ranking_dsc + weight_recent_ranking * target_asc_ranking
    )
    adj_bars_since_buy_ranking_asc = ranking_ASC_within_universe(
        adj_bars_since_buy_ranking_val_asc, universe_frm_ranking
    ).astype(np.float64)
    # adj_bars_since_buy_ranking_asc = adj_bars_since_buy_ranking_asc[adj_bars_since_buy_ranking_asc<=buy_top_smallest]
    return adj_bars_since_buy_ranking_asc


def sig_ranking_ASC_within_universe(
    target_: pd.DataFrame,
    universe: pd.DataFrame,
    buy_top_smallest: int,
    sell_drop_from_top: int,
    weight_recent_ranking=0.11,
) -> pd.DataFrame:
    # buy top smallest
    dsc_oldest_ranking_ASC_within_universe_ = dsc_oldest_ranking_ASC_within_universe(
        target_, universe, buy_top_smallest, sell_drop_from_top, weight_recent_ranking
    )

    adj_bars_since_buy_ranking_asc = dsc_oldest_ranking_ASC_within_universe_[
        dsc_oldest_ranking_ASC_within_universe_ <= buy_top_smallest
    ]

    return adj_bars_since_buy_ranking_asc


def normalize_row_by_perK(
    target_: pd.DataFrame, universe_: pd.DataFrame
) -> pd.DataFrame:
    exist = data_exist_1_nan(target_)
    target_ = target_ * universe_
    # target_ = clean_target(target_)
    LLV_row = target_.apply(lambda row: row.min(), axis=1)
    HHV_row = target_.apply(lambda row: row.max(), axis=1)

    LLV_df = series_to_df(LLV_row, target_)

    HHV_df = series_to_df(HHV_row, target_)
    output = (target_ - LLV_df) / (HHV_df - LLV_df)
    output[
        (HHV_df - LLV_df) == init_df(target_, 0)
    ] = 1.0  # case only one existing value in a row
    output[(HHV_df - LLV_df) == init_df(target_, np.nan)] = 1.0
    output[
        (HHV_df == init_df(target_, np.nan)) | (LLV_df == init_df(target_, np.nan))
    ] = 0.00
    output[(HHV_df == init_df(target_, 0.0)) & (LLV_df == init_df(target_, 0.0))] = 0.00

    output = output * exist * universe_
    return output.replace(np.nan, 0)


def newMA(target_: pd.DataFrame, period_: int) -> pd.DataFrame:
    period_ = int(period_)
    data_exist = (target_.isnull() == False).astype(np.float64).replace(0, np.nan)
    data_exist = data_exist * (data_exist.shift(period_))
    target_np = np.array(clean_target(target_)).astype(np.float64)
    output_np = np.apply_along_axis(ta.SMA, 0, target_np, period_)
    output_df = pd.DataFrame(output_np, index=target_.index, columns=target_.columns)
    return output_df * data_exist


def convert_weight_to_share(
    weight_: pd.DataFrame, price_: pd.DataFrame, deposit_: float
) -> pd.DataFrame:
    target_weight = weight_.astype(np.float64)
    target_shares = target_weight / price_ * deposit_
    # target_shares = target_weight*100
    return normalize_shares(target_shares)


def txn_reduction__change_share_only_when_weight_change(
    shares_: pd.DataFrame, weight_: pd.DataFrame, allow_thres_: float
) -> pd.DataFrame:

    weight_change_bool = init_df(weight_, np.nan)

    weight_change_bool[(weight_ - weight_.shift()).abs() > allow_thres_] = 1.0

    adjusted_shares = weight_change_bool * shares_

    adjusted_shares.ffill(inplace=True)

    adjusted_shares = clean_target(adjusted_shares)

    return adjusted_shares.astype(np.int32)


def txn_reduction__change_share_only_new_sig(
    shares_: pd.DataFrame, weight_: pd.DataFrame
) -> pd.DataFrame:

    weight_change_bool = init_df(weight_, np.nan)

    weight_change_bool[(weight_ > 0) & (weight_.shift() == 0)] = 1.0
    weight_change_bool[(weight_ == 0) & (weight_.shift() > 0)] = 1.0

    adjusted_shares = weight_change_bool * shares_

    adjusted_shares.ffill(inplace=True)

    adjusted_shares = clean_target(adjusted_shares)

    return adjusted_shares  # adjusted_shares.astype(np.int32)


# def cap_maximal_weight(weight_: pd.DataFrame, max_weight_ = 0.10) -> pd.DataFrame:
#     max_df = init_df(weight_, max_weight_)
#     return weight_.apply(lambda col: np.minimum(col, max_df[col.name])).astype(np.float64)


def cap_maximal_value(target_: pd.DataFrame, max_value_=0.10) -> pd.DataFrame:
    max_df = init_df(target_, max_value_)
    return target_.apply(lambda col: np.minimum(col, max_df[col.name])).astype(
        np.float64
    )


def cap_minimal_value(target_: pd.DataFrame, min_value_=-0.10) -> pd.DataFrame:
    min_df = init_df(target_, min_value_)
    return target_.apply(lambda col: np.maximum(col, min_df[col.name])).astype(
        np.float64
    )


def count_existing_value_in_a_row(target_: pd.DataFrame) -> pd.Series:
    return (target_ / target_).sum(axis=1)


def AND_dict(input_: dict):
    values_view = input_.values()
    value_iterator = iter(values_view)
    sample_df = next(value_iterator)
    output_ = init_df(sample_df, True)
    for key, value in input_.items():
        output_ = output_ & value
    return output_


def OR_dict(input_: dict):
    values_view = input_.values()
    value_iterator = iter(values_view)
    sample_df = next(value_iterator)
    output_ = init_df(sample_df, False)
    for key, value in input_.items():
        output_ = output_ | value
    return output_


def allow_only_weight_decline(weight_: pd.DataFrame) -> pd.DataFrame:
    weight_ = weight_.round(3)
    original_weight = init_df(weight_, np.nan)
    original_weight[(weight_ > 0.000) & (weight_.shift() == 0.000)] = 1.000
    original_weight[(weight_ == 0.000) & (weight_.shift() > 0.000)] = 0.000
    original_weight = original_weight.round(0)
    original_weight = original_weight * weight_
    original_weight.ffill(inplace=True)
    original_weight = original_weight.round(3)
    output_ = init_df(weight_, np.nan)
    output_[(weight_ > 0.000) & (weight_.shift() == 0.000)] = 1.000
    output_[(weight_ < original_weight) & (weight_ < weight_.shift())] = 1.000
    output_[weight_ == 0.000] = 0.000
    output_ = output_.round(3)
    output_ = output_ * weight_
    output_.ffill(inplace=True)
    return output_


def allow_only_shares_decline(
    shares_: pd.DataFrame, percent_chg_from_original=10, percent_chg_from_prev=5
) -> pd.DataFrame:
    shares_.ffill(inplace=True)
    shares_ = shares_.round(0)
    original_shares = init_df(shares_, np.nan)
    original_shares[(shares_ > int(0)) & (shares_.shift() == int(0))] = int(1)
    original_shares[(shares_ == int(0)) & (shares_.shift() > int(0))] = int(0)
    original_shares = original_shares.round(0)
    original_shares = original_shares * shares_
    original_shares.ffill(inplace=True)
    original_shares = original_shares.round(0)
    output_ = init_df(shares_, np.nan)
    output_[(shares_ > int(0)) & (shares_.shift() == int(0))] = int(1)
    output_[
        (shares_ < original_shares)
        & (shares_ / shares_.shift() - 1.0 < -1.0 * percent_chg_from_prev / 100)
        & (shares_ / original_shares - 1.0 < -1.0 * percent_chg_from_original / 100.0)
    ] = int(1)
    # output_[(shares_  < original_shares)&(shares_/shares_.shift() - 1.0 < -1.0*percent_chg_from_original )] = int(1)
    output_[shares_ == int(0)] = int(0)
    output_ = output_.round(0)
    output_ = output_ * shares_
    output_.ffill(inplace=True)
    return output_


def even_better_sinewave(target_: pd.DataFrame, duration_=20):
    import math

    exist = data_exist_1_nan(target_)
    target_ = target_.ffill().bfill()
    target_np = np.array(target_)
    HP = target_np.copy()
    Filt = HP.copy()
    Wave = Filt.copy()
    Pwr = Filt.copy()

    PI = math.pi
    alpha1 = (1 - math.sin(2 * PI / duration_)) / math.cos(2 * PI / duration_)
    a1 = math.exp(-1.414 * PI / 10)
    b1 = 2 * a1 * math.cos(1.414 * PI / 10)
    c2 = b1
    c3 = -a1 * a1
    c1 = 1 - c2 - c3

    for row_i in range(2, target_.shape[0]):

        HP[row_i, :] = (
            0.5 * (1 + alpha1) * (target_np[row_i, :] - target_np[row_i - 1, :])
            + alpha1 * HP[row_i - 1, :]
        )

        Filt[row_i, :] = (
            c1 * (HP[row_i, :] + HP[row_i - 1, :]) / 2
            + c2 * Filt[row_i - 1, :]
            + c3 * Filt[row_i - 2, :]
        )

    Filt = pd.DataFrame(index=target_.index, columns=target_.columns, data=Filt)

    Wave = (Filt + Filt.shift() + Filt.shift(2)) / 3
    Pwr = (
        Filt * Filt + Filt.shift() * Filt.shift() + Filt.shift(2) * Filt.shift(2)
    ) / 3

    Wave = Wave / SQRT(Pwr)

    return Wave * exist


def TN_v(c_: pd.DataFrame, v_: pd.DataFrame, period_=240):
    period_ = int(period_)
    TN_v_raw = init_df(c_, -v_)
    TN_v_raw[c_ > c_.shift()] = v_
    return MA(SUM(TN_v_raw.astype(np.float64), period_), 7)


def NT_v_PLUS(c_: pd.DataFrame, v_: pd.DataFrame, period_=240):
    period_ = int(period_)
    TN_v_raw = init_df(c_, 0)
    TN_v_raw[c_ > c_.shift()] = v_
    return MA(SUM(TN_v_raw, period_).astype(np.float64), 7)


def NT_v_MINUS(c_: pd.DataFrame, v_: pd.DataFrame, period_=240):
    period_ = int(period_)
    TN_v_raw = init_df(c_, 0)
    TN_v_raw[c_ < c_.shift()] = v_
    return SUM(TN_v_raw, period_).astype(np.float64)


def TN_f(c_: pd.DataFrame, v_: pd.DataFrame, period_=240):

    TN_f_ = (c_ - VMA(c_, v_, 30)) * v_

    return MA(SUM(TN_f_, period_), 7)


def bool_extend_TRUE(target_: pd.DataFrame, period_=5):
    period_ = int(period_)
    target_.replace(np.nan, False)
    extended = init_df(target_, np.nan)
    extended = target_.copy()
    extended = extended.astype(np.float64)
    extended.replace(0, np.nan, inplace=True)
    extended.ffill(limit=period_, inplace=True)
    extended.replace(np.nan, 0, inplace=True)
    extended = extended.astype(np.bool)
    return extended


def display_comparison_2_axis(
    stock_name: str,
    target_01_: pd.Series,
    target_01_name_: str,
    target_02_: pd.Series,
    target_02_name_: str,
    time_from="2014-01-01",
    time_end="2021-01-01",
):
    """
    example: display_comparison_2_axis('AOT', c, 'close', MA(c, 10), 'moving average')
    """

    fig, ax = plt.subplots()
    # make a plot

    target_01_ = target_01_.copy().loc[:, stock_name]
    target_02_ = target_02_.copy().loc[:, stock_name]

    ax.plot(
        target_01_[time_from:time_end].index,
        target_01_[time_from:time_end].values,
        label=target_01_name_,
    )
    ax.legend(loc="upper left")

    # twin object for two different y-axis on the sample plot
    ax2 = ax.twinx()
    # make a plot with different y-axis using second axis object
    ax2.plot(
        target_02_[time_from:time_end].index,
        target_02_[time_from:time_end].values,
        color="red",
        label=target_02_name_,
    )  #  alpha=0.3
    # ax2.plot(drawdown_percent_NAV.index, drawdown_percent_NAV.values,color="orange", label = 'DD_NAV')
    ax2.legend(loc="upper right")
    fig.text(
        0.5,
        0.05,
        stock_name + ": " + target_01_name_ + " vs " + target_02_name_,
        ha="center",
        va="bottom",
    )


def display_comparison_2_axis_left_list(
    stock_name: str,
    target_01_list: list,
    target_01_name_: str,
    target_02_: pd.Series,
    target_02_name_: str,
    time_from="2014-01-01",
    time_end="2021-01-01",
):
    """
    example: display_comparison_2_axis_left_list('AOT', [MA(c, 5), MA(c, 10), MA(c, 20)],'ma team', c, 'close')
    """

    fig, ax = plt.subplots()
    # make a plot

    target_02_ = target_02_.copy().loc[:, stock_name]

    for item in target_01_list:
        item = item.copy().loc[:, stock_name]
        ax.plot(
            item[time_from:time_end].index,
            item[time_from:time_end].values,
            label=target_01_name_,
        )

    ax.legend(loc="upper left")

    # twin object for two different y-axis on the sample plot
    ax2 = ax.twinx()
    # make a plot with different y-axis using second axis object
    ax2.plot(
        target_02_[time_from:time_end].index,
        target_02_[time_from:time_end].values,
        color="red",
        label=target_02_name_,
    )  #  alpha=0.3
    # ax2.plot(drawdown_percent_NAV.index, drawdown_percent_NAV.values,color="orange", label = 'DD_NAV')
    ax2.legend(loc="upper right")

    fig.text(
        0.5,
        0.05,
        stock_name + ": " + target_01_name_ + " vs " + target_02_name_,
        ha="center",
        va="bottom",
    )


def df_pivot_ranking_by_alphabet(target_):

    #### example:
    # stably_strong_union_test = df_pivot_ranking_by_alphabet(stably_strong_union)
    target_ = target_.copy()
    target_melt = pd.melt(target_, value_name="rank", ignore_index=False)
    target_melt["index"] = target_melt.index

    target_melt.dropna(inplace=True)
    target_melt["rank"] = target_melt["rank"].astype(np.int32)
    target_melt_pivot = target_melt.pivot(
        values="Ticker", columns="rank", index="index"
    )
    target_melt_pivot_alphabet = target_melt_pivot.apply(
        lambda row: row.sort_values(), axis=1
    )
    return target_melt_pivot_alphabet


def df_pivot_ranking(target_):

    #### example:
    # stably_strong_union_test = df_pivot_ranking_by_alphabet(stably_strong_union)
    target_ = target_.copy()
    target_melt = pd.melt(target_, value_name="rank", ignore_index=False)
    target_melt["index"] = target_melt.index

    target_melt.dropna(inplace=True)
    target_melt["rank"] = target_melt["rank"].astype(np.int32)
    target_melt_pivot = target_melt.pivot(
        values="Ticker", columns="rank", index="index"
    )

    return target_melt_pivot


# def get_avg_all__avg_m__avg_p__var_m__var_p__es_m__es_p__max_m__max_p(target_: pd.DataFrame, period_ = 240, confident_percent = 0.05)-> (pd.DataFrame, pd.DataFrame):


#     # target_ = daily_return.copy()
#     # period_ = 240
#     # confident_percent_ = 0.05

#     avg_all_df_, avg_m_df_, avg_p_df_, var_m_df_, var_p_df_, es_m_df_, es_p_df_, max_m_df_, max_p_df_ = init_df(target_, np.nan) \
#                                                 ,init_df(target_, np.nan), init_df(target_, np.nan), init_df(target_, np.nan)\
#                                                 ,init_df(target_, np.nan), init_df(target_, np.nan), init_df(target_, np.nan)\
#                                                 ,init_df(target_, np.nan), init_df(target_, np.nan)


#     range_of_array = period_
#     stock_set = set(target_.columns)
#     stock_set_len = len(stock_set)
#     # 4.04 PM 4.27 PM ->23

#     for stock in stock_set:
#         target_stock_np = target_[stock].to_numpy()
#         col_index = target_.columns.get_loc(stock)
#         print(stock)

#         # print(col_index)
#         # print(target_stock_np)

#         for i in range(period_-1,len(target_),1):
#             if(np.isnan(target_stock_np[i]).sum() < int(1)):
#                 # print(i)
#                 # raise
#                 sliced_target_np = target_stock_np[i-(range_of_array-1):i+1]

#                 # print(len(sliced_target_np))
#                 if(len(sliced_target_np) == 0):
#                     print('zero found')
#                     raise
#                 # cal_avg_all__avg_m__avg_p__var_m__var_p__es_m__es_p__max_m__max_p_from_np(sliced_target_np, 0.02)
#                 # print(cal_avg_all__avg_m__avg_p__var_m__var_p__es_m__es_p__max_m__max_p_from_np(sliced_target_np, 0.02))

#                 avg_all_df_.iloc[i, col_index], avg_m_df_.iloc[i, col_index], avg_p_df_.iloc[i, col_index]\
#                     , var_m_df_.iloc[i, col_index], var_p_df_.iloc[i, col_index], es_m_df_.iloc[i, col_index], es_p_df_.iloc[i, col_index], max_m_df_.iloc[i, col_index], max_p_df_.iloc[i, col_index] \
#                         = cal_avg_all__avg_m__avg_p__var_m__var_p__es_m__es_p__max_m__max_p_from_np(sliced_target_np, 0.02)
#             else:
#                 # print('nan')
#                 avg_all_df_.iloc[i, col_index], avg_m_df_.iloc[i, col_index], avg_p_df_.iloc[i, col_index]\
#                     , var_m_df_.iloc[i, col_index], var_p_df_.iloc[i, col_index], es_m_df_.iloc[i, col_index], es_p_df_.iloc[i, col_index], max_m_df_.iloc[i, col_index], max_p_df_.iloc[i, col_index] \
#                         = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
#     return (avg_all_df_, avg_m_df_, avg_p_df_, var_m_df_, var_p_df_, es_m_df_, es_p_df_, max_m_df_, max_p_df_ )


def get_avg_all__avg_m__avg_p__var_m__var_p__es_m__es_p__max_m__max_p(
    target_: pd.DataFrame, period_=240, confident_percent=0.05
):

    target_np = target_.ffill(limit=5).to_numpy()
    period_ = int(period_)

    (
        avg_all_np,
        avg_m_np,
        avg_p_np,
        var_m_np,
        var_p_np,
        es_m_np,
        es_p_np,
        max_m_np,
        max_p_np,
    ) = (
        init_np(target_np, np.nan),
        init_np(target_np, np.nan),
        init_np(target_np, np.nan),
        init_np(target_np, np.nan),
        init_np(target_np, np.nan),
        init_np(target_np, np.nan),
        init_np(target_np, np.nan),
        init_np(target_np, np.nan),
        init_np(target_np, np.nan),
    )

    stock_num = target_np.shape[1]
    bar_num = target_np.shape[0]

    for stock in range(stock_num):
        target_of_a_stock = target_np[:, stock]
        print(stock)
        for row in range(period_ - 1, bar_num, 1):
            sliced_np = target_of_a_stock[row - (period_ - 1) : row + 1]
            # print(len(sliced_np))
            # sliced_target_np = price_of_a_stock[i-(range_of_array-1):i+1]
            if np.isnan(sliced_np).sum() < 1:
                (
                    avg_all_np[row, stock],
                    avg_m_np[row, stock],
                    avg_p_np[row, stock],
                    var_m_np[row, stock],
                    var_p_np[row, stock],
                    es_m_np[row, stock],
                    es_p_np[row, stock],
                    max_m_np[row, stock],
                    max_p_np[row, stock],
                ) = cal_avg_all__avg_m__avg_p__var_m__var_p__es_m__es_p__max_m__max_p_from_np(
                    sliced_np, confident_percent
                )

    (
        avg_all_df_,
        avg_m_df_,
        avg_p_df_,
        var_m_df_,
        var_p_df_,
        es_m_df_,
        es_p_df_,
        max_m_df_,
        max_p_df_,
    ) = (
        np_to_df(avg_all_np, target_),
        np_to_df(avg_m_np, target_),
        np_to_df(avg_p_np, target_),
        np_to_df(var_m_np, target_),
        np_to_df(var_p_np, target_),
        np_to_df(es_m_np, target_),
        np_to_df(es_p_np, target_),
        np_to_df(max_m_np, target_),
        np_to_df(max_p_np, target_),
    )

    return (
        avg_all_df_,
        avg_m_df_,
        avg_p_df_,
        var_m_df_,
        var_p_df_,
        es_m_df_,
        es_p_df_,
        max_m_df_,
        max_p_df_,
    )


def cal_avg_all__avg_m__avg_p__var_m__var_p__es_m__es_p__max_m__max_p_from_np(
    target_: np.array, confident_percent_=0.05
) -> (
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
):
    target_list = target_.tolist()
    # print(target_list)
    # print(type(target_list))
    n = len(target_list)
    # print(n)
    if n == 0:
        print("shit n is 0")
        raise

    tail = int(n * confident_percent_)
    # print(n)
    # print(confident_percent_)
    # print(tail)
    ret_p_ = []
    ret_m_ = []
    target_list_sorted = sorted(target_list)
    # print( sum(target_list_sorted))
    for r in target_list_sorted:
        if r > 0:
            ret_p_.append(r)
        if r < 0:
            ret_m_.append(r)
    avg_all_ = sum(target_list_sorted) / n
    # print( avg_all_)
    avg_m_ = sum(ret_m_) / n
    avg_p_ = sum(ret_p_) / n
    var_m_ = target_list_sorted[tail]
    var_p_ = target_list_sorted[-tail]
    es_m_ = sum(target_list_sorted[:tail]) / tail
    es_p_ = sum(target_list_sorted[-tail:]) / tail
    max_m_ = min(target_list_sorted)
    max_p_ = max(target_list_sorted)
    return (avg_all_, avg_m_, avg_p_, var_m_, var_p_, es_m_, es_p_, max_m_, max_p_)


def cal_avg_all__avg_m__avg_p__var_m__var_p__es_m__es_p__max_m__max_p_from_series(
    target_: pd.Series, confident_percent_=0.05
) -> (
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
):
    target_list = target_.to_list()
    n = len(target_list)
    tail = int(n * confident_percent_)
    ret_p_ = []
    ret_m_ = []
    target_list_sorted = sorted(target_list)
    for r in target_list_sorted:
        if r > 0:
            ret_p_.append(r)
        if r < 0:
            ret_m_.append(r)
    avg_all_ = sum(target_list_sorted) / n
    avg_m_ = sum(ret_m_) / n
    avg_p_ = sum(ret_p_) / n
    var_m_ = target_list_sorted[tail]
    var_p_ = target_list_sorted[-tail]
    es_m_ = sum(target_list_sorted[:tail]) / tail
    es_p_ = sum(target_list_sorted[-tail:]) / tail
    max_m_ = min(target_list_sorted)
    max_p_ = max(target_list_sorted)
    return (avg_all_, avg_m_, avg_p_, var_m_, var_p_, es_m_, es_p_, max_m_, max_p_)


def convert_key_to_list(dict):
    list = []
    for key in dict.keys():
        list.append(key)
    return list


def bool_more(target_: pd.DataFrame):
    return target_ > target_.shift()


def bool_less(target_: pd.DataFrame):
    return target_ < target_.shift()


def init_np(mother_np_: np.array, value_):
    inited_np = np.empty(mother_np_.shape)
    inited_np[:] = np.nan
    return inited_np


def np_to_df(mother_: np.array, df_blueprint_: pd.DataFrame):
    return pd.DataFrame(
        index=df_blueprint_.index, columns=df_blueprint_.columns, data=mother_
    )


from sklearn.linear_model import LinearRegression


def cal_slope__r_sq(X_: np.array, y_: np.array) -> (float, float):
    X_ = X_.reshape(-1, 1)
    y_ = y_.reshape(-1, 1)
    reg = LinearRegression().fit(X_, y_)
    slope_ = float(reg.coef_)
    r_sq_ = reg.score(X_, y_)
    return (slope_, r_sq_)


def get_ss01_model_slope__r_sq(c_: pd.DataFrame, period_: int):

    period_ = int(period_)
    c_np = c_.ffill(limit=5).to_numpy()

    stock_num = c_np.shape[1]
    bar_num = c_np.shape[0]
    X = np.arange(0, period_)

    slope_np = np.empty(c_np.shape)
    slope_np[:] = np.nan
    r_sq_np = np.empty(c_np.shape)
    r_sq_np[:] = np.nan

    for stock in range(stock_num):
        price_of_a_stock = c_np[:, stock]
        print(stock)
        for row in range(period_ - 1, bar_num, 1):
            sliced_np = price_of_a_stock[row - (period_ - 1) : row + 1]
            if np.isnan(sliced_np).sum() < 1:
                y = sliced_np / sliced_np[0] - 1
                slope_np[row, stock], r_sq_np[row, stock] = cal_slope__r_sq(X, y)

    slope_ = pd.DataFrame(index=c_.index, columns=c_.columns, data=slope_np)
    r_sq_ = pd.DataFrame(index=c_.index, columns=c_.columns, data=r_sq_np)

    return slope_, r_sq_


def cal_downside_deviation(target_: np.array):
    mean = target_.mean()
    relative = target_ - mean
    smaller_than_mean = relative < 0
    downside = np.zeros(target_.shape)
    downside = smaller_than_mean * relative * relative
    downside = downside.sum() / (target_.shape[0])
    downside = np.sqrt(downside)
    return downside


def rollingRankOnSeries(array):
    s = pd.Series(array)
    return s.rank(method="min", ascending=False)[len(s) - 1]


def ts_rank_pct(input_df: pd.DataFrame, period_) -> pd.DataFrame:
    # https://www.programmersought.com/article/63348086022/
    period_ = int(period_)
    return input_df.rolling(period_).apply(
        lambda row: rollingRankOnSeries(row) / period_
    )


def get_downside_deviationold(target_: pd.DataFrame, period_=60):
    data_exist = (target_.isnull() == False).astype(np.float64).replace(0, np.nan)
    data_exist = data_exist * (data_exist.shift(period_))
    target_ = clean_target(target_)
    period_ = int(period_)
    target_np = target_.ffill(limit=5).to_numpy()

    stock_num = target_np.shape[1]
    bar_num = target_np.shape[0]

    downside_deviation_np = np.empty(target_np.shape)
    downside_deviation_np[:] = np.nan

    for stock in range(stock_num):
        price_of_a_stock = target_np[:, stock]
        if stock % 100 == 0:
            print(stock)
        for row in range(period_ - 1, bar_num, 1):
            sliced_np = price_of_a_stock[row - (period_ - 1) : row + 1]
            if np.isnan(sliced_np).sum() < 1:
                downside_deviation_np[row, stock] = cal_downside_deviation(sliced_np)

    downside_deviation_ = pd.DataFrame(
        index=target_.index, columns=target_.columns, data=downside_deviation_np
    )
    downside_deviation_ = downside_deviation_ * data_exist
    return downside_deviation_


def count_existing_in_row(target_: pd.DataFrame):
    target_temp = target_.copy()
    target_temp = target_temp + 1000
    target_temp = target_temp / target_temp
    return target_temp.sum(axis=1)


def gen_target_shares_bool_sig_equal_weight(
    BUY_: pd.DataFrame,
    SELL_: pd.DataFrame,
    trade_universe_: pd.DataFrame,
    c_: pd.DataFrame,
    setting_backtest_: dict,
    method="one_way",
    cap_weight_=0.25,
):

    sig_one_zero_ = one_zero_gen_frm_BUYSELL(BUY_, SELL_)
    sig = sig_one_zero_ * trade_universe_
    sig_sum = series_to_df(sig.sum(axis=1), sig)
    equal_weight = (sig / sig_sum).replace(np.nan, 0)
    weight = equal_weight.copy()

    weight = cap_maximal_value(weight, cap_weight_)
    weight = clean_target(weight)
    target_shares = convert_weight_to_share(
        weight, c_, setting_backtest_["initial_deposit"]
    )

    weight_change = (weight != weight.shift()).astype(np.float64)
    weight_change_sum = weight_change.sum(axis=1)
    weight_change_trigger = weight_change_sum / weight_change_sum
    weight_change_trigger = series_to_df(weight_change_trigger, c_).replace(0, np.nan)
    target_shares = target_shares * weight_change_trigger
    if method == "one_way":
        target_shares = allow_only_shares_decline(target_shares, 0.0, 0.0)
    elif method == "two_ways":
        pass
    else:
        print("invalid method")
        raise

    target_shares.ffill(inplace=True)

    return target_shares


def clean_weight_over_1(target_weight):
    target_sum = target_weight.sum(axis=1)

    target_sum_extend = series_to_df(target_sum, target_weight)

    target_weight[target_sum > 1.0000000000000] = (
        target_weight[target_sum > 1.0000000000000]
        / target_sum_extend[target_sum > 1.0000000000000]
    )
    target_weight = target_weight * 1000
    target_weight = target_weight.apply(np.floor)
    target_weight = target_weight / 1000
    return target_weight
