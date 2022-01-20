# %%

from numpy import *


import numpy as np


import numba as nb

import talib as ta

"""

welcome to numpy world

1. df.to_numpy()
2. col_map = df.columns.to_numpy()
3. index_map = df.index.to_numpy()
How to manipulate time series data with rolling windows

# go vectorize for basic thing
# go guvectorize/jit for for loop

"""


def replace_na_traditional(input_arr: np.array, value):
    # 4.182005405426025
    return np.nan_to_num(input_arr, nan=value)




@nb.vectorize(["float64(float64, float64)"])
def replace_na_vector(input_element, value):
    # replace_na_vector
    # 0.9864423274993896
    output = input_element
    if np.isnan(input_element):
        output = value
    return output
    
@nb.guvectorize(
    [(nb.float64[:, :], nb.float64, nb.float64[:, :])],
    "(m, n ), ()->(m, n )",
)
def replace_na_guvectorize(input_arr, value, result):
    # replace_na_guvectorize
    # 0.9060289859771729
    for row in range(input_arr.shape[0]):
        for col in range(input_arr.shape[1]):
            res_element = input_arr[row][col]
            if np.isnan(res_element):
                res_element = value
            result[row][col] = res_element
    
@nb.njit(["float64[:, :](float64[:, :], float64)"])
def replace_na_njit(input_arr, value):
    # replace_na_for_loop
    # 0.8731958866119385
    result = np.empty(input_arr.shape)
    for row in range(input_arr.shape[0]):
        for col in range(input_arr.shape[1]):
            res_element = input_arr[row][col]
            if np.isnan(res_element):
                res_element = value
            result[row][col] = res_element    
    return result


@nb.njit(["float64[:, :](float64[:, :], float64)"])
def replace_na(input_arr, value):
    # replace_na_for_loop
    # 0.8731958866119385
    result = np.empty(input_arr.shape)
    for row in range(input_arr.shape[0]):
        for col in range(input_arr.shape[1]):
            res_element = input_arr[row][col]
            if np.isnan(res_element):
                res_element = value
            result[row][col] = res_element    
    return result







@calculate_time
@nb.jit(nopython = False, fastmath = True,  parallel=True, cache=True)
def add(a, b):
    result = a + b
    return result
"""    
result = tn_process_time(add(1, 3))   
"""
# %%
large_arr = init_np_random(10000, 10000)
large_arr = np.where(large_arr > 50, large_arr, np.nan)
# %%
measure_time_cls = Measure_time()
measure_time_cls.print_elapsed_time()
x1 = replace_na_traditional(large_arr, 0.00)
measure_time_cls.print_elapsed_time()

x2 = replace_na_vector(large_arr, 0.00)
measure_time_cls.print_elapsed_time()

x3 = replace_na_guvectorize(large_arr, 0.00)
measure_time_cls.print_elapsed_time()

x4 = replace_na_njit(large_arr, 0.00)
measure_time_cls.print_elapsed_time()



# %%

def ffill_traditional(input_arr: np.array):
    # 5.060262203216553
    input_arr = input_arr.T
    mask = np.isnan(input_arr)
    idx = np.where(~mask, np.arange(mask.shape[1]), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    output_arr = input_arr[np.arange(idx.shape[0])[:, None], idx]
    return output_arr.T

@nb.guvectorize(
    ["void(float64[:, :], float64[:, :])"],
    "(m, n) -> (m, n)",
    nopython=False,
)
def ffill_guvectorize(input_arr, result):
    # guvectorize
    # 2.0188186168670654
    result[:] = input_arr
    for row in range(1, input_arr.shape[0]):
        for col in range(input_arr.shape[1]):
            res_element = input_arr[row][col]
            if np.isnan(res_element):
                res_element = result[row-1][col]
            result[row][col] = res_element

@nb.njit(
    ["float64[:, :](float64[:, :])"])
def ffill_njit(input_arr):
    # ffill_njit
    # 1.9829812049865723
    result = input_arr.copy()
    for row in range(1, input_arr.shape[0]):
        for col in range(input_arr.shape[1]):
            res_element = input_arr[row][col]
            if np.isnan(res_element):
                res_element = result[row-1][col]
            result[row][col] = res_element
    return result
            
@nb.njit(
    ["float64[:, :](float64[:, :])"])
def ffill(input_arr):
    # ffill_njit
    # 1.9829812049865723
    result = input_arr.copy()
    for row in range(1, input_arr.shape[0]):
        for col in range(input_arr.shape[1]):
            res_element = input_arr[row][col]
            if np.isnan(res_element):
                res_element = result[row-1][col]
            result[row][col] = res_element
    return result
            
from numba import stencil
# https://numba.pydata.org/numba-doc/dev/user/stencil.html



def rolling_window_1d(input_arr: np.array, window: int) -> np.array:

    shape = input_arr.shape[:-1] + (input_arr.shape[-1] - window + 1, window)
    shape_output = input_arr.shape[:-1] + (input_arr.shape[-1], window)
    strides = input_arr.strides + (input_arr.strides[-1],)
    output_arr = np.full(shape_output, np.nan)
    output_arr[window - 1 :] = np.lib.stride_tricks.as_strided(
        input_arr, shape=shape, strides=strides
    )
    return output_arr

@calculate_time
def MA_stencil_1d_arr(a, p):
    loop_start = p-1
    @nb.stencil(neighborhood = ((-loop_start, 0),))
    def window_with_length_p_inner(a, p):
        cumul = 0
        for i in range(-loop_start, 1):
            cumul += a[i]
        return cumul / p
    return window_with_length_p_inner(a, p)



@calculate_time
def MA_rollin_1d_arr(a, p):

    temp = rolling_window_1d(a, p)
    output = np.full(a.shape, np.nan)
    for row in range(temp.shape[0]):
        output[row] = temp[row].mean()
    return output

@nb.njit
def insert_mean(A):
    output = np.full(A.shape[0], np.nan)
    for row in range(A.shape[0]):
        output[row] = A[row].mean()  
    return output


def MA_rollin_1d_arr_v2(a, p):
    temp = rolling_window_1d(a, p)
    return insert_mean(temp)

@calculate_time
def MA_arr_rolling_jit(A, p):
    output = np.full(A.shape, np.nan)
    for col_index in range(A.shape[1]):
        output[:, col_index] = MA_rollin_1d_arr_v2(A[:, col_index], p)
    return output


@calculate_time
def MA_arr_multi_thread(A, p):
    from multiprocessing.pool import ThreadPool
    output = np.full(c_arr.shape, np.nan)
    pool = ThreadPool()    
    def f(col_index):
        output[:, col_index] = MA_rollin_1d_arr_v2(c_arr[:, col_index], p)
    pool.map(f, range(c_arr.shape[1]))
    return output

@calculate_time
def MA_arr_multi_processes(A, p):
    # https://stackoverflow.com/questions/32816410/parallelize-loop-over-numpy-rows
    from multiprocessing  import Pool
    output = np.full(c_arr.shape, np.nan)
    pool = ThreadPool()    
    def f(col_index):
        output[:, col_index] = MA_rollin_1d_arr_v2(c_arr[:, col_index], p) 
    pool.map(f, range(c_arr.shape[1]))
    return output

@calculate_time
def MA(input_: NdType, periods: int = 20) -> NdType:
    # https://stackoverflow.com/questions/32816410/parallelize-loop-over-numpy-rows
    exist = data_exist_1_nan(input_)
    exist = exist*(exist.shift(periods))
    target_ = input_
    target_clean = clean_target(target_).astype(np.float64)
    return nd_universal_adapter(sma_np_1d, (target_clean,), (periods,))*exist

c_long = pd.concat([c]*10, axis = 1)
c_long_arr = c_long.to_numpy()


test_1 = MA_arr_rolling_jit(c_long_arr, 240)
test_2 = MA_arr_multi_thread(c_long_arr, 240) 
test_3 = MA_arr_multi_processes(c_long_arr, 240) # the best of the best
test_4 = MA(c_long, 240)

# %%



# %%

import dask.dataframe as ddf
c_t = c.T
c_t_ddf = ddf.from_pandas(c_t, npartitions = 5)

measure_time_cls = Measure_time()
measure_time_cls.print_elapsed_time()
MA_arr_rolling_jit(c_arr, 20)
measure_time_cls.print_elapsed_time()
c_t.rolling(5, axis = 1).mean()
MA(c, 20)
measure_time_cls.print_elapsed_time()

c_ddf.apply

x = da.from_array(ar, chunks=(1, arr.shape[1]))
x.map_blocks(function, *args)
states = x.compute()



def MA_stencil_2d_arr(A, p):
    return A.apply(lambda col: MA_stencil_1d_arr(col.values, p))

c_AOT_arr = c.AOT.to_numpy()

c_ddf = ddf.from_pandas(c, npartitions = 1)
c_ddf.AOT.rolling(5).mean()
c_ddf.apply

result = np.full(c_arr.shape, np.nan)
for col_index in range(c_arr.shape[1]):
    print(col_index)
    result[:, col_index] = MA_stencil_1d_arr(c_arr[:, col_index], 2)
    

def MA_stencil_2d_arr(input_: NdType, periods_, period_ = 20):
    """
        pseudo rank:
            it ranges from 0 to 1
    """
    exist = data_exist_1_nan(input_)
    exist = exist*(exist.shift(periods_))
    target_ = input_
    target_clean = clean_target(target_)
    output = nd_universal_adapter(MA_stencil_1d_arr, (target_clean,), (periods_,))
    return output*exist





import multiprocessing as mp
pool = mp.Pool(processes=mp.cpu_count())

result = pool.map( MA_stencil_1d_arr, [c_arr[:, col_idx] for col_idx in range(c_arr.shape[1])])

# %%
measure_time_cls = Measure_time()
measure_time_cls.print_elapsed_time()
x1 = ffill_traditional(large_arr)
measure_time_cls.print_elapsed_time()

x4 = ffill_guvectorize(large_arr)
measure_time_cls.print_elapsed_time()

x5 = ffill_njit(large_arr)
measure_time_cls.print_elapsed_time()


# %%

@nb.njit
def bfill(input_arr: np.array):
    input_arr_inverse = np.flipud(input_arr)
    output_arr = ffill(input_arr_inverse)
    return np.flipud(output_arr)




def rolling_window_2d(input_arr: np.array, window: int) -> np.array:
    # https://stackoverflow.com/questions/6811183/rolling-window-for-1d-arrays-in-numpy by
    # 0.043853044509887695
    shape_output = input_arr.shape[:-1] + (input_arr.shape[-1], window)
    input_arr = input_arr.T
    shape = input_arr.shape[:-1] + (input_arr.shape[-1] - window + 1, window)
    strides = input_arr.strides + (input_arr.strides[-1],)
    output_arr = np.full(shape_output, np.nan)
    output_arr[window - 1 :] = np.moveaxis(
        np.lib.stride_tricks.as_strided(input_arr, shape=shape, strides=strides), 0, 1
    )
    return output_arr



# %%

large_arr = init_np_random(10000, 10000)
large_arr = np.where(large_arr > 50, large_arr, np.nan)

# %%
measure_time_cls = Measure_time()
measure_time_cls.print_elapsed_time()
y1 = rolling_window_2d_traditional(large_arr, 10)
measure_time_cls.print_elapsed_time()

# %%
def rolling_window(input_arr: np.array, window: int) -> np.array:
    arr_dimension_int = len(input_arr.shape)
    assert arr_dimension_int < 3, "Dimension exceeds 2d"
    if abs(arr_dimension_int - 2) < 1:
        return rolling_window_2d(input_arr, window)
    else:
        return rolling_window_1d(input_arr, window)


def apply_func_to_col_2d_arr(input_arr: np.array, func_implement_to_each_col, *args):

    """
    Example 1 :
    temp_01_arr = init_np_random(1000, 700)
    def col_MA_np(input_1d_arr: np.array, window: int) -> np.array:
        output_01_2d_arr = rolling_window_1d(input_1d_arr, window)
        return output_01_2d_arr.mean(axis = 1)

    mean_test_func = lambda input_arr, param: apply_func_to_col_2d_arr(input_arr, col_MA_np, param)

    print(mean_test_func(temp_01_arr, 5))
    """

    return np.apply_along_axis(func_implement_to_each_col, 0, input_arr, *args)


@nb.njit
def apply_func_to_row_2d_arr(input_arr: np.array, func, *args):

    return np.apply_along_axis(func, 1, input_arr, *args)


"""
Example of writing function ts
"""


def find_rank_asc_n_ts(input_2d_arr: np.array, window: int, rank_n: int) -> np.array:
    def transformation_each_col(input_1d_arr: np.array, window: int, rank_n: int):

        # Go through rows
        # V
        # V
        # V

        # column as 1d input
        output_01_2d_arr = rolling_window_1d(input_1d_arr, window)
        # output as 2d array . Each row has an array containing the previous values as "Window"

        # Use the following function to transform the array of each row into a value of each row

        def rank_asc_n(input_1d_arr_, n):

            return np.sort(input_1d_arr_)[-n]

        # implementation of converting a column containing windows to a column containing values
        output_final_1d_arr = np.apply_along_axis(
            rank_asc_n, 1, output_01_2d_arr, rank_n
        )

        return output_final_1d_arr

    # Go through columns >>>>>>>>>>>>>>>>>>>>>

    return apply_func_to_col_2d_arr(
        input_2d_arr, transformation_each_col, window, rank_n
    )


# %%


def find_rank_asc_n_ts_fast1(
    input_2d_arr: np.array, window: int, rank_n: int
) -> np.array:
    @nb.njit
    def rank_asc_n(input_1d_arr_, n):
        return np.sort(input_1d_arr_)[-n]

    window_3d_arr = rolling_window_2d(input_2d_arr, window)
    window_3d_arr_2d = window_3d_arr.reshape(
        window_3d_arr.shape[0] * window_3d_arr.shape[1], window_3d_arr.shape[2]
    )

    @nb.njit
    def for_loop(window_3d_arr_2d, rank_asc_n):
        output_1d_arr = np.full((window_3d_arr_2d.shape[0]), np.nan)
        for row in range(window - 1, window_3d_arr_2d.shape[0], 1):
            output_1d_arr[row] = rank_asc_n(window_3d_arr_2d[row], rank_n)
        return output_1d_arr

    output_1d_arr = for_loop(window_3d_arr_2d, rank_asc_n)

    output_2d_arr = output_1d_arr.reshape(
        window_3d_arr.shape[0], window_3d_arr.shape[1]
    )

    return output_2d_arr


# %%


def find_rank_asc_n_ts_fast2(
    input_2d_arr: np.array, window: int, rank_n: int
) -> np.array:
    @nb.njit
    def rank_asc_n(input_1d_arr_, n):
        return np.sort(input_1d_arr_)[-n]

    window_3d_arr = rolling_window_2d(input_2d_arr, window)
    window_3d_arr_2d = window_3d_arr.reshape(
        window_3d_arr.shape[0] * window_3d_arr.shape[1], window_3d_arr.shape[2]
    )

    @nb.njit
    def for_loop(window_3d_arr_2d, rank_asc_n):
        output_1d_arr = np.full((window_3d_arr_2d.shape[0]), np.nan)
        for index, item in enumerate(window_3d_arr_2d):
            if index >= window:
                output_1d_arr[index] = rank_asc_n(item, rank_n)
        return output_1d_arr

    output_1d_arr = for_loop(window_3d_arr_2d, rank_asc_n)
    output_2d_arr = output_1d_arr.reshape(
        window_3d_arr.shape[0], window_3d_arr.shape[1]
    )

    return output_2d_arr


# %%


def find_rank_asc_n_ts_fast3(
    input_2d_arr: np.array, window: int, rank_n: int
) -> np.array:
    @nb.njit
    def rank_asc_n(input_1d_arr_, n):
        return np.sort(input_1d_arr_)[-n]

    window_3d_arr = rolling_window_2d(input_2d_arr, window)
    window_3d_arr_2d = window_3d_arr.reshape(
        window_3d_arr.shape[0] * window_3d_arr.shape[1], window_3d_arr.shape[2]
    )

    @nb.njit(parallel=True)
    def for_loop(window_3d_arr_2d, rank_asc_n):
        output_1d_arr = np.full((window_3d_arr_2d.shape[0]), np.nan)
        for row in range(window - 1, window_3d_arr_2d.shape[0], 1):
            output_1d_arr[row] = rank_asc_n(window_3d_arr_2d[row], rank_n)
        return output_1d_arr

    output_1d_arr = for_loop(window_3d_arr_2d, rank_asc_n)

    output_2d_arr = output_1d_arr.reshape(
        window_3d_arr.shape[0], window_3d_arr.shape[1]
    )

    return output_2d_arr


# %%
# measure_time_cls = Measure_time()
# x1 = find_rank_asc_n_ts(temp_01_arr, 10, 2)
# measure_time_cls.print_elapsed_time()
# x2 = find_rank_asc_n_ts_fast1(temp_01_arr, 10, 2)
# measure_time_cls.print_elapsed_time()
# x4 = find_rank_asc_n_ts_fast2(temp_01_arr, 10, 2)
# measure_time_cls.print_elapsed_time()
# x5 = find_rank_asc_n_ts_fast3(temp_01_arr, 10, 2)
# measure_time_cls.print_elapsed_time()

# %%
# @nb.vectorize([nb.float64(nb.float64, nb.int64)])
# def rank_asc_n(input_1d_arr_, n):
#     return np.sort(input_1d_arr_)[-n]

# window_3d_arr = tn_np.rolling_window_2d(temp_01_arr.astype(np.float64), 10)
# rank_asc_n(window_3d_arr[-1, 0, :], 2)


@guvectorize(["void(float64[:,:], float64[:,:], float64[:,:])"], "(m,n),(n,p)->(m,p)")
def f(a, b, result):
    """Fill-in *result* matrix such as result := a * b"""
    ...


def HHV_np(input_2d_arr: np.array, window: int) -> np.array:
    def transformation_each_col(input_1d_arr: np.array, window: int):
        # Go through rows
        # V
        # V
        # V
        return ta.MAX(input_1d_arr, window)

    # Go through columns >>>>>>>>>>>>>>>>>>>>>

    return apply_func_to_col_2d_arr(input_2d_arr, transformation_each_col, window)


def apply_func_to_col_3d_arr(func, input_arr: np.array, param=(None,)):

    """
    Example:
    HHV_2d_func = lambda input_arr, param: apply_func_to_col_3d_arr(ta.MAX, input_arr, param)
    temp  = HHV_2d_func(np.nan_to_num(c_arr, 0), 20)
    """

    return np.apply_along_axis(func, 2, input_arr, param)


# mean_test_func = lambda input_arr: apply_func_to_col_3d_arr(np.mean, input_arr, 0)


# measure_time_cls = Measure_time()
# c_arr = c.to_numpy()
# HHV_2d_func = lambda input_arr, param: apply_func_to_col_2d_arr(ta.MAX, input_arr, param)
# temp  = HHV_2d_func(np.nan_to_num(c_arr, 0), 20)
# measure_time_cls.print_elapsed_time()
# temp  = HHV(c, 20)
# measure_time_cls.print_elapsed_time()
