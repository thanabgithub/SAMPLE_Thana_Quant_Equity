import timeit
import time
import pandas as pd
import numpy as np

"""

pandas for reindex
np for calculation

high performance technique

1. Use np.where for conditional operation is faster     df[condition]=xx
    Ex. Case 1 is faster than Case 2. Case 2 is faster than Case 3
    
    Case 1:
        temp_1_df['C'] = np.where(temp_1_df.B > temp_1_df.A, 1, 0)
    Case 2:
        temp_1_df['D'] = 0
        temp_1_df['D'][temp_1_df.B > temp_1_df.A] = 1
    Case 3
        temp_1_df['E'] = temp_1_df.apply(lambda row: 1 if row['B'] > row['A'] else 0, axis = 1)


2. Use np.apply_over_axes is faster than                df.sum, df.mean

    Ex. Case 1 is faster than Case 2.
    
    Case 1:
        print(np.apply_over_axes(np.sum, temp_1_arr, [1]))
    Case 2:
        print(temp_1_df.sum(axis = 1))

3. use np.apply_along_axis is faster than               df.apply

    Ex. Case 1 is faster than Case 2.
    
    Case 1:
        c_arr = np.array(c)
        c_new_arr = np.where(np.isnan(c_arr), 0, c_arr)
        print(np.apply_along_axis(HHV_np, 0, c_new_arr, p))
    Case 2:
        print(clean_target(c).apply(lambda col: ta.MAX(col, p)))

4. lookup by np.where is superfast
https://www.w3schools.com/python/numpy/numpy_array_search.asp


5. 
@guvectorize(["void(float64[:,:], float64[:,:], float64[:,:])"],
             "(m,n),(n,p)->(m,p)")
def f(a, b, result):

    ...


"""

# %%
"""
Template area

def wrapper_template_col_np(input_arr, param):
    
    return np.apply_along_axis(func, 0, input_arr, param)

def wrapper_template_row_np(input_arr, param):
    
    return np.apply_along_axis(func, 1, input_arr, param)



"""
# %%



class Measure_time:
    """
    Examples:
    >>> measure_time_cls = Measure_time()
        get_description_pandas_01("Country", schema)
        measure_time_cls.print_elapsed_time()
        get_description_pandas_02("Country", schema)
        measure_time_cls.print_elapsed_time()
        get_description_pandas_02("Country", schema)
        measure_time_cls.print_elapsed_time()
    """

    def __init__(self):
        self.prev = time.time()

    def init(self):
        pass

    def update_prev(self):
        self.prev = time.time()

    def print_elapsed_time(self):
        print(time.time() - self.prev)
        self.update_prev()


def get_value_of_different_column_in_the_same_row(
    input_df: pd.DataFrame,
    input_col_name_str: str,
    search_value,
    output_col_name_str: str,
):
    """

    can use only with unique value in those columns

    """
    input_df_np = np.array(input_df)
    input_col_index_int, output_col_index_int = input_df.columns.get_loc(
        input_col_name_str
    ), input_df.columns.get_loc(output_col_name_str)
    interesting_row = input_df_np[input_df_np[:, input_col_index_int] == search_value]
    desc = interesting_row[0, output_col_index_int]
    return desc

def init_np_random(ax_0, ax_1):
    
    """
    Examples:
    >>> temp_1_arr = init_np_random(20, 2)
    """
    return np.random.randint(100, size = (ax_0*ax_1)).reshape(ax_0, ax_1)

def init_df_random(ax_0, ax_1):
    return pd.DataFrame(init_np_random(ax_0, ax_1))










