
import timeit
import time

class Template(object):

    def __init__(self):
        self.value = "hello"
        self.other_value = "bonjour"
        self.constant_value = 42
        current_class = self.__class__
        inits = []
        while (current_class.__name__ != "Template"):
            inits.append(current_class.init)
            current_class = current_class.__bases__[0]
        for i in reversed(inits):
            i(self)

    def init(self):
        pass

    def info(self):
        print(self.value)
        print(self.other_value)
        print(self.constant_value)
        print("")
        
class Template(object):

    def __init__(self):
        self.value = "hello"
        self.other_value = "bonjour"
        self.constant_value = 42
        current_class = self.__class__
        inits = []
        while (current_class.__name__ != "Template"):
            inits.append(current_class.init)
            current_class = current_class.__bases__[0]
        for i in reversed(inits):
            i(self)

    def init(self):
        pass

    def info(self):
        print(self.value)
        print(self.other_value)
        print(self.constant_value)
        print("")

class Deep(Template):
    def init(self):
        self.value = "howdy"
        self.other_value = "salut"

class Deeep(Deep):
    def init(self):
        self.value = "hi"

class Deeeep(Deeep):
    def init(self):
        self.value = "'sup"

very_deep = Deeeep()
not_so_deep = Deep()
very_deep.info()
not_so_deep.info()        
        


def get_description_pandas_01(column_name, schema=schema):
    '''
    INPUT - schema - pandas dataframe with the schema of the developers survey
            column_name - string - the name of the column you would like to know about
    OUTPUT - 
            desc - string - the description of the column
    '''
    index_lst = schema.Column == column_name
    interesting_index = index_lst[index_lst].index.item()
    desc = schema.at[interesting_index, "Question"]
    return desc

def get_description_pandas_02(column_name, schema=schema):
    '''
    INPUT - schema - pandas dataframe with the schema of the developers survey
            column_name - string - the name of the column you would like to know about
    OUTPUT - 
            desc - string - the description of the column
    '''
    index_lst = schema.Column == column_name
    interesting_row = schema[index_lst]
    desc = interesting_row["Question"]
    return desc.item()

def get_description_numpy(column_name, schema=schema):
    schema_np = np.array(schema)
    input_col_index_int = schema.columns.get_loc("Column")
    output_col_index_int = schema.columns.get_loc("Question")
    interesting_row = schema_np[schema_np[:, input_col_index_int] == column_name]
    desc = interesting_row[0, output_col_index_int]
    return desc

class Measure_time():
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
    import time    
    def __init__(self):
        self.prev = time.time()
    def init(self):
        pass
    def update_prev(self):
        self.prev = time.time()
    def print_elapsed_time(self):
        print(time.time() - self.prev)
        self.update_prev()
        

measure_time_cls = Measure_time()
get_description_pandas_01("Country", schema)
measure_time_cls.print_elapsed_time()
get_description_pandas_02("Country", schema)
measure_time_cls.print_elapsed_time()
get_description_pandas_02("Country", schema)
measure_time_cls.print_elapsed_time()

# %%
"""
Multi Thread
"""

from threading import Thread

def func_simple(length):
    sum_fl = 0
    for x in range(0, length):
        sum_fl += x
    print("Normal sum is {}".format(sum_fl))
    
def func_square(length):
    sum_fl = 0
    for x in range(0, length):
        sum_fl += x*x
    print("Square sum is {}".format(sum_fl))

def func_cubes(length):
    sum_fl = 0
    for x in range(0, length):
        sum_fl += x*x*x
    print("Cube sum is {}".format(sum_fl))

def do_threading():
    length = 3
    thread_simple = Thread(target = func_simple, args = (length,))
    thread_square = Thread(target = func_square, args = (length,))
    thread_cube = Thread(target = func_cubes, args = (length,))

    # start execution
    thread_simple.start()
    thread_square.start()
    thread_cube.start()


    # wait for the other threads to finish
    thread_simple.join()
    thread_square.join()
    thread_cube.join()

do_threading()

# %%
"""
Multi Thread
Locks
"""
from threading import Thread, Lock

thread_lock = Lock()

my_global_string = "Hello World"

def add_prefix(prefix_to_add):
    
    global my_global_string
    
    thread_lock.acquire()
    
    my_global_string = prefix_to_add + " " + my_global_string
    
    thread_lock.release()
    
def add_suffix(suffix_to_add):
    
    global my_global_string
    
    thread_lock.acquire()
    
    my_global_string = my_global_string + " " + suffix_to_add
    
    thread_lock.release()
        
def do_threading():  
  
    thread_prefix = Thread(target = add_prefix, args = ("YOLO",))
    thread_suffix = Thread(target = add_suffix, args = ("BYE!!",))
    

    thread_prefix.start()
    thread_suffix.start()
    
    thread_prefix.join()
    thread_suffix.join()
    
    global my_global_string
    print("Final string is {}".format(my_global_string))


do_threading()

# %%

"""
Numba
"""

def factorial(n):
    fact = 1
    for i in range(1, n + 1):
        fact = fact * i
    return fact

%timeit factorial(100000)

from numba import jit
@jit
def factorial_numba(n):
    fact = 1
    for i in range(1, n + 1):
        fact = fact * i
    return fact

%timeit factorial_numba(100000)

from numba import jit
@jit(nopython=True)
def factorial_numba(n):
    fact = 1
    for i in range(1, n + 1):
        fact = fact * i
    return fact

%timeit factorial_numba(100000)

# %%


from numba import jit, vectorize, int64


def do_some_op(x, y, z):
    # print(x)
    # print(y)
    # print(z)
    
    ouput = x + y + z
    return ouput

arr1 = np.array([1, 2, 3])
arr2 = np.array([[10, 20, 30], [40, 50, 60]])
c = 100

%timeit do_some_op(arr1, arr2, c)
print(do_some_op(arr1, arr2, c))
# %%


@vectorize([int64(int64, int64, int64)], nopython = True)
def do_some_op(x, y, z):
    # x, y, z become an element
    # print(x)
    # print(y)
    # print(z)
    
    ouput = x + y + z
    return ouput
%timeit do_some_op(arr1, arr2, c)
print(do_some_op(arr1, arr2, c))
# %%
temp = np.array([[1, 2]
                 , [3, 4]])

print(temp)

# %%

import numpy as np
import pandas as pd

def init_np_random(ax_0, ax_1):
    return np.random.randint(100, size = (ax_0*ax_1)).reshape(ax_0, ax_1)

def init_df_random(ax_0, ax_1):
    return pd.DataFrame(init_np_random(ax_0, ax_1))


temp_1_arr = init_np_random(100000, 2)
temp_1_df = pd.DataFrame(temp_1_arr) # init_df_random(100000, 2)
temp_1_df.columns = ['A', 'B']

# %%

measure_time_cls = Measure_time()

temp_1_df['C'] = np.where(temp_1_df.B > temp_1_df.A, 1, 0)
measure_time_cls.print_elapsed_time()
temp_1_df['D'] = 0
temp_1_df['D'][temp_1_df.B > temp_1_df.A] = 1
measure_time_cls.print_elapsed_time()
temp_1_df['E'] = temp_1_df.apply(lambda row: 1 if row['B'] > row['A'] else 0, axis = 1)
measure_time_cls.print_elapsed_time()

# %%
import numpy as np
def my_func(a):
    """Average first and last element of a 1-D array"""
    return (a[0] + a[-1]) * 0.5
b = np.array([[1,2,3],
              [4,5,6],
              [7,8,9]])

print(np.apply_along_axis(my_func, 0, b))
print(np.apply_along_axis(my_func, 1, b))

# %%

import talib as ta

def HHV_np(arr, period):
    return ta.MAX(arr, period)
p = 20
measure_time_cls = Measure_time()
c_arr = np.array(c)
c_new_arr = np.where(np.isnan(c_arr), 0, c_arr)
print(np.apply_along_axis(HHV_np, 0, c_new_arr, p))
measure_time_cls.print_elapsed_time()
print(clean_target(c).apply(lambda col: ta.MAX(col, p)))
measure_time_cls.print_elapsed_time()

# %%

def rolling_window(input_arr, window):
    shape = input_arr.shape[:-1] + (input_arr.shape[-1] - window + 1, window)
    strides = input_arr.strides + (input_arr.strides[-1],)
    return np.lib.stride_tricks.as_strided(input_arr, shape=shape, strides=strides)


windows = rolling_window(c_arr, 5)