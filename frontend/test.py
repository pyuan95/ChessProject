import numpy as np
from numpy.ctypeslib import load_library
from numpyctypes import c_ndarray

mylib = load_library('extension', '../backend/output')       # '.' is the directory of the C++ lib  

def myfunc(array1, array2):
    arg1 = c_ndarray(array1, dtype=np.double, ndim = 3, shape = array1.shape)
    arg2 = c_ndarray(array2, dtype=np.double, ndim = 3, shape = array2.shape)
    z =5
    s = bytes("abcde hello world!", encoding='utf8')
    return mylib.myfunc(arg1, arg2, s, z)

def get_mcts(array1, array2):
    arg1 = c_ndarray(array1, dtype=int, ndim = 3, shape = array1.shape)
    arg2 = c_ndarray(array2, dtype=int, ndim = 2, shape = array2.shape)
    return mylib.createBatchMCTS(arg1, arg2)

x = np.arange(100).reshape([5, 5, -1])
y = np.arange(100).reshape([5, 5, -1])
print(myfunc(x, y))

# a = np.arange(32 * 8 * 8).reshape([32, 8, 8])
# b = np.arange(32 * 5).reshape([32, 5])

# print(get_mcts(a, b))