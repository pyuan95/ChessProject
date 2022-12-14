import numpy as np
from numpy.ctypeslib import load_library
from numpyctypes import c_ndarray

mylib = load_library("extension_BatchMCTS", "../backend/output")


def myfunc(array1, array2):
    arg1 = c_ndarray(array1)
    arg2 = c_ndarray(array2)
    z = 5
    s = bytes("abcde hello world!", encoding="utf8")
    print(array1)
    k = mylib.myfunc(arg1, arg2, s, z)
    print(array1)
    return k


def get_mcts(array1, array2):
    arg1 = c_ndarray(array1, dtype=int, ndim=3, shape=array1.shape)
    arg2 = c_ndarray(array2, dtype=int, ndim=2, shape=array2.shape)
    return mylib.createBatchMCTS(arg1, arg2)


x = np.arange(100).reshape([5, 5, -1]).astype(np.float32) * np.e
y = np.arange(100).reshape([5, 5, -1]).astype(np.float32) * np.e
print(myfunc(x, y))
