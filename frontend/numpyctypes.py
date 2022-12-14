import numpy as N
import ctypes as C


ctypesDict = {
    "d": C.c_double,
    "f": C.c_float,
    "b": C.c_char,
    "h": C.c_short,
    "i": C.c_int,
    "l": C.c_long,
    "q": C.c_longlong,
    "B": C.c_ubyte,
    "H": C.c_ushort,
    "I": C.c_uint,
    "L": C.c_ulong,
    "Q": C.c_ulonglong,
}


def c_ndarray(array):

    """
    PURPOSE: Given an array, return a ctypes structure containing the
             arrays info (data, shape, strides, ndim). A check is made to ensure
             that the array has the specified dtype and requirements.

    INPUT: a: an array: something that is or can be converted to a numpy array

    OUTPUT: ctypes structure with the fields:
             . data: pointer to the data : the type is determined with the dtype of the
                     array, and with ctypesDict.
             . shape: pointer to long array : size of each of the dimensions
             . strides: pointer to long array : strides in elements (not bytes)
    """

    # Define a class that serves as interface of an ndarray to ctypes.
    # Part of the type depends on the array's dtype.

    class ndarrayInterfaceToCtypes(C.Structure):
        pass

    typechar = array.dtype.char

    if typechar in ctypesDict:
        ndarrayInterfaceToCtypes._fields_ = [
            ("data", C.POINTER(ctypesDict[typechar])),
            ("shape", C.POINTER(C.c_long)),
            ("strides", C.POINTER(C.c_long)),
        ]
    else:
        raise TypeError("dtype of input ndarray not supported")

    # Instantiate the interface class and attach the ndarray's internal info.
    # Ctypes does automatic conversion between (c_long * #) arrays and POINTER(c_long).

    ndarrayInterface = ndarrayInterfaceToCtypes()
    ndarrayInterface.data = array.ctypes.data_as(C.POINTER(ctypesDict[typechar]))
    ndarrayInterface.shape = (C.c_long * array.ndim)(*array.shape)
    ndarrayInterface.strides = (C.c_long * array.ndim)(*array.strides)
    for n in range(array.ndim):
        ndarrayInterface.strides[n] //= array.dtype.itemsize

    return ndarrayInterface
