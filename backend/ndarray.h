//==========================================================
// ndarray.h: C++ interface for easy access to numpy arrays
//
// J. De Ridder
//==========================================================

#ifndef NDARRAY_H
#define NDARRAY_H
#include <cstring>
#include <iostream>

// The C-struct to retrieve the ctypes structure.
// Note: Order of struct members must be the same as in Python.
//       Member names are not recognized!

template <typename T>
struct numpyArray
{
    T *data;
    long *shape;
    long *strides;
};

// Traits need to used because the return type of the []-operator can be
// a subarray or an element, depending whether all the axes are exhausted.

template <typename datatype, int ndim>
class Ndarray; // forward declaration

template <typename datatype, int ndim>
struct getItemTraits
{
    typedef Ndarray<datatype, ndim - 1> returnType;
};

template <typename datatype>
struct getItemTraits<datatype, 1>
{
    typedef datatype &returnType;
};

// Ndarray definition

template <typename datatype, int ndim>
class Ndarray
{
private:
    datatype *data;
    long *shape;
    long *strides;

public:
    Ndarray(datatype *data, long *shape, long *strides);
    Ndarray(const Ndarray<datatype, ndim> &array);
    Ndarray(const numpyArray<datatype> &array);
    long getShape(const int axis);
    Ndarray<datatype, ndim> deepcopy();
    typename getItemTraits<datatype, ndim>::returnType operator[](unsigned long i);
    void destroy();
    void init(datatype t);
};

// Ndarray constructor

template <typename datatype, int ndim>
Ndarray<datatype, ndim>::Ndarray(datatype *data, long *shape, long *strides)
{
    this->data = data;
    this->shape = shape;
    this->strides = strides;
}

// Ndarray copy constructor

template <typename datatype, int ndim>
Ndarray<datatype, ndim>::Ndarray(const Ndarray<datatype, ndim> &array)
{
    this->data = array.data;
    this->shape = array.shape;
    this->strides = array.strides;
}

// Ndarray constructor from ctypes structure

template <typename datatype, int ndim>
Ndarray<datatype, ndim>::Ndarray(const numpyArray<datatype> &array)
{
    this->data = array.data;
    this->shape = array.shape;
    this->strides = array.strides;
}

// Ndarray method to get length of given axis

template <typename datatype, int ndim>
long Ndarray<datatype, ndim>::getShape(const int axis)
{
    return this->shape[axis];
}

template <typename datatype, int ndim>
inline Ndarray<datatype, ndim> Ndarray<datatype, ndim>::deepcopy()
{
    size_t size = this->strides[0] * this->shape[0];
    datatype *newdata = new datatype[size];
    long *newstrides = new long[ndim];
    long *newshape = new long[ndim];
    memcpy(newdata, this->data, sizeof(datatype) * size);
    memcpy(newstrides, this->strides, sizeof(long) * ndim);
    memcpy(newshape, this->shape, sizeof(long) * ndim);

    return Ndarray<datatype, ndim>(newdata, newshape, newstrides);
}

template <typename datatype, int ndim>
inline void Ndarray<datatype, ndim>::destroy()
{
    delete[] data;
    delete[] shape;
    delete[] strides;
}

template <typename datatype, int ndim>
inline void Ndarray<datatype, ndim>::init(datatype t)
{
    size_t size = this->strides[0] * this->shape[0];
    for (int i = 0; i < size; i++)
        this->data[i] = t;
}

// Ndarray overloaded []-operator.
// The [i][j][k] selection is recursively replaced by i*strides[0]+j*strides[1]+k*strides[2]
// at compile time, using template meta-programming. If the axes are not exhausted, return
// a subarray, else return an element.

template <typename datatype, int ndim>
typename getItemTraits<datatype, ndim>::returnType
Ndarray<datatype, ndim>::operator[](unsigned long i)
{
    return Ndarray<datatype, ndim - 1>(&this->data[i * this->strides[0]], &this->shape[1], &this->strides[1]);
}

// Template partial specialisation of Ndarray.
// For 1D Ndarrays, the [] operator should return an element, not a subarray, so it needs
// to be special-cased. In principle only the operator[] method should be specialised, but
// for some reason my gcc version seems to require that then the entire class with all its
// methods are specialised.

template <typename datatype>
class Ndarray<datatype, 1>
{
private:
    datatype *data;
    long *shape;
    long *strides;

public:
    Ndarray(datatype *data, long *shape, long *strides);
    Ndarray(const Ndarray<datatype, 1> &array);
    Ndarray(const numpyArray<datatype> &array);
    long getShape(const int axis);
    Ndarray<datatype, 1> deepcopy();
    typename getItemTraits<datatype, 1>::returnType operator[](unsigned long i);
    void destroy();
    void init(datatype t);
};

// Ndarray partial specialised constructor

template <typename datatype>
Ndarray<datatype, 1>::Ndarray(datatype *data, long *shape, long *strides)
{
    this->data = data;
    this->shape = shape;
    this->strides = strides;
}

// Ndarray partially specialised copy constructor

template <typename datatype>
Ndarray<datatype, 1>::Ndarray(const Ndarray<datatype, 1> &array)
{
    this->data = array.data;
    this->shape = array.shape;
    this->strides = array.strides;
}

// Ndarray partially specialised constructor from ctypes structure

template <typename datatype>
Ndarray<datatype, 1>::Ndarray(const numpyArray<datatype> &array)
{
    this->data = array.data;
    this->shape = array.shape;
    this->strides = array.strides;
}

// Ndarray method to get length of given axis

template <typename datatype>
long Ndarray<datatype, 1>::getShape(const int axis)
{
    return this->shape[axis];
}

template <typename datatype>
inline Ndarray<datatype, 1> Ndarray<datatype, 1>::deepcopy()
{
    size_t size = this->strides[0] * this->shape[0];
    datatype *newdata = new datatype[size];
    long *newstrides = new long[1];
    long *newshape = new long[1];
    memcpy(newdata, this->data, sizeof(datatype) * size);
    memcpy(newstrides, this->strides, sizeof(long) * 1);
    memcpy(newshape, this->shape, sizeof(long) * 1);

    return Ndarray<datatype, 1>(newdata, newshape, newstrides);
}

template <typename datatype>
inline void Ndarray<datatype, 1>::destroy()
{
    delete[] data;
    delete[] shape;
    delete[] strides;
}

template <typename datatype>
inline void Ndarray<datatype, 1>::init(datatype t)
{
    for (int i = 0; i < this->shape[0]; i++)
        this->data[i] = t;
}

// Partial specialised [] operator: for 1D arrays, return an element rather than a subarray

template <typename datatype>
typename getItemTraits<datatype, 1>::returnType
Ndarray<datatype, 1>::operator[](unsigned long i)
{
    return this->data[i * this->strides[0]];
}

template <typename datatype>
inline std::ostream &operator<<(std::ostream &out, Ndarray<datatype, 1> arr)
{
    out << "[";
    for (int i = 0; i < arr.getShape(0); i++)
    {
        out << arr[i];
        if (i < arr.getShape(0) - 1)
            out << ", ";
    }
    out << "]";
    return out;
}

template <typename datatype, int ndim>
inline std::ostream &operator<<(std::ostream &out, Ndarray<datatype, ndim> arr)
{
    out << "[";
    for (int i = 0; i < arr.getShape(0); i++)
    {
        out << arr[i];
        if (i < arr.getShape(0) - 1)
            out << "\n";
    }
    out << "]";
    return out;
}

#endif