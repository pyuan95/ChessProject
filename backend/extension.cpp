#include <iostream>
#include "ndarray.h"
#include "BatchMCTS.h"
extern "C" {
int myfunc(numpyArray<double> array1, numpyArray<double> array2, char* test, int x)
{
    Ndarray<double,3> a(array1);
    Ndarray<double,3> b(array2);

    double sum=0.0;

    for (int i = 0; i < a.getShape(0); i++)
    {
        for (int j = 0; j < a.getShape(1); j++)
        {
            for (int k = 0; k < a.getShape(2); k++)
            {
                a[i][j][k] = 2.0 * b[i][j][k];
                sum += a[i][j][k];
           }
        }
    }
    std::cout << "test! " << test << "\n";
    std::cout << "x! " << x << "\n";
    return sum;
}

BatchMCTS* createBatchMCTS(numpyArray<int> array1, numpyArray<int> array2) {
    Ndarray<int,3> a(array1);
    Ndarray<int,2> b(array2);
    std::string out = "output";
    BatchMCTS* m = new BatchMCTS(1600, 1.0, true, out, 4, 16, 2, 0.05, a, b);
    return m;
}
} // end extern "C"
