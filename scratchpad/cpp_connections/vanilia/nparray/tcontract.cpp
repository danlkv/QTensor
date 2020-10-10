#include "Python.h"
#include <iostream>
#include "numpy/arrayobject.h"
#include <math.h>
#include <chrono>
#include <complex>

#include "mkl.h"

using namespace std::chrono;
using namespace std;


// Helper function to parse numpy arguments complex-valued matrices A, B and C
//
int python_abc_complex_args(PyObject *dummy, PyObject *args, PyObject **Obj, std::complex<double> **Data) {
    PyObject *argA, *argB, *argC;
    int fail = 0;

    if (!PyArg_ParseTuple(args, "OOO!", &argA, &argB,
        &PyArray_Type, &argC)) return 1;

    Obj[0]= PyArray_FROM_OTF(argA, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    if (Obj[0] == NULL) fail = 1;
    Obj[1] = PyArray_FROM_OTF(argB, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    if (Obj[1] == NULL) fail = 1;
#if NPY_API_VERSION >= 0x0000000c
    Obj[2] = PyArray_FROM_OTF(argC, NPY_COMPLEX128, NPY_ARRAY_INOUT_ARRAY2);
#else
    Obj[2] = PyArray_FROM_OTF(argC, NPY_COMPLEX128, NPY_ARRAY_INOUT_ARRAY);
#endif
    if (Obj[2] == NULL)  fail = 1;

    if (fail != 0) {
        for (int i=0; i<3; i++) {
            Py_XDECREF(Obj[i]);
        }
        return fail;
    }else{

        for (int i=0; i<3; i++) {
            Data[i] = (std::complex<double> *)PyArray_DATA( Obj[i] ); 
        }
        return 0;
    }
}
//

// DEBUG
static PyObject *
debug_mkl_contract_sum(PyObject *dummy, PyObject *args)
{
    std::complex<double> alpha(1, 0);
    std::complex<double> beta(0, 0);
    // -- Parse Python arguments
    PyObject *Obj[3];
    std::complex<double> *Data[3];
    int parse_fail;
    parse_fail = python_abc_complex_args(dummy, args, Obj, Data);

    if (parse_fail != 0) {
        std::cerr << "Failed to parse arguments" << std::endl;
        return NULL;
    }
    // -- 
    PyObject *A, *B, *C;
    A = Obj[0]; B = Obj[1]; C = Obj[2];

    std::complex<double> *Aptr, *Bptr, *Cptr;
    Aptr = Data[0]; Bptr = Data[1]; Cptr = Data[2]; 
    npy_intp *dimC = PyArray_DIMS(C);
    npy_intp *dimA = PyArray_DIMS(A);

    int m = dimC[1]; // Row length of A, third index
    int n = dimC[2]; // Row length of B, third index
    int k = dimA[0]; // Summation length, first index of A and B
    int f = dimA[1]; // Multiplication-only index, second index of A and B

    std::cerr << "Dimensions: f:" << f << " k:" << k << " n:" << n << " m:" << m << std::endl;
    auto start = high_resolution_clock::now();
    /*
     * Performs opearation
     * \sum_k A_{kfm} * B_{kfn} = C_{fmn}
     */

    for (int i=0; i<f; i++){
       cblas_zgemm(CblasRowMajor,
                CblasTrans,
                CblasNoTrans,
                m, n, k, &alpha,
                Aptr + i*m, f*m,
                Bptr + i*n, f*n, &beta,
                Cptr + i*n*m, n); 
    }
    auto stop = high_resolution_clock::now();
    auto millis = duration_cast<milliseconds>(stop - start).count();
    std::cerr << "Duration: " << millis << " milliseconds" << std::endl;

    /*
     * Works as well:
       cblas_zgemm(CblasColMajor,
                CblasNoTrans,
                CblasTrans,
                n, m, k, &alpha,
                Bptr + i*n, f*n,
                Aptr + i*m, f*m,
                &beta,
                Cptr + i*n*m, n); 
    */
    // -- Clean up python pointers
    Py_DECREF(A);
    Py_DECREF(B);
    Py_DECREF(C);
    Py_INCREF(Py_None);
    return Py_None;
}


static PyObject *
mkl_contract_sum(PyObject *dummy, PyObject *args)
{
    std::complex<double> alpha(1, 0);
    std::complex<double> beta(0, 0);

    // -- Parse Python arguments
    PyObject *Obj[3];
    std::complex<double> *Data[3];
    int parse_fail;
    parse_fail = python_abc_complex_args(dummy, args, Obj, Data);

    if (parse_fail != 0) {
        std::cerr << "Failed to parse arguments" << std::endl;
        return NULL;
    }
    // -- 
    PyObject *A, *B, *C;
    A = Obj[0]; B = Obj[1]; C = Obj[2];
    std::complex<double> *Aptr, *Bptr, *Cptr;
    Aptr = Data[0]; Bptr = Data[1]; Cptr = Data[2]; 
    npy_intp *dimC = PyArray_DIMS(C);
    npy_intp *dimA = PyArray_DIMS(A);

    int m = dimC[1]; // Row length of A, third index
    int n = dimC[2]; // Row length of B, third index
    int k = dimA[0]; // Summation length, first index of A and B
    int f = dimA[1]; // Multiplication-only index, second index of A and B


    /*
     * Performs opearation
     * \sum_k A_{kfm} * B_{kfn} = C_{fmn}
     */

    for (int i=0; i<f; i++){
       cblas_zgemm(CblasRowMajor,
                CblasTrans,
                CblasNoTrans,
                m, n, k, &alpha,
                Aptr + i*m, f*m,
                Bptr + i*n, f*n, &beta,
                Cptr + i*n*m, n); 
    }
    // -- Clean up python pointers
    Py_DECREF(A);
    Py_DECREF(B);
    Py_DECREF(C);
    Py_INCREF(Py_None);
    return Py_None;
}

// DEBUG
static PyObject *
debug_mkl_contract_complex(PyObject *dummy, PyObject *args)
{
    std::complex<double> alpha(1, 0);
    std::complex<double> beta(0, 0);

    // -- Parse Python arguments
    PyObject *Obj[3];
    std::complex<double> *Data[3];
    int parse_fail;
    parse_fail = python_abc_complex_args(dummy, args, Obj, Data);

    if (parse_fail != 0) {
        std::cerr << "Failed to parse arguments" << std::endl;
        return NULL;
    }
    // -- 

    PyObject *A, *B, *C;
    A = Obj[0]; B = Obj[1]; C = Obj[2];

    std::complex<double> *Aptr, *Bptr, *Cptr;
    Aptr = Data[0]; Bptr = Data[1]; Cptr = Data[2]; 

    //auto now = high_resolution_clock::now();
    //auto millis = duration_cast<milliseconds>(now - epoch).count();
    //std::cout << "after convert. duration (μs) = " << millis << std::endl;
    
    npy_intp *dimC = PyArray_DIMS(C);

    std::cerr << "Dimensions: C[0]:" << dimC[0] << " C[1]:" << dimC[1] << " C[2]:" << dimC[2] << std::endl;
    auto start = high_resolution_clock::now();

    for (int i=0; i<dimC[0]; i++){
       cblas_zgemm(CblasColMajor,
                CblasNoTrans,
                CblasTrans,
                dimC[2], dimC[1], 1, &alpha,
                Bptr + i*dimC[2], dimC[2],
                Aptr + i*dimC[1], dimC[1], &beta,
                Cptr + i*dimC[2]*dimC[1], dimC[2]); 
    }

    auto stop = high_resolution_clock::now();
    auto millis = duration_cast<milliseconds>(stop - start).count();
    std::cerr << "Duration: " << millis << " milliseconds" << std::endl;

    Py_DECREF(A);
    Py_DECREF(B);
    Py_DECREF(C);
    Py_INCREF(Py_None);
    return Py_None;
}


static PyObject *
mkl_contract_complex(PyObject *dummy, PyObject *args)
{
    std::complex<double> alpha(1, 0);
    std::complex<double> beta(0, 0);

    // -- Parse Python arguments
    PyObject *Obj[3];
    std::complex<double> *Data[3];
    int parse_fail;
    parse_fail = python_abc_complex_args(dummy, args, Obj, Data);

    if (parse_fail != 0) {
        std::cerr << "Failed to parse arguments" << std::endl;
        return NULL;
    }
    // -- 

    PyObject *A, *B, *C;
    A = Obj[0]; B = Obj[1]; C = Obj[2];

    std::complex<double> *Aptr, *Bptr, *Cptr;
    Aptr = Data[0]; Bptr = Data[1]; Cptr = Data[2]; 

    //auto now = high_resolution_clock::now();
    //auto millis = duration_cast<milliseconds>(now - epoch).count();
    //std::cout << "after convert. duration (μs) = " << millis << std::endl;
    
    npy_intp *dimC = PyArray_DIMS(C);

    for (int i=0; i<dimC[0]; i++){
       cblas_zgemm(CblasColMajor,
                CblasNoTrans,
                CblasTrans,
                dimC[2], dimC[1], 1, &alpha,
                Bptr + i*dimC[2], dimC[2],
                Aptr + i*dimC[1], dimC[1], &beta,
                Cptr + i*dimC[2]*dimC[1], dimC[2]); 
    }

    Py_DECREF(A);
    Py_DECREF(B);
    Py_DECREF(C);
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *
mkl_dotmul(PyObject *dummy, PyObject *args)
{
    PyObject *argA=NULL, *argB, *argC;
    PyObject *A=NULL, *B, *C;
    std::complex<double> *Aptr, *Bptr, *Cptr;
    std::complex<double> alpha(1, 0);
    std::complex<double> beta(0, 0);

    int nd;
    npy_intp * dimC;
    npy_intp * dimA;

    if (!PyArg_ParseTuple(args, "OOO!", &argA, &argB,
        &PyArray_Type, &argC)) return NULL;

    A = PyArray_FROM_OTF(argA, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    if (A == NULL) return NULL;
    B = PyArray_FROM_OTF(argB, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    if (B == NULL) goto fail;
#if NPY_API_VERSION >= 0x0000000c
    C = PyArray_FROM_OTF(argC, NPY_COMPLEX128, NPY_ARRAY_INOUT_ARRAY2);
#else
    C = PyArray_FROM_OTF(argC, NPY_COMPLEX128, NPY_ARRAY_INOUT_ARRAY);
#endif
    if (C == NULL) goto fail;
    


    //auto now = high_resolution_clock::now();
    //auto millis = duration_cast<milliseconds>(now - epoch).count();
    
    nd = PyArray_NDIM(C);
    if (nd!=2) goto fail;
    dimC = PyArray_DIMS(C);
    dimA = PyArray_DIMS(A);
    Aptr = (std::complex<double> *)PyArray_DATA(A);
    Bptr = (std::complex<double> *)PyArray_DATA(B);
    Cptr = (std::complex<double> *)PyArray_DATA(C);
    std::cout << "A[0][1]" << Aptr[1] << std::endl;
    std::cout << "dimC" << dimC[0] << "," <<dimC[1] << std::endl;


   cblas_zgemm(CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                dimC[0], dimC[1], dimA[1], &alpha,
                Aptr, dimA[1],
                Bptr, dimC[1], &beta,
                Cptr, dimC[1]); 

    std::cout << "C[0][1]" << Cptr[1] << std::endl;


    Py_DECREF(A);
    Py_DECREF(B);
    Py_DECREF(C);
    Py_INCREF(Py_None);
    return Py_None;

 fail:
    Py_XDECREF(A);
    Py_XDECREF(B);
    Py_XDECREF(C);
    return NULL;

}

static PyObject *
mkl_contract(PyObject *dummy, PyObject *args)
{
    PyObject *argA=NULL, *argB, *argC;
    PyObject *A=NULL, *B, *C;
    double *Aptr, *Bptr, *Cptr;

    auto epoch = high_resolution_clock::now();
    int nd;
    npy_intp * dimC;

    if (!PyArg_ParseTuple(args, "OOO!", &argA, &argB,
        &PyArray_Type, &argC)) return NULL;

    A = PyArray_FROM_OTF(argA, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (A == NULL) return NULL;
    B = PyArray_FROM_OTF(argB, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (B == NULL) goto fail;
#if NPY_API_VERSION >= 0x0000000c
    C = PyArray_FROM_OTF(argC, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);
#else
    C = PyArray_FROM_OTF(argC, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY);
#endif
    if (C == NULL) goto fail;
    


    //auto now = high_resolution_clock::now();
    //auto millis = duration_cast<milliseconds>(now - epoch).count();
    //std::cout << "after convert. duration (μs) = " << millis << std::endl;
    
    nd = PyArray_NDIM(C);
    if (nd!=3) goto fail;
    dimC = PyArray_DIMS(C);
    Aptr = (double *)PyArray_DATA(A);
    Bptr = (double *)PyArray_DATA(B);
    Cptr = (double *)PyArray_DATA(C);

    for (int i=0; i<dimC[0]; i++){
//      for (int j=0; j<dimC[1]; j++){
//          for (int k=0; k<dimC[2]; k++){
//              Cptr[i*dimC[1]*dimC[2] + j*dimC[2] + k] = 
//                  Aptr[i*dimC[1] + j]*Bptr[i*dimC[2] + k];
//          }
//      }
       cblas_dgemm(CblasColMajor,
                CblasNoTrans,
                CblasTrans,
                dimC[2], dimC[1], 1, 1.0,
                Bptr + i*dimC[2], dimC[2],
                Aptr + i*dimC[1], dimC[1], 0.0,
                Cptr + i*dimC[2]*dimC[1], dimC[2]); 
    }


    /* code that makes use of arguments */
    /* You will probably need at least
       nd = PyArray_NDIM(<..>)    -- number of dimensions
       dims = PyArray_DIMS(<..>)  -- npy_intp array of length nd
                                     showing length in each dim.
       dptr = (double *)PyArray_DATA(<..>) -- pointer to data.

       If an error occurs goto fail.
     */

    Py_DECREF(A);
    Py_DECREF(B);
    Py_DECREF(C);
    Py_INCREF(Py_None);
    return Py_None;

 fail:
    Py_XDECREF(A);
    Py_XDECREF(B);
    Py_XDECREF(C);
    return NULL;

}

static PyObject *
triple_loop_contract(PyObject *dummy, PyObject *args)
{
    PyObject *argA=NULL, *argB, *argC;
    PyObject *A=NULL, *B, *C;
    double *Aptr, *Bptr, *Cptr;

    auto epoch = high_resolution_clock::now();
    int nd;
    npy_intp * dimC;

    if (!PyArg_ParseTuple(args, "OOO!", &argA, &argB,
        &PyArray_Type, &argC)) return NULL;

    A = PyArray_FROM_OTF(argA, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (A == NULL) return NULL;
    B = PyArray_FROM_OTF(argB, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (B == NULL) goto fail;
#if NPY_API_VERSION >= 0x0000000c
    C = PyArray_FROM_OTF(argC, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);
#else
    C = PyArray_FROM_OTF(argC, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY);
#endif
    if (C == NULL) goto fail;
    


    //auto now = high_resolution_clock::now();
    //auto millis = duration_cast<milliseconds>(now - epoch).count();
    //std::cout << "after convert. duration (μs) = " << millis << std::endl;
    
    nd = PyArray_NDIM(C);
    if (nd!=3) goto fail;
    dimC = PyArray_DIMS(C);
    Aptr = (double *)PyArray_DATA(A);
    Bptr = (double *)PyArray_DATA(B);
    Cptr = (double *)PyArray_DATA(C);

    for (int i=0; i<dimC[0]; i++){
        for (int j=0; j<dimC[1]; j++){
            for (int k=0; k<dimC[2]; k++){
                Cptr[i*dimC[1]*dimC[2] + j*dimC[2] + k] = 
                    Aptr[i*dimC[1] + j]*Bptr[i*dimC[2] + k];
            }
        }
    }



    Py_DECREF(A);
    Py_DECREF(B);
    Py_DECREF(C);
    Py_INCREF(Py_None);
    return Py_None;

 fail:
    Py_XDECREF(A);
    Py_XDECREF(B);
    Py_XDECREF(C);
    return NULL;
}

static PyObject *
print_4(PyObject *dummy, PyObject *args)
{
    PyObject *arg=NULL;
    PyObject *arr=NULL;
    double *dptr;

    if (!PyArg_ParseTuple(args, "O", &arg)) return NULL;

    auto epoch = high_resolution_clock::now();
    arr = PyArray_FROM_OTF(arg, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (arr == NULL) return NULL;
    auto now = high_resolution_clock::now();
    auto millis = duration_cast<milliseconds>(now - epoch).count();
    std::cout << "after convert. duration (μs) = " << millis << std::endl;
    

    dptr = (double *)PyArray_DATA(arr);
    std::cout << "arr[0] = " << *dptr << std::endl;
    std::cout << "arr[1] = " << *(dptr+1) << std::endl;
    std::cout << "arr[2] = " << *(dptr+2) << std::endl;
    std::cout << "arr[3] = " << *(dptr+3) << std::endl;


    Py_DECREF(arr);
    Py_INCREF(Py_None);
    return Py_None;

}

// -- Examples

static PyObject * integrate3(PyObject * module, PyObject * args)
{
    PyObject * argy=NULL;        // Regular Python/C API
    PyArrayObject * yarr=NULL;   // Extended Numpy/C API
    double dx,dy,dz;

    std::cout << "in func" <<std::endl;

    // "O" format -> read argument as a PyObject type into argy (Python/C API)
    if (!PyArg_ParseTuple(args, "Oddd", &argy,&dx,&dy,&dz))
    {
        PyErr_SetString(PyExc_ValueError, "Error parsing arguments.");
        return NULL;
    }

    std::cout << "parsed" << std::endl;
    // Determine if it's a complex number array (Numpy/C API)
    int DTYPE = PyArray_ObjectType(argy, NPY_FLOAT); 
    int iscomplex = PyTypeNum_ISCOMPLEX(DTYPE);      
    std::cout << "Is complex" << iscomplex << std::endl;

    // parse python object into numpy array (Numpy/C API)
    yarr = (PyArrayObject *)PyArray_FROM_OTF(argy, DTYPE, NPY_ARRAY_IN_ARRAY);
    if (yarr==NULL) {
        Py_INCREF(Py_None);
        return Py_None;
    }

    //just assume this for 3 dimensional array...you can generalize to N dims
    if (PyArray_NDIM(yarr) != 3) {
        Py_CLEAR(yarr);
        PyErr_SetString(PyExc_ValueError, "Expected 3 dimensional integrand");
        return NULL;
    }

    npy_intp * dims = PyArray_DIMS(yarr);
    npy_intp i,j,k,m;
    double * p;

    //initialize variable to hold result
    Py_complex result = {.real = 0, .imag = 0};
    std::cout << "Is complex" << iscomplex << std::endl;

    if (iscomplex) {
        for (i=0;i<dims[0];i++) 
            for (j=0;j<dims[1];j++) 
                for (k=0;k<dims[1];k++) {
                    p = (double*)PyArray_GETPTR3(yarr, i,j,k);
                    result.real += *p;
                    result.imag += *(p+1);
                }
    } else {
        for (i=0;i<dims[0];i++) 
            for (j=0;j<dims[1];j++) 
                for (k=0;k<dims[1];k++) {
                    p = (double*)PyArray_GETPTR3(yarr, i,j,k);
                    result.real += *p;
                }
    }

    //multiply by step size
    result.real *= (dx*dy*dz);
    result.imag *= (dx*dy*dz);

    Py_CLEAR(yarr);

    //copy result into returnable type with new reference
    if (iscomplex) {
        return Py_BuildValue("D", &result);
    } else {
        return Py_BuildValue("d", result.real);
    }

};

static PyObject *
example_wrapper(PyObject *dummy, PyObject *args)
{
    PyObject *arg1=NULL, *arg2=NULL, *out=NULL;
    PyObject *arr1=NULL, *arr2=NULL, *oarr=NULL;
    double *dptr;

    if (!PyArg_ParseTuple(args, "OOO!", &arg1, &arg2,
        &PyArray_Type, &out)) return NULL;

    arr1 = PyArray_FROM_OTF(arg1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (arr1 == NULL) return NULL;
    arr2 = PyArray_FROM_OTF(arg2, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (arr2 == NULL) goto fail;
#if NPY_API_VERSION >= 0x0000000c
    oarr = PyArray_FROM_OTF(out, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);
#else
    oarr = PyArray_FROM_OTF(out, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY);
#endif
    if (oarr == NULL) goto fail;
    

    std::cout << "arr1" << std::endl;
    dptr = (double *)PyArray_DATA(arr1);
    std::cout << "arrval = " << *dptr << std::endl;
    std::cout << "arrval = " << *(dptr+1) << std::endl;
    std::cout << "arrval = " << *(dptr+2) << std::endl;
    std::cout << "arrval = " << *dptr+3 << std::endl;
    std::cout << "arrval = " << *dptr+4 << std::endl;


    /* code that makes use of arguments */
    /* You will probably need at least
       nd = PyArray_NDIM(<..>)    -- number of dimensions
       dims = PyArray_DIMS(<..>)  -- npy_intp array of length nd
                                     showing length in each dim.
       dptr = (double *)PyArray_DATA(<..>) -- pointer to data.

       If an error occurs goto fail.
     */

    Py_DECREF(arr1);
    Py_DECREF(arr2);
    Py_DECREF(oarr);
    Py_INCREF(Py_None);
    return Py_None;

 fail:
    Py_XDECREF(arr1);
    Py_XDECREF(arr2);
    Py_XDECREF(oarr);
    return NULL;
}

// --

static PyMethodDef tcontract_Methods[] = {
    {"integrate3",  integrate3, METH_VARARGS,
     "Pass 3D numpy array (double or complex) and dx,dy,dz step size. Returns Reimman integral"},
    {"example",  example_wrapper, METH_VARARGS,
     "Example from https://numpy.org/doc/stable/user/c-info.how-to-extend.html"},
    {"print_4",  print_4, METH_VARARGS,
     "Prints first 4 values of numpy array"},
    {"mkl_dotmul",  mkl_dotmul, METH_VARARGS,
     "Matrix multiplication"},

    {"triple_loop_contract",  triple_loop_contract, METH_VARARGS,
     "Contracts two arrays with first common index"},
    {"mkl_contract",  mkl_contract, METH_VARARGS,
     "Contracts two arrays with first common index using MKL"},
    {"mkl_contract_complex",  mkl_contract_complex, METH_VARARGS,
     "Contracts two arrays with first common index using MKL"},
    {"mkl_contract_sum",  mkl_contract_sum, METH_VARARGS,
     "Performs opearation:\
     \\sum_k A_{kfm} * B_{kfn} = C_{fmn}"},

    {"debug_mkl_contract_sum",  debug_mkl_contract_sum, METH_VARARGS,
     "DEBUG Performs opearation:\
     \\sum_k A_{kfm} * B_{kfn} = C_{fmn}"},
    {"debug_mkl_contract_complex",  debug_mkl_contract_complex, METH_VARARGS,
     "DEBUG Contracts two arrays with first common index using MKL"},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};


static struct PyModuleDef module = {
   PyModuleDef_HEAD_INIT,
   "tcontract",   /* name of module */
   NULL, /* module documentation, may be NULL */
   -1,       /* size of per-interpreter state of the module,
                or -1 if the module keeps state in global variables. */
   tcontract_Methods
};

PyMODINIT_FUNC
PyInit_tcontract(void)
{
    // Called on import
    // Returns a pointer to module, which is insected into `sys.modules`
    import_array(); // needed for numpy to work
    return PyModule_Create(&module);
}
