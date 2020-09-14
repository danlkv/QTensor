//-- Pull the Python API
#define PY_SSIZE_T_CLEAN // What this does? https://docs.python.org/3/extending/extending.html#parsetuple
#include <Python.h>
//--

// The self argument points to the module object for module-level functions
static PyObject *
spam_system(PyObject *self, PyObject *args) {
    const char * command;
    int sts;

    // int PyArg_ParseTuple(PyObject *args, const char *format, ...)
    if (!PyArg_ParseTuple(args, "s", &command))
        return NULL;
    // Python.h above includes stdlib.h and other libraries
    sts = system(command);
    // Will return an integer object. (Yes, even integers are objects on the heap in Python!)
    return PyLong_FromLong(sts);
}

// "Method table"
static PyMethodDef SpamMethods[] = {
    {"system",  spam_system, METH_VARARGS, "Execute a shell command."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

// Module definition struct
static struct PyModuleDef spammodule = {
    PyModuleDef_HEAD_INIT,
    "spam",   /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    SpamMethods
};

/*
 PyMODINIT_FUNC declares the function as PyObject * return type,
 declares any special linkage declarations required by the platform, and for
 C++ declares the function as extern "C".  PyMODINIT_FUNC
*/
PyMODINIT_FUNC
PyInit_spam(void)
{
    // Called on import
    // Returns a pointer to module, which is insected into `sys.modules`
    return PyModule_Create(&spammodule);
}
