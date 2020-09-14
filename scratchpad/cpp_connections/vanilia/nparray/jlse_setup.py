from setuptools import setup, Extension # use setuptools instead of distutils from tutorial
import numpy as np

extra_link_args = ['-I', '/soft/compilers/intel-2019/compilers_and_libraries/linux/mkl/include', '-l', 'mkl_intel_lp64', '-l', 'mkl_intel_thread', '-l', 'mkl_core', '-l', 'iomp5', '-l', 'pthread', '-l', 'm', '-l', 'dl', '-L', '/soft/compilers/intel-2019/compilers_and_libraries/linux/mkl/lib/intel64', '-L', '/soft/compilers/intel-2019/compilers_and_libraries/linux/mkl/../compiler/lib/intel64']

module  = Extension('tcontract'
                    , sources=['tcontract.cpp']
                    , include_dirs=[np.get_include(), '/soft/compilers/intel-2019/compilers_and_libraries/linux/mkl/include']
                    , extra_link_args=extra_link_args
                    , extra_compile_args=extra_link_args
                   )

setup(
    name='tcontract',
    version='0.0.0',
    description='Contract two tensors',
    ext_modules=[module]
)
