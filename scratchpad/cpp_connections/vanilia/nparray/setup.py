from setuptools import setup, Extension # use setuptools instead of distutils from tutorial
import numpy as np

"""
Use this before:

export LD_PRELOAD=/opt/intel/mkl/lib/intel64/libmkl_def.so:/opt/intel/mkl/lib/intel64/libmkl_avx2.so:/opt/intel/mkl/lib/intel64/libmkl_core.so:/opt/intel/mkl/lib/intel64/libmkl_intel_lp64.so:/opt/intel/mkl/lib/intel64/libmkl_intel_thread.so:/usr/lib/libomp.so
"""

extra_link_args = ['-I', '/opt/intel/mkl/include'
                  , '-L', '/opt/intel/mkl/lib/intel64/'
                   , '-Wl,--no-as-needed'
                   , '-lmkl_intel_lp64'
                   , '-lmkl_gnu_thread'
                   , '-lmkl_core'
                   , '-lpthread'
                   , '-lgomp'
                   , '-lm'
                   , '-ldl'
                  ]

extra_compile_args = ['-I','/opt/intel/mkl/include'
                      ,'-std=c++11'
                      ,'-m64'
                      ,'-fopenmp'
                     ]

module  = Extension('tcontract'
                    , sources=['tcontract.cpp']
                    , include_dirs=[np.get_include()]
                    , extra_compile_args=extra_compile_args
                    , extra_link_args=extra_link_args
                   )

setup(
    name='tcontract',
    version='0.0.0',
    description='Contract two tensors',
    ext_modules=[module]
)

