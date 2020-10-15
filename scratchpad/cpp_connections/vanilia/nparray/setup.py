from setuptools import setup, Extension # use setuptools instead of distutils from tutorial
import numpy as np
import os

"""
Use this before:

export LD_PRELOAD=$MKLROOT/lib/intel64/libmkl_def.so:$MKLROOT/lib/intel64/libmkl_avx2.so:$MKLROOT/lib/intel64/libmkl_core.so:$MKLROOT/lib/intel64/libmkl_intel_lp64.so:$MKLROOT/lib/intel64/libmkl_intel_thread.so
"""

# :/usr/lib/libomp.so

mklroot = os.environ['MKLROOT']
mklinclude = mklroot + '/include'
mkllib = mklroot + '/lib/intel64'

extra_link_args = ['-I', mklinclude
                  , '-L', mkllib
                   , '-Wl,--no-as-needed'
                   , '-lmkl_intel_lp64'
                   , '-lmkl_gnu_thread'
                   , '-lmkl_core'
                   , '-lpthread'
                   , '-lgomp'
                   , '-lm'
                   , '-ldl'
                  ]

extra_compile_args = ['-I', mklinclude
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

