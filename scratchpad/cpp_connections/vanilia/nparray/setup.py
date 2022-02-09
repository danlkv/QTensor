from setuptools import setup, Extension # use setuptools instead of distutils from tutorial
import numpy as np
import os

"""
Use this before:

export LD_PRELOAD=$MKLROOT/lib/intel64/libmkl_def.so:$MKLROOT/lib/intel64/libmkl_avx2.so:$MKLROOT/lib/intel64/libmkl_core.so:$MKLROOT/lib/intel64/libmkl_intel_lp64.so
"""

# Using gcc compiler requires linking to mkl_gnu_thread and gomp. 
# gomp is part of gcc toolset, and should resolve automatically.
# However it may be that it's not automatically found.

# In this case, run `ldd build/lib.linux-x86_64-3.8/tcontract.cpython-38-x86_64-linux-gnu.so`
# and see where does ldd resolve the gomp library.
# Put this path into LD_PRELOAD

# :/usr/lib/libomp.so

mklroot = '/'
mklinclude = mklroot + '/include'
mkllib = mklroot + '/lib/intel64'

extra_link_args = [ '-L', mkllib
                   , '-Wl,--no-as-needed'
                   , '-lmkl_intel_lp64'
                   , '-lmkl_core'
                   , '-lpthread'
                   , '-lmkl_gnu_thread'
                   , '-lgomp'
                   , '-lm'
                   , '-ldl'
                  ]

extra_compile_args = ['-I', mklinclude
                     ,'-std=c++11'
                     ,'-m64'
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

