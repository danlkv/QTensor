from setuptools import setup, Extension # use setuptools instead of distutils from tutorial
import numpy as np

module  = Extension('tcontract'
                    , sources=['tcontract.cpp']
                    , include_dirs=[np.get_include()]
                   )

setup(
    name='tcontract',
    version='0.0.0',
    description='Contract two tensors',
    ext_modules=[module]
)
