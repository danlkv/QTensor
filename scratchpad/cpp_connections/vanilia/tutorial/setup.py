from setuptools import setup, Extension # use setuptools instead of distutils from tutorial

module  = Extension('spam', sources=['spammodule.c'])

setup(
    name='spam',
    version='0.0.0',
    description='Binding python to cpp',
    ext_modules=[module]
)
