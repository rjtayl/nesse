import sys
import os

from setuptools import Extension, setup
import numpy

try:
    from Cython.Build import cythonize
    USE_CYTHON = True
except ImportError:
    USE_CYTHON = False

ext = '.pyx' if USE_CYTHON else '.c'

extensions = [Extension("nesse.interp", ["src/nesse/interp"+ext])]

if USE_CYTHON:
    extensions = cythonize(extensions)

setup(
    ext_modules = extensions,
    include_dirs=[numpy.get_include()],
)