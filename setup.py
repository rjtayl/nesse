import sys
import os

from setuptools import Extension, setup
import numpy
from Cython.Build import cythonize

extensions = cythonize([Extension("nesse.interp", ["src/nesse/interp.pyx"])])

setup(
    ext_modules = extensions,
    include_dirs=[numpy.get_include()],
)