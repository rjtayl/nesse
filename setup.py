from setuptools import setup
from Cython.Build import cythonize
import numpy
import sys

# Apparently building cython code on windows is a massive pain and pyMVSC at least makes your build environment correct
# this is mostly so that you can develop the code on windows without rebuilding the entire thing constantly.
if sys.platform.startswith('win'):
    import pyMSVC
    environment = pyMSVC.setup_environment()

setup(
    ext_modules=cythonize("src/nesse/interp.pyx"),
    include_dirs=[numpy.get_include()],
)