import sys
import os

# Apparently building cython code on windows is a massive pain and pyMVSC at least makes your build environment correct
# this is mostly so that you can develop the code on windows without rebuilding the entire thing constantly.
if (
    sys.platform.startswith('win')
    and os.environ.get("NESSE_DEV_USE_PYMSVC") == "1"
):
    import pyMSVC
    environment = pyMSVC.setup_environment()

# from Cython.Build import cythonize
# import numpy
# from setuptools import Extension, setup

# setup(
#     ext_modules=cythonize("src/nesse/interp.pyx"),
#     include_dirs=[numpy.get_include()],
# )

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