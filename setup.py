from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

ext = [Extension('interp',sources=["interp.pyx"], include_dirs=[numpy.get_include()])]

setup(
  name = 'interp',
    ext_modules = cythonize(ext)
)
