from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("two.pyx", annotate=True),
)
